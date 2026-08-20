//! A small ONNX-subset executor on candle.
//!
//! ## Why an interpreter and not three hand-written ports
//!
//! Carmenta implements two stages of the five-stage document pipeline the field
//! uses (§44): text detection and text recognition. The three missing stages —
//! LAYOUT, FORMULA and TABLE — all ship as ONNX, and all three are ordinary
//! CNN/transformer graphs over the same primitives `svtr.rs` already
//! implements. Hand-transcribing 400+ nodes three times is the error class
//! §8.167 documents: two of the three hypotheses that died in the SVTR port
//! "matched every shape" and were still structurally wrong.
//!
//! So the graph is DATA, not code. `.tools-bench/onnx_export.py` reads the ONNX
//! protobuf with no framework installed and emits `<name>_arch.json` (nodes in
//! topological order) plus `<name>.safetensors` (every initializer, including
//! the ones exports inline as `Constant` nodes). This module walks that.
//!
//! ## What it deliberately does NOT do
//!
//! Detection post-processing — NMS, top-k, box packing — is the tail of the
//! graph (nodes 378..411 of PP-DocLayout-S) and is far clearer written directly
//! in Rust than interpreted. The executor stops at the tensors that feed NMS and
//! hands them back; `doclayout.rs` finishes the job.

use candle_core::{DType, Device, Result as CResult, Tensor};
use std::collections::HashMap;

/// Hard cap on Loop iterations. The table decoder emits at most 500 structure
/// tokens and the formula decoder a similar order; an unbounded interpreter
/// loop on a malformed graph is a hang, not an error.
pub const LOOP_CAP: i64 = 600;

#[derive(serde::Deserialize)]
pub struct Body {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub nodes: Vec<Node>,
    #[serde(default)]
    pub alias: HashMap<String, String>,
}

#[derive(serde::Deserialize)]
pub struct Node {
    pub op: String,
    #[serde(rename = "in")]
    pub inputs: Vec<String>,
    #[serde(rename = "out")]
    pub outputs: Vec<String>,
    #[serde(default)]
    pub attr: HashMap<String, serde_json::Value>,
    /// `Loop` body / `If` branches — the decoder lives here, not in the top
    /// level. SLANet's is 144 nodes; PP-FormulaNet's is 1743.
    #[serde(default)]
    pub body: Option<Box<Body>>,
    #[serde(default)]
    pub then_branch: Option<Box<Body>>,
    #[serde(default)]
    pub else_branch: Option<Box<Body>>,
}

#[derive(serde::Deserialize)]
pub struct Arch {
    pub name: String,
    #[serde(default)]
    pub alias: HashMap<String, String>,
    pub nodes: Vec<Node>,
}

impl Node {
    fn ints(&self, k: &str) -> Option<Vec<usize>> {
        self.attr.get(k)?.as_array().map(|a| {
            a.iter().map(|v| v.as_i64().unwrap_or(0).max(0) as usize).collect()
        })
    }
    fn int(&self, k: &str, d: i64) -> i64 {
        self.attr.get(k).and_then(|v| v.as_i64()).unwrap_or(d)
    }
    fn float(&self, k: &str, d: f64) -> f64 {
        self.attr.get(k).and_then(|v| v.as_f64()).unwrap_or(d)
    }
    fn string(&self, k: &str) -> Option<&str> {
        self.attr.get(k).and_then(|v| v.as_str())
    }

    /// Explicit `[top, left, bottom, right]` padding for a windowed op.
    ///
    /// ONNX lets a Conv/Pool state its padding EITHER as `pads` or as
    /// `auto_pad: SAME_UPPER|SAME_LOWER`, and the second form carries no `pads`
    /// at all — so an executor that reads only `pads` silently treats it as
    /// VALID (no padding) and every such layer comes out one or two pixels
    /// small. In PP-FormulaNet's stem that surfaced as
    /// `Concat: shape mismatch, [1,48,191,191] vs [1,48,190,190]`, two branches
    /// that are supposed to be identical; on a net where the branches happen to
    /// agree it would have been a silent crop instead.
    fn pad4(&self, ks: &[usize], st: &[usize], dil: &[usize], hw: (usize, usize)) -> [usize; 4] {
        match self.string("auto_pad") {
            Some(ap @ ("SAME_UPPER" | "SAME_LOWER")) => {
                let mut out = [0usize; 4];
                for i in 0..2 {
                    let inp = if i == 0 { hw.0 } else { hw.1 };
                    let (k, s) = (*ks.get(i).unwrap_or(&1), *st.get(i).unwrap_or(&1));
                    let eff = (k - 1) * *dil.get(i).unwrap_or(&1) + 1;
                    let need = (inp.div_ceil(s) - 1) * s + eff;
                    let tot = need.saturating_sub(inp);
                    // SAME_UPPER puts the extra pixel at the END, SAME_LOWER at
                    // the start; on an odd total the two differ by one row.
                    let (b, e) = if ap == "SAME_UPPER" {
                        (tot / 2, tot - tot / 2)
                    } else {
                        (tot - tot / 2, tot / 2)
                    };
                    out[i] = b;
                    out[i + 2] = e;
                }
                out
            }
            _ => {
                let p = self.ints("pads").unwrap_or_default();
                if p.len() >= 4 { [p[0], p[1], p[2], p[3]] } else { [0; 4] }
            }
        }
    }
}

/// Apply `[top, left, bottom, right]` to a NCHW tensor. Padding is materialised
/// rather than handed to candle's single symmetric `pad` argument, because ONNX
/// padding is per-EDGE and the asymmetric case is common once `auto_pad` is in
/// play.
fn pad_nchw(a: &Tensor, p: [usize; 4]) -> CResult<Tensor> {
    let a = if p[0] + p[2] > 0 { a.pad_with_zeros(2, p[0], p[2])? } else { a.clone() };
    if p[1] + p[3] > 0 { a.pad_with_zeros(3, p[1], p[3]) } else { Ok(a) }
}

/// A stack of alias maps, outermost first; `resolve` searches it innermost-out
/// and then falls back to the graph's own map.
type Scope<'a> = &'a [&'a HashMap<String, String>];

pub struct Graph {
    pub arch: Arch,
    pub weights: HashMap<String, Tensor>,
}

impl Graph {
    /// Resolve an Identity chain. The exporter folds `Identity` nodes into an
    /// alias map rather than executing them; every consumer name must be walked
    /// back to the tensor that actually produced it.
    fn resolve<'a>(&'a self, name: &'a str, scope: Scope<'a>) -> &'a str {
        let mut n = name;
        for _ in 0..32 {
            // Innermost scope first, then each ENCLOSING one, then the graph's.
            // Walking only (innermost, graph) skips the intermediate levels: an
            // `If` nested in a `Loop` saw its own 10 aliases and the top-level
            // 286, but not the loop body's 792, so a name defined by the body
            // and read inside the branch came back missing.
            match scope
                .iter()
                .rev()
                .find_map(|m| m.get(n))
                .or_else(|| self.arch.alias.get(n))
            {
                Some(next) => n = next.as_str(),
                None => break,
            }
        }
        n
    }

    fn get_s(
        &self,
        env: &HashMap<String, Tensor>,
        name: &str,
        scope: Scope<'_>,
    ) -> CResult<Tensor> {
        // A BOUND NAME SHADOWS ITS ALIAS — and the check has to happen at EVERY
        // hop of the chain, not just at the ends.
        //
        // Inside a Loop body the carried buffers are bound under their formal
        // names, but those names are themselves `Identity` aliases of the
        // tensors the caller passed on iteration zero, which are still present
        // in the cloned enclosing environment. Resolving the chain to its root
        // first walks straight past the live formal and lands on the stale
        // original, so the body restarts from the initial value every step: the
        // table decoder's [1,501,50] logit buffer held exactly one row for all
        // 500 iterations, and PP-FormulaNet's token buffer and KV caches grew
        // from [1,1] to [1,2] and then froze for all 600.
        //
        // Binding each formal under its root name as well is NOT the fix: the
        // roots collide. Four of PP-FormulaNet's KV caches resolve to the same
        // `Expand.5`, so that collapses four distinct buffers into one. Stopping
        // the walk at the first bound name is what shadowing actually means.
        let mut n = name;
        for _ in 0..32 {
            if let Some(t) = env.get(n) {
                return Ok(t.clone());
            }
            if let Some(t) = self.weights.get(n) {
                return Ok(t.clone());
            }
            match scope
                .iter()
                .rev()
                .find_map(|m| m.get(n))
                .or_else(|| self.arch.alias.get(n))
            {
                Some(next) => n = next.as_str(),
                None => break,
            }
        }
        Err(candle_core::Error::Msg(format!("missing tensor `{name}` (-> `{n}`)")))
    }

    fn get(&self, env: &HashMap<String, Tensor>, name: &str) -> CResult<Tensor> {
        self.get_s(env, name, &[])
    }

    /// Run nodes `[0, stop)` and return the environment.
    ///
    /// `stop` exists because the graph tail is detection post-processing; the
    /// caller names the last node it wants and finishes in Rust.
    pub fn run(
        &self,
        inputs: HashMap<String, Tensor>,
        stop: usize,
        dev: &Device,
    ) -> CResult<HashMap<String, Tensor>> {
        let mut env = inputs;
        for (i, n) in self.arch.nodes.iter().enumerate() {
            if i >= stop {
                break;
            }
            let out = self.exec(n, &env, dev).map_err(|e| {
                candle_core::Error::Msg(format!("node {i} `{}`: {e}", n.op))
            })?;
            for (name, t) in n.outputs.iter().zip(out) {
                env.insert(name.clone(), t);
            }
        }
        Ok(env)
    }

    /// Execute a subgraph (a `Loop` body or an `If` branch) over its own scope.
    ///
    /// The body sees the enclosing environment plus its formal inputs bound to
    /// the caller's actuals, and its own `Identity` aliases layered on top.
    fn run_body(
        &self,
        b: &Body,
        binds: Vec<(String, Tensor)>,
        outer: &HashMap<String, Tensor>,
        dev: &Device,
        parent: Scope<'_>,
    ) -> CResult<Vec<Tensor>> {
        let mut env = outer.clone();
        for (k, v) in binds {
            env.insert(k, v);
        }
        // the enclosing chain, with this body's own aliases innermost
        let mut chain: Vec<&HashMap<String, String>> = parent.to_vec();
        chain.push(&b.alias);
        let inner: Scope<'_> = &chain;
        let dbg = std::env::var("FFAI_BODY_DEBUG").is_ok();
        for (bi, n) in b.nodes.iter().enumerate() {
            let out = self.exec_in(n, &env, dev, &inner).map_err(|e| {
                candle_core::Error::Msg(format!("body node {bi} `{}`: {e}", n.op))
            })?;
            if dbg {
                let d: Vec<String> = out.iter().map(|t| format!("{:?}", t.dims())).collect();
                eprintln!("  b{bi:<4} {:<22} {}", n.op, d.join(" "));
            }
            for (name, t) in n.outputs.iter().zip(out) {
                env.insert(name.clone(), t);
            }
        }
        // Same shadow-aware walk the body's own reads use: an output name can
        // be an alias chain, and it can name an INITIALIZER rather than
        // anything the branch computed — an `If` arm that returns a constant
        // does exactly that.
        b.outputs
            .iter()
            .map(|o| {
                self.get_s(&env, o, inner)
                    .map_err(|_| candle_core::Error::Msg(format!("body output `{o}` missing")))
            })
            .collect()
    }

    fn exec(
        &self,
        n: &Node,
        env: &HashMap<String, Tensor>,
        dev: &Device,
    ) -> CResult<Vec<Tensor>> {
        self.exec_in(n, env, dev, &[])
    }

    fn exec_in(
        &self,
        n: &Node,
        env: &HashMap<String, Tensor>,
        dev: &Device,
        scope: Scope<'_>,
    ) -> CResult<Vec<Tensor>> {
        let x = |k: usize| self.get_s(env, &n.inputs[k], scope);
        // control flow first — these consult subgraphs, not tensors
        match n.op.as_str() {
            "If" => {
                let c = x(0)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .first()
                    .copied()
                    .unwrap_or(0.0);
                let br = if c != 0.0 { n.then_branch.as_ref() } else { n.else_branch.as_ref() };
                let br = br.ok_or_else(|| candle_core::Error::Msg("If without branch".into()))?;
                return self.run_body(br, Vec::new(), env, dev, scope);
            }
            "Loop" => {
                let body = n
                    .body
                    .as_ref()
                    .ok_or_else(|| candle_core::Error::Msg("Loop without body".into()))?;
                // ONNX Loop: inputs are (M, cond, v_initial...); an empty name
                // means "absent". The body takes (iter, cond, v...) and returns
                // (cond, v..., scan...).
                let max_trip = if !n.inputs[0].is_empty() {
                    self.get_s(env, &n.inputs[0], scope)
                        .and_then(|t| Ok(t.to_dtype(DType::I64)?.flatten_all()?.to_vec1::<i64>()?[0]))
                        .unwrap_or(i64::MAX)
                } else {
                    i64::MAX
                };
                let mut cond = if n.inputs.len() > 1 && !n.inputs[1].is_empty() {
                    self.get_s(env, &n.inputs[1], scope)?
                } else {
                    Tensor::from_vec(vec![1i64], 1, dev)?
                };
                let mut carried: Vec<Tensor> = Vec::new();
                for i in 2..n.inputs.len() {
                    carried.push(self.get_s(env, &n.inputs[i], scope)?);
                }
                if std::env::var("FFAI_LOOP_DEBUG").is_ok() {
                    for (i, c) in carried.iter().enumerate().take(8) {
                        eprintln!("  carry[{i}] `{}` {:?}", n.inputs[i + 2], c.dims());
                    }
                }
                let cap = max_trip.min(LOOP_CAP);
                let mut scans: Vec<Vec<Tensor>> = Vec::new();
                for it in 0..cap {
                    let on = cond
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?
                        .first()
                        .copied()
                        .unwrap_or(0.0);
                    if on == 0.0 {
                        break;
                    }
                    let mut binds: Vec<(String, Tensor)> = Vec::new();
                    if !body.inputs.is_empty() {
                        binds.push((
                            body.inputs[0].clone(),
                            Tensor::from_vec(vec![it], 1, dev)?,
                        ));
                    }
                    if body.inputs.len() > 1 {
                        binds.push((body.inputs[1].clone(), cond.clone()));
                    }
                    for (k, v) in body.inputs.iter().skip(2).zip(carried.iter()) {
                        binds.push((k.clone(), v.clone()));
                    }
                    let out = self.run_body(body, binds, env, dev, scope)?;
                    if out.is_empty() {
                        break;
                    }
                    if std::env::var("FFAI_LOOP_DEBUG").is_ok() && (it < 2 || it % 150 == 0 || it > 497) {
                        let stat = |t: &Tensor| -> String {
                            match t.to_dtype(DType::F32).and_then(|x| x.flatten_all())
                                   .and_then(|x| x.to_vec1::<f32>()) {
                                Ok(v) => {
                                    let (mut mn, mut mx, mut s) = (f32::MAX, f32::MIN, 0f32);
                                    for &q in &v { mn = mn.min(q); mx = mx.max(q); s += q.abs(); }
                                    format!("{:?} min{:.3} max{:.3} absum{:.3}", t.dims(), mn, mx, s)
                                }
                                Err(_) => format!("{:?}", t.dims()),
                            }
                        };
                        eprintln!("  loop it{it}: cond={} outs={}", stat(&out[0]), out.len());
                        for (j, o) in out.iter().enumerate().skip(1).take(9) {
                            eprintln!("     out[{j}] {}", stat(o));
                        }
                    }
                    cond = out[0].clone();
                    let n_carry = carried.len().min(out.len().saturating_sub(1));
                    carried = out[1..1 + n_carry].to_vec();
                    let sc: Vec<Tensor> = out[1 + n_carry..].to_vec();
                    if !sc.is_empty() {
                        scans.push(sc);
                    }
                }
                let mut res = carried;
                if !scans.is_empty() {
                    let k = scans[0].len();
                    for j in 0..k {
                        let parts: Vec<Tensor> = scans
                            .iter()
                            .filter_map(|s| s.get(j).cloned())
                            .map(|t| t.unsqueeze(0))
                            .collect::<CResult<Vec<_>>>()?;
                        res.push(Tensor::cat(&parts, 0)?);
                    }
                }
                return Ok(res);
            }
            _ => {}
        }
        Ok(match n.op.as_str() {
            "Conv" => {
                let (a, w) = (x(0)?, x(1)?);
                let st = n.ints("strides").unwrap_or(vec![1, 1]);
                let dil = n.ints("dilations").unwrap_or(vec![1, 1]);
                let g = n.int("group", 1).max(1) as usize;
                let wd = w.dims();
                let ks = n.ints("kernel_shape").unwrap_or(wd[2..].to_vec());
                let ad = a.dims();
                let p = n.pad4(&ks, &st, &dil, (ad[ad.len() - 2], ad[ad.len() - 1]));
                // candle takes ONE symmetric padding; every ONNX form — explicit
                // per-edge `pads` and `auto_pad` alike — is materialised first.
                let a = pad_nchw(&a, p)?;
                let mut y = a.conv2d(&w, 0, st[0], dil[0], g)?;
                if n.inputs.len() > 2 {
                    let b = x(2)?;
                    let c = b.elem_count();
                    y = y.broadcast_add(&b.reshape((1, c, 1, 1))?)?;
                }
                vec![y]
            }
            // The transformer decoder's normaliser. Unlike BatchNormalization
            // this reduces over the TRAILING dims from `axis` on, using the
            // batch's own statistics rather than stored running estimates.
            "LayerNormalization" => {
                let a = x(0)?.to_dtype(DType::F32)?;
                let eps = n.float("epsilon", 1e-5);
                let rank = a.rank();
                let ax = n.int("axis", -1);
                let ax = if ax < 0 { (rank as i64 + ax) as usize } else { ax as usize };
                let red = |t: &Tensor| -> CResult<Tensor> {
                    let mut r = t.clone();
                    for d in ax..rank {
                        r = r.mean_keepdim(d)?;
                    }
                    Ok(r)
                };
                let mean = red(&a)?;
                let c = a.broadcast_sub(&mean)?;
                let var = red(&c.sqr()?)?;
                let inv = (var + eps)?.sqrt()?.recip()?;
                let mut y = c.broadcast_mul(&inv)?;
                if n.inputs.len() > 1 {
                    y = y.broadcast_mul(&x(1)?.to_dtype(DType::F32)?)?;
                }
                if n.inputs.len() > 2 {
                    y = y.broadcast_add(&x(2)?.to_dtype(DType::F32)?)?;
                }
                // ONNX also defines Mean and InvStdDev outputs; emit them so a
                // consumer of either does not read a shifted output slot.
                vec![y, mean, inv]
            }
            "BatchNormalization" => {
                let (a, s, b, m, v) = (x(0)?, x(1)?, x(2)?, x(3)?, x(4)?);
                let eps = n.float("epsilon", 1e-5);
                let c = s.elem_count();
                let inv = (v + eps)?.sqrt()?;
                let scale = (s / &inv)?;
                let shift = (b - (m * &scale)?)?;
                vec![a.broadcast_mul(&scale.reshape((1, c, 1, 1))?)?
                      .broadcast_add(&shift.reshape((1, c, 1, 1))?)?]
            }
            "Relu" => vec![x(0)?.relu()?],
            "Sigmoid" => vec![candle_nn::ops::sigmoid(&x(0)?)?],
            "HardSwish" => {
                let a = x(0)?;
                vec![(((&a + 3.0)?.clamp(0.0, 6.0)? * &a)? / 6.0)?]
            }
            "HardSigmoid" => {
                let a = x(0)?;
                let (al, be) = (n.float("alpha", 0.2), n.float("beta", 0.5));
                vec![((a * al)? + be)?.clamp(0.0, 1.0)?]
            }
            "Add" => vec![bcast(&x(0)?, &x(1)?, |a, b| a.broadcast_add(b))?],
            "Sub" => vec![bcast(&x(0)?, &x(1)?, |a, b| a.broadcast_sub(b))?],
            "Mul" => vec![bcast(&x(0)?, &x(1)?, |a, b| a.broadcast_mul(b))?],
            "Div" => vec![bcast(&x(0)?, &x(1)?, |a, b| a.broadcast_div(b))?],
            "Sqrt" => vec![x(0)?.sqrt()?],
            "Concat" => {
                let ts: CResult<Vec<Tensor>> =
                    (0..n.inputs.len()).map(|i| self.get_s(env, &n.inputs[i], scope)).collect();
                let ts = ts?;
                let ax = n.int("axis", 0);
                let ax = if ax < 0 { (ts[0].rank() as i64 + ax) as usize } else { ax as usize };
                vec![Tensor::cat(&ts, ax)?]
            }
            "GlobalAveragePool" => {
                let a = x(0)?;
                vec![a.mean_keepdim(2)?.mean_keepdim(3)?]
            }
            "Softmax" => {
                let a = x(0)?;
                let ax = n.int("axis", -1);
                let ax = if ax < 0 { (a.rank() as i64 + ax) as usize } else { ax as usize };
                vec![candle_nn::ops::softmax(&a, ax)?]
            }
            "MatMul" => {
                // The PicoDet head decodes boxes by projecting an 8-bin
                // distribution onto its expectation, which ONNX expresses as a
                // [.., 8] x [8] matmul. candle needs both sides >= 2-D, so a
                // 1-D operand is lifted to a column and the extra axis dropped
                // again — numerically the same contraction.
                let (a, b) = (x(0)?, x(1)?);
                let m = match (a.rank(), b.rank()) {
                    (ra, 1) if ra >= 2 => {
                        let k = b.elem_count();
                        a.broadcast_matmul(&b.reshape((k, 1))?)?.squeeze(ra - 1)?
                    }
                    (1, rb) if rb >= 2 => {
                        let k = a.elem_count();
                        a.reshape((1, k))?.broadcast_matmul(&b)?.squeeze(0)?
                    }
                    _ => a.broadcast_matmul(&b)?,
                };
                vec![m]
            }
            "Transpose" => {
                let a = x(0)?;
                let p = n.ints("perm").unwrap_or_else(|| (0..a.rank()).rev().collect());
                vec![a.permute(p.as_slice())?]
            }
            "Reshape" => {
                let a = x(0)?;
                let shp = x(1)?.to_dtype(DType::I64)?.flatten_all()?.to_vec1::<i64>()?;
                let n_elem = a.elem_count();
                let mut dims: Vec<usize> = Vec::with_capacity(shp.len());
                let mut neg = None;
                let mut known = 1usize;
                for (i, &d) in shp.iter().enumerate() {
                    if d == -1 {
                        neg = Some(i);
                        dims.push(1);
                    } else if d == 0 {
                        let v = a.dim(i)?;
                        dims.push(v);
                        known *= v;
                    } else {
                        dims.push(d as usize);
                        known *= d as usize;
                    }
                }
                if let Some(i) = neg {
                    dims[i] = n_elem / known.max(1);
                }
                vec![a.reshape(dims)?]
            }
            "Squeeze" => {
                let a = x(0)?;
                let mut t = a;
                let axes = if n.inputs.len() > 1 {
                    x(1)?.to_dtype(DType::I64)?.flatten_all()?.to_vec1::<i64>()?
                } else {
                    n.ints("axes").unwrap_or_default().iter().map(|&v| v as i64).collect()
                };
                let mut axes: Vec<usize> = axes
                    .iter()
                    .map(|&v| if v < 0 { (t.rank() as i64 + v) as usize } else { v as usize })
                    .collect();
                axes.sort_unstable();
                for a_ in axes.into_iter().rev() {
                    t = t.squeeze(a_)?;
                }
                vec![t]
            }
            "Cast" => {
                let a = x(0)?;
                let to = n.int("to", 1);
                vec![match to {
                    1 => a.to_dtype(DType::F32)?,
                    6 | 7 => a.to_dtype(DType::I64)?,
                    _ => a,
                }]
            }
            "Resize" => {
                // FPN upsampling. `mode: nearest`, and the scale arrives either
                // as `scales` (input 2 or 3, float) or `sizes` (input 3, int).
                let a = x(0)?;
                let (h, w) = (a.dim(2)?, a.dim(3)?);
                let mut th = h;
                let mut tw = w;
                for i in (2..n.inputs.len()).rev() {
                    let Ok(t) = self.get_s(env, &n.inputs[i], scope) else { continue };
                    if t.elem_count() < 4 {
                        continue;
                    }
                    if t.dtype() == DType::I64 {
                        let v = t.flatten_all()?.to_vec1::<i64>()?;
                        th = v[2].max(1) as usize;
                        tw = v[3].max(1) as usize;
                    } else {
                        let v = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                        th = ((h as f32) * v[2]).round().max(1.0) as usize;
                        tw = ((w as f32) * v[3]).round().max(1.0) as usize;
                    }
                    break;
                }
                vec![a.upsample_nearest2d(th, tw)?]
            }
            "Split" => {
                let a = x(0)?;
                let ax = n.int("axis", 0);
                let ax = if ax < 0 { (a.rank() as i64 + ax) as usize } else { ax as usize };
                let total = a.dim(ax)?;
                // sizes come from input 1 when present, else an equal split
                let sizes: Vec<usize> = if n.inputs.len() > 1 {
                    match self.get_s(env, &n.inputs[1], scope) {
                        Ok(t) => t
                            .to_dtype(DType::I64)?
                            .flatten_all()?
                            .to_vec1::<i64>()?
                            .iter()
                            .map(|&v| v.max(0) as usize)
                            .collect(),
                        Err(_) => vec![total / n.outputs.len().max(1); n.outputs.len()],
                    }
                } else {
                    vec![total / n.outputs.len().max(1); n.outputs.len()]
                };
                let mut out = Vec::with_capacity(sizes.len());
                let mut off = 0usize;
                for s in sizes {
                    out.push(a.narrow(ax, off, s)?);
                    off += s;
                }
                out
            }
            "Shape" => {
                let a = x(0)?;
                let d: Vec<i64> = a.dims().iter().map(|&v| v as i64).collect();
                let n_ = d.len();
                vec![Tensor::from_vec(d, n_, a.device())?]
            }
            "Unsqueeze" => {
                let a = x(0)?;
                let axes = if n.inputs.len() > 1 {
                    x(1)?.to_dtype(DType::I64)?.flatten_all()?.to_vec1::<i64>()?
                } else {
                    n.ints("axes").unwrap_or_default().iter().map(|&v| v as i64).collect()
                };
                let mut t = a;
                let mut ax: Vec<usize> = axes
                    .iter()
                    .map(|&v| if v < 0 { (t.rank() as i64 + 1 + v) as usize } else { v as usize })
                    .collect();
                ax.sort_unstable();
                for a_ in ax {
                    t = t.unsqueeze(a_.min(t.rank()))?;
                }
                vec![t]
            }
            "Slice" => {
                let a = x(0)?;
                let gv = |i: usize| -> CResult<Vec<i64>> {
                    Ok(self.get_s(env, &n.inputs[i], scope)?
                        .to_dtype(DType::I64)?
                        .flatten_all()?
                        .to_vec1::<i64>()?)
                };
                let starts = gv(1)?;
                let ends = gv(2)?;
                let axes = if n.inputs.len() > 3 {
                    gv(3)?
                } else {
                    (0..starts.len() as i64).collect()
                };
                let mut t = a;
                for (k, &ax) in axes.iter().enumerate() {
                    let ax = if ax < 0 { (t.rank() as i64 + ax) as usize } else { ax as usize };
                    let d = t.dim(ax)? as i64;
                    let s = starts[k].clamp(-d, d);
                    let s = if s < 0 { (d + s).max(0) } else { s.min(d) } as usize;
                    let e = ends[k].clamp(-d - 1, d);
                    let e = if e < 0 { (d + e).max(0) } else { e.min(d) } as usize;
                    t = t.narrow(ax, s, e.saturating_sub(s).max(1).min(t.dim(ax)? - s))?;
                }
                vec![t]
            }
            "Gather" => {
                let (a, idx) = (x(0)?, x(1)?);
                let ax = n.int("axis", 0);
                let ax = if ax < 0 { (a.rank() as i64 + ax) as usize } else { ax as usize };
                let iv = idx.to_dtype(DType::I64)?.flatten_all()?.to_vec1::<i64>()?;
                let d = a.dim(ax)? as i64;
                let iv: Vec<u32> =
                    iv.iter().map(|&v| (if v < 0 { d + v } else { v }).clamp(0, d - 1) as u32).collect();
                let n_i = iv.len();
                let sel = a.index_select(&Tensor::from_vec(iv, n_i, a.device())?, ax)?;
                // ONNX Gather's output rank is
                //   data.shape[:axis] + INDICES.shape + data.shape[axis+1:]
                // whereas candle's `index_select` always takes a 1-D index and
                // returns the data rank unchanged. Keeping candle's shape drops
                // the index's own rank: an embedding lookup with ids `[1, 1]`
                // came back `[1, 512]` instead of `[1, 1, 512]`, and the missing
                // sequence axis then propagated through the whole decoder — the
                // failure surfaced 90 nodes later as a cross-attention reshape
                // mismatch, nowhere near its cause.
                let ad = a.dims();
                let mut out: Vec<usize> = ad[..ax].to_vec();
                out.extend_from_slice(idx.dims());
                out.extend_from_slice(&ad[ax + 1..]);
                vec![if out == sel.dims() { sel } else { sel.reshape(out)? }]
            }
            "Expand" => {
                let a = x(0)?;
                let shp = x(1)?.to_dtype(DType::I64)?.flatten_all()?.to_vec1::<i64>()?;
                let mut t = a;
                while t.rank() < shp.len() {
                    t = t.unsqueeze(0)?;
                }
                let target: Vec<usize> = shp
                    .iter()
                    .enumerate()
                    // Only a NEGATIVE entry means "keep this dim"; a literal 0 is
                    // a genuinely empty axis. Folding 0 into "keep" gave the
                    // decoder a KV cache of length 1 where it should have begun
                    // empty, and the cache then ran one step ahead of the
                    // attention mask forever after: `[1,16,1,3]` vs `[1,1,1,2]`.
                    .map(|(i, &v)| if v < 0 { t.dim(i).unwrap_or(1) } else { v as usize })
                    .collect();
                vec![t.broadcast_as(target.as_slice())?]
            }
            "Tile" => {
                let a = x(0)?;
                let reps = x(1)?.to_dtype(DType::I64)?.flatten_all()?.to_vec1::<i64>()?;
                let mut t = a;
                for (ax, &r) in reps.iter().enumerate() {
                    if r > 1 {
                        let parts: Vec<Tensor> = (0..r).map(|_| t.clone()).collect();
                        t = Tensor::cat(&parts, ax)?;
                    }
                }
                vec![t]
            }
            "MaxPool" => {
                let a = x(0)?;
                let k = n.ints("kernel_shape").unwrap_or(vec![2, 2]);
                let s = n.ints("strides").unwrap_or_else(|| k.clone());
                let ad = a.dims();
                let p = n.pad4(&k, &s, &[1, 1], (ad[ad.len() - 2], ad[ad.len() - 1]));
                let y = if p == [0; 4] {
                    a.max_pool2d_with_stride(k[0], s[0])?
                } else {
                    // A max-pool must pad with -inf, not 0, or a zero border
                    // wins the window over genuinely negative activations.
                    // `pad_with_zeros` is the only padder candle offers, so
                    // shift the tensor to be non-negative first and shift back
                    // after: max(x - m) + m == max(x) exactly, for any m below
                    // the minimum.
                    let m = a.min_all()?.to_dtype(DType::F32)?.to_scalar::<f32>()? as f64;
                    let z = pad_nchw(&a.affine(1.0, -m)?, p)?;
                    z.max_pool2d_with_stride(k[0], s[0])?.affine(1.0, m)?
                };
                vec![y]
            }
            // ---- elementwise / reduction ----------------------------------
            "Tanh" => vec![x(0)?.tanh()?],
            "Erf" => vec![x(0)?.erf()?],
            "Not" => vec![x(0)?.eq(0f64)?],
            // On the BOOL tensors the decoder's stopping logic uses — which
            // reach us as candle U8, since candle has no bool dtype — the
            // bitwise forms are the logical ones. Applied to a wider integer
            // `~x` would be `-x - 1`, but ONNX only emits these on masks here.
            // Running sum along one axis — the decoder counts how many sequences
            // have already emitted `</s>` to decide when to stop.
            "CumSum" => {
                let a = x(0)?.to_dtype(DType::F32)?;
                let ax = x(1)?.to_dtype(DType::I64)?.flatten_all()?.to_vec1::<i64>()?[0];
                let ax = if ax < 0 { (a.rank() as i64 + ax) as usize } else { ax as usize };
                let len = a.dim(ax)?;
                let mut acc: Option<Tensor> = None;
                let mut parts: Vec<Tensor> = Vec::with_capacity(len);
                for i in 0..len {
                    let s = a.narrow(ax, i, 1)?;
                    let t = match &acc {
                        Some(p) => (p + &s)?,
                        None => s,
                    };
                    parts.push(t.clone());
                    acc = Some(t);
                }
                vec![Tensor::cat(&parts, ax)?]
            }
            "BitwiseNot" => vec![x(0)?.eq(0f64)?],
            "BitwiseAnd" => vec![bcast(&x(0)?, &x(1)?, |a, b| {
                a.to_dtype(DType::F32)?.broadcast_mul(&b.to_dtype(DType::F32)?)?.ne(0f64)
            })?],
            "BitwiseOr" | "Or" => vec![bcast(&x(0)?, &x(1)?, |a, b| {
                a.to_dtype(DType::F32)?.broadcast_add(&b.to_dtype(DType::F32)?)?.ne(0f64)
            })?],
            "And" => vec![bcast(&x(0)?, &x(1)?, |a, b| {
                a.to_dtype(DType::F32)?.broadcast_mul(&b.to_dtype(DType::F32)?)?.ne(0f64)
            })?],
            // Comparisons and min/max: ONNX allows mixed integer widths where
            // candle requires identical dtypes (`dtype mismatch in cmp, lhs:
            // I64, rhs: I32`). Both sides go to F32 first — every value these
            // ops see in these graphs is an index or a small count, well inside
            // exact f32 range.
            "Less" => vec![bcast(&f32c(&x(0)?)?, &f32c(&x(1)?)?, |a, b| a.broadcast_lt(b))?],
            "Greater" => vec![bcast(&f32c(&x(0)?)?, &f32c(&x(1)?)?, |a, b| a.broadcast_gt(b))?],
            "Equal" => vec![bcast(&f32c(&x(0)?)?, &f32c(&x(1)?)?, |a, b| a.broadcast_eq(b))?],
            "LessOrEqual" => vec![bcast(&f32c(&x(0)?)?, &f32c(&x(1)?)?, |a, b| a.broadcast_le(b))?],
            "GreaterOrEqual" => {
                vec![bcast(&f32c(&x(0)?)?, &f32c(&x(1)?)?, |a, b| a.broadcast_ge(b))?]
            }
            "Min" => {
                let (a, b) = (x(0)?, x(1)?);
                let d = a.dtype();
                vec![bcast(&f32c(&a)?, &f32c(&b)?, |p, q| p.broadcast_minimum(q))?.to_dtype(d)?]
            }
            "Max" => {
                let (a, b) = (x(0)?, x(1)?);
                let d = a.dtype();
                vec![bcast(&f32c(&a)?, &f32c(&b)?, |p, q| p.broadcast_maximum(q))?.to_dtype(d)?]
            }
            "Where" => {
                let (c, a, b) = (x(0)?, x(1)?, x(2)?);
                // candle's where_cond needs matching shapes; broadcast all three
                let shape = if a.rank() >= b.rank() { a.dims().to_vec() } else { b.dims().to_vec() };
                let c = c.to_dtype(DType::U8)?.broadcast_as(shape.as_slice())?;
                let a = a.broadcast_as(shape.as_slice())?;
                let b = b.broadcast_as(shape.as_slice())?;
                vec![c.where_cond(&a, &b)?]
            }
            "ReduceMax" | "ReduceMin" => {
                let a = x(0)?;
                let axes = n.ints("axes").unwrap_or_else(|| vec![a.rank() - 1]);
                let keep = n.int("keepdims", 1) != 0;
                let mut t = a;
                for &ax in axes.iter() {
                    t = if n.op == "ReduceMax" { t.max_keepdim(ax)? } else { t.min_keepdim(ax)? };
                }
                if !keep {
                    for &ax in axes.iter().rev() {
                        t = t.squeeze(ax)?;
                    }
                }
                vec![t]
            }
            "ArgMax" => {
                let a = x(0)?;
                let ax = n.int("axis", -1);
                let ax = if ax < 0 { (a.rank() as i64 + ax) as usize } else { ax as usize };
                let i = a.argmax_keepdim(ax)?;
                vec![if n.int("keepdims", 1) != 0 { i } else { i.squeeze(ax)? }]
            }
            "Range" => {
                // start, limit, delta — all scalars.
                //
                // `get_s(.., scope)`, NOT `get()`: `get` hardcodes the TOP-LEVEL
                // alias map, so inside a Loop body it resolves against the wrong
                // 286-entry table instead of the body's 792 and every aliased
                // name comes back missing. This was the only such call left, and
                // it only surfaced because PP-FormulaNet is the first graph with
                // a `Range` inside a subgraph.
                let sc = |k: usize| -> CResult<f64> {
                    Ok(self.get_s(env, &n.inputs[k], scope)?
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?[0] as f64)
                };
                let (s, l, d) = (sc(0)?, sc(1)?, sc(2)?);
                let mut v: Vec<i64> = Vec::new();
                let mut c = s;
                while (d > 0.0 && c < l) || (d < 0.0 && c > l) {
                    v.push(c as i64);
                    c += d;
                }
                let ln = v.len().max(1);
                if v.is_empty() {
                    v.push(0);
                }
                vec![Tensor::from_vec(v, ln, x(0)?.device())?]
            }
            "OneHot" => {
                let (idx, depth, vals) = (x(0)?, x(1)?, x(2)?);
                let d = depth.to_dtype(DType::I64)?.flatten_all()?.to_vec1::<i64>()?[0] as usize;
                let v = vals.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                let (off, on) = (v[0], *v.get(1).unwrap_or(&1.0));
                let iv = idx.to_dtype(DType::I64)?.flatten_all()?.to_vec1::<i64>()?;
                let mut out = vec![off; iv.len() * d];
                for (r, &i) in iv.iter().enumerate() {
                    let i = if i < 0 { i + d as i64 } else { i };
                    if i >= 0 && (i as usize) < d {
                        out[r * d + i as usize] = on;
                    }
                }
                let mut shape = idx.dims().to_vec();
                shape.push(d);
                vec![Tensor::from_vec(out, shape.as_slice(), idx.device())?]
            }
            "ScatterElements" => {
                // Writes one decode step into a carried buffer.
                //
                // ONNX broadcasts the index against the update; candle's
                // `scatter` requires index and src to have IDENTICAL shape and
                // the same rank as self. A [1,1,1] index against a [1,1,50]
                // update therefore wrote ONE element instead of fifty, which is
                // why the table decoder's [1,501,50] logit buffer held exactly
                // one row's worth (absum 511) for all 500 iterations while its
                // [1,501] token buffer accumulated correctly.
                let (data, idx, upd) = (x(0)?, x(1)?, x(2)?);
                let ax = n.int("axis", 0);
                let ax = if ax < 0 { (data.rank() as i64 + ax) as usize } else { ax as usize };
                let idx = if idx.dims() != upd.dims() {
                    idx.broadcast_as(upd.dims())?.contiguous()?
                } else {
                    idx
                };
                if std::env::var("FFAI_SCATTER_DEBUG").is_ok() {
                    let iv = idx.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                    eprintln!("    scatter self{:?} idx{:?} upd{:?} ax{ax} idx[0]={} idxmax={}",
                              data.dims(), idx.dims(), upd.dims(), iv[0],
                              iv.iter().cloned().fold(f32::MIN, f32::max));
                }
                vec![data.scatter(&idx.to_dtype(DType::U32)?, &upd, ax)?]
            }
            "Identity" => vec![x(0)?],
            other => {
                return Err(candle_core::Error::Msg(format!(
                    "unimplemented op `{other}` — the executor stops before the \
                     detection tail; if this fires inside the network the graph \
                     needs it implemented, not skipped"
                )))
            }
        })
    }
}

/// Promote to f32 so mixed-width integer operands compare.
fn f32c(t: &Tensor) -> CResult<Tensor> {
    if t.dtype() == DType::F32 { Ok(t.clone()) } else { t.to_dtype(DType::F32) }
}

/// Broadcasting that tolerates rank mismatch by left-padding the smaller rank,
/// which is what ONNX numpy-style broadcasting does and candle does not.
fn bcast(
    a: &Tensor,
    b: &Tensor,
    f: impl Fn(&Tensor, &Tensor) -> CResult<Tensor>,
) -> CResult<Tensor> {
    // ONNX arithmetic promotes mixed integer widths; candle requires identical
    // dtypes and errors with `dtype mismatch in mul, lhs: U32, rhs: I64`. Index
    // arithmetic in a decoder mixes them freely — a shape read (I64) multiplied
    // by a Gather result (U32) — so unify on the wider signed type first.
    if a.dtype() != b.dtype() && a.dtype().is_int() && b.dtype().is_int() {
        return bcast(&a.to_dtype(DType::I64)?, &b.to_dtype(DType::I64)?, f);
    }
    if a.rank() == b.rank() {
        return f(a, b);
    }
    let (big, small, flip) = if a.rank() > b.rank() { (a, b, false) } else { (b, a, true) };
    let mut s = small.clone();
    while s.rank() < big.rank() {
        s = s.unsqueeze(0)?;
    }
    if flip { f(&s, big) } else { f(big, &s) }
}
