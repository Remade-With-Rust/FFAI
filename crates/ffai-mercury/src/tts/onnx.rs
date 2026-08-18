//! A minimal ONNX reader: enough of the format to lift a piper voice's
//! weights and convolution geometry, in Rust, with no Python and no ONNX
//! runtime.
//!
//! **Why this exists.** The voice files piper ships are ONNX, and Mercury
//! runs candle. Until now the bridge was `corpora/refs/dump_piper_weights.py`
//! — which is fine inside this repo and useless to someone who typed
//! `cargo add ffai-mercury`: it made the crate's first run depend on Python,
//! onnx, and a checkout. Reading the file here restores principle 1 (pure
//! Rust) and principle 4 (weights are data, fetched from a manifest) for
//! consumers, and makes any of the 40+ piper voices loadable with no
//! per-voice conversion step.
//!
//! **Scope, deliberately small.** This is not an ONNX implementation. It
//! decodes four message types (`ModelProto`, `GraphProto`, `NodeProto`,
//! `TensorProto`/`AttributeProto`) far enough to answer two questions: what
//! are the float initializers called, and what geometry does each convolution
//! use. Every other field is skipped by wire type. Executing a graph is
//! candle's job; `vits.rs` already knows the architecture.
//!
//! The gate is byte equality with the Python converter's output — see
//! `examples/onnx_vs_safetensors.rs`.

use std::collections::HashMap;

use ffai_core::error::{Error, Result};

// ---------------------------------------------------------------------------
// protobuf wire format — the four things the encoding can hold
// ---------------------------------------------------------------------------

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

/// One field's payload, already sliced.
// Narrow allow: `Bytes`'s payload is navigated past rather than read, because
// length-delimited fields are skipped by this parser. Scoped to this enum so
// dead code elsewhere still surfaces.
#[allow(dead_code)]
enum Wire<'a> {
    Varint(u64),
    Fixed64(u64),
    // The payload is sliced and skipped rather than read: length-delimited
    // fields are navigated, not consumed, by this parser. Kept so the enum
    // documents the whole wire format.
    Bytes(&'a [u8]),
    Fixed32(u32),
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }

    fn done(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn varint(&mut self) -> Result<u64> {
        let mut out = 0u64;
        let mut shift = 0u32;
        loop {
            let b = *self.buf.get(self.pos).ok_or_else(|| trunc("varint"))?;
            self.pos += 1;
            out |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                return Ok(out);
            }
            shift += 7;
            if shift > 63 {
                return Err(Error::Model("onnx: varint longer than 64 bits".into()));
            }
        }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| trunc("length overflow"))?;
        let slice = self.buf.get(self.pos..end).ok_or_else(|| trunc("bytes"))?;
        self.pos = end;
        Ok(slice)
    }

    /// Next (field number, payload), or `None` at the end of the message.
    fn field(&mut self) -> Result<Option<(u32, Wire<'a>)>> {
        if self.done() {
            return Ok(None);
        }
        let tag = self.varint()?;
        let field = (tag >> 3) as u32;
        let wire = match tag & 7 {
            0 => Wire::Varint(self.varint()?),
            1 => {
                let b = self.take(8)?;
                Wire::Fixed64(u64::from_le_bytes(b.try_into().expect("8 bytes")))
            }
            2 => {
                let n = self.varint()? as usize;
                Wire::Bytes(self.take(n)?)
            }
            5 => {
                let b = self.take(4)?;
                Wire::Fixed32(u32::from_le_bytes(b.try_into().expect("4 bytes")))
            }
            other => {
                return Err(Error::Model(format!("onnx: unsupported wire type {other}")));
            }
        };
        Ok(Some((field, wire)))
    }
}

fn trunc(what: &str) -> Error {
    Error::Model(format!("onnx: truncated while reading {what}"))
}

fn utf8(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

/// Repeated int64 that may arrive packed (one length-delimited field) or
/// unpacked (one varint field per element). Both are legal; exporters differ.
fn push_ints(out: &mut Vec<i64>, wire: &Wire<'_>) -> Result<()> {
    match wire {
        Wire::Varint(v) => out.push(*v as i64),
        Wire::Bytes(b) => {
            let mut r = Reader::new(b);
            while !r.done() {
                out.push(r.varint()? as i64);
            }
        }
        _ => {}
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// the ONNX subset
// ---------------------------------------------------------------------------

/// A float initializer: the weights themselves.
pub struct Initializer {
    pub name: String,
    pub dims: Vec<usize>,
    pub data: Vec<f32>,
}

/// One graph node, reduced to what naming and geometry need.
pub struct Node {
    pub op_type: String,
    pub name: String,
    pub inputs: Vec<String>,
    /// `kernel_shape`, `strides`, `pads`, `dilations`, `group` — first element
    /// only, since the voice is 1-D throughout.
    pub ints: HashMap<String, Vec<i64>>,
}

pub struct Graph {
    pub nodes: Vec<Node>,
    pub initializers: Vec<Initializer>,
}

/// ONNX `TensorProto.data_type`.
const DT_FLOAT: i64 = 1;

fn parse_tensor(buf: &[u8]) -> Result<Option<Initializer>> {
    let (mut dims, mut name, mut raw, mut floats) =
        (Vec::new(), String::new(), None::<&[u8]>, Vec::<f32>::new());
    let mut data_type = 0i64;
    let mut r = Reader::new(buf);
    while let Some((field, wire)) = r.field()? {
        match (field, &wire) {
            (1, w) => push_ints(&mut dims, w)?, // dims
            (2, Wire::Varint(v)) => data_type = *v as i64,
            (4, Wire::Bytes(b)) => {
                // packed float_data
                let mut fr = Reader::new(b);
                while !fr.done() {
                    let raw = fr.take(4)?;
                    floats.push(f32::from_le_bytes(raw.try_into().expect("4 bytes")));
                }
            }
            (4, Wire::Fixed32(v)) => floats.push(f32::from_bits(*v)),
            (8, Wire::Bytes(b)) => name = utf8(b),
            (9, Wire::Bytes(b)) => raw = Some(*b),
            _ => {}
        }
    }
    // Weights only. Int64 shape constants and scalars belong to the graph, not
    // to the model — the same rule the Python converter settled on after a
    // size filter silently ate the [2,1] ElementwiseAffine parameters.
    if data_type != DT_FLOAT || dims.is_empty() {
        return Ok(None);
    }
    let data = match raw {
        Some(bytes) => {
            if bytes.len() % 4 != 0 {
                return Err(Error::Model(format!(
                    "onnx: raw_data for `{name}` is {} bytes, not a whole number of f32",
                    bytes.len()
                )));
            }
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
                .collect()
        }
        None => floats,
    };
    let dims: Vec<usize> = dims.iter().map(|d| *d as usize).collect();
    let expected: usize = dims.iter().product();
    if data.len() != expected {
        return Err(Error::Model(format!(
            "onnx: `{name}` declares {dims:?} ({expected} values) but carries {}",
            data.len()
        )));
    }
    Ok(Some(Initializer { name, dims, data }))
}

fn parse_attribute(buf: &[u8]) -> Result<(String, Vec<i64>)> {
    let (mut name, mut ints) = (String::new(), Vec::new());
    let mut r = Reader::new(buf);
    while let Some((field, wire)) = r.field()? {
        match (field, &wire) {
            (1, Wire::Bytes(b)) => name = utf8(b),
            (3, Wire::Varint(v)) => ints.push(*v as i64), // single int (`group`)
            (8, w) => push_ints(&mut ints, w)?,           // repeated ints
            _ => {}
        }
    }
    Ok((name, ints))
}

fn parse_node(buf: &[u8]) -> Result<Node> {
    let (mut op_type, mut name, mut inputs, mut ints) =
        (String::new(), String::new(), Vec::new(), HashMap::new());
    let mut r = Reader::new(buf);
    while let Some((field, wire)) = r.field()? {
        match (field, &wire) {
            (1, Wire::Bytes(b)) => inputs.push(utf8(b)),
            (3, Wire::Bytes(b)) => name = utf8(b),
            (4, Wire::Bytes(b)) => op_type = utf8(b),
            (5, Wire::Bytes(b)) => {
                let (k, v) = parse_attribute(b)?;
                if !v.is_empty() {
                    ints.insert(k, v);
                }
            }
            _ => {}
        }
    }
    Ok(Node {
        op_type,
        name,
        inputs,
        ints,
    })
}

fn parse_graph(buf: &[u8]) -> Result<Graph> {
    let (mut nodes, mut initializers) = (Vec::new(), Vec::new());
    let mut r = Reader::new(buf);
    while let Some((field, wire)) = r.field()? {
        match (field, &wire) {
            (1, Wire::Bytes(b)) => nodes.push(parse_node(b)?),
            (5, Wire::Bytes(b)) => {
                if let Some(t) = parse_tensor(b)? {
                    initializers.push(t);
                }
            }
            _ => {}
        }
    }
    Ok(Graph {
        nodes,
        initializers,
    })
}

/// Parse an `.onnx` file down to its graph.
pub fn parse(bytes: &[u8]) -> Result<Graph> {
    let mut r = Reader::new(bytes);
    while let Some((field, wire)) = r.field()? {
        if let (7, Wire::Bytes(b)) = (field, &wire) {
            return parse_graph(b);
        }
    }
    Err(Error::Model(
        "onnx: no graph in the file — is it an ONNX model?".into(),
    ))
}

// ---------------------------------------------------------------------------
// name recovery: ONNX export mangles names, node paths carry the module tree
// ---------------------------------------------------------------------------

/// Convolution geometry, read from the graph rather than assumed.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Geometry {
    pub transpose: bool,
    pub stride: usize,
    pub pad: usize,
    pub dilation: usize,
}

/// Weights under canonical module names, plus per-conv geometry.
pub struct VoiceWeights {
    pub tensors: HashMap<String, Initializer>,
    pub geometry: HashMap<String, Geometry>,
}

/// `/flow/flows.6/enc/in_layers.0/Conv` → `flow.flows.6.enc.in_layers.0`
fn module_path(node_name: &str, op_type: &str) -> String {
    let mut parts: Vec<&str> = node_name.split('/').filter(|p| !p.is_empty()).collect();
    if parts.last().is_some_and(|last| last.starts_with(op_type)) {
        parts.pop();
    }
    parts.join(".")
}

/// Recover canonical weight names and conv geometry from a parsed graph.
///
/// Three export quirks are handled here, each of which was a silent
/// wrong-output bug when it was first missed:
///
/// 1. the phoneme embedding is exported under the name `sid`;
/// 2. `ElementwiseAffine`'s `logs` is constant-folded into `exp(-logs)`;
/// 3. weight-norm folding leaves 32 convolution weights named `onnx::Conv_*`,
///    recoverable only through the node that consumes them.
///
/// Takes the graph **by value** so the weights are moved, not copied. The
/// first version borrowed and cloned, which meant the file bytes, the parsed
/// initializers, the recovered copies and the candle tensors were all live at
/// once — ~4× the model in peak memory, enough to flip the footprint gate.
pub fn recover(graph: Graph) -> Result<VoiceWeights> {
    let mut rename: HashMap<String, String> = HashMap::new();
    let mut geometry = HashMap::new();
    let initializer_names: std::collections::HashSet<&str> =
        graph.initializers.iter().map(|t| t.name.as_str()).collect();

    for node in &graph.nodes {
        match node.op_type.as_str() {
            "Conv" | "ConvTranspose" => {
                let base = module_path(&node.name, &node.op_type);
                let first = |k: &str, d: usize| {
                    node.ints
                        .get(k)
                        .and_then(|v| v.first())
                        .map(|v| *v as usize)
                        .unwrap_or(d)
                };
                geometry.insert(
                    base.clone(),
                    Geometry {
                        transpose: node.op_type == "ConvTranspose",
                        stride: first("strides", 1),
                        pad: first("pads", 0),
                        dilation: first("dilations", 1),
                    },
                );
                // Inputs are (x, weight, bias?); only the initializers are ours.
                let weights: Vec<&String> = node
                    .inputs
                    .iter()
                    .filter(|i| initializer_names.contains(i.as_str()))
                    .collect();
                for (k, input) in weights.iter().enumerate() {
                    let suffix = if k == 0 { "weight" } else { "bias" };
                    rename.insert((*input).clone(), format!("{base}.{suffix}"));
                }
            }
            "Gather" if node.name == "/enc_p/emb/Gather" => {
                for input in &node.inputs {
                    if initializer_names.contains(input.as_str()) {
                        rename.insert(input.clone(), "enc_p.emb.weight".to_string());
                    }
                }
            }
            "Mul" if node.name == "/dp/flows.0/Mul" => {
                for input in &node.inputs {
                    if initializer_names.contains(input.as_str()) && input.ends_with("Exp_output_0")
                    {
                        rename.insert(input.clone(), "dp.flows.0.exp_neg_logs".to_string());
                    }
                }
            }
            _ => {}
        }
    }

    let mut tensors = HashMap::new();
    for init in graph.initializers {
        let canonical = rename
            .get(&init.name)
            .cloned()
            .unwrap_or_else(|| init.name.clone());
        // Float constants the graph owns rather than the model: keep them
        // under a stable name so nothing is silently dropped, exactly as the
        // Python converter does.
        let canonical = if canonical.starts_with("onnx::") || canonical.starts_with('/') {
            format!(
                "graph_const.{}",
                canonical.replace('/', "_").replace("::", "_")
            )
        } else {
            canonical
        };
        if tensors.contains_key(&canonical) {
            return Err(Error::Model(format!(
                "onnx: two tensors both named `{canonical}`"
            )));
        }
        tensors.insert(canonical, init);
    }
    Ok(VoiceWeights { tensors, geometry })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-built protobuf: one TensorProto with dims [2,1] and raw f32 data,
    /// wrapped in a GraphProto, wrapped in a ModelProto. Exercises the whole
    /// nesting and the raw_data path without needing a 63 MB voice file.
    #[test]
    fn parses_a_hand_built_model() {
        fn len_delim(field: u32, payload: &[u8]) -> Vec<u8> {
            let mut out = varint((field as u64) << 3 | 2);
            out.extend(varint(payload.len() as u64));
            out.extend_from_slice(payload);
            out
        }
        fn varint(mut v: u64) -> Vec<u8> {
            let mut out = Vec::new();
            loop {
                let b = (v & 0x7f) as u8;
                v >>= 7;
                if v == 0 {
                    out.push(b);
                    return out;
                }
                out.push(b | 0x80);
            }
        }

        let mut tensor = Vec::new();
        for d in [2u64, 1] {
            tensor.extend(varint(1 << 3)); // field 1, varint
            tensor.extend(varint(d));
        }
        tensor.extend(varint(2 << 3)); // data_type
        tensor.extend(varint(DT_FLOAT as u64));
        tensor.extend(len_delim(8, b"dp.flows.0.m")); // name
        let raw: Vec<u8> = [1.5f32, -2.25]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        tensor.extend(len_delim(9, &raw)); // raw_data

        let graph = len_delim(5, &tensor);
        let model = len_delim(7, &graph);

        let parsed = parse(&model).expect("parses");
        assert_eq!(parsed.initializers.len(), 1);
        let t = &parsed.initializers[0];
        assert_eq!(t.name, "dp.flows.0.m");
        assert_eq!(t.dims, vec![2, 1]);
        assert_eq!(t.data, vec![1.5, -2.25]);
    }

    #[test]
    fn module_path_strips_the_op_suffix() {
        assert_eq!(
            module_path("/flow/flows.6/enc/in_layers.0/Conv", "Conv"),
            "flow.flows.6.enc.in_layers.0"
        );
        assert_eq!(
            module_path("/dec/ups.0/ConvTranspose", "ConvTranspose"),
            "dec.ups.0"
        );
        // A numbered repeat of the op is stripped too (`Conv_1` → same base),
        // which is what the Python converter does and therefore what byte
        // equality with it requires. Two convs under one module would then
        // collide — `recover` errors on that rather than silently keeping one.
        assert_eq!(module_path("/dp/flows.0/Mul_1", "Mul"), "dp.flows.0");
        // A last segment that merely contains the op name is NOT stripped.
        assert_eq!(
            module_path("/enc_p/proj/PreConv", "Conv"),
            "enc_p.proj.PreConv"
        );
    }

    #[test]
    fn a_truncated_file_is_an_error_not_a_panic() {
        assert!(parse(&[0x3a, 0x7f]).is_err());
        assert!(parse(&[]).is_err());
    }
}
