//! The two primitives the depth head needs and the detect graph never did.
//!
//! Both are gated against fixtures dumped from `PyTorch` itself
//! (`corpora/refs/fixtures/diana_depth_ops.json`), not against a reading of
//! the documentation. That distinction earned its place elsewhere in this
//! crate: an FMA that passed a 1e-6 unit test still breached a full-graph
//! oracle, because the unit test and the graph disagreed about what "the same
//! function" meant.

use candle_core::{Result, Tensor};

/// Bilinear upsample by exactly 2, with `PyTorch`'s `align_corners=True`.
///
/// # Why this convention and not the other one
///
/// Ultralytics' `Depth.forward` says it plainly: *"`align_corners=True` is baked
/// into the released depth weights."* It is therefore not a free choice, and
/// the two conventions differ in a way that will not fail loudly.
///
/// * `align_corners=True` maps output pixel `i` to input coordinate
///   `i * (in - 1) / (out - 1)` — the corner SAMPLES are pinned, so the first
///   and last output pixels equal the first and last input pixels exactly.
/// * The half-pixel convention used by most resize code maps `i` to
///   `(i + 0.5) * in / out - 0.5`, which pins the corner AREAS instead.
///
/// At `scale_factor = 2` and an input width of 4 the two disagree by up to
/// half an input pixel, and the depth head applies this twice — once per
/// pyramid level — so the error compounds before anything downstream can
/// notice. A depth map shifted by a pixel still looks like a depth map.
///
/// With `out = 2 * in`, the factor is `(in - 1) / (2 * in - 1)`, which is NOT
/// 0.5 — that is the whole subtlety, and it is why this is a named function
/// with a fixture rather than three lines inline.
pub fn bilinear2x_align_corners(x: &Tensor) -> Result<Tensor> {
    let (n, c, h, w) = x.dims4()?;
    let (oh, ow) = (h * 2, w * 2);
    let xs = x.flatten_all()?.to_vec1::<f32>()?;
    let mut out = vec![0f32; n * c * oh * ow];

    // Corner-pinned scale. Guard the degenerate 1-pixel axis, where the
    // formula divides by zero and every output must simply repeat the input.
    let sy = if oh > 1 { (h as f32 - 1.0) / (oh as f32 - 1.0) } else { 0.0 };
    let sx = if ow > 1 { (w as f32 - 1.0) / (ow as f32 - 1.0) } else { 0.0 };

    for plane in 0..n * c {
        let src = &xs[plane * h * w..(plane + 1) * h * w];
        let dst = &mut out[plane * oh * ow..(plane + 1) * oh * ow];
        for oy in 0..oh {
            let fy = oy as f32 * sy;
            let y0 = fy.floor() as usize;
            let y1 = (y0 + 1).min(h - 1);
            let ty = fy - y0 as f32;
            for ox in 0..ow {
                let fx = ox as f32 * sx;
                let x0 = fx.floor() as usize;
                let x1 = (x0 + 1).min(w - 1);
                let tx = fx - x0 as f32;
                let top = src[y0 * w + x0] * (1.0 - tx) + src[y0 * w + x1] * tx;
                let bot = src[y1 * w + x0] * (1.0 - tx) + src[y1 * w + x1] * tx;
                dst[oy * ow + ox] = top * (1.0 - ty) + bot * ty;
            }
        }
    }
    Tensor::from_vec(out, (n, c, oh, ow), x.device())
}

/// `ConvTranspose2d` with `kernel = stride = 2`, `padding = 0`.
///
/// The easy half of this module. Because stride equals kernel size the output
/// tiles never overlap, so there is no accumulation and no scatter: each input
/// pixel contributes exactly one 2x2 output block, and each output pixel is
/// touched exactly once. That makes it a gather —
///
/// ```text
/// out[oc][2*y + ky][2*x + kx] = bias[oc] + sum_ic in[ic][y][x] * w[ic][oc][ky][kx]
/// ```
///
/// — which is a plain reduction over input channels, not the general
/// transposed-convolution accumulation.
///
/// Weight layout is `PyTorch`'s for this op: `[in_channels, out_channels, kh,
/// kw]`, with `in` FIRST. That is transposed relative to `Conv2d`'s
/// `[out, in, kh, kw]` and is exactly the kind of thing that produces a
/// plausible-looking wrong answer, so the fixture below pins it.
pub fn convtranspose2x(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let (n, c_in, h, w) = x.dims4()?;
    let (wc_in, c_out, kh, kw) = weight.dims4()?;
    debug_assert_eq!(wc_in, c_in, "convtranspose2x: weight is [in, out, kh, kw]");
    debug_assert!(kh == 2 && kw == 2, "convtranspose2x implements k=2 s=2 only");

    let xs = x.flatten_all()?.to_vec1::<f32>()?;
    let ws = weight.flatten_all()?.to_vec1::<f32>()?;
    let bs = match bias {
        Some(b) => b.flatten_all()?.to_vec1::<f32>()?,
        None => vec![0f32; c_out],
    };
    let (oh, ow) = (h * 2, w * 2);
    let mut out = vec![0f32; n * c_out * oh * ow];

    for b in 0..n {
        for oc in 0..c_out {
            let dst = &mut out[((b * c_out) + oc) * oh * ow..((b * c_out) + oc + 1) * oh * ow];
            for y in 0..h {
                for x_ in 0..w {
                    // One 2x2 tile, four independent sums over c_in.
                    let mut acc = [bs[oc]; 4];
                    for ic in 0..c_in {
                        let v = xs[((b * c_in) + ic) * h * w + y * w + x_];
                        let wbase = (ic * c_out + oc) * 4;
                        for (t, a) in acc.iter_mut().enumerate() {
                            *a += v * ws[wbase + t];
                        }
                    }
                    for ky in 0..2 {
                        for kx in 0..2 {
                            dst[(2 * y + ky) * ow + 2 * x_ + kx] = acc[ky * 2 + kx];
                        }
                    }
                }
            }
        }
    }
    Tensor::from_vec(out, (n, c_out, oh, ow), x.device())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn fixture() -> serde_json::Value {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .unwrap()
            .join("corpora/refs/fixtures/diana_depth_ops.json");
        serde_json::from_str(&std::fs::read_to_string(p).expect("depth op fixture")).unwrap()
    }

    fn floats(v: &serde_json::Value) -> Vec<f32> {
        v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
    }

    fn dims(v: &serde_json::Value) -> (usize, usize, usize, usize) {
        let d: Vec<usize> = v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        (d[0], d[1], d[2], d[3])
    }

    #[test]
    fn bilinear_matches_pytorch_align_corners() {
        let f = fixture();
        let b = &f["bilinear"];
        let x = Tensor::from_vec(floats(&b["input"]), dims(&b["in_shape"]), &Device::Cpu).unwrap();
        let got = bilinear2x_align_corners(&x).unwrap();
        assert_eq!(got.dims4().unwrap(), dims(&b["out_shape"]));
        let g = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let e = floats(&b["expect"]);
        for (i, (a, b)) in g.iter().zip(&e).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "bilinear diverges at {i}: {a} vs {b} — check align_corners, \
                 the scale is (in-1)/(out-1), NOT 0.5"
            );
        }
    }

    #[test]
    fn convtranspose_matches_pytorch() {
        let f = fixture();
        let c = &f["convtranspose"];
        let x = Tensor::from_vec(floats(&c["input"]), dims(&c["in_shape"]), &Device::Cpu).unwrap();
        let w = Tensor::from_vec(floats(&c["weight"]), dims(&c["weight_shape"]), &Device::Cpu).unwrap();
        let bias = floats(&c["bias"]);
        let bt = Tensor::from_vec(bias.clone(), (bias.len(),), &Device::Cpu).unwrap();
        let got = convtranspose2x(&x, &w, Some(&bt)).unwrap();
        assert_eq!(got.dims4().unwrap(), dims(&c["out_shape"]));
        let g = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let e = floats(&c["expect"]);
        for (i, (a, b)) in g.iter().zip(&e).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "convtranspose diverges at {i}: {a} vs {b} — weight layout is \
                 [in, out, kh, kw], transposed from Conv2d's"
            );
        }
    }

    /// A 1-pixel axis divides by zero in the corner-pinned formula. PyTorch
    /// returns the input value everywhere; so must we.
    #[test]
    fn bilinear_handles_degenerate_axis() {
        let x = Tensor::from_vec(vec![3.5f32, -1.25], (1, 2, 1, 1), &Device::Cpu).unwrap();
        let got = bilinear2x_align_corners(&x).unwrap();
        assert_eq!(got.dims4().unwrap(), (1, 2, 2, 2));
        let g = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(&g[..4], &[3.5, 3.5, 3.5, 3.5]);
        assert_eq!(&g[4..], &[-1.25, -1.25, -1.25, -1.25]);
    }
}
