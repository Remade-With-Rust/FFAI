//! CRNN recognition (`english_g2`) on candle: VGG feature extractor →
//! 2× bidirectional LSTM → linear → CTC greedy decode.
//!
//! Ported layer-for-layer from EasyOCR's generation-2 model
//! (JaidedAI/EasyOCR, Apache-2.0: `model/vgg_model.py` + `model/modules.py`),
//! weights `english_g2` converted by `tools/carmenta_crnn_prepare.py` under
//! `load_state_dict(strict=True)`.
//!
//! This seat was TrOCR's until the oracle fixture measured trocr-small-
//! printed reading mixed-case text as ALL CAPS (SROIE-trained) — a lineage
//! that can't approach a case-scoring CER gate. The g2 charset below is the
//! reason this model won: 96 characters covering digits, punctuation, and
//! BOTH cases. Oracle: `tests/crnn_oracle.rs` reproduces the reference's CTC
//! ids exactly on the pinned fixture tensor.

use candle_core::{Result, Tensor};
use candle_nn::rnn::{lstm, Direction, LSTMConfig, LSTM, RNN};
use candle_nn::{batch_norm, conv2d, linear, BatchNorm, Conv2d, Conv2dConfig, Linear, Module, ModuleT, VarBuilder};

/// Index i in CTC output = charset[i-1]; index 0 is the CTC blank.
pub const CHARSET: &str = "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

struct BiLstm {
    fwd: LSTM,
    bwd: LSTM,
    linear: Linear,
}

impl BiLstm {
    fn new(vb: &VarBuilder, name: &str, input: usize, hidden: usize, out: usize) -> Result<Self> {
        let vb = vb.pp(name);
        Ok(BiLstm {
            fwd: lstm(input, hidden, LSTMConfig::default(), vb.pp("rnn"))?,
            bwd: lstm(
                input,
                hidden,
                LSTMConfig { direction: Direction::Backward, ..Default::default() },
                vb.pp("rnn"),
            )?,
            linear: linear(hidden * 2, out, vb.pp("linear"))?,
        })
    }

    /// (1, T, input) -> (1, T, out)
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let t = x.dim(1)?;
        let fwd_states = self.fwd.seq(x)?;
        let fwd: Vec<Tensor> = fwd_states.iter().map(|s| s.h().clone()).collect();
        let fwd = Tensor::stack(&fwd, 1)?;

        // Backward direction: reverse time, run, reverse back.
        let rev_idx: Vec<u32> = (0..t as u32).rev().collect();
        let idx = Tensor::from_vec(rev_idx, t, x.device())?;
        let xrev = x.index_select(&idx, 1)?;
        let bwd_states = self.bwd.seq(&xrev)?;
        let bwd: Vec<Tensor> = bwd_states.iter().map(|s| s.h().clone()).collect();
        let bwd = Tensor::stack(&bwd, 1)?.index_select(&idx, 1)?;

        Tensor::cat(&[&fwd, &bwd], 2)?.apply(&self.linear)
    }
}

pub struct Crnn {
    convs: [Conv2d; 7],
    bns: [BatchNorm; 2],
    rnn0: BiLstm,
    rnn1: BiLstm,
    prediction: Linear,
}

impl Crnn {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let f = vb.pp("FeatureExtraction").pp("ConvNet");
        let p1 = Conv2dConfig { padding: 1, ..Default::default() };
        let seq = vb.pp("SequenceModeling");
        Ok(Crnn {
            convs: [
                conv2d(1, 32, 3, p1, f.pp("0"))?,
                conv2d(32, 64, 3, p1, f.pp("3"))?,
                conv2d(64, 128, 3, p1, f.pp("6"))?,
                conv2d(128, 128, 3, p1, f.pp("8"))?,
                candle_nn::conv2d_no_bias(128, 256, 3, p1, f.pp("11"))?,
                candle_nn::conv2d_no_bias(256, 256, 3, p1, f.pp("14"))?,
                conv2d(256, 256, 2, Conv2dConfig::default(), f.pp("18"))?,
            ],
            bns: [batch_norm(256, 1e-5, f.pp("12"))?, batch_norm(256, 1e-5, f.pp("15"))?],
            rnn0: BiLstm::new(&seq, "0", 256, 256, 256)?,
            rnn1: BiLstm::new(&seq, "1", 256, 256, 256)?,
            prediction: linear(256, CHARSET.chars().count() + 1, vb.pp("Prediction"))?,
        })
    }

    /// (1, 1, 64, W) normalized crop -> per-timestep logits (T, num_class).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let t0 = std::time::Instant::now();
        let x = self.convs[0].forward(x)?.relu()?.max_pool2d_with_stride(2, 2)?;
        let x = self.convs[1].forward(&x)?.relu()?.max_pool2d_with_stride(2, 2)?;
        let x = self.convs[2].forward(&x)?.relu()?;
        let x = self.convs[3].forward(&x)?.relu()?.max_pool2d_with_stride((2, 1), (2, 1))?;
        let x = self.bns[0].forward_t(&self.convs[4].forward(&x)?, false)?.relu()?;
        let x = self
            .bns[1]
            .forward_t(&self.convs[5].forward(&x)?, false)?
            .relu()?
            .max_pool2d_with_stride((2, 1), (2, 1))?;
        let x = self.convs[6].forward(&x)?.relu()?; // (1, 256, H', W')

        // AdaptiveAvgPool2d((None, 1)) after permute(0,3,1,2) == mean over
        // the height axis, sequence along width: (1, C, H, W) -> (1, W, C).
        let x = x.mean(2)?.transpose(1, 2)?.contiguous()?;
        crate::profile::profile().rec_cnn.add(t0.elapsed().as_nanos() as u64);

        let t1 = std::time::Instant::now();
        let x = self.rnn0.forward(&x)?;
        let x = self.rnn1.forward(&x)?;
        crate::profile::profile().rec_rnn.add(t1.elapsed().as_nanos() as u64);

        let t2 = std::time::Instant::now();
        let out = x.apply(&self.prediction)?.squeeze(0);
        crate::profile::profile().rec_head.add(t2.elapsed().as_nanos() as u64);
        out
    }
}

/// CTC greedy decode: argmax per step, collapse repeats, drop blanks.
/// Returns (text, mean softmax probability of the kept steps).
pub fn ctc_greedy(logits: &Tensor) -> Result<(String, Option<f32>)> {
    let charset: Vec<char> = CHARSET.chars().collect();
    let probs = candle_nn::ops::softmax(logits, 1)?;
    let (t, _) = logits.dims2()?;
    let ids = probs.argmax(1)?.to_vec1::<u32>()?;
    let pv = probs.to_vec2::<f32>()?;

    let mut out = String::new();
    let mut confs = Vec::new();
    let mut prev = 0u32;
    for step in 0..t {
        let id = ids[step];
        if id != 0 && id != prev {
            out.push(charset[id as usize - 1]);
            confs.push(pv[step][id as usize]);
        }
        prev = id;
    }
    let conf = if confs.is_empty() {
        None
    } else {
        Some(confs.iter().sum::<f32>() / confs.len() as f32)
    };
    Ok((out, conf))
}
