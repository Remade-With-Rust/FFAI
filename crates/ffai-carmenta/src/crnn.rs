//! CRNN recognition (`english_g2`) on candle: VGG feature extractor →
//! 2× bidirectional LSTM → linear → CTC greedy decode.
//!
//! Ported layer-for-layer from `EasyOCR`'s generation-2 model
//! (JaidedAI/EasyOCR, Apache-2.0: `model/vgg_model.py` + `model/modules.py`),
//! weights `english_g2` converted by `tools/carmenta_crnn_prepare.py` under
//! `load_state_dict(strict=True)`.
//!
//! This seat was `TrOCR`'s until the oracle fixture measured trocr-small-
//! printed reading mixed-case text as ALL CAPS (SROIE-trained) — a lineage
//! that can't approach a case-scoring CER gate. The g2 charset below is the
//! reason this model won: 96 characters covering digits, punctuation, and
//! BOTH cases. Oracle: `tests/crnn_oracle.rs` reproduces the reference's CTC
//! ids exactly on the pinned fixture tensor.

use candle_core::{Result, Tensor};
use candle_nn::rnn::{lstm, Direction, LSTMConfig, LSTM, RNN};
use candle_nn::{batch_norm, conv2d, linear, BatchNorm, Conv2d, Conv2dConfig, Linear, Module, ModuleT, VarBuilder};

/// Index i in CTC output = charset[i-1]; index 0 is the CTC blank.
///
/// This is the ENGLISH charset and the default. `FFAI_REC_LANG=zh` selects
/// `zh_sim_g2` instead, whose 6 718-character set covers the same ASCII plus
/// 6 614 CJK characters — see [`charset_for`] and §8.143.
pub const CHARSET: &str = "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

struct BiLstm {
    fwd: LSTM,
    bwd: LSTM,
    linear: Linear,
}

impl BiLstm {
    fn new(vb: &VarBuilder, name: &str, input: usize, hidden: usize, out: usize) -> Result<Self> {
        let vb = vb.pp(name);
        Ok(Self {
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

/// Which recognizer the engine loads, chosen at runtime by `FFAI_REC_LANG`.
///
/// §8.142 measured 11 holdout pages of 236 as **51 % of the competitive gap**
/// for one reason: they contain Chinese and a 96-class CTC head cannot emit a
/// character it has no class for. The two checkpoints are architecturally
/// identical — same 44 tensors, same names — and differ only in the head
/// (97 vs 6 719), so the same code runs both.
///
/// **English is the default and stays the oracle.** A 6 719-class head is
/// normally weaker on Latin than a 97-class specialist, and trading English CER
/// for CJK coverage is a bad bargain until measured. Opt in, measure, and only
/// then consider changing the default — the shape `--features mimalloc` and
/// `FFAI_CONV3X3=0` already use here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecLang {
    English,
    ChineseSimplified,
}

impl RecLang {
    /// `FFAI_REC_LANG=zh` (or `zh_sim`, `chinese`) selects the Chinese model.
    /// Anything else, including unset, is English.
    #[must_use] 
    pub fn from_env() -> Self {
        match std::env::var("FFAI_REC_LANG").as_deref() {
            Ok("zh") | Ok("zh_sim") | Ok("chinese") => Self::ChineseSimplified,
            _ => Self::English,
        }
    }

    /// The model-registry name whose `crnn.safetensors` this variant loads.
    #[must_use] 
    pub fn model_name(self) -> &'static str {
        match self {
            Self::English => "crnn-english-g2",
            Self::ChineseSimplified => "crnn-zh-sim-g2",
        }
    }
}

/// The charset a variant decodes with.
///
/// English is compiled in; Chinese is read from `charset.txt` beside the
/// weights, because 6 718 characters is 20 KB of UTF-8 and a generated `const`
/// that large is a merge hazard for no benefit. The file MUST have as many
/// characters as the head has classes minus one — a mismatch shifts every
/// decoded character, which is §8.113's failure in a form far harder to notice,
/// so `Crnn::new_with_charset` refuses rather than decoding nonsense.
pub fn charset_for(lang: RecLang, dir: Option<&std::path::Path>) -> Result<Vec<char>> {
    match lang {
        RecLang::English => Ok(CHARSET.chars().collect()),
        RecLang::ChineseSimplified => {
            let d = dir.ok_or_else(|| candle_core::Error::Msg(
                "FFAI_REC_LANG=zh needs the model directory to read charset.txt".into()))?;
            let f = d.join("charset.txt");
            let txt = std::fs::read_to_string(&f).map_err(|e| candle_core::Error::Msg(
                format!("FFAI_REC_LANG=zh: cannot read {}: {e}", f.display())))?;
            Ok(txt.chars().filter(|c| *c != '\n' && *c != '\r').collect())
        }
    }
}

pub struct Crnn {
    /// What index `i` in the CTC output means. Carried per instance rather than
    /// read from a global, because the two models disagree about it.
    charset: Vec<char>,
    convs: [Conv2d; 7],
    bns: [BatchNorm; 2],
    rnn0: BiLstm,
    rnn1: BiLstm,
    prediction: Linear,
}

impl Crnn {
    /// The English recognizer — the default and the oracle.
    pub fn new(vb: VarBuilder) -> Result<Self> {
        Self::new_with_charset(vb, CHARSET.chars().collect())
    }

    /// Build with an explicit charset; the head is sized from it.
    pub fn new_with_charset(vb: VarBuilder, charset: Vec<char>) -> Result<Self> {
        if charset.is_empty() {
            return Err(candle_core::Error::Msg("empty charset".into()));
        }
        let n_class = charset.len() + 1;
        let f = vb.pp("FeatureExtraction").pp("ConvNet");
        let p1 = Conv2dConfig { padding: 1, ..Default::default() };
        let seq = vb.pp("SequenceModeling");
        Ok(Self {
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
            prediction: linear(256, n_class, vb.pp("Prediction"))?,
            charset,
        })
    }

    /// CTC greedy decode against THIS model's charset.
    pub fn decode(&self, logits: &Tensor) -> Result<(String, Option<f32>)> {
        ctc_greedy_with(logits, &self.charset)
    }

    /// How many characters this instance can emit — 96 for English, 6 718 for
    /// Chinese. Exposed so a caller can assert which model it actually got.
    #[must_use] 
    pub fn charset_len(&self) -> usize {
        self.charset.len()
    }

    /// (1, 1, 64, W) normalized crop -> per-timestep logits (T, `num_class`).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let t0 = std::time::Instant::now();
        let x = crate::conv3x3::apply(x, &self.convs[0])?.relu()?.max_pool2d_with_stride(2, 2)?;
        let x = crate::conv3x3::apply(&x, &self.convs[1])?.relu()?.max_pool2d_with_stride(2, 2)?;
        let x = crate::conv3x3::apply(&x, &self.convs[2])?.relu()?;
        let x = crate::conv3x3::apply(&x, &self.convs[3])?.relu()?.max_pool2d_with_stride((2, 1), (2, 1))?;
        let x = self.bns[0].forward_t(&crate::conv3x3::apply(&x, &self.convs[4])?, false)?.relu()?;
        let x = self
            .bns[1]
            .forward_t(&crate::conv3x3::apply(&x, &self.convs[5])?, false)?
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
    ctc_greedy_with(logits, &charset)
}

/// The same decode against an explicit charset.
pub fn ctc_greedy_with(logits: &Tensor, charset: &[char]) -> Result<(String, Option<f32>)> {
    let probs = candle_nn::ops::softmax(logits, 1)?;
    let (t, _) = logits.dims2()?;
    let ids = probs.argmax(1)?.to_vec1::<u32>()?;
    // The kept confidence is the probability OF THE ARGMAX, which is the row
    // maximum — so take `max(1)`, a T-vector, instead of materialising the whole
    // (T, n_class) matrix as a `Vec<Vec<f32>>` to read one number per row.
    //
    // With English's 97 classes that was merely wasteful. With `zh_sim_g2`'s
    // 6 719 it is 69x worse and it showed: peak working set went 2 119 -> 2 854
    // MB (+35 %) and two pages of a 236-page run died with an access violation
    // and a stack-buffer overrun. Fixing it here helps BOTH models (§8.144).
    let maxp = probs.max(1)?.to_vec1::<f32>()?;

    let mut out = String::new();
    let mut confs = Vec::new();
    let mut prev = 0u32;
    for step in 0..t {
        let id = ids[step];
        if id != 0 && id != prev {
            out.push(charset[id as usize - 1]);
            confs.push(maxp[step]);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The switch is OFF unless asked for, and only for the values documented.
    #[test]
    fn rec_lang_defaults_to_english() {
        assert_eq!(RecLang::English.model_name(), "crnn-english-g2");
        assert_eq!(RecLang::ChineseSimplified.model_name(), "crnn-zh-sim-g2");
        let cs = charset_for(RecLang::English, None).unwrap();
        assert_eq!(cs.len(), CHARSET.chars().count());
        assert_eq!(cs.len(), 96);
        // Chinese without a directory must ERROR, never silently fall back to
        // the English charset — that would decode every CJK class as Latin.
        assert!(charset_for(RecLang::ChineseSimplified, None).is_err());
    }

    /// A charset and a head that disagree shift EVERY decoded character, so the
    /// constructor sizes the head FROM the charset rather than trusting both.
    #[test]
    fn empty_charset_is_refused() {
        let dev = candle_core::Device::Cpu;
        let vb = VarBuilder::zeros(candle_core::DType::F32, &dev);
        assert!(Crnn::new_with_charset(vb, Vec::new()).is_err());
    }
}
