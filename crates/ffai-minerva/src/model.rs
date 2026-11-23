//! A `BitNet` b1.58 model loaded from an `I2_S` `GGUF`: its hyperparameters and where each
//! tensor's bytes are. The file's bytes are held once; projections are read in place.

use rusty_ternary::gguf::{Gguf, TYPE_F16, TYPE_F32, TYPE_I2_S, Value};
use rusty_ternary::i2s;

/// What can go wrong loading a model.
#[derive(Debug)]
pub enum LoadError {
    /// The file could not be read.
    Io(std::io::Error),
    /// The file is not a `GGUF` `rusty_ternary` reads.
    Format(rusty_ternary::Error),
    /// The file is a `GGUF`, but not a `BitNet` b1.58 model this engine runs.
    Model(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::Format(e) => write!(f, "not a readable GGUF: {e:?}"),
            Self::Model(s) => f.write_str(s),
        }
    }
}

impl std::error::Error for LoadError {}

fn bad(s: impl Into<String>) -> LoadError {
    LoadError::Model(s.into())
}

/// Hyperparameters, from the file's metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// Vocabulary size (rows of the embedding / head).
    pub vocab: usize,
    /// Hidden width.
    pub embd: usize,
    /// Transformer blocks.
    pub layers: usize,
    /// Query heads.
    pub heads: usize,
    /// Key/value heads (grouped-query attention).
    pub kv_heads: usize,
    /// Width of one head.
    pub head_dim: usize,
    /// Feed-forward width.
    pub ffn: usize,
    /// RMS norm epsilon.
    pub rms_eps: f32,
    /// `RoPE` base frequency.
    pub rope_base: f32,
    /// End-of-sequence token, when the file names one.
    pub eos: Option<u32>,
}

impl Config {
    /// The key/value width of one layer.
    #[must_use]
    pub const fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }
}

/// An `I2_S` projection: `rows x cols`, its codes at `codes..codes + rows * cols / 4` in the
/// file, and its scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ternary {
    /// Byte offset of the codes in the file.
    pub codes: usize,
    /// Output rows.
    pub rows: usize,
    /// Input columns (a multiple of 128).
    pub cols: usize,
    /// The tensor's scale (`y = sum / act_scale * scale`).
    pub scale: f32,
}

/// One transformer block.
#[derive(Clone, Debug)]
pub struct Layer {
    /// Pre-attention RMS norm weight.
    pub attn_norm: Vec<f32>,
    /// RMS norm weight on the attention output, before `o`.
    pub attn_sub_norm: Vec<f32>,
    /// Pre-FFN RMS norm weight.
    pub ffn_norm: Vec<f32>,
    /// RMS norm weight on the FFN's inner activation, before `down`.
    pub ffn_sub_norm: Vec<f32>,
    /// Projections.
    pub q: Ternary,
    /// Key projection.
    pub k: Ternary,
    /// Value projection.
    pub v: Ternary,
    /// Attention output projection.
    pub o: Ternary,
    /// FFN gate.
    pub gate: Ternary,
    /// FFN up.
    pub up: Ternary,
    /// FFN down.
    pub down: Ternary,
}

/// A loaded model.
pub struct Model {
    bytes: Vec<u8>,
    /// Hyperparameters.
    pub cfg: Config,
    /// Byte offset of the `f16` token embedding (`vocab x embd`), which is also the head when
    /// the file has no `output.weight`.
    pub embd: usize,
    /// Byte offset of the `f16` head (`vocab x embd`).
    pub head: usize,
    /// Final RMS norm weight.
    pub output_norm: Vec<f32>,
    /// The blocks.
    pub layers: Vec<Layer>,
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model").field("cfg", &self.cfg).field("bytes", &self.bytes.len()).finish_non_exhaustive()
    }
}

fn meta_u64(g: &Gguf<'_>, key: &str) -> Result<u64, LoadError> {
    g.get(key).and_then(Value::as_u64).ok_or_else(|| bad(format!("missing metadata `{key}`")))
}

fn meta_f32(g: &Gguf<'_>, key: &str) -> Option<f32> {
    match g.get(key)? {
        #[expect(clippy::cast_possible_truncation, reason = "stored as f32 in the file")]
        Value::Float(v) => Some(*v as f32),
        _ => None,
    }
}

fn usize_of(v: u64) -> Result<usize, LoadError> {
    usize::try_from(v).map_err(|_| bad("dimension does not fit usize"))
}

fn offset_in(bytes: &[u8], part: &[u8]) -> usize {
    part.as_ptr() as usize - bytes.as_ptr() as usize
}

impl Model {
    /// Read and check a model file.
    ///
    /// # Errors
    /// When the file cannot be read, is not a `GGUF`, or is not a `BitNet` b1.58 model with the
    /// tensors, types and shapes this engine expects.
    pub fn load(path: &std::path::Path) -> Result<Self, LoadError> {
        Self::from_bytes(std::fs::read(path).map_err(LoadError::Io)?)
    }

    /// Check a model held in memory.
    ///
    /// # Errors
    /// As [`Self::load`].
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, LoadError> {
        let g = Gguf::parse(&bytes).map_err(LoadError::Format)?;
        let arch = g.get("general.architecture").and_then(Value::as_str).ok_or_else(|| bad("no architecture"))?;
        if arch != rusty_ternary::gguf::ARCHITECTURE {
            return Err(bad(format!("architecture `{arch}`, not `{}`", rusty_ternary::gguf::ARCHITECTURE)));
        }
        let key = |k: &str| format!("{arch}.{k}");
        // Scaling this engine does not implement must not load silently.
        for k in ["rope.scaling.type", "rope.scaling.factor"] {
            if let Some(v) = g.get(&key(k))
                && !matches!(v, Value::Text(s) if s == "none")
            {
                return Err(bad(format!("`{}` = {v:?}: RoPE scaling is not supported", key(k))));
            }
        }
        if g.tensor("rope_freqs.weight").is_some() {
            return Err(bad("`rope_freqs.weight` (frequency factors) is not supported"));
        }
        let embd = usize_of(meta_u64(&g, &key("embedding_length"))?)?;
        let layers = usize_of(meta_u64(&g, &key("block_count"))?)?;
        let heads = usize_of(meta_u64(&g, &key("attention.head_count"))?)?;
        let kv_heads = usize_of(meta_u64(&g, &key("attention.head_count_kv")).unwrap_or(heads as u64))?;
        let ffn = usize_of(meta_u64(&g, &key("feed_forward_length"))?)?;
        if heads == 0 || kv_heads == 0 || embd % heads != 0 || heads % kv_heads != 0 {
            return Err(bad(format!("heads {heads} / kv heads {kv_heads} do not divide width {embd}")));
        }
        let head_dim = embd / heads;
        if let Ok(rot) = meta_u64(&g, &key("rope.dimension_count"))
            && usize_of(rot)? != head_dim
        {
            return Err(bad(format!("partial RoPE ({rot} of {head_dim}) is not supported")));
        }
        let rms_eps = meta_f32(&g, &key("attention.layer_norm_rms_epsilon")).ok_or_else(|| bad("no rms epsilon"))?;
        let rope_base = meta_f32(&g, &key("rope.freq_base")).unwrap_or(10_000.0);
        let eos = g.get("tokenizer.ggml.eos_token_id").and_then(Value::as_u64).and_then(|v| u32::try_from(v).ok());

        let tensor = |name: &str| g.tensor(name).ok_or_else(|| bad(format!("missing tensor `{name}`")));
        let f16_matrix = |name: &str, cols: usize| -> Result<(usize, usize), LoadError> {
            let t = tensor(name)?;
            if t.ty != TYPE_F16 || t.dims.len() != 2 || usize_of(t.dims[0])? != cols {
                return Err(bad(format!("`{name}`: want f16 [{cols}, vocab], got type {} {:?}", t.ty, t.dims)));
            }
            let rows = usize_of(t.dims[1])?;
            Ok((offset_in(&bytes, g.bytes_of(t).map_err(LoadError::Format)?), rows))
        };
        let (embd_off, vocab) = f16_matrix("token_embd.weight", embd)?;
        let head = if g.tensor("output.weight").is_some() {
            let (off, rows) = f16_matrix("output.weight", embd)?;
            if rows != vocab {
                return Err(bad("output.weight rows differ from the vocabulary"));
            }
            off
        } else {
            embd_off
        };
        let norm = |name: &str, len: usize| -> Result<Vec<f32>, LoadError> {
            let t = tensor(name)?;
            if t.ty != TYPE_F32 || t.elements() != Some(len) {
                return Err(bad(format!("`{name}`: want f32 [{len}], got type {} {:?}", t.ty, t.dims)));
            }
            let b = g.bytes_of(t).map_err(LoadError::Format)?;
            Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
        };
        let ternary = |name: &str, rows: usize, cols: usize| -> Result<Ternary, LoadError> {
            let t = tensor(name)?;
            if t.ty != TYPE_I2_S || t.dims.len() != 2 || usize_of(t.dims[0])? != cols || usize_of(t.dims[1])? != rows {
                return Err(bad(format!("`{name}`: want I2_S [{cols}, {rows}], got type {} {:?}", t.ty, t.dims)));
            }
            if !cols.is_multiple_of(i2s::BLOCK) {
                return Err(bad(format!("`{name}`: {cols} columns is not a whole number of blocks")));
            }
            let b = g.bytes_of(t).map_err(LoadError::Format)?;
            let scale = i2s::scale_of(b, rows, cols).map_err(LoadError::Format)?;
            if !scale.is_finite() {
                return Err(bad(format!("`{name}`: scale {scale}")));
            }
            Ok(Ternary { codes: offset_in(&bytes, b), rows, cols, scale })
        };
        let q_dim = heads * head_dim;
        let kv_dim = kv_heads * head_dim;
        let layers = (0..layers)
            .map(|i| {
                let n = |s: &str| format!("blk.{i}.{s}.weight");
                Ok(Layer {
                    attn_norm: norm(&n("attn_norm"), embd)?,
                    attn_sub_norm: norm(&n("attn_sub_norm"), q_dim)?,
                    ffn_norm: norm(&n("ffn_norm"), embd)?,
                    ffn_sub_norm: norm(&n("ffn_sub_norm"), ffn)?,
                    q: ternary(&n("attn_q"), q_dim, embd)?,
                    k: ternary(&n("attn_k"), kv_dim, embd)?,
                    v: ternary(&n("attn_v"), kv_dim, embd)?,
                    o: ternary(&n("attn_output"), embd, q_dim)?,
                    gate: ternary(&n("ffn_gate"), ffn, embd)?,
                    up: ternary(&n("ffn_up"), ffn, embd)?,
                    down: ternary(&n("ffn_down"), embd, ffn)?,
                })
            })
            .collect::<Result<Vec<_>, LoadError>>()?;
        let output_norm = norm("output_norm.weight", embd)?;
        let cfg = Config {
            vocab,
            embd,
            layers: layers.len(),
            heads,
            kv_heads,
            head_dim,
            ffn,
            rms_eps,
            rope_base,
            eos,
        };
        drop(g);
        Ok(Self { bytes, cfg, embd: embd_off, head, output_norm, layers })
    }

    /// A projection's codes (`rows * cols / 4` bytes).
    #[must_use]
    pub fn codes(&self, t: &Ternary) -> &[u8] {
        &self.bytes[t.codes..t.codes + t.rows * t.cols / 4]
    }

    /// Row `r` of an `f16` matrix at `off` with `cols` columns, as raw half bits.
    #[must_use]
    pub fn f16_row(&self, off: usize, cols: usize, r: usize) -> &[u8] {
        let start = off + r * cols * 2;
        &self.bytes[start..start + cols * 2]
    }

    /// The whole `f16` head as raw bytes (`vocab * embd * 2`).
    #[must_use]
    pub fn head_bytes(&self) -> &[u8] {
        &self.bytes[self.head..self.head + self.cfg.vocab * self.cfg.embd * 2]
    }
}
