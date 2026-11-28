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

/// Where a model's bytes live: a read-only map of the file (native targets: nothing is copied,
/// and a page is read from the OS's cache only when a kernel touches it), or the heap (wasm32,
/// which has no map, and models handed over in memory).
enum Store {
    Heap(Vec<u8>),
    #[cfg(not(target_arch = "wasm32"))]
    Mapped(memmap2::Mmap),
}

impl Store {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Heap(v) => v,
            #[cfg(not(target_arch = "wasm32"))]
            Self::Mapped(m) => m,
        }
    }

    /// Map `path` read-only.
    ///
    /// The contract a map carries: the file must not be modified or truncated while the model is
    /// alive. A model file is an immutable artifact (an export, a download checked by hash), so
    /// this holds by how Minerva is used; on Windows the OS also refuses to truncate a mapped
    /// file.
    #[cfg(not(target_arch = "wasm32"))]
    #[allow(unsafe_code)]
    fn map(path: &std::path::Path) -> std::io::Result<Self> {
        let f = std::fs::File::open(path)?;
        // SAFETY: read-only map of a file no one modifies while it is mapped (the contract
        // above). Every access then goes through `as_slice`, which bounds-checks.
        let m = unsafe { memmap2::Mmap::map(&f)? };
        Self::prefetch(&m);
        Ok(Self::Mapped(m))
    }

    /// Ask the OS to read the whole map into its file cache now, in large asynchronous reads: on
    /// a cold cache a map otherwise faults the file in a page at a time as the first token
    /// touches it (measured: 1.16x slower than a streamed read). A hint, non-blocking; it moves
    /// no bytes into this process and changes no results.
    #[cfg(not(target_arch = "wasm32"))]
    #[allow(unsafe_code)]
    fn prefetch(m: &memmap2::Mmap) {
        #[cfg(windows)]
        {
            #[repr(C)]
            struct Range {
                address: *const u8,
                size: usize,
            }
            unsafe extern "system" {
                fn GetCurrentProcess() -> isize;
                fn PrefetchVirtualMemory(
                    process: isize,
                    n: usize,
                    ranges: *const Range,
                    flags: u32,
                ) -> i32;
            }
            let r = Range {
                address: m.as_ptr(),
                size: m.len(),
            };
            // SAFETY: one range covering exactly the live mapping; the call only schedules reads.
            let _ = unsafe { PrefetchVirtualMemory(GetCurrentProcess(), 1, &raw const r, 0) };
        }
        #[cfg(unix)]
        {
            let _ = m.advise(memmap2::Advice::WillNeed);
        }
    }

    /// Ask the OS to drop the pages wholly inside `range` from this process's resident set. The
    /// bytes stay valid: a later read faults them back from the file (the OS's cache). A heap
    /// store keeps its bytes. A hint: a failure changes memory use, never results.
    #[allow(unsafe_code)]
    fn release(&self, range: std::ops::Range<usize>) {
        #[cfg(not(target_arch = "wasm32"))]
        if let Self::Mapped(m) = self {
            const PAGE: usize = 4096;
            let base = m.as_ptr() as usize;
            let lo = (base + range.start).next_multiple_of(PAGE) - base;
            let hi = ((base + range.end.min(m.len())) / PAGE * PAGE).saturating_sub(base);
            if hi <= lo {
                return;
            }
            #[cfg(windows)]
            {
                unsafe extern "system" {
                    fn VirtualUnlock(address: *const u8, size: usize) -> i32;
                }
                // SAFETY: `lo..hi` lies inside the live mapping. On pages that are not locked,
                // VirtualUnlock removes them from the working set and fails with
                // ERROR_NOT_LOCKED, which is the documented effect wanted here.
                let _ = unsafe { VirtualUnlock(m.as_ptr().add(lo), hi - lo) };
            }
            #[cfg(unix)]
            {
                // SAFETY: a read-only file-backed map: DONTNEED drops the pages, and the next
                // read faults the file's bytes back. Nothing is written, so nothing is lost.
                let _ = unsafe {
                    m.unchecked_advise_range(memmap2::UncheckedAdvice::DontNeed, lo, hi - lo)
                };
            }
        }
        let _ = range;
    }
}

/// A loaded model.
pub struct Model {
    store: Store,
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
    /// The greedy head's `int8` screen, built on first use ([`crate::head`]).
    pub screen: crate::head::Lazy,
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("cfg", &self.cfg)
            .field("bytes", &self.store.as_slice().len())
            .finish_non_exhaustive()
    }
}

fn meta_u64(g: &Gguf<'_>, key: &str) -> Result<u64, LoadError> {
    g.get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| bad(format!("missing metadata `{key}`")))
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

/// The whole file, read 16 MB at a time (wasm32, which cannot map): a WASI host (wasmtime) refuses one 1.2 GB `fd_read`
/// with `ENOMEM`, where many bounded reads into one buffer work on every target.
#[cfg_attr(
    not(target_arch = "wasm32"),
    allow(dead_code, reason = "wasm32's load path")
)]
fn read_chunked(path: &std::path::Path) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    const CHUNK: usize = 16 << 20;
    let mut f = std::fs::File::open(path)?;
    let len = usize::try_from(f.metadata()?.len()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::OutOfMemory,
            "file larger than the address space",
        )
    })?;
    let mut buf = vec![0u8; len];
    for part in buf.chunks_mut(CHUNK) {
        f.read_exact(part)?;
    }
    Ok(buf)
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
        // `MINERVA_LOAD=read` streams the file onto the heap instead of mapping it: the A/B arm
        // for cold-cache loads (a map faults pages in as kernels touch them; a read streams).
        #[cfg(not(target_arch = "wasm32"))]
        let store = if std::env::var("MINERVA_LOAD").as_deref() == Ok("read") {
            Store::Heap(read_chunked(path).map_err(LoadError::Io)?)
        } else {
            Store::map(path).map_err(LoadError::Io)?
        };
        #[cfg(target_arch = "wasm32")]
        let store = Store::Heap(read_chunked(path).map_err(LoadError::Io)?);
        Self::from_store(store)
    }

    /// Check a model held in memory.
    ///
    /// # Errors
    /// As [`Self::load`].
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, LoadError> {
        Self::from_store(Store::Heap(bytes))
    }

    fn from_store(store: Store) -> Result<Self, LoadError> {
        let bytes = store.as_slice();
        let g = Gguf::parse_lazy(bytes).map_err(LoadError::Format)?;
        let arch = g
            .get("general.architecture")
            .and_then(Value::as_str)
            .ok_or_else(|| bad("no architecture"))?;
        if arch != rusty_ternary::gguf::ARCHITECTURE {
            return Err(bad(format!(
                "architecture `{arch}`, not `{}`",
                rusty_ternary::gguf::ARCHITECTURE
            )));
        }
        let key = |k: &str| format!("{arch}.{k}");
        // Scaling this engine does not implement must not load silently.
        for k in ["rope.scaling.type", "rope.scaling.factor"] {
            if let Some(v) = g.get(&key(k))
                && !matches!(v, Value::Text(s) if s == "none")
            {
                return Err(bad(format!(
                    "`{}` = {v:?}: RoPE scaling is not supported",
                    key(k)
                )));
            }
        }
        if g.tensor("rope_freqs.weight").is_some() {
            return Err(bad(
                "`rope_freqs.weight` (frequency factors) is not supported",
            ));
        }
        let embd = usize_of(meta_u64(&g, &key("embedding_length"))?)?;
        let layers = usize_of(meta_u64(&g, &key("block_count"))?)?;
        let heads = usize_of(meta_u64(&g, &key("attention.head_count"))?)?;
        let kv_heads =
            usize_of(meta_u64(&g, &key("attention.head_count_kv")).unwrap_or(heads as u64))?;
        let ffn = usize_of(meta_u64(&g, &key("feed_forward_length"))?)?;
        if heads == 0 || kv_heads == 0 || embd % heads != 0 || heads % kv_heads != 0 {
            return Err(bad(format!(
                "heads {heads} / kv heads {kv_heads} do not divide width {embd}"
            )));
        }
        let head_dim = embd / heads;
        if let Ok(rot) = meta_u64(&g, &key("rope.dimension_count"))
            && usize_of(rot)? != head_dim
        {
            return Err(bad(format!(
                "partial RoPE ({rot} of {head_dim}) is not supported"
            )));
        }
        let rms_eps = meta_f32(&g, &key("attention.layer_norm_rms_epsilon"))
            .ok_or_else(|| bad("no rms epsilon"))?;
        let rope_base = meta_f32(&g, &key("rope.freq_base")).unwrap_or(10_000.0);
        let eos = g
            .get("tokenizer.ggml.eos_token_id")
            .and_then(Value::as_u64)
            .and_then(|v| u32::try_from(v).ok());

        let tensor = |name: &str| {
            g.tensor(name)
                .ok_or_else(|| bad(format!("missing tensor `{name}`")))
        };
        let f16_matrix = |name: &str, cols: usize| -> Result<(usize, usize), LoadError> {
            let t = tensor(name)?;
            if t.ty != TYPE_F16 || t.dims.len() != 2 || usize_of(t.dims[0])? != cols {
                return Err(bad(format!(
                    "`{name}`: want f16 [{cols}, vocab], got type {} {:?}",
                    t.ty, t.dims
                )));
            }
            let rows = usize_of(t.dims[1])?;
            Ok((
                offset_in(bytes, g.bytes_of(t).map_err(LoadError::Format)?),
                rows,
            ))
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
                return Err(bad(format!(
                    "`{name}`: want f32 [{len}], got type {} {:?}",
                    t.ty, t.dims
                )));
            }
            let b = g.bytes_of(t).map_err(LoadError::Format)?;
            Ok(b.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        };
        let ternary = |name: &str, rows: usize, cols: usize| -> Result<Ternary, LoadError> {
            let t = tensor(name)?;
            if t.ty != TYPE_I2_S
                || t.dims.len() != 2
                || usize_of(t.dims[0])? != cols
                || usize_of(t.dims[1])? != rows
            {
                return Err(bad(format!(
                    "`{name}`: want I2_S [{cols}, {rows}], got type {} {:?}",
                    t.ty, t.dims
                )));
            }
            if !cols.is_multiple_of(i2s::BLOCK) {
                return Err(bad(format!(
                    "`{name}`: {cols} columns is not a whole number of blocks"
                )));
            }
            let b = g.bytes_of(t).map_err(LoadError::Format)?;
            let scale = i2s::scale_of(b, rows, cols).map_err(LoadError::Format)?;
            if !scale.is_finite() {
                return Err(bad(format!("`{name}`: scale {scale}")));
            }
            // Code 3 is not a weight (no writer emits it). Refused here, once, so every kernel
            // may take `code <= 2` as given, and a corrupt file fails instead of reading 2a.
            i2s::check_codes(&b[..rows * cols / 4]).map_err(|_| {
                bad(format!(
                    "`{name}`: holds code 3, which is not an I2_S weight"
                ))
            })?;
            Ok(Ternary {
                codes: offset_in(bytes, b),
                rows,
                cols,
                scale,
            })
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
        Ok(Self {
            store,
            cfg,
            embd: embd_off,
            head,
            output_norm,
            layers,
            screen: crate::head::Lazy::default(),
        })
    }

    /// A projection's codes (`rows * cols / 4` bytes).
    #[must_use]
    pub fn codes(&self, t: &Ternary) -> &[u8] {
        &self.store.as_slice()[t.codes..t.codes + t.rows * t.cols / 4]
    }

    /// Row `r` of an `f16` matrix at `off` with `cols` columns, as raw half bits.
    #[must_use]
    pub fn f16_row(&self, off: usize, cols: usize, r: usize) -> &[u8] {
        let start = off + r * cols * 2;
        &self.store.as_slice()[start..start + cols * 2]
    }

    /// Drop the head's rows `rows` from the resident set (see `Store::release`): after the screen
    /// is built, greedy decoding reads only a few rows of the `f16` head a token.
    pub fn release_head_rows(&self, rows: std::ops::Range<usize>) {
        let row = self.cfg.embd * 2;
        self.store
            .release(self.head + rows.start * row..self.head + rows.end * row);
    }

    /// The whole `f16` head as raw bytes (`vocab * embd * 2`).
    #[must_use]
    pub fn head_bytes(&self) -> &[u8] {
        &self.store.as_slice()[self.head..self.head + self.cfg.vocab * self.cfg.embd * 2]
    }

    /// The whole model file (its metadata holds the tokenizer: [`crate::tokenizer`]).
    #[must_use]
    pub fn file_bytes(&self) -> &[u8] {
        self.store.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real model, from `MINERVA_GGUF` (the ignored tests need it).
    fn ours() -> String {
        std::env::var("MINERVA_GGUF").expect("set MINERVA_GGUF to a BitNet b1.58 I2_S GGUF")
    }

    /// A real model with one byte of `blk.0.attn_q`'s codes set to `0xff` (four code-3 weights)
    /// must be refused at load, naming the tensor; the untouched file must load.
    #[test]
    #[ignore = "needs the model in MINERVA_GGUF"]
    fn a_tensor_holding_code_3_is_refused() {
        let bytes = std::fs::read(ours()).expect("model file");
        let at = {
            let g = Gguf::parse(&bytes).expect("parse");
            let t = g.tensor("blk.0.attn_q.weight").expect("tensor");
            offset_in(&bytes, g.bytes_of(t).expect("bytes")) + 1000
        };
        let mut bad = bytes.clone();
        bad[at] = 0xff;
        let err = Model::from_bytes(bad).expect_err("code 3 must be refused");
        assert!(err.to_string().contains("blk.0.attn_q.weight"), "{err}");
        Model::from_bytes(bytes).expect("the real file loads");
    }
}
