//! Minerva in the browser: the whole BitNet b1.58 model in WebAssembly, behind wasm-bindgen.
//!
//! The page (or, better, a Web Worker) streams the model file into a [`ModelBuffer`] in the
//! module's own memory (no second copy in JavaScript), builds an [`Engine`] from it, then chats.
//! The engine is the same code as the native one; built with `+simd128` it runs the simd128 tier,
//! and the logits hash ([`Engine::full_head_fnv`]) equals the native `minerva generate` hash for
//! the same file and ids, which is how the browser build is checked.

use ffai_minerva::engine::{self, HeadMode, Session};
use ffai_minerva::model::Model;
use ffai_minerva::sampling::{Rng, Sampling, sample, sample_from};
use ffai_minerva::tokenizer::{StreamDecoder, Tokenizer};
use serde_json::{Value, json};
use wasm_bindgen::prelude::*;

/// With the `threads` feature: `initThreadPool(n)` starts rayon's pool on `n` Web Workers. Await it
/// once, before building the [`Engine`].
#[cfg(feature = "threads")]
pub use wasm_bindgen_rayon::init_thread_pool;

/// The model file's bytes, written in place by JavaScript through [`Self::ptr`].
#[wasm_bindgen]
pub struct ModelBuffer {
    bytes: Vec<u8>,
}

#[wasm_bindgen]
impl ModelBuffer {
    /// A zeroed buffer of `len` bytes (the model file's size).
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new(len: usize) -> Self {
        Self {
            bytes: vec![0; len],
        }
    }

    /// Where the bytes start in the module's memory: JavaScript writes each fetched chunk at
    /// `ptr + offset` through a `Uint8Array` over `memory.buffer`.
    pub fn ptr(&mut self) -> *mut u8 {
        self.bytes.as_mut_ptr()
    }

    /// The buffer's length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

/// A loaded model, its tokenizer and one conversation's session.
#[wasm_bindgen]
pub struct Engine {
    m: &'static Model,
    tok: Tokenizer,
    s: Session<'static>,
    /// Tokens whose keys and values the session holds.
    cached: Vec<u32>,
    eog: Vec<u32>,
}

fn js(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

fn now_ms() -> f64 {
    js_sys::Date::now()
}

#[wasm_bindgen]
impl Engine {
    /// Check and load the model in `buf`, with a context of `ctx` positions.
    ///
    /// # Errors
    /// When the bytes are not a `BitNet` b1.58 `I2_S` model with a byte-level BPE tokenizer.
    #[wasm_bindgen(constructor)]
    pub fn new(buf: ModelBuffer, ctx: usize) -> Result<Engine, JsError> {
        let m = Model::from_bytes(buf.bytes).map_err(js)?;
        // One model per page, alive as long as the page: leaking it gives the session its
        // 'static borrow.
        let m: &'static Model = Box::leak(Box::new(m));
        let tok = Tokenizer::from_gguf(m.file_bytes()).map_err(js)?;
        let mut eog: Vec<u32> = tok.eos.into_iter().collect();
        for name in ["<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"] {
            if let Some(id) = tok.special(name)
                && !eog.contains(&id)
            {
                eog.push(id);
            }
        }
        Ok(Self {
            m,
            tok,
            s: Session::new(m, ctx.max(2)),
            cached: Vec::new(),
            eog,
        })
    }

    /// The instruction set the session runs (`simd128` or `scalar`).
    #[must_use]
    pub fn isa(&self) -> String {
        self.s.isa().name().to_owned()
    }

    /// The context size.
    #[must_use]
    pub fn ctx(&self) -> usize {
        self.s.ctx()
    }

    /// Token ids for `text` (special-token strings parsed), with BOS first when `bos`.
    #[must_use]
    pub fn tokenize(&self, text: &str, bos: bool) -> Vec<u32> {
        self.tok.encode(text, bos && self.tok.add_bos)
    }

    /// Text for `ids`.
    #[must_use]
    pub fn detokenize(&self, ids: Vec<u32>) -> String {
        self.tok.decode(&ids)
    }

    /// A conversation (`[{"role": ..., "content": ...}, ...]` as JSON) rendered through the
    /// model's chat template, ending with the assistant's turn.
    ///
    /// # Errors
    /// On bad JSON or a template this renderer does not know.
    pub fn apply_chat_template(&self, messages_json: &str) -> Result<String, JsError> {
        let v: Value = serde_json::from_str(messages_json).map_err(js)?;
        let msgs: Vec<(String, String)> = v
            .as_array()
            .ok_or_else(|| js("messages must be an array"))?
            .iter()
            .map(|m| {
                let f = |k: &str| m.get(k).and_then(Value::as_str).unwrap_or("").to_owned();
                (f("role"), f("content"))
            })
            .collect();
        self.tok.apply_chat_template(&msgs, true).map_err(js)
    }

    /// Generate from `prompt` (text, tokenized with BOS), calling `on_piece(text)` as each piece
    /// of the reply completes. Stops at an end-of-turn token or after `n_predict` tokens.
    /// `temperature <= 0` is greedy. The prompt cache keeps the prefix shared with the last call.
    /// Returns the run's statistics as JSON.
    ///
    /// # Errors
    /// When the prompt does not fit the context, or the callback throws.
    #[expect(clippy::too_many_arguments, reason = "the sampling settings, flat for JavaScript")]
    pub fn complete(
        &mut self,
        prompt: &str,
        n_predict: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        min_p: f32,
        seed: f64,
        on_piece: &js_sys::Function,
    ) -> Result<String, JsError> {
        let ids = self.tok.encode(prompt, self.tok.add_bos);
        let ctx = self.s.ctx();
        if ids.is_empty() || ids.len() >= ctx {
            return Err(js(format!(
                "the prompt ({} tokens) does not fit the context ({ctx})",
                ids.len()
            )));
        }
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a seed from JavaScript"
        )]
        let sp = Sampling {
            temperature,
            top_k,
            top_p,
            min_p,
            seed: seed.max(0.0) as u64,
        };
        let n_predict = n_predict.min(ctx - ids.len());
        let shared = self.cached.iter().zip(&ids).take_while(|(a, b)| a == b).count();
        let keep = shared.min(ids.len() - 1).min(self.s.pos());
        self.s.rewind(keep);
        self.cached.truncate(keep);
        let t0 = now_ms();
        self.s.prefill(&ids[keep..ids.len() - 1]).map_err(js)?;
        self.cached.extend_from_slice(&ids[keep..ids.len() - 1]);
        let mut rng = Rng::new(sp.seed);
        let mut dec = StreamDecoder::new();
        let mut next = ids[ids.len() - 1];
        let (mut n, mut first_ms, mut stop) = (0usize, 0.0, "limit");
        for i in 0..n_predict {
            let t = if sp.is_greedy() {
                self.s.step_argmax(next).map_err(js)?
            } else if sp.top_k > 0 {
                let (c, _) = self.s.step_top_k(next, sp.top_k).map_err(js)?;
                sample_from(&c, &sp, &mut rng)
            } else {
                let logits = self.s.step(next, true).map_err(js)?.unwrap_or_default();
                sample(logits, &sp, &mut rng)
            };
            self.cached.push(next);
            if i == 0 {
                first_ms = now_ms() - t0;
            }
            if self.eog.contains(&t) {
                stop = "eos";
                break;
            }
            n += 1;
            let piece = dec.push(&self.tok, t);
            if !piece.is_empty() {
                on_piece
                    .call1(&JsValue::NULL, &JsValue::from_str(&piece))
                    .map_err(|e| js(format!("{e:?}")))?;
            }
            next = t;
        }
        let rest = dec.finish();
        if !rest.is_empty() {
            on_piece
                .call1(&JsValue::NULL, &JsValue::from_str(&rest))
                .map_err(|e| js(format!("{e:?}")))?;
        }
        let total = now_ms() - t0;
        #[expect(clippy::cast_precision_loss, reason = "token counts")]
        let stats = json!({
            "prompt_n": ids.len() - keep,
            "cached": keep,
            "prompt_ms": first_ms,
            "predicted_n": n,
            "predicted_ms": total - first_ms,
            "tokens_per_second": n.saturating_sub(1) as f64 / ((total - first_ms) / 1e3).max(1e-9),
            "stop": stop,
            "seed": sp.seed,
            "isa": self.s.isa().name(),
        });
        Ok(stats.to_string())
    }

    /// Forget the conversation's cache.
    pub fn reset(&mut self) {
        self.s.rewind(0);
        self.cached.clear();
    }

    /// The check against native: greedy decode `new_tokens` after `ids` through the full head
    /// and return the FNV-1a over every generated step's logit bits, as `minerva generate` with
    /// `MINERVA_HEAD=full` prints it (`logits-fnv`), plus the ids.
    ///
    /// # Errors
    /// When the ids do not fit the context or are outside the vocabulary.
    pub fn full_head_fnv(&mut self, ids: Vec<u32>, new_tokens: usize) -> Result<String, JsError> {
        self.reset();
        let mut fnv = 0xcbf2_9ce4_8422_2325_u64;
        let t0 = now_ms();
        let out = engine::generate(&mut self.s, &ids, new_tokens, HeadMode::Full, |_, logits, _| {
            for l in logits.unwrap_or_default() {
                for b in l.to_bits().to_le_bytes() {
                    fnv = (fnv ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
                }
            }
        })
        .map_err(js)?;
        let ms = now_ms() - t0;
        self.reset();
        Ok(json!({"fnv": format!("{fnv:016x}"), "ids": out, "ms": ms, "isa": self.s.isa().name(), "vocab": self.m.cfg.vocab}).to_string())
    }
}
