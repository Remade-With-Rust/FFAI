//! The decode loop: `inputs_embeds` -> generated tokens, greedy, on candle.
//!
//! Step 5 of `docs/plans/argus-launch-plan.md`, whose gate is *"greedy decode
//! matches reference greedy decode"*.
//!
//! # Why a `candle` loop rather than a `mistral.rs` call
//!
//! The plan offers both ("decode loop (or `mistral.rs` call)"), and the house
//! doctrine's rule is *don't hand-roll an LLM SERVING loop on raw `candle`*,
//! because paging, quantization, sampling and constrained decoding are solved
//! there. That rule is about serving. What Argus needs here is a **greedy
//! prefill plus one-token-at-a-time decode for a single sequence** — which is
//! precisely what `mercury::asr::whisper_candle` already does in this tree,
//! and which `candle` supports directly:
//!
//! * `candle_transformers::models::llama` is `SmolVLM`'s text tower verbatim —
//!   `text_config.model_type` is literally `llama`;
//! * `Llama::forward_input_embed` takes INJECTED embeddings, which is exactly
//!   what a VLM needs and what `forward` (which embeds ids itself) cannot do;
//! * `llama::Cache` is the `KV` cache.
//!
//! Three things decided it:
//!
//! 1. **Publication.** `mistralrs` is on crates.io at 0.8.1, but the version
//!    proven to serve `SmolVLM` here is 0.9.0 from git — and `cargo publish`
//!    refuses a git dependency outright. `ffai-media`'s manifest records that
//!    this exact constraint once made every downstream `FFai` crate
//!    unpublishable and had to be undone. Taking a git dependency now would
//!    re-import that problem into `ffai-argus`.
//! 2. **It composes with work already gated.** Steps 3 and 4 produce
//!    `inputs_embeds` that the reference decoder turns into 32/32 identical
//!    tokens. The only missing piece is the loop itself.
//! 3. **Size.** This is ~100 lines against a large optional dependency, for a
//!    256M model.
//!
//! **`mistral.rs` is not rejected** — it remains the documented path for the
//! serving concerns it owns (quantized weights, grammar-constrained JSON
//! decoding — §2.3's v2 item), it is already proven to load and generate for
//! this checkpoint (Gate 1.2), and `ffai-argus` keeps the reserved
//! `mistralrs-backend` feature for it.
//!
//! # The weight-name adapter
//!
//! `SmolVLM` stores its text tower under `model.text_model.*` while `candle`'s
//! `Llama::load` looks for `model.*`; `lm_head` matches on both sides. That is
//! handled with `VarBuilder::rename_f`, which rewrites the LOOKUP rather than
//! copying tensors — the checkpoint stays memory-mapped.

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use ffai_core::engine::Decoding;
use candle_nn::VarBuilder;
use candle_transformers::models::llama;

/// Where the time went inside one generation.
///
/// Milliseconds, because that is the unit a reader can compare against their
/// own patience. Per-STEP rather than an average: the first token after a
/// prefill behaves differently from the fiftieth, and an average of the two
/// describes neither.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DecodeTrace {
    /// One forward pass over the entire prompt.
    pub prefill_ms: f64,
    /// Prompt length in tokens — for a VLM, mostly image tokens.
    pub prompt_tokens: usize,
    /// One entry per generated token.
    pub steps_ms: Vec<f64>,
}

impl DecodeTrace {
    /// Total time in the decode loop, excluding prefill.
    #[must_use]
    pub fn decode_ms(&self) -> f64 {
        self.steps_ms.iter().sum()
    }

    /// Generated tokens per second, prefill EXCLUDED.
    ///
    /// Excluded because including it makes the rate depend on the size of the
    /// picture, which is not what "tokens per second" means to anyone reading
    /// it. The prefill is reported separately and in full.
    #[must_use]
    pub fn tokens_per_sec(&self) -> f64 {
        let ms = self.decode_ms();
        if ms <= 0.0 {
            return 0.0;
        }
        self.steps_ms.len() as f64 / (ms / 1e3)
    }
}

/// `SmolVLM`'s text tower plus its `KV` cache.
pub struct TextDecoder {
    /// candle's tower — loaded ONLY when it is the one that will run.
    ///
    /// # This used to be loaded unconditionally, and it cost 540 MB
    ///
    /// The comment here previously claimed that holding both towers "costs
    /// address space rather than memory, because the weights are mmapped".
    /// That is wrong: `VarBuilder::get_unchecked` calls candle's `convert`,
    /// which **allocates a tensor and copies** out of the mapping — the same
    /// fact that motivated `ffai-carmenta`'s SVTR weight cache. Two towers
    /// therefore meant two full f32 copies of a 135M-parameter model.
    ///
    /// Measured: the footprint gate went from **PASS 0.71x** to **FAIL 1.20x**
    /// when our tower landed, steady resident rising 1309 -> 2126 MiB. A second
    /// copy of the text weights is 540 MB of that.
    model: Option<llama::Llama>,
    cache: llama::Cache,
    /// A pristine copy of the cache, cloned back over `cache` before every
    /// generation.
    ///
    /// `candle`'s `llama::Cache` has no `reset` and its `kvs` are private. An
    /// engine that generates twice from one `&self` would otherwise have its
    /// SECOND caption prefixed by the first one's keys and values — the same
    /// image and prompt producing a different answer depending on what ran
    /// before it. Cloning is cheap: the cos/sin tables are `Tensor`s (an `Arc`
    /// bump) and a pristine `kvs` is a vector of `None`.
    pristine: llama::Cache,
    config: llama::Config,
    device: Device,
    /// Our own tower — the one that actually runs, unless the toggle says
    /// otherwise.
    ///
    /// candle's `llama` stays loaded beside it as the ORACLE: the A/B in
    /// `examples/text_ab` and the `FFAI_ARGUS_CANDLE_TEXT=1` arm both need a
    /// reference in the same process, and a reference you cannot run is not a
    /// reference. The weights are mmapped, so holding both costs address
    /// space rather than memory.
    ours: Option<crate::text::TextTower>,
}

/// Force candle's text tower instead of ours.
///
/// Read ONCE and cached — a toggle inside a per-element loop is a
/// vectorisation barrier, which is a mistake this workspace has already paid
/// for once (`ffai-diana`'s `silu`, 1.92x).
fn use_candle_text() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static C: AtomicU8 = AtomicU8::new(u8::MAX);
    match C.load(Ordering::Relaxed) {
        u8::MAX => {
            let on = std::env::var("FFAI_ARGUS_CANDLE_TEXT").is_ok_and(|v| v == "1");
            C.store(u8::from(on), Ordering::Relaxed);
            on
        }
        v => v == 1,
    }
}

impl TextDecoder {
    /// Load the text tower from a checkpoint.
    ///
    /// # Errors
    /// Propagates `candle`'s load errors; a missing tensor names itself, which
    /// is what a wrong prefix produces.
    pub fn load(weights: &std::path::Path, config_json: &str, device: &Device) -> Result<Self, String> {
        // SAFETY: the mapped file is owned by the model cache and is not
        // mutated while this process holds it.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&weights), DType::F32, device)
        }
        .map_err(|e| format!("load {}: {e}", weights.display()))?;
        Self::load_vb(vb, config_json, device)
    }

    /// Build the decoder from a `VarBuilder` the caller already has.
    ///
    /// The path constructor above is written in terms of this, so a browser
    /// and a server build the same decoder from the same tensors.
    ///
    /// **One builder, cloned — not two loads.** The path version used to map
    /// the file twice, once renamed for candle's tower and once raw for ours.
    /// A `VarBuilder` is cheap to clone (its backend is shared), and on wasm a
    /// second load would mean a second COPY of the checkpoint in a 32-bit
    /// address space that is already the binding constraint.
    pub fn load_vb(
        vb: VarBuilder<'static>,
        config_json: &str,
        device: &Device,
    ) -> Result<Self, String> {
        let config = text_config_from_json(config_json)?;
        // Our tower reads the checkpoint's own names, so it keeps a builder
        // WITHOUT the rename applied below.
        let raw = vb.clone();

        // Rewrite the lookup, do not copy the weights: candle asks for
        // `model.embed_tokens`, the checkpoint stores
        // `model.text_model.embed_tokens`, and `lm_head` is the same on both
        // sides. Renaming the QUERY keeps the mmap intact.
        let vb = vb.rename_f(|name: &str| {
            if let Some(rest) = name.strip_prefix("model.") {
                format!("model.text_model.{rest}")
            } else {
                name.to_string()
            }
        });

        let cache = llama::Cache::new(true, DType::F32, &config, device)
            .map_err(|e| format!("kv cache: {e}"))?;
        // Deferred: built below only if ours could not be, so the losing
        // tower's weights are never materialised.
        let load_candle = |vb: VarBuilder<'static>| {
            llama::Llama::load(vb, &config).map_err(|e| format!("text tower: {e}"))
        };

        let v: serde_json::Value =
            serde_json::from_str(config_json).map_err(|e| format!("config.json: {e}"))?;
        let t = v.get("text_config").unwrap_or(&v);
        let gu = |k: &str, d: u64| t.get(k).and_then(serde_json::Value::as_u64).unwrap_or(d);
        let gfl = |k: &str, d: f64| t.get(k).and_then(serde_json::Value::as_f64).unwrap_or(d);
        let heads = gu("num_attention_heads", 9) as usize;
        let hidden = gu("hidden_size", 576) as usize;
        let cfg = crate::text::Cfg {
            layers: gu("num_hidden_layers", 30) as usize,
            hidden,
            heads,
            kv_heads: gu("num_key_value_heads", 3) as usize,
            head_dim: hidden / heads.max(1),
            inter: gu("intermediate_size", 1536) as usize,
            eps: gfl("rms_norm_eps", 1e-5),
            rope_theta: gfl("rope_theta", 100_000.0) as f32,
            max_pos: gu("max_position_embeddings", 8192) as usize,
        };
        // A tower that fails to load is a fallback to candle's, not an error:
        // the engine's contract is a caption, and candle's path is gated too.
        let ours = if use_candle_text() {
            None
        } else {
            crate::text::TextTower::load(&raw, cfg, device).ok()
        };
        // EXACTLY ONE tower is resident. `ours` is preferred; candle's is built
        // only when ours is absent — because the toggle asked for it, or
        // because ours failed to load and the engine must still caption.
        let model = if ours.is_some() { None } else { Some(load_candle(vb)?) };

        Ok(Self {
            model,
            pristine: cache.clone(),
            cache,
            config,
            device: device.clone(),
            ours,
        })
    }

    /// Load with candle's tower forced — the ORACLE arm.
    ///
    /// `examples/text_ab` needs both implementations live in ONE process to
    /// compare them. The env toggle cannot do that: it is read once and cached
    /// (deliberately — a toggle re-read per call is the barrier
    /// `ffai-diana`'s `silu` paid 1.92x for). Without this constructor the A/B
    /// silently compared our tower against itself and reported a max logit
    /// delta of exactly 0.000e0, which is what a broken instrument looks like
    /// when it looks like a pass.
    ///
    /// # Errors
    /// Same as [`Self::load`].
    pub fn load_reference(
        weights: &std::path::Path,
        config_json: &str,
        device: &Device,
    ) -> Result<Self, String> {
        // Ask for candle's tower up front rather than loading ours and then
        // discarding it — dropping a tower still pays for having built it.
        // SAFETY: same mapped file, same ownership as `load`.
        let raw = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&weights),
                DType::F32,
                device,
            )
        }
        .map_err(|e| format!("load {}: {e}", weights.display()))?;
        let config = text_config_from_json(config_json)?;
        let vb = raw.rename_f(|name: &str| {
            if let Some(rest) = name.strip_prefix("model.") {
                format!("model.text_model.{rest}")
            } else {
                name.to_string()
            }
        });
        let cache = llama::Cache::new(true, DType::F32, &config, device)
            .map_err(|e| format!("kv cache: {e}"))?;
        let model = llama::Llama::load(vb, &config).map_err(|e| format!("text tower: {e}"))?;
        Ok(Self {
            model: Some(model),
            pristine: cache.clone(),
            cache,
            config,
            device: device.clone(),
            ours: None,
        })
    }

    /// Drop everything the previous generation left in the `KV` cache.
    ///
    /// [`Self::generate`] calls this unconditionally at its top, so a caller
    /// cannot forget it.
    pub fn reset(&mut self) {
        self.cache = self.pristine.clone();
        if let Some(t) = self.ours.as_mut() {
            t.reset();
        }
    }

    /// Logits for the LAST position, given a slice of the sequence.
    ///
    /// `index_pos` is where this slice starts in the whole sequence — 0 for the
    /// prefill, then the running length. Getting it wrong does not error: `RoPE`
    /// simply rotates by the wrong amount and the output degrades, which is the
    /// same silent class as a mis-assembled prompt.
    pub fn forward_embeds(&mut self, embeds: &Tensor, index_pos: usize) -> CandleResult<Tensor> {
        if let Some(t) = self.ours.as_mut() {
            return t.forward(embeds, index_pos);
        }
        let Some(m) = self.model.as_ref() else {
            return Err(candle_core::Error::Msg("no text tower loaded".into()));
        };
        m.forward_input_embed(embeds, index_pos, &mut self.cache)
    }

    /// Embed token ids through the tower's own table.
    pub fn embed(&self, ids: &Tensor) -> CandleResult<Tensor> {
        if let Some(t) = self.ours.as_ref() {
            return t.embed(ids);
        }
        let Some(m) = self.model.as_ref() else {
            return Err(candle_core::Error::Msg("no text tower loaded".into()));
        };
        m.embed(ids)
    }

    /// Greedy generation from a prefilled embedding sequence.
    ///
    /// Deterministic by construction — `argmax`, no sampling, no seed needed.
    /// That is the plan's §2 Gate 2 requirement (`Decoding::Greedy` is the
    /// default and the only variant that needs no seed) and it is also what
    /// makes step 5's gate a token-equality check rather than a distribution
    /// comparison.
    ///
    /// # Errors
    /// Propagates `candle` errors from the forward passes.
    pub fn generate_greedy(
        &mut self,
        inputs_embeds: &Tensor,
        max_new_tokens: usize,
        stop_ids: &[u32],
    ) -> CandleResult<Vec<u32>> {
        self.generate(inputs_embeds, max_new_tokens, stop_ids, &Decoding::Greedy, None)
    }

    /// Generation under any [`Decoding`] strategy.
    ///
    /// [`Decoding::Greedy`] takes the `argmax` path and needs no seed;
    /// [`Decoding::Sampled`] builds `candle`'s `LogitsProcessor` from the
    /// caller's seed, so two runs with the same seed produce the same text.
    /// That is what Gate 2 bought by putting the seed in the TYPE rather than
    /// in an engine's private state.
    ///
    /// # Errors
    /// Propagates `candle` errors from the forward passes.
    pub fn generate(
        &mut self,
        inputs_embeds: &Tensor,
        max_new_tokens: usize,
        stop_ids: &[u32],
        decoding: &Decoding,
        repetition_penalty: Option<f32>,
    ) -> CandleResult<Vec<u32>> {
        self.generate_traced(
            inputs_embeds,
            max_new_tokens,
            stop_ids,
            decoding,
            repetition_penalty,
            None,
        )
    }

    /// [`Self::generate`], optionally filling in a per-step timing trace.
    ///
    /// The split it records is the one that matters for understanding VLM
    /// latency: **prefill is one pass over the whole prompt, decode is one
    /// pass per token.** For a VLM the prompt is mostly image tokens — 1088 of
    /// them for a single split still — so prefill is a large, fixed cost that
    /// has nothing to do with how long the answer is. Reporting a single
    /// "generation" number hides that, and hiding it is how people conclude
    /// the decoder is slow when the picture is what cost them.
    ///
    /// # Errors
    /// Propagates `candle` errors from the forward passes.
    pub fn generate_traced(
        &mut self,
        inputs_embeds: &Tensor,
        max_new_tokens: usize,
        stop_ids: &[u32],
        decoding: &Decoding,
        repetition_penalty: Option<f32>,
        mut trace: Option<&mut DecodeTrace>,
    ) -> CandleResult<Vec<u32>> {
        // Unconditional, at the top. A stale cache presents as "the second
        // caption is wrong", which is not a symptom anyone attributes to a
        // cache.
        self.reset();

        let mut sampler = match decoding {
            Decoding::Greedy => None,
            Decoding::Sampled {
                temperature,
                top_p,
                top_k,
                seed,
            } => {
                let t = f64::from(*temperature);
                Some(LogitsProcessor::from_sampling(
                    *seed,
                    match (top_k, top_p) {
                        (Some(k), Some(p)) => Sampling::TopKThenTopP {
                            k: *k,
                            p: f64::from(*p),
                            temperature: t,
                        },
                        (Some(k), None) => Sampling::TopK {
                            k: *k,
                            temperature: t,
                        },
                        (None, Some(p)) => Sampling::TopP {
                            p: f64::from(*p),
                            temperature: t,
                        },
                        (None, None) => Sampling::All { temperature: t },
                    },
                ))
            }
        };

        let (_b, prefill_len, _d) = inputs_embeds.dims3()?;
        // Prefill: the whole prompt in one pass, populating the KV cache.
        let t_prefill = crate::clock::Instant::now();
        let mut logits = self.forward_embeds(inputs_embeds, 0)?;
        if let Some(t) = trace.as_deref_mut() {
            t.prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;
            t.prompt_tokens = prefill_len;
        }
        let mut out: Vec<u32> = Vec::with_capacity(max_new_tokens);
        let mut pos = prefill_len;

        for _ in 0..max_new_tokens {
            let t_step = crate::clock::Instant::now();
            let mut step = logits.flatten_all()?;
            // A LOGIT transform, so it applies to greedy too — which is why it
            // is not a field of `Decoding::Sampled`. Small models loop under
            // greedy more than under sampling, not less.
            if let Some(p) = repetition_penalty {
                // SITE-REVIEWED false positive. This is not an equality
                // test on a computed float, it is a SENTINEL check: 1.0 is the
                // documented "penalty disabled" default and arrives as that
                // literal from the config. An epsilon window would make a
                // penalty of 1.0000001 silently do nothing.
                #[allow(clippy::float_cmp)]
                if p != 1.0 && !out.is_empty() {
                    step = candle_transformers::utils::apply_repeat_penalty(&step, p, &out)?;
                }
            }
            let next = match sampler.as_mut() {
                Some(s) => s.sample(&step)?,
                None => argmax(&step)?,
            };
            if stop_ids.contains(&next) {
                break;
            }
            out.push(next);
            // One token at a time, cache-appended — the step the KV cache
            // exists for. Re-running the whole prefix each step would be
            // correct and quadratic.
            let ids = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
            let emb = self.embed(&ids)?;
            logits = self.forward_embeds(&emb, pos)?;
            pos += 1;
            if let Some(t) = trace.as_deref_mut() {
                t.steps_ms.push(t_step.elapsed().as_secs_f64() * 1e3);
            }
        }
        Ok(out)
    }

    #[must_use]
    pub const fn config(&self) -> &llama::Config {
        &self.config
    }
}

/// Argmax over a `(1, vocab)` or `(vocab,)` logits tensor.
///
/// Written out rather than using a sort: it is O(vocab) against O(v log v),
/// runs once per generated token, and — more importantly — makes the tie rule
/// explicit. `>` keeps the FIRST maximum, which is what `torch.argmax` does;
/// `>=` would keep the last and could disagree with the reference on an exact
/// tie. Ties are rare and this is exactly the kind of detail that produces a
/// one-token difference nobody can explain later.
fn argmax(logits: &Tensor) -> CandleResult<u32> {
    let v = logits.flatten_all()?.to_vec1::<f32>()?;
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    u32::try_from(best).map_err(|e| candle_core::Error::Msg(format!("token id overflow: {e}")))
}

/// Read `text_config` out of a checkpoint's config.json into `candle`'s shape.
///
/// # Errors
/// If the JSON is malformed or `text_config` is missing/incompatible.
pub fn text_config_from_json(config_json: &str) -> Result<llama::Config, String> {
    let v: serde_json::Value =
        serde_json::from_str(config_json).map_err(|e| format!("config.json: {e}"))?;
    let tc = v
        .get("text_config")
        .ok_or("config.json has no text_config")?;
    let get_usize = |k: &str| -> Result<usize, String> {
        tc.get(k)
            .and_then(serde_json::Value::as_u64)
            .map(|x| x as usize)
            .ok_or_else(|| format!("text_config has no {k}"))
    };
    // Built field by field rather than deserialized whole: SmolVLM's
    // `text_config` carries ~70 keys of HF generation boilerplate that
    // candle's `Config` has no fields for, and a strict deserialize would
    // reject the lot. Naming what we read also makes it visible WHICH
    // properties the decoder depends on.
    Ok(llama::Config {
        hidden_size: get_usize("hidden_size")?,
        intermediate_size: get_usize("intermediate_size")?,
        vocab_size: get_usize("vocab_size")?,
        num_hidden_layers: get_usize("num_hidden_layers")?,
        num_attention_heads: get_usize("num_attention_heads")?,
        num_key_value_heads: get_usize("num_key_value_heads")
            .or_else(|_| get_usize("num_attention_heads"))?,
        rms_norm_eps: tc
            .get("rms_norm_eps")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1e-5),
        rope_theta: tc
            .get("rope_theta")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(10000.0) as f32,
        bos_token_id: tc.get("bos_token_id").and_then(serde_json::Value::as_u64).map(|x| x as u32),
        eos_token_id: tc
            .get("eos_token_id")
            .and_then(serde_json::Value::as_u64)
            .map(|x| llama::LlamaEosToks::Single(x as u32)),
        rope_scaling: None,
        max_position_embeddings: get_usize("max_position_embeddings").unwrap_or(8192),
        tie_word_embeddings: tc
            .get("tie_word_embeddings")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        use_flash_attn: false,
    })
}
