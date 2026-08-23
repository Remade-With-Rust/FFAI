//! The `VlmEngine` implementation: an `ImageBuffer` in, a caption out.
//!
//! Step 6 of `docs/plans/argus-launch-plan.md`. Steps 3, 4 and 5 each gated one
//! brick against the reference in isolation; this is the one that composes them
//! and is therefore the one where a *plumbing* mistake — a stale cache, a
//! dropped alpha channel, a grid the prompt and the pixels disagree about —
//! can appear without any brick being wrong.
//!
//! # The four things this file does, in order
//!
//! 1. **RGB8.** Every image arrives as an [`ImageBuffer`] in one of three pixel
//!    formats. The tower takes one. Converting here, once, means no other stage
//!    has to know that grayscale exists.
//! 2. **Preprocess + tower**, per tile, into a `(tiles, tokens_per_tile, d)`
//!    block of image embeddings.
//! 3. **Assemble** the chat turn, tokenize it, embed the ids, and splice the
//!    image embeddings over the `<image>` placeholders.
//! 4. **Decode** greedily (or sampled, per [`Decoding`]) and detokenize.
//!
//! # Why the geometry is carried and not recomputed
//!
//! [`preprocess_rgb8`] returns the tile grid it chose, and that grid is what
//! builds the `<row_r_col_c>` markers. The alternative — deriving the grid
//! twice, once for pixels and once for text — is how the two silently disagree,
//! and the failure mode is not an error but a caption of the wrong thing.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use candle_core::{Device, Tensor};
use ffai_core::engine::{
    EngineInfo, EngineStatus, Task, VlmEngine, VlmOptions, VlmPart, VlmPrompt,
};
use ffai_core::error::{Error, Result};
use ffai_core::types::{ImageBuffer, PixelFormat, TimedSegment, VideoFrame};

use crate::decode::TextDecoder;
use crate::preprocess::preprocess_rgb8_opts;
use crate::prompt::{merge_image_embeddings, PromptLayout};
use crate::vision::SmolVlmVision;

/// The checkpoint this engine is gated against.
///
/// Named as a constant because it is not a default that can drift: every
/// number in the launch plan's steps 3-6 was measured on this exact
/// checkpoint, and the prompt template in `prompt.rs` is *its* template.
pub const MODEL: &str = "smolvlm-256m-instruct";

/// Generation budget when the caller gives none.
const DEFAULT_MAX_NEW_TOKENS: usize = 256;

/// Frames per video caption when the caller gives none.
///
/// Eight unsplit frames is 512 image tokens — comfortably inside the tower's
/// 8192 positions with room for the question and the answer — and eight
/// samples is enough to see a change rather than a moment. It is a default,
/// not a limit: [`VlmOptions::frames_per_window`] overrides it, and the error
/// on overflow names this knob.
const DEFAULT_FRAMES_PER_WINDOW: usize = 8;

/// Where the time went, stage by stage, for one caption.
///
/// # Why this exists on the engine rather than in the caller
///
/// A caller *can* time `describe_image` and get one number. That number is
/// almost useless for a VLM, because it is dominated by a cost the caller
/// cannot see: **the vision tower runs once per tile, and a still image is
/// seventeen tiles.** A reader looking at "4.2 seconds" concludes the language
/// model is slow. The truth is usually that they handed it a picture.
///
/// So the split is reported by the code that knows it, and the alternative —
/// a demo or a profiler reassembling the pipeline out of the public pieces to
/// time each one — is exactly the drift this crate spent §16 avoiding.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CaptionTrace {
    /// Image -> `pixel_values`: two Lanczos resizes, the tile cut, normalise.
    pub preprocess_ms: f64,
    /// `SigLIP` + connector, summed over every tile of every image.
    pub tower_ms: f64,
    /// One entry per tile, in the order the tower saw them. The last belongs
    /// to the global thumbnail.
    pub tower_per_tile_ms: Vec<f64>,
    /// Chat template, tokenize, embed, splice.
    pub assemble_ms: f64,
    /// One forward pass over the whole prompt.
    pub prefill_ms: f64,
    /// One entry per generated token.
    pub step_ms: Vec<f64>,
    /// Detokenize and trim.
    pub detokenize_ms: f64,

    pub tiles: usize,
    pub rows: usize,
    pub cols: usize,
    /// Tile edge in pixels (512 for this checkpoint).
    pub tile: usize,
    /// `tiles * tokens_per_tile` — what the picture costs the prompt.
    pub image_tokens: usize,
    /// Whatever is left of the prompt once the images are counted out.
    pub text_tokens: usize,
    pub prompt_tokens: usize,
    /// The tower's own token budget, so a reader can see the headroom.
    pub max_positions: usize,
    /// Was tile splitting on? Off means video framing — 1 tile per image.
    pub split: bool,
    /// The size each image was resized to before tiling, in order.
    pub resized_to: Vec<(usize, usize)>,
}

impl CaptionTrace {
    /// Time in the decode loop, excluding prefill.
    #[must_use]
    pub fn decode_ms(&self) -> f64 {
        self.step_ms.iter().sum()
    }

    /// Everything this engine spent, decode of the source file excluded —
    /// that happens before the engine is called.
    #[must_use]
    pub fn total_ms(&self) -> f64 {
        self.preprocess_ms
            + self.tower_ms
            + self.assemble_ms
            + self.prefill_ms
            + self.decode_ms()
            + self.detokenize_ms
    }

    /// Generated tokens per second, prefill excluded — see [`DecodeTrace`].
    #[must_use]
    pub fn tokens_per_sec(&self) -> f64 {
        let ms = self.decode_ms();
        if ms <= 0.0 {
            return 0.0;
        }
        self.step_ms.len() as f64 / (ms / 1e3)
    }
}

/// `SmolVLM`-256M-Instruct on candle: `SigLIP` tower, pixel-shuffle connector,
/// Llama decoder.
///
/// Chosen in Gate 1.2 over `SmolVLM2` for one decidable reason: only v1 has a
/// published `OCRBench` row to be scored against, which is what makes Arm 1 an
/// external number rather than our own claim.
pub struct SmolVlm {
    manifest_dir: PathBuf,
    /// Loaded on first use. A registry installs every engine at startup, and
    /// loading ~500 MB of weights because someone ran `ffai --help` is not a
    /// cost the user asked for.
    model: OnceLock<std::result::Result<Model, String>>,
}

struct Model {
    vision: SmolVlmVision,
    /// `&mut` because generation walks a `KV` cache; `VlmEngine::describe`
    /// takes `&self`, so the interior mutability lives here rather than
    /// forcing every caller to own the engine mutably.
    decoder: Mutex<TextDecoder>,
    tokenizer: tokenizers::Tokenizer,
    layout: PromptLayout,
    image_token_id: i64,
    /// Every id that ends a turn. `<end_of_utterance>` is the one that
    /// actually fires for this checkpoint; the config's `eos_token_id` is kept
    /// beside it because a checkpoint that disagreed with its own tokenizer
    /// would otherwise run to the token budget every time.
    stop_ids: Vec<u32>,
    /// The text tower's position budget. Read from the checkpoint so the
    /// overflow error can name the real number rather than a guess, and so a
    /// larger SmolVLM raises the ceiling without a code change.
    max_positions: usize,
    device: Device,
}

impl SmolVlm {
    /// An engine reading manifests from the workspace's `models/` directory.
    #[must_use]
    pub fn new() -> Self {
        Self::with_manifest_dir(PathBuf::from("models"))
    }

    #[must_use]
    pub fn with_manifest_dir(dir: PathBuf) -> Self {
        Self {
            manifest_dir: dir,
            model: OnceLock::new(),
        }
    }

    /// Are the weights already resident?
    ///
    /// Loading is ~1 GB of safetensors and happens once, on first use. A
    /// caller timing a caption needs to know whether that cost landed inside
    /// the measurement, because a first call that includes it is not the same
    /// event as a warm one and averaging the two describes neither.
    #[must_use]
    pub fn is_loaded(&self) -> bool {
        self.model.get().is_some()
    }

    /// Force the load now, so a later timed call does not pay for it.
    ///
    /// # Errors
    /// Whatever loading the checkpoint would return.
    pub fn warm(&self) -> Result<()> {
        self.model().map(|_| ())
    }

    fn model(&self) -> Result<&Model> {
        match self
            .model
            .get_or_init(|| load(&self.manifest_dir).map_err(|e| e.to_string()))
        {
            Ok(m) => Ok(m),
            Err(e) => Err(Error::Model(e.clone())),
        }
    }
}

impl SmolVlm {
    /// [`VlmEngine::describe_image`], with a stage-by-stage timing trace.
    ///
    /// Same code path as the untraced call — it *is* the untraced call, with a
    /// trace threaded through — so the numbers describe what actually runs
    /// rather than a parallel implementation that exists to be measured.
    ///
    /// # Errors
    /// Whatever `describe_image` would return.
    pub fn describe_image_traced(
        &self,
        image: &ImageBuffer,
        opts: &VlmOptions,
    ) -> Result<(String, CaptionTrace)> {
        let mut trace = CaptionTrace::default();
        let mut pieces = vec![Piece::Image(image)];
        if let Some(t) = opts.prompt.as_deref() {
            pieces.push(Piece::Text(t));
        }
        let text = self.caption_traced(&pieces, true, opts, Some(&mut trace))?;
        Ok((text, trace))
    }
}

impl Default for SmolVlm {
    fn default() -> Self {
        Self::new()
    }
}

impl VlmEngine for SmolVlm {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            name: "smolvlm".into(),
            task: Task::Vlm,
            // `Stable` means "oracle-gated against a reference
            // implementation", and that is precisely what steps 3-6 bought:
            // the whole content path reproduces the reference's tokens 32/32
            // from a raw image. Anything less would be `Experimental`.
            status: EngineStatus::Stable,
            description:
                "SmolVLM-256M-Instruct on candle — SigLIP tower, pixel-shuffle connector, Llama decoder"
                    .into(),
        }
    }

    fn describe(&self, prompt: &VlmPrompt<'_>, opts: &VlmOptions) -> Result<String> {
        if prompt.image_count() == 0 {
            return Err(Error::Other(
                "argus: describe() needs at least one image — this is a VLM, not a chat model"
                    .into(),
            ));
        }
        let pieces: Vec<Piece<'_>> = prompt
            .parts
            .iter()
            .map(|p| match p {
                VlmPart::Text(t) => Piece::Text(t),
                VlmPart::Image(i) => Piece::Image(i),
            })
            .collect();
        // Stills SPLIT: seventeen tiles is what lets the model read fine
        // print, and a single still has all 8192 positions to itself.
        self.caption(&pieces, true, opts)
    }

    fn describe_video(
        &self,
        frames: &[VideoFrame],
        opts: &VlmOptions,
    ) -> Result<Vec<TimedSegment<String>>> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        let window = opts
            .frames_per_window
            .unwrap_or(DEFAULT_FRAMES_PER_WINDOW)
            .max(1);
        let step = median_step(frames);
        let mut out = Vec::with_capacity(frames.len().div_ceil(window));

        for (w, chunk) in frames.chunks(window).enumerate() {
            // Every frame of the window in ONE prompt, then the question.
            // This is the difference between video understanding and captioning
            // stills in a loop: the model sees the frames together, so it can
            // answer about change rather than about one moment.
            let mut pieces: Vec<Piece<'_>> =
                chunk.iter().map(|f| Piece::Image(&f.image)).collect();
            if let Some(t) = opts.prompt.as_deref() {
                pieces.push(Piece::Text(t));
            }
            // Video does NOT split — see `preprocess_rgb8_opts`. With splitting
            // on, a four-frame window is 4352 image tokens and an eight-frame
            // window does not fit at all.
            let value = self.caption(&pieces, false, opts)?;

            let start = chunk[0].timestamp;
            // A window runs until the next window starts. The last one has no
            // successor, so it takes the sampling step rather than claiming a
            // zero-length segment no player would show.
            let end = frames
                .get((w + 1) * window)
                .map_or_else(|| chunk[chunk.len() - 1].timestamp + step, |f| f.timestamp);
            out.push(TimedSegment {
                start,
                end,
                value,
                confidence: None,
            });
        }
        Ok(out)
    }
}

/// How many tiles to run at once.
///
/// `FFAI_ARGUS_TILE_WORKERS` overrides; 0 or unset picks the default below.
fn tile_workers(tiles: usize) -> usize {
    if let Some(n) = std::env::var("FFAI_ARGUS_TILE_WORKERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
    {
        return n.min(tiles).max(1);
    }
    // Measured on a 24-core box, 17 tiles, min-of-3, bit-identical throughout:
    //
    //   workers |  1     2     4     6     8    12    17
    //   speedup | 1.03  1.64  2.19  2.50  2.37  2.44  2.69
    //
    // 17 is fastest but every concurrent tower materialises its own
    // `(1, 12, 1024, 1024)` attention matrix — **50 MiB each** — so 17 workers
    // is ~850 MiB of transient peak against a footprint gate this engine
    // currently PASSES at 0.71x. Six workers keeps 93 % of the win for ~35 % of
    // that memory, which is the right side of a trade between a gate we win and
    // a gate we lose.
    //
    // Scaled by cores rather than fixed, so a 4-core laptop does not spawn six
    // towers into four cores.
    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    (cores / 4).clamp(1, 6).min(tiles.max(1))
}

/// Run the vision tower over every tile, concurrently.
///
/// # Why threads and not a bigger batch
///
/// candle's CPU backend uses rayon for `conv2d` and nothing else — its
/// elementwise and layout kernels are **single-threaded**. Half of a `SigLIP`
/// encoder layer is exactly those ops (`GELU` alone is 19.7 % of one, measured
/// by `examples/vision_ops_probe`), so for half the tower a 24-core box runs
/// one core.
///
/// Batching cannot fix that: the kernel stays single-threaded however long the
/// array is, which is why `examples/tile_batching_ab` measured only 1.07x.
/// Threading can, because seventeen tiles are seventeen independent
/// single-threaded workloads.
///
/// **Bit-identical**, and not by assumption: `examples/tile_parallel_ab`
/// compares the threaded result against the sequential one tensor by tensor and
/// asserts `max_abs == 0`. Each tile is its own forward pass over shared,
/// immutable weights — there is no accumulation order to change.
fn run_tower(
    vision: &crate::vision::SmolVlmVision,
    pre: &crate::preprocess::Preprocessed,
    device: &Device,
) -> Result<(Vec<Tensor>, Vec<f64>)> {
    let per = 3 * pre.tile * pre.tile;
    let workers = tile_workers(pre.tiles);
    if workers <= 1 || pre.tiles <= 1 {
        let mut out = Vec::with_capacity(pre.tiles);
        let mut ms = Vec::with_capacity(pre.tiles);
        for t in 0..pre.tiles {
            let t0 = std::time::Instant::now();
            let px = pre.pixel_values[t * per..(t + 1) * per].to_vec();
            let tensor = Tensor::from_vec(px, (1, 3, pre.tile, pre.tile), device)?;
            out.push(vision.forward(&tensor)?.squeeze(0)?);
            ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        return Ok((out, ms));
    }

    // The kernels stand down: this loop is the parallelism now. Restored
    // before returning so a later single-tile call gets them back.
    let prev = crate::siglip::set_kernels_parallel(false);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let slots: Mutex<Vec<Option<(Tensor, f64)>>> = Mutex::new((0..pre.tiles).map(|_| None).collect());
    let failed: Mutex<Option<String>> = Mutex::new(None);

    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                // Claimed by an atomic counter rather than a fixed split: the
                // tiles cost the same in principle, and in practice the last
                // worker of a fixed split waits on whichever core the OS
                // descheduled.
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= pre.tiles {
                    break;
                }
                let t0 = std::time::Instant::now();
                let px = pre.pixel_values[i * per..(i + 1) * per].to_vec();
                let done = Tensor::from_vec(px, (1, 3, pre.tile, pre.tile), device)
                    .and_then(|t| vision.forward(&t))
                    .and_then(|o| o.squeeze(0));
                match done {
                    Ok(block) => {
                        if let Ok(mut g) = slots.lock() {
                            g[i] = Some((block, t0.elapsed().as_secs_f64() * 1e3));
                        }
                    }
                    Err(e) => {
                        // Record and KEEP DRAINING. Returning early would leave
                        // the other workers writing into a scope that is trying
                        // to unwind, and the first error is the informative one
                        // anyway.
                        if let Ok(mut f) = failed.lock() {
                            f.get_or_insert_with(|| e.to_string());
                        }
                    }
                }
            });
        }
    });

    crate::siglip::set_kernels_parallel(prev);
    if let Some(e) = failed.into_inner().ok().flatten() {
        return Err(Error::Model(format!("vision tower: {e}")));
    }
    let done = slots
        .into_inner()
        .map_err(|_| Error::Other("argus: tile results poisoned".into()))?;
    let mut out = Vec::with_capacity(pre.tiles);
    let mut ms = Vec::with_capacity(pre.tiles);
    for (i, slot) in done.into_iter().enumerate() {
        let (block, t) = slot.ok_or_else(|| Error::Model(format!("tile {i} produced nothing")))?;
        out.push(block);
        ms.push(t);
    }
    Ok((out, ms))
}

/// One element of an assembled prompt.
///
/// A private mirror of [`VlmPart`] so the shared assembly path can be driven
/// by `describe_video`, whose inputs are [`VideoFrame`]s rather than a
/// [`VlmPrompt`]. Copying two lines here is cheaper than making the trait's
/// prompt type do double duty.
enum Piece<'a> {
    Text(&'a str),
    Image(&'a ImageBuffer),
}

impl SmolVlm {
    /// The one assembly path: images through the tower, text interleaved in
    /// place, spliced, decoded.
    ///
    /// Both entry points funnel through here so they cannot drift in the
    /// things that are easy to get subtly different — the chat template, the
    /// order images are consumed in, the stop handling — while still differing
    /// in the one thing that should differ, `split`.
    fn caption(&self, pieces: &[Piece<'_>], split: bool, opts: &VlmOptions) -> Result<String> {
        self.caption_traced(pieces, split, opts, None)
    }

    fn caption_traced(
        &self,
        pieces: &[Piece<'_>],
        split: bool,
        opts: &VlmOptions,
        mut trace: Option<&mut CaptionTrace>,
    ) -> Result<String> {
        let m = self.model()?;

        // Price the prompt FIRST, from geometry alone. Every image here costs
        // a resize and a vision-tower pass, so discovering the prompt does not
        // fit after paying for two hundred of them is four minutes spent to
        // learn something two integers could have said.
        let images: Vec<&ImageBuffer> = pieces
            .iter()
            .filter_map(|p| match p {
                Piece::Image(i) => Some(*i),
                Piece::Text(_) => None,
            })
            .collect();
        let planned: usize = images
            .iter()
            .map(|i| {
                crate::preprocess::tile_geometry(i.width as usize, i.height as usize, split).0
            })
            .sum();
        let planned_tokens = planned * m.layout.tokens_per_tile;
        if planned_tokens >= m.max_positions {
            return Err(Error::Other(format!(
                "argus: {} image(s) would contribute {planned_tokens} image tokens \
                 ({planned} tiles x {}), and the text tower holds {}. {}",
                images.len(),
                m.layout.tokens_per_tile,
                m.max_positions,
                if split {
                    "Stills are split into 17 tiles each; pass fewer images."
                } else {
                    "Reduce frames_per_window (--window)."
                }
            )));
        }

        // Run the tower and build the text in the SAME walk, so the Nth image
        // block in the string is the Nth image's blocks in the tensor. Two
        // separate walks is how those get out of order.
        let mut blocks: Vec<Tensor> = Vec::new();
        let mut text = String::new();
        for piece in pieces {
            match piece {
                Piece::Text(t) => text.push_str(t),
                Piece::Image(img) => {
                    let t_pre = std::time::Instant::now();
                    let rgb = to_rgb8(img)?;
                    let pre = preprocess_rgb8_opts(
                        &rgb,
                        img.width as usize,
                        img.height as usize,
                        split,
                    );
                    if let Some(tr) = trace.as_deref_mut() {
                        tr.preprocess_ms += t_pre.elapsed().as_secs_f64() * 1e3;
                        tr.tiles += pre.tiles;
                        tr.rows = pre.rows;
                        tr.cols = pre.cols;
                        tr.tile = pre.tile;
                        tr.resized_to.push(crate::preprocess::resized_size(
                            img.width as usize,
                            img.height as usize,
                        ));
                    }
                    let t_tower = std::time::Instant::now();
                    let (tiles, per_tile_ms) = run_tower(&m.vision, &pre, &m.device)?;
                    blocks.extend(tiles);
                    if let Some(tr) = trace.as_deref_mut() {
                        // WALL time, not the sum of the per-tile timings. The
                        // tiles now overlap, so summing them would report more
                        // milliseconds than actually elapsed — a timeline that
                        // adds up to more than the clock is worse than no
                        // timeline. The per-tile figures are kept for the
                        // distribution and are explicitly concurrent.
                        tr.tower_ms += t_tower.elapsed().as_secs_f64() * 1e3;
                        tr.tower_per_tile_ms.extend(per_tile_ms);
                    }
                    text.push_str(&m.layout.image_block(pre.rows, pre.cols));
                }
            }
        }

        // The chat turn. `user_turn` would rebuild the image block, so the
        // template is applied to the text already assembled.
        let t_asm = std::time::Instant::now();
        let templated = format!("<|im_start|>User:{text}<end_of_utterance>\nAssistant:");
        let enc = m
            .tokenizer
            .encode(templated.as_str(), true)
            .map_err(|e| Error::Model(format!("tokenize: {e}")))?;
        let ids: Vec<i64> = enc.get_ids().iter().map(|&i| i64::from(i)).collect();
        // Backstop. The geometry check above bounds the IMAGE tokens; this
        // catches a prompt whose TEXT pushes it over — a long question, or a
        // template that grew. Kept because the two failures have different
        // causes and a caller shouldn't have to guess which one they hit.
        if ids.len() >= m.max_positions {
            return Err(Error::Other(format!(
                "argus: assembled prompt is {} tokens but the text tower holds {} — \
                 {planned_tokens} of those are image tokens, so the rest is text",
                ids.len(),
                m.max_positions
            )));
        }

        let image_hidden = Tensor::stack(&blocks, 0)?;
        let id_tensor = Tensor::from_vec(enc.get_ids().to_vec(), (1, ids.len()), &m.device)?;

        let mut dec = m
            .decoder
            .lock()
            .map_err(|_| Error::Other("argus: decoder mutex poisoned".into()))?;
        let text_embeds = dec.embed(&id_tensor)?;
        let merged = merge_image_embeddings(&text_embeds, &image_hidden, &ids, m.image_token_id)?;

        let mut dtrace = crate::decode::DecodeTrace::default();
        if let Some(tr) = trace.as_deref_mut() {
            tr.assemble_ms += t_asm.elapsed().as_secs_f64() * 1e3;
            tr.image_tokens = planned_tokens;
            tr.prompt_tokens = ids.len();
            tr.text_tokens = ids.len().saturating_sub(planned_tokens);
            tr.max_positions = m.max_positions;
            tr.split = split;
        }
        let out = dec.generate_traced(
            &merged,
            opts.max_new_tokens.unwrap_or(DEFAULT_MAX_NEW_TOKENS),
            &m.stop_ids,
            &opts.decoding,
            opts.repetition_penalty,
            Some(&mut dtrace),
        )?;
        drop(dec);

        let t_detok = std::time::Instant::now();
        let decoded = m
            .tokenizer
            .decode(&out, true)
            .map_err(|e| Error::Model(format!("detokenize: {e}")))?;
        let answer = truncate_at_stop(&decoded, &opts.stop).trim().to_string();
        if let Some(tr) = trace.as_deref_mut() {
            tr.prefill_ms = dtrace.prefill_ms;
            tr.step_ms = dtrace.steps_ms;
            tr.detokenize_ms = t_detok.elapsed().as_secs_f64() * 1e3;
        }
        Ok(answer)
    }
}

/// Median spacing between sampled frames, for the final segment's duration.
fn median_step(frames: &[VideoFrame]) -> f64 {
    if frames.len() < 2 {
        return 1.0;
    }
    let mut steps: Vec<f64> = frames.windows(2).map(|w| w[1].timestamp - w[0].timestamp).collect();
    steps.sort_by(f64::total_cmp);
    steps[steps.len() / 2]
}

/// Any supported pixel format to packed RGB8.
///
/// Grayscale is replicated across the three channels rather than being fed to
/// a one-channel tower: `SigLIP` has three input channels and there is no
/// grayscale variant of it. Alpha is DROPPED, not composited — compositing
/// would need a background colour, and inventing one changes the picture.
fn to_rgb8(img: &ImageBuffer) -> Result<Vec<u8>> {
    if img.width == 0 || img.height == 0 {
        // Traced rather than guessed: a 0x0 buffer survives the whole content
        // path (every resample window is empty, so nothing is indexed) and
        // comes out as a black 512x512 thumbnail the model captions happily.
        // A confident caption of a nonexistent image is worse than an error.
        return Err(Error::Media(format!(
            "argus: image is {}x{} — nothing to describe",
            img.width, img.height
        )));
    }
    let n = img.width as usize * img.height as usize;
    let want = n * img.format.bytes_per_pixel();
    if img.data.len() < want {
        return Err(Error::Media(format!(
            "argus: {}x{} {:?} needs {want} bytes, buffer has {}",
            img.width,
            img.height,
            img.format,
            img.data.len()
        )));
    }
    Ok(match img.format {
        PixelFormat::Rgb8 => img.data[..want].to_vec(),
        PixelFormat::Rgba8 => img.data[..want]
            .chunks_exact(4)
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect(),
        PixelFormat::Gray8 => img.data[..want].iter().flat_map(|&g| [g, g, g]).collect(),
    })
}

/// Cut the text at the first stop string.
///
/// The stop string itself is not included, which is what [`VlmOptions::stop`]
/// documents. Applied to the DECODED text rather than to token ids because a
/// stop string need not be a token boundary.
fn truncate_at_stop<'a>(text: &'a str, stops: &[String]) -> &'a str {
    let cut = stops
        .iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| text.find(s.as_str()))
        .min();
    cut.map_or(text, |i| &text[..i])
}

fn load(manifest_dir: &Path) -> Result<Model> {
    let manifests = ffai_models::load_dir(manifest_dir)?;
    let manifest = manifests
        .iter()
        .find(|m| m.name == MODEL)
        .ok_or_else(|| {
            Error::Model(format!(
                "no model manifest named `{MODEL}` in {}",
                manifest_dir.display()
            ))
        })?;
    let resolved = manifest.fetch()?;
    let weights = resolved.file("model.safetensors")?.to_path_buf();
    let config_path = resolved.file("config.json")?;
    let tokenizer_path = resolved.file("tokenizer.json")?;
    let config_json = std::fs::read_to_string(config_path)?;

    let device = Device::Cpu;
    let vision = crate::vision::load(&weights, &config_json, &device).map_err(Error::Model)?;
    let decoder = TextDecoder::load(&weights, &config_json, &device).map_err(Error::Model)?;
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
        .map_err(|e| Error::Model(format!("tokenizer: {e}")))?;

    // Geometry from the checkpoint, never constants: a different SmolVLM size
    // changes tokens_per_tile, and a hard-coded 64 would be silently wrong
    // there — the same defect class prompt.rs exists to guard against.
    let cfg: serde_json::Value = serde_json::from_str(&config_json)
        .map_err(|e| Error::Model(format!("config.json: {e}")))?;
    let vision_cfg = cfg.get("vision_config");
    let get = |k: &str, d: usize| -> usize {
        vision_cfg
            .and_then(|v| v.get(k))
            .and_then(serde_json::Value::as_u64)
            .map_or(d, |x| x as usize)
    };
    let scale_factor = cfg
        .get("scale_factor")
        .and_then(serde_json::Value::as_u64)
        .map_or(4, |x| x as usize);
    let layout = PromptLayout::default().with_geometry(
        get("image_size", 512),
        get("patch_size", 16),
        scale_factor,
    );

    let id_of = |t: &str| tokenizer.token_to_id(t).map(i64::from);
    let image_token_id = id_of("<image>")
        .ok_or_else(|| Error::Model("tokenizer has no `<image>` token".into()))?;

    let mut stop_ids: Vec<u32> = Vec::new();
    for t in ["<end_of_utterance>", "<|im_end|>", "<|endoftext|>"] {
        if let Some(id) = tokenizer.token_to_id(t) {
            stop_ids.push(id);
        }
    }
    if let Some(id) = cfg
        .get("text_config")
        .and_then(|t| t.get("eos_token_id"))
        .and_then(serde_json::Value::as_u64)
    {
        let id = id as u32;
        if !stop_ids.contains(&id) {
            stop_ids.push(id);
        }
    }
    if stop_ids.is_empty() {
        return Err(Error::Model(
            "no end-of-turn token found in the tokenizer or config — every caption would run to the token budget".into(),
        ));
    }

    let max_positions = cfg
        .get("text_config")
        .and_then(|t| t.get("max_position_embeddings"))
        .and_then(serde_json::Value::as_u64)
        .map_or(8192, |x| x as usize);

    Ok(Model {
        vision,
        decoder: Mutex::new(decoder),
        tokenizer,
        layout,
        image_token_id,
        stop_ids,
        max_positions,
        device,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffai_core::engine::Decoding;

    fn img(format: PixelFormat, data: Vec<u8>) -> ImageBuffer {
        ImageBuffer {
            width: 2,
            height: 1,
            format,
            data,
        }
    }

    #[test]
    fn grayscale_is_replicated_and_alpha_is_dropped() {
        assert_eq!(
            to_rgb8(&img(PixelFormat::Gray8, vec![10, 200])).unwrap(),
            vec![10, 10, 10, 200, 200, 200]
        );
        assert_eq!(
            to_rgb8(&img(PixelFormat::Rgba8, vec![1, 2, 3, 255, 4, 5, 6, 0])).unwrap(),
            vec![1, 2, 3, 4, 5, 6],
            "alpha is dropped, not composited — compositing needs a background \
             colour and inventing one changes the picture"
        );
        assert_eq!(
            to_rgb8(&img(PixelFormat::Rgb8, vec![1, 2, 3, 4, 5, 6])).unwrap(),
            vec![1, 2, 3, 4, 5, 6]
        );
    }

    #[test]
    fn a_zero_dimension_image_is_refused() {
        let e = to_rgb8(&ImageBuffer {
            width: 0,
            height: 0,
            format: PixelFormat::Rgb8,
            data: Vec::new(),
        })
        .unwrap_err();
        assert!(format!("{e}").contains("nothing to describe"), "{e}");
    }

    #[test]
    fn a_short_buffer_is_an_error_not_a_panic() {
        // A truncated decode must name the shortfall rather than slice-panic
        // three stages later inside the resampler.
        let e = to_rgb8(&img(PixelFormat::Rgb8, vec![1, 2, 3])).unwrap_err();
        assert!(format!("{e}").contains("needs 6 bytes"), "{e}");
    }

    #[test]
    fn stops_cut_before_the_marker_and_take_the_earliest() {
        let stops = vec!["\nUser:".to_string(), "###".to_string()];
        assert_eq!(truncate_at_stop("a cat ### b", &stops), "a cat ");
        assert_eq!(truncate_at_stop("x\nUser: y ### z", &stops), "x");
        assert_eq!(truncate_at_stop("no marker", &stops), "no marker");
        // An empty stop string would match at 0 and blank every caption.
        assert_eq!(truncate_at_stop("keep me", &[String::new()]), "keep me");
    }

    #[test]
    fn the_last_video_segment_gets_the_median_spacing() {
        let f = |t: f64| VideoFrame {
            image: img(PixelFormat::Gray8, vec![0, 0]),
            timestamp: t,
        };
        assert_eq!(median_step(&[f(0.0), f(0.5), f(1.0)]), 0.5);
        // One frame has no spacing to take a median of; 1.0 s beats a
        // zero-length segment that no player would show.
        assert_eq!(median_step(&[f(0.0)]), 1.0);
    }

    #[test]
    fn the_default_decoding_is_greedy_so_a_caption_is_reproducible() {
        assert_eq!(VlmOptions::default().decoding, Decoding::Greedy);
    }
}
