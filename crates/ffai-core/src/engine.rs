//! One trait per task, many engines per trait.
//!
//! An *engine* is a named, swappable implementation of a task — exactly the
//! role a codec plays in ffmpeg. Engines are registered in an
//! [`crate::registry::EngineRegistry`] and selected by name.

use std::fmt;
use std::str::FromStr;

use crate::error::Result;
use crate::types::{
    AudioBuffer, DetectOutput, ImageBuffer, OcrOutput, TimedSegment, Transcript, VideoFrame,
};

/// The tasks `FFai` knows about (the "stream types" of the toolkit).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Task {
    Asr,
    Tts,
    Ocr,
    Vlm,
    /// Object detection (Diana). The task exists ahead of its first engine so
    /// `ffai bench detect` can baseline the world references (M-D0); the
    /// `DetectEngine` trait and `DetectOutput` type land with the first
    /// engine at M-D1.
    Detect,
    /// Monocular depth estimation (Diana). Shares Diana's backbone and neck
    /// with [`Task::Detect`] — only the final head differs — but the output
    /// is a dense metric map rather than boxes, so it is its own task.
    Depth,
}

impl fmt::Display for Task {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Asr => "asr",
            Self::Tts => "tts",
            Self::Ocr => "ocr",
            Self::Vlm => "vlm",
            Self::Detect => "detect",
            Self::Depth => "depth",
        })
    }
}

impl FromStr for Task {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "asr" => Ok(Self::Asr),
            "tts" => Ok(Self::Tts),
            "ocr" => Ok(Self::Ocr),
            "vlm" => Ok(Self::Vlm),
            "detect" => Ok(Self::Detect),
            other => Err(format!(
                "unknown task `{other}` (expected asr, tts, ocr, vlm, or detect)"
            )),
        }
    }
}

/// Honesty marker shown in `ffai engines` — stubs are visible, not hidden.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineStatus {
    /// Registered and selectable, returns `Error::NotImplemented`.
    Stub,
    /// Works, not yet oracle-gated against a reference implementation.
    Experimental,
    /// Oracle-gated against a reference implementation.
    Stable,
}

impl fmt::Display for EngineStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Stub => "stub",
            Self::Experimental => "experimental",
            Self::Stable => "stable",
        })
    }
}

/// Metadata every engine exposes for discovery (`ffai engines`).
#[derive(Debug, Clone)]
pub struct EngineInfo {
    pub name: String,
    pub task: Task,
    pub status: EngineStatus,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct AsrOptions {
    /// Force a language instead of auto-detecting.
    pub language: Option<String>,
    /// Word-level timestamps (WhisperX-style forced alignment).
    pub word_timestamps: bool,
    /// Speaker diarization (WhisperX-style).
    pub diarize: bool,
    /// Keep speaker identities across calls, so a voice heard in one chunk
    /// keeps its label in the next.
    ///
    /// Off by default, and the default is the batch behaviour every
    /// diarization system has: labels are arbitrary names for clusters within
    /// ONE call, and `SPEAKER_00` in two separate calls need not be the same
    /// person. That is fine for a file and useless for a stream.
    ///
    /// With this set, the engine keeps a speaker registry between calls. Call
    /// `WhisperCandle::reset_speakers()` when a new recording begins — a new
    /// session is a new set of people, and carrying identities across is
    /// worse than starting fresh.
    ///
    /// Matching is deliberately stricter than in-call clustering: a registry
    /// merge is permanent, and two people who share a centroid stay merged for
    /// the rest of the session.
    pub persist_speakers: bool,
    /// Known speaker count, when the caller has one ("this is an interview,
    /// two people"). Overrides the clustering threshold.
    ///
    /// Measured caution: with the threshold tuned this is **not** the safer
    /// choice. Blind clustering scores 4.21 % DER against 5.00 % with the
    /// true count supplied, because forcing a count forces a merge, and a bad
    /// merge attributes one speaker's words to another. Set it when the count
    /// is certain, not as insurance.
    pub max_speakers: Option<usize>,
    /// Cosine-distance threshold for merging speaker clusters.
    ///
    /// Swept against DER on a 6-conversation corpus: the minimum sits at
    /// 0.85 (2.71 %), and 0.80 (4.21 %) ships instead because over-merging
    /// fails catastrophically (44.7 % at 0.95) while over-splitting fails
    /// gently. See `ffai_mercury::asr::diarize::DEFAULT_THRESHOLD`.
    pub diarize_threshold: f32,
    /// Translate to English instead of transcribing.
    pub translate: bool,
    /// Segment on speech before transcribing, so silence never reaches the
    /// model.
    ///
    /// **On by default, for measured speed — not for quality.**
    ///
    /// - Audio with trailing silence: 2.2–4.2× faster, transcript byte-identical.
    /// - Silent input: empty transcript, with no encoder pass at all.
    /// - A live sliding window stops spending five encoder passes to produce
    ///   nothing.
    ///
    /// Corpus WER *does* move with this on (test-clean 7.99 → 6.79,
    /// test-other 16.79 → 16.43), and that is **not** a quality improvement —
    /// do not cite it as one. Per-clip decomposition over 400 clips gives 38
    /// improved and 38 worsened, a sign test of z = 0.00. VAD shifts where
    /// speech sits inside Whisper's fixed 30 s context by ~0.2 s, which
    /// re-rolls the decode on about a fifth of clips, half each way; the
    /// aggregate moved because WER is dominated by a few high-delta clips.
    /// Full descent: `docs/whys/vad-quality.md`.
    ///
    /// Set `false` for the unsegmented fixed-30 s-grid behaviour.
    pub vad: bool,
    /// Speech threshold, 0..1, higher being stricter. Only read when
    /// [`Self::vad`] is set.
    pub vad_threshold: f32,
    /// Pack speech regions into windows of at most this many seconds.
    pub vad_chunk_secs: f32,
    /// Where this buffer starts in the wider stream, in seconds.
    ///
    /// Only meaningful for a streaming caller that re-sends a sliding window
    /// (a live transcriber sending the trailing N seconds every tick). It
    /// costs nothing to leave at `0.0`.
    ///
    /// **What it buys.** Diarization sub-segments each speech region into
    /// 1.5 s windows and embeds each one — the dominant cost, ~172 ms apiece.
    /// Those windows are placed relative to the region, and a region clipped
    /// by the buffer's leading edge is anchored to the *buffer*, which moves.
    /// So consecutive ticks re-cut the same audio at shifted offsets and every
    /// embedding is recomputed. Measured on a 10 s window at a 1 s tick: the
    /// window grids realign only every 3 s (`lcm(1.0, 0.75)`), and the cache
    /// hit rate sat at ~24 %.
    ///
    /// Given this, windows are placed on an ABSOLUTE grid, so the same audio
    /// yields the same window bounds no matter where the buffer happens to
    /// start — which is what makes the embedding cache actually hit.
    pub stream_offset_secs: f64,
}

impl Default for AsrOptions {
    fn default() -> Self {
        // Written out rather than derived: `vad_threshold` and
        // `vad_chunk_secs` have meaningful defaults, and `#[derive(Default)]`
        // would silently make them 0.0 — a threshold that calls everything
        // speech and a window width that holds nothing.
        Self {
            language: None,
            word_timestamps: false,
            diarize: false,
            persist_speakers: false,
            max_speakers: None,
            diarize_threshold: 0.80,
            translate: false,
            vad: true,
            vad_threshold: 0.5,
            vad_chunk_secs: 30.0,
            stream_offset_secs: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TtsOptions {
    pub voice: Option<String>,
    /// Playback-rate multiplier, 1.0 = normal.
    pub speed: f32,
    /// Acoustic variation (VITS prior noise); `None` = the voice's own
    /// default. 0.0 is fully deterministic audio.
    pub noise_scale: Option<f32>,
    /// Duration variation (stochastic duration predictor noise); `None` =
    /// voice default, 0.0 = deterministic timing.
    pub noise_w: Option<f32>,
    /// Seed for all sampled noise. Mercury synthesis is byte-stable per
    /// (input, options, seed) — a capability the references do not offer.
    pub seed: u64,
    /// Silence inserted between sentences of long-form input, in seconds.
    pub sentence_silence_s: f32,
}

impl Default for TtsOptions {
    fn default() -> Self {
        Self {
            voice: None,
            speed: 1.0,
            noise_scale: None,
            noise_w: None,
            seed: 0,
            sentence_silence_s: 0.2,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct OcrOptions {
    /// Language hints (engine-specific tags); empty = engine default.
    pub languages: Vec<String>,
    /// The image IS one text line: skip detection, recognize the whole
    /// frame as a single line (tesseract's `--psm 7`). LIVE's dirty-band
    /// path sets this when a band's known geometry is a single line —
    /// detection becomes async maintenance, recognition the only
    /// synchronous work.
    pub single_line: bool,
}

/// Detection options (Diana).
///
/// `confidence` defaults to 0.25 — the threshold a person looking at boxes
/// wants. Benchmarks that need the low-confidence tail for mAP set it to
/// ~0.001 explicitly, the way `corpora/refs/*_ref.py` do; the default is
/// not tuned for the scorer, and the scorer does not inherit it silently.
#[derive(Debug, Clone)]
pub struct DetectOptions {
    /// Minimum confidence to report.
    pub confidence: f32,
    /// Maximum detections returned, highest confidence first.
    pub max_detections: usize,
    /// Class-wise NMS `IoU`. `None` for the NMS-free one-to-one path, which
    /// is YOLO26's default and needs no suppression.
    pub iou: Option<f32>,
    /// Restrict to these class ids; empty = every class.
    pub classes: Vec<u32>,
}

impl Default for DetectOptions {
    fn default() -> Self {
        Self {
            confidence: 0.25,
            max_detections: 300,
            iou: None,
            classes: Vec::new(),
        }
    }
}

/// How a VLM decoder picks each next token.
///
/// **This is an enum rather than a bag of `Option` fields on purpose, and the
/// reason is the whole of Gate 2's determinism requirement: there is no way to
/// spell "sampling without a seed".** A `temperature: Option<f32>` beside a
/// `seed: Option<u64>` lets a caller set one and forget the other, and the
/// result is output that cannot be reproduced — silently, and only noticed
/// when someone tries to re-run a ledger line.
///
/// Byte-stability is the one property every other `FFai` component already
/// holds. `Mercury` TTS ships it as a competitive claim its reference
/// structurally cannot match ([`TtsOptions::seed`]); Carmenta gates on
/// byte-identity; `Diana` matches `PyTorch` detection-for-detection. `Argus` does
/// not get to be the exception, so the type makes the exception
/// unrepresentable.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Decoding {
    /// Always the argmax. Deterministic by construction, and **the default**.
    ///
    /// `#[default]` rather than a hand-written `impl Default`: the default
    /// belongs to the type, and a separate impl is one more place for it to
    /// drift away from what this doc comment promises.
    #[default]
    Greedy,
    /// Stochastic — and always seeded, because `seed` is not optional here.
    ///
    /// Same input + same options + same seed = same bytes.
    Sampled {
        /// Logit temperature. `1.0` is the model's own distribution.
        temperature: f32,
        /// Nucleus cutoff, `None` = disabled.
        top_p: Option<f32>,
        /// Top-k cutoff, `None` = disabled.
        top_k: Option<usize>,
        /// Not optional. See the type-level note above.
        seed: u64,
    },
}

/// One piece of a multimodal prompt.
///
/// Borrowed rather than owned: an `ImageBuffer` is the decoded raster, and a
/// tiling VLM will re-encode it into a dozen-plus tiles anyway. Cloning it to
/// build a prompt would copy megabytes for nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VlmPart<'a> {
    Text(&'a str),
    Image(&'a ImageBuffer),
}

/// An ordered, interleaved multimodal prompt — `text <img> text <img> text`.
///
/// **Order is the payload.** "Compare the first image to the second" is not
/// expressible as a set of images plus a question, and a model that receives
/// the images in the wrong order answers the wrong question fluently. So the
/// prompt is a sequence and the engine splices its image-token blocks at the
/// positions the sequence gives it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VlmPrompt<'a> {
    pub parts: Vec<VlmPart<'a>>,
}

impl<'a> VlmPrompt<'a> {
    /// The single-image case: the image, then the instruction if there is one.
    #[must_use]
    pub fn single(image: &'a ImageBuffer, text: Option<&'a str>) -> Self {
        let mut parts = vec![VlmPart::Image(image)];
        if let Some(t) = text {
            parts.push(VlmPart::Text(t));
        }
        Self { parts }
    }

    /// Number of images in the prompt — what an engine checks against the
    /// image-token placeholders it is about to splice.
    #[must_use]
    pub fn image_count(&self) -> usize {
        self.parts
            .iter()
            .filter(|p| matches!(p, VlmPart::Image(_)))
            .count()
    }
}

/// Options for a VLM call (Argus).
///
/// # Gate 2 — the v1 surface, and what is deliberately NOT here
///
/// Settled once, before implementation, because every field is cheap now and a
/// breaking change later (`docs/plans/argus-launch-plan.md` §2 Gate 2).
///
/// **Excluded from v1, as decisions rather than oversights:**
///
/// - **Streaming.** Tokens as produced. It changes the return type of every
///   method, so it is a trait redesign and not a field; it waits until there
///   is a consumer that needs it.
/// - **Grounding.** Region-in ("what is in this box") and grounded-out ("the
///   dog `[x,y,w,h]`"). Diana already returns boxes; a second, weaker box
///   source in the toolkit needs a reason beyond "the model can".
/// - **Structured / JSON output.** Needs constrained decoding to be worth
///   anything — an unconstrained "please reply in JSON" is a request, not a
///   guarantee. High value for a tooling product, so it is v2 rather than
///   never; mistral.rs already provides grammar-constrained decoding, which is
///   what makes it cheap when it lands.
/// - **Token-level confidence (logprobs).** Carmenta uses recognition
///   confidence for its verifier and the analog is genuinely valuable, but it
///   is only meaningful once an engine exists to produce it honestly.
/// - **Conversation history / multi-turn.** [`VlmPrompt`] can already express
///   an interleaved turn sequence; a typed history is a chat-session concern,
///   and Argus is a media op.
///
/// **Not here because it belongs to the ENGINE, not the caller:** the chat
/// template, the special vision tokens (`<image>`, `<|vision_start|>`), and
/// the M-RoPE time axis for video. Those are properties of a specific
/// checkpoint. Exposing them as options would invite a caller to set them
/// wrong — and the launch plan measured what that costs: **43 of 50 answers
/// changed** on identical weights purely from prompt formatting, with no error
/// raised.
#[derive(Debug, Clone, Default)]
pub struct VlmOptions {
    /// Instruction for the model; `None` = plain captioning.
    ///
    /// Read by [`VlmEngine::describe_image`] only. [`VlmEngine::describe`]
    /// takes its text from the prompt's own [`VlmPart::Text`] parts, because
    /// position matters there and a separate field could not say where the
    /// text goes.
    pub prompt: Option<String>,
    /// Role/behaviour framing, when the checkpoint's template has a slot for
    /// it. Engines without one must ignore it rather than concatenate it into
    /// the user turn, which changes the prompt the model was tuned on.
    pub system_prompt: Option<String>,
    /// Token-selection strategy. Defaults to [`Decoding::Greedy`].
    pub decoding: Decoding,
    /// Token budget. `None` = the engine's own default.
    pub max_new_tokens: Option<usize>,
    /// Stop strings. Generation halts as soon as one is produced; the string
    /// itself is not included in the returned text.
    ///
    /// EOS is not here — that is the tokenizer's, and an engine that needed to
    /// be told its own EOS would be misconfigured.
    pub stop: Vec<String>,
    /// Frames shown to the model per video caption. `None` = the engine's own
    /// default.
    ///
    /// This is the video knob that decides what a caption can be ABOUT, and
    /// the arithmetic is worth stating because it is not obvious. A still
    /// image is split into tiles so fine print survives — for `SmolVLM` that
    /// is 17 tiles at 64 tokens, **1088 tokens per image**. A text tower with
    /// 8192 positions therefore holds seven such frames, which is not a window
    /// so much as a slideshow.
    ///
    /// Video engines turn splitting off, making a frame **one** tile, and the
    /// same 8192 positions then hold a hundred. So this number trades fine
    /// detail within a frame against temporal context across frames, and the
    /// right value depends on the question being asked — which is why it is a
    /// caller's option and not a constant.
    ///
    /// `1` degenerates to per-frame captioning: correct, and blind to motion.
    pub frames_per_window: Option<usize>,
    /// Repetition penalty, `None` = disabled.
    ///
    /// Sits beside [`Self::decoding`] rather than inside
    /// [`Decoding::Sampled`] because it is a *logit* transform, not a
    /// sampling one: it applies to greedy decoding too, and small models loop
    /// under greedy more than under sampling.
    pub repetition_penalty: Option<f32>,
}

/// Speech → text (Mercury).
pub trait AsrEngine: Send + Sync {
    fn info(&self) -> EngineInfo;
    fn transcribe(&self, audio: &AudioBuffer, opts: &AsrOptions) -> Result<Transcript>;
}

/// Text → speech (Mercury).
pub trait TtsEngine: Send + Sync {
    fn info(&self) -> EngineInfo;
    fn synthesize(&self, text: &str, opts: &TtsOptions) -> Result<AudioBuffer>;
}

/// Image → text (Carmenta).
pub trait OcrEngine: Send + Sync {
    fn info(&self) -> EngineInfo;
    fn recognize(&self, image: &ImageBuffer, opts: &OcrOptions) -> Result<OcrOutput>;
}

/// Image → objects (Diana).
/// What a depth engine returns: a dense map plus what is needed to place it
/// back on the source image.
#[derive(Debug, Clone)]
pub struct DepthOutput {
    /// Row-major depth, `height * width` values, in **metres**.
    pub depth: Vec<f32>,
    pub width: usize,
    pub height: usize,
    /// The letterbox that produced this map, when one was applied — the same
    /// role it plays in [`DetectOutput`], and needed for the same reason: the
    /// map is in letterboxed space and means nothing without it.
    pub letterbox: Option<crate::types::Letterbox>,
}

impl DepthOutput {
    /// Depth at a pixel of the map. `None` when out of range.
    #[must_use]
    pub fn at(&self, x: usize, y: usize) -> Option<f32> {
        if x >= self.width || y >= self.height {
            return None;
        }
        self.depth.get(y * self.width + x).copied()
    }

    /// `(min, max)` over the map, for callers normalising it for display.
    /// Returns `None` for an empty map rather than a nonsense range.
    #[must_use]
    pub fn range(&self) -> Option<(f32, f32)> {
        if self.depth.is_empty() {
            return None;
        }
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        for &v in &self.depth {
            if v.is_finite() {
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        (lo <= hi).then_some((lo, hi))
    }
}

/// Options for a depth run.
#[derive(Debug, Clone, Default)]
pub struct DepthOptions {
    /// Resize the map to the SOURCE image's resolution and undo the
    /// letterbox, instead of returning the raw network output at stride 4.
    ///
    /// Off by default: the raw map is what the model computed, and resizing
    /// is a lossy convenience the caller may want to do differently.
    pub full_resolution: bool,
}

/// Monocular depth estimation.
pub trait DepthEngine: Send + Sync {
    fn info(&self) -> EngineInfo;
    fn depth(&self, image: &ImageBuffer, opts: &DepthOptions) -> Result<DepthOutput>;
}

pub trait DetectEngine: Send + Sync {
    fn info(&self) -> EngineInfo;
    fn detect(&self, image: &ImageBuffer, opts: &DetectOptions) -> Result<DetectOutput>;

    /// Detect over many images, using the whole machine.
    ///
    /// **This is a first-class path, not a convenience wrapper**, because it
    /// is where a detector's throughput actually lives. Measured on Diana:
    /// running images concurrently is worth **3.6-5.3x** over calling
    /// [`Self::detect`] in a loop, with byte-identical output and no change
    /// whatever to the per-image path — because intra-image parallelism is
    /// nearly exhausted (24 cores buy one image only 1.42x) while the images
    /// themselves are independent.
    ///
    /// The trait is `Send + Sync`, so one loaded model serves every thread.
    /// That is the structural advantage over the Python reference: measured
    /// on the same machine, `PyTorch` gets *slower* under threading (0.68-0.72x)
    /// because of the GIL, and its escape hatch — multiprocessing — pays a
    /// full model copy per worker.
    ///
    /// The default implementation is sequential so existing engines keep
    /// working; an engine that can do better should override it.
    fn detect_batch(
        &self,
        images: &[ImageBuffer],
        opts: &DetectOptions,
    ) -> Result<Vec<DetectOutput>> {
        images.iter().map(|i| self.detect(i, opts)).collect()
    }
    /// Class names for the ids in [`DetectOutput`], in id order. They come
    /// from the weight manifest, so they belong to the engine rather than
    /// to every output it produces.
    fn class_names(&self) -> &[String];
}

/// Image/video → description (Argus).
///
/// See [`VlmOptions`] for the Gate-2 surface decision and the written-down v1
/// exclusions.
pub trait VlmEngine: Send + Sync {
    fn info(&self) -> EngineInfo;

    /// The general path: an ordered, interleaved multimodal prompt.
    ///
    /// **This is the required method, and `describe_image` is derived from
    /// it, rather than the other way round.** An engine that implemented only
    /// the single-image case would still compile against a multi-image
    /// prompt — and would then silently answer using the first image, or the
    /// last, or a concatenation. Making the general case the one an
    /// implementor must write means multi-image support is a compile-time
    /// obligation instead of a runtime surprise.
    fn describe(&self, prompt: &VlmPrompt<'_>, opts: &VlmOptions) -> Result<String>;

    /// One image, with [`VlmOptions::prompt`] as the instruction.
    ///
    /// Provided: builds a single-image [`VlmPrompt`] and calls
    /// [`Self::describe`]. Zero-copy — the prompt borrows the image.
    fn describe_image(&self, image: &ImageBuffer, opts: &VlmOptions) -> Result<String> {
        self.describe(&VlmPrompt::single(image, opts.prompt.as_deref()), opts)
    }

    /// Video understanding over sampled frames → a timed caption track.
    ///
    /// Frames arrive with their timestamps ([`VideoFrame::timestamp`]) so an
    /// engine can encode *time* and not merely order — M-RoPE's third axis.
    /// Whether it does is the engine's business; the trait's job is to make
    /// sure the information reaches it.
    fn describe_video(
        &self,
        frames: &[VideoFrame],
        opts: &VlmOptions,
    ) -> Result<Vec<TimedSegment<String>>>;
}

#[cfg(test)]
mod vlm_surface_tests {
    use super::*;
    use crate::types::PixelFormat;

    fn img(byte: u8) -> ImageBuffer {
        ImageBuffer {
            width: 1,
            height: 1,
            format: PixelFormat::Rgb8,
            data: vec![byte, byte, byte],
        }
    }

    /// An engine that implements ONLY the required method and records what it
    /// was handed — which is how we check `describe_image` really routes
    /// through `describe` rather than being a second, divergent path.
    struct Recorder(std::sync::Mutex<Vec<String>>);

    impl VlmEngine for Recorder {
        fn info(&self) -> EngineInfo {
            EngineInfo {
                name: "recorder".into(),
                task: Task::Vlm,
                status: EngineStatus::Stub,
                description: String::new(),
            }
        }
        fn describe(&self, prompt: &VlmPrompt<'_>, _opts: &VlmOptions) -> Result<String> {
            let shape: Vec<String> = prompt
                .parts
                .iter()
                .map(|p| match p {
                    VlmPart::Text(t) => format!("text:{t}"),
                    VlmPart::Image(i) => format!("image:{}", i.data[0]),
                })
                .collect();
            self.0.lock().unwrap().push(shape.join("|"));
            Ok(shape.join("|"))
        }
        fn describe_video(
            &self,
            _frames: &[VideoFrame],
            _opts: &VlmOptions,
        ) -> Result<Vec<TimedSegment<String>>> {
            Ok(Vec::new())
        }
    }

    /// Gate 2's determinism requirement, checked rather than asserted in prose:
    /// a caller who sets nothing gets deterministic decoding.
    #[test]
    fn the_default_decoding_is_deterministic() {
        assert_eq!(VlmOptions::default().decoding, Decoding::Greedy);
        // ...and the default options carry no stochasticity anywhere else.
        assert!(VlmOptions::default().stop.is_empty());
        assert_eq!(VlmOptions::default().repetition_penalty, None);
    }

    /// There is no way to construct stochastic decoding without a seed. This
    /// test cannot fail at runtime — the point is that the alternative does
    /// not COMPILE, and this is where that intent is recorded.
    #[test]
    fn sampling_always_carries_a_seed() {
        let d = Decoding::Sampled {
            temperature: 0.7,
            top_p: Some(0.9),
            top_k: None,
            seed: 42,
        };
        // Every Sampled value has a seed by construction; matching proves it.
        let Decoding::Sampled { seed, .. } = d else {
            panic!("expected Sampled")
        };
        assert_eq!(seed, 42);
    }

    #[test]
    fn describe_image_routes_through_describe() {
        let e = Recorder(std::sync::Mutex::new(Vec::new()));
        let opts = VlmOptions {
            prompt: Some("what is this?".into()),
            ..VlmOptions::default()
        };
        let out = e.describe_image(&img(7), &opts).unwrap();
        // Image FIRST, then the instruction — the order most VLM chat
        // templates expect, and the order `VlmPrompt::single` documents.
        assert_eq!(out, "image:7|text:what is this?");
        assert_eq!(e.0.lock().unwrap().len(), 1, "describe must have been used");
    }

    #[test]
    fn a_prompt_with_no_instruction_is_just_the_image() {
        let e = Recorder(std::sync::Mutex::new(Vec::new()));
        let out = e.describe_image(&img(3), &VlmOptions::default()).unwrap();
        assert_eq!(out, "image:3");
    }

    /// Order is the payload: "compare the first to the second" is not
    /// expressible as a set, so the sequence must survive intact.
    #[test]
    fn interleaved_order_is_preserved() {
        let (a, b) = (img(1), img(2));
        let e = Recorder(std::sync::Mutex::new(Vec::new()));
        let prompt = VlmPrompt {
            parts: vec![
                VlmPart::Text("before"),
                VlmPart::Image(&a),
                VlmPart::Text("between"),
                VlmPart::Image(&b),
                VlmPart::Text("after"),
            ],
        };
        assert_eq!(prompt.image_count(), 2);
        let out = e.describe(&prompt, &VlmOptions::default()).unwrap();
        assert_eq!(
            out,
            "text:before|image:1|text:between|image:2|text:after",
            "the sequence an engine receives must be the sequence it was given"
        );
    }
}
