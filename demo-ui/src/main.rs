//! FFai side-by-side demo: Mercury and whisper.cpp on the same microphone.
//!
//! Click Start, talk, and both engines transcribe the same 5-second chunks so
//! their output can be read against each other in real time. Stop flushes
//! whatever is left in the buffer so the last sentence is not lost.
//!
//! **Audio capture lives in JavaScript, not in Rust.** Getting PCM out of the
//! browser means `getUserMedia`, an `AudioContext`, a processor node and a
//! resampler; doing that through `web-sys` is several hundred lines of
//! plumbing for something the platform hands you in thirty lines of JS. The
//! Rust side owns the state and the rendering and receives finished JSON over
//! the eval channel.
//!
//! The blob is a `const` with `__PLACEHOLDER__` substitution rather than a
//! `format!`, because `format!` would require doubling every brace in the JS.

use dioxus::prelude::*;

fn main() {
    dioxus::launch(App);
}

/// One engine's answer for one chunk.
#[derive(Clone, PartialEq)]
struct Line {
    text: String,
    ms: f64,
    error: Option<String>,
    /// A committed line is finished and will not be rewritten; the last
    /// uncommitted one is the sentence currently being spoken.
    committed: bool,
    /// The same content as `text`, but per segment and with the times and the
    /// speaker kept apart. Empty when the engine sent none (whisper.cpp always,
    /// Mercury with speakers off), in which case `text` is what gets rendered.
    turns: Vec<Turn>,
}

/// One segment of one chunk: who said it, and when, in SESSION time.
///
/// Times are absolute rather than chunk-relative because the browser posts a
/// sliding window — a time measured from the buffer's start slides with it,
/// so two chunks' numbers would not be comparable and nothing could be laid
/// out against anything else.
#[derive(Clone, PartialEq)]
struct Turn {
    /// `None` when the diarizer found no turn covering this segment, or when
    /// speakers are off. Rendered unattributed rather than guessed at.
    speaker: Option<String>,
    text: String,
    start: f64,
    end: f64,
}

/// How many distinct colours speakers cycle through.
const SPEAKER_COLOURS: usize = 8;

/// How much trailing audio each pass sees, in seconds.
///
/// One constant for two consumers: the recorder is templated with it, and the
/// ribbons are drawn against it. Scaling each ribbon to its OWN span instead
/// would stretch every chunk to full width, so a 1.3 s exchange and a 4 s one
/// would draw the same picture — the layout would look like a timeline without
/// being one. Against a fixed scale, width means seconds everywhere in the
/// pane.
const WINDOW_SECS: f64 = 10.0;

/// Stable colour slot for a diarizer label.
///
/// The label's own number is used where there is one, so `SPEAKER_02` is the
/// same colour in the legend, in the ribbon and on every row — and stays that
/// colour for the whole session, which is the entire point of the persistent
/// registry behind it. Anything else falls back to a byte sum so an unexpected
/// label still gets a consistent colour instead of a panic or a default.
fn speaker_slot(label: &str) -> usize {
    label
        .rsplit(['_', '-', ' '])
        .next()
        .and_then(|n| n.parse::<usize>().ok())
        .unwrap_or_else(|| label.bytes().map(usize::from).sum())
        % SPEAKER_COLOURS
}

/// `SPEAKER_02` -> `S2`. The full label rides in the row's tooltip.
///
/// Six repetitions of `SPEAKER_` down the left margin is most of the width of
/// the pane spent on a prefix that never varies, which is exactly what pushed
/// the turns into one paragraph in the first place.
fn short_label(label: &str) -> String {
    match label.rsplit(['_', '-', ' ']).next().map(str::trim) {
        Some(n) if !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) => {
            // `SPEAKER_00` is a real label and `S` is not a name: strip the
            // zero padding, never the last digit.
            let n = n.trim_start_matches('0');
            format!("S{}", if n.is_empty() { "0" } else { n })
        }
        _ => label.to_string(),
    }
}

/// Session seconds as `m:ss`, so a row can be found in the recording.
fn clock(secs: f64) -> String {
    let secs = if secs.is_finite() && secs > 0.0 { secs } else { 0.0 };
    let whole = secs as u64;
    format!("{}:{:02}", whole / 60, whole % 60)
}

/// Capture 16 kHz mono PCM, POST each chunk, and hand the JSON back to Rust.
///
/// `CHUNK_SECS` is the latency/accuracy trade: Whisper is trained on 30 s
/// windows and gets *better* with more context, but a demo that answers every
/// 30 s feels broken. Five seconds is short enough to feel live and long
/// enough to hold a clause.
const RECORDER_JS: &str = r####"
(async () => {
  const S = (window.__ffai = window.__ffai || {});
  S.stop = false;
  const TICK = __TICK__;      // how often we re-transcribe, seconds
  const WINDOW = __WINDOW__;  // how much trailing audio each pass sees

  function flatten(chunks) {
    let n = 0; for (const c of chunks) n += c.length;
    const out = new Float32Array(n); let o = 0;
    for (const c of chunks) { out.set(c, o); o += c.length; }
    return out;
  }
  // Linear resample. Good enough for speech at these rates, and it keeps the
  // browser side dependency-free.
  function resample(input, from, to) {
    if (from === to) return input;
    const ratio = from / to;
    const out = new Float32Array(Math.floor(input.length / ratio));
    for (let i = 0; i < out.length; i++) {
      const src = i * ratio;
      const i0 = Math.floor(src), i1 = Math.min(i0 + 1, input.length - 1);
      const t = src - i0;
      out[i] = input[i0] * (1 - t) + input[i1] * t;
    }
    return out;
  }
  // 16-bit PCM WAV: what ffai-media reads and what whisper-cli reads.
  function encodeWav(samples, rate) {
    const buf = new ArrayBuffer(44 + samples.length * 2);
    const v = new DataView(buf);
    const put = (off, s) => { for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i)); };
    put(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); put(8, 'WAVE');
    put(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true);
    v.setUint16(22, 1, true); v.setUint32(24, rate, true);
    v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    put(36, 'data'); v.setUint32(40, samples.length * 2, true);
    let off = 44;
    for (let i = 0; i < samples.length; i++, off += 2) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      v.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return buf;
  }

  let stream, ctx;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 } });
    ctx = new (window.AudioContext || window.webkitAudioContext)();
  } catch (e) {
    dioxus.send(JSON.stringify({ fatal: 'microphone unavailable: ' + e }));
    return;
  }

  const src = ctx.createMediaStreamSource(stream);
  const node = ctx.createScriptProcessor(4096, 1, 1);
  let pending = [];
  let seen = 0;   // total samples captured this session, for the stream offset
  node.onaudioprocess = (e) => {
    if (S.stop) return;
    const frame = new Float32Array(e.inputBuffer.getChannelData(0));
    pending.push(frame);
    seen += frame.length;   // absolute position in the session; see `offset`
  };
  src.connect(node); node.connect(ctx.destination);

  // Root-mean-square, to spot a pause. A cheap stand-in for the real VAD the
  // roadmap schedules for Phase 1.5 — enough to end a line at a natural break
  // rather than mid-word, which is what a fixed cadence gets wrong.
  function rms(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s / a.length); }

  // Re-transcribe the trailing window every tick and REPLACE the live line.
  // Affordable only because Whisper pads every window to 30 s regardless: 1 s
  // of audio costs 213 ms and 10 s costs 311 ms, so a pass WITH full context
  // is essentially the same price as one without. Sending more audio buys
  // accuracy for free; sending it more often buys latency.
  // Drop audio older than the window, EVERY tick, regardless of what the
  // request did. Bounding this on a successful response was a leak: `pass()`
  // early-returns while a request is inflight, and the buffer reset sat after
  // the `await fetch` inside the `try`, so a hung or failing server meant the
  // recorder kept appending and nothing ever cleared — ~192 KB/s at 48 kHz,
  // about 690 MB an hour, until the tab died. Transient errors were enough.
  // Now the cap is structural and the commit only decides what to DISPLAY.
  function trimToWindow() {
    let total = 0;
    for (const c of pending) total += c.length;
    const cap = Math.floor(ctx.sampleRate * WINDOW);
    while (pending.length > 1 && total - pending[0].length >= cap) {
      total -= pending.shift().length;
    }
  }

  let inflight = false;
  async function pass(commit) {
    if (inflight) return;
    inflight = true;
    try {
      const pcm = flatten(pending);
      if (pcm.length >= ctx.sampleRate * 0.3) {
        const span = Math.floor(ctx.sampleRate * WINDOW);
        const tail = pcm.length > span ? pcm.subarray(pcm.length - span) : pcm;
        const wav = encodeWav(resample(tail, ctx.sampleRate, 16000), 16000);
        // Where this trailing window starts in the session. The diarizer
        // places its embedding windows on an ABSOLUTE grid from this, so a
        // sliding buffer stops re-cutting the same audio at new offsets.
        const offset = Math.max(0, (seen - tail.length) / ctx.sampleRate);
        const res = await fetch(
          '/transcribe?diarize=__DIARIZE__&offset=' + offset.toFixed(3),
          { method: 'POST', body: wav });
        const j = JSON.parse(await res.text());
        j.commit = commit;
        dioxus.send(JSON.stringify(j));
      }
      if (commit) pending = [];
    } catch (e) {
      dioxus.send(JSON.stringify({ fatal: 'transcribe request failed: ' + e }));
    } finally { inflight = false; }
  }

  let quiet = 0;
  while (!S.stop) {
    await new Promise((r) => setTimeout(r, TICK * 1000));
    trimToWindow();
    const recent = pending.length ? pending[pending.length - 1] : null;
    const silent = recent ? rms(recent) < 0.008 : true;
    quiet = silent ? quiet + TICK : 0;
    const held = pending.reduce((a, b) => a + b.length, 0) / ctx.sampleRate;
    // End the line on a ~0.8 s pause, or when the window is full and holding
    // more would start dropping the start of the sentence.
    const commit = (quiet >= 0.8 && held > 1.0) || held >= WINDOW;
    await pass(commit);
    if (commit) quiet = 0;
  }
  // Stop was pressed: send whatever is left rather than dropping the last
  // sentence mid-word.
  await pass(true);
  stream.getTracks().forEach((t) => t.stop());
  ctx.close();
  dioxus.send(JSON.stringify({ done: true }));
})();
"####;

/// Take an image from a file picker, a drop, or the clipboard; POST the raw
/// bytes to `/read`; hand back the server's JSON with a data-URL preview
/// attached so Rust can render the image it actually sent.
///
/// PNG only: `ffai-media` decodes PNG until the rff image decoders land, so
/// anything else is re-encoded to PNG through a canvas here rather than
/// rejected — a user pasting a JPEG screenshot should not have to care.
const READ_JS: &str = r####"
(async () => {
  const send = (o) => dioxus.send(JSON.stringify(o));
  const toPng = (blob) => new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const c = document.createElement('canvas');
      c.width = img.naturalWidth; c.height = img.naturalHeight;
      c.getContext('2d').drawImage(img, 0, 0);
      c.toBlob((b) => b ? resolve(b) : reject('canvas encode failed'), 'image/png');
    };
    img.onerror = () => reject('not a decodable image');
    img.src = URL.createObjectURL(blob);
  });
  const post = async (blob) => {
    try {
      const png = blob.type === 'image/png' ? blob : await toPng(blob);
      const buf = await png.arrayBuffer();
      const res = await fetch('/read', { method: 'POST', body: buf });
      const out = JSON.parse(await res.text());
      const fr = new FileReader();
      fr.onload = () => { out.preview = fr.result; send(out); };
      fr.readAsDataURL(png);
    } catch (e) {
      send({ error: 'read failed: ' + e });
    }
  };
  // One-shot: whichever source fires first wins, then the listeners go away.
  const input = document.createElement('input');
  input.type = 'file'; input.accept = 'image/*';
  input.onchange = () => { if (input.files[0]) post(input.files[0]); };
  const onPaste = (e) => {
    for (const it of (e.clipboardData || {}).items || []) {
      if (it.type.startsWith('image/')) { post(it.getAsFile()); cleanup(); return; }
    }
  };
  const cleanup = () => document.removeEventListener('paste', onPaste);
  document.addEventListener('paste', onPaste);
  input.click();
})();
"####;

/// Pick or paste an image, POST it to `/describe`, hand back the JSON plus a
/// data-URL preview.
///
/// The knobs ride in the QUERY STRING and the body stays raw image bytes. The
/// alternative — base64 inside a JSON envelope — inflates a screenshot by a
/// third for the sake of carrying two integers and a sentence.
const SEE_JS: &str = r####"
(async () => {
  const send = (o) => dioxus.send(JSON.stringify(o));
  // ffai-media decodes PNG and JPEG. Anything else (webp, gif, avif) is
  // re-encoded through a canvas rather than rejected: a user pasting a
  // screenshot should not have to know what their OS chose to put on the
  // clipboard.
  const toPng = (blob) => new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const c = document.createElement('canvas');
      c.width = img.naturalWidth; c.height = img.naturalHeight;
      c.getContext('2d').drawImage(img, 0, 0);
      c.toBlob((b) => b ? resolve(b) : reject('canvas encode failed'), 'image/png');
    };
    img.onerror = () => reject('not a decodable image');
    img.src = URL.createObjectURL(blob);
  });
  const post = async (blob) => {
    try {
      const ok = blob.type === 'image/png' || blob.type === 'image/jpeg';
      const img = ok ? blob : await toPng(blob);
      const buf = await img.arrayBuffer();
      const res = await fetch('/describe?__QUERY__', { method: 'POST', body: buf });
      const out = JSON.parse(await res.text());
      const fr = new FileReader();
      fr.onload = () => { out.preview = fr.result; send(out); };
      fr.readAsDataURL(img);
    } catch (e) {
      send({ error: 'describe failed: ' + e });
    }
  };
  const input = document.createElement('input');
  input.type = 'file'; input.accept = 'image/*';
  input.onchange = () => { if (input.files[0]) post(input.files[0]); };
  const onPaste = (e) => {
    for (const it of (e.clipboardData || {}).items || []) {
      if (it.type.startsWith('image/')) { post(it.getAsFile()); cleanup(); return; }
    }
  };
  const cleanup = () => document.removeEventListener('paste', onPaste);
  document.addEventListener('paste', onPaste);
  input.click();
})();
"####;

const STOP_JS: &str = "window.__ffai = window.__ffai || {}; window.__ffai.stop = true;";

/// POST the synthesis request and hand the JSON back. Playback is a data URL
/// on an `<audio>` element (Rust owns that), so this only moves JSON.
const SPEAK_JS: &str = r####"
(async () => {
  try {
    const res = await fetch('/synthesize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: __PAYLOAD__,
    });
    dioxus.send(await res.text());
  } catch (e) {
    dioxus.send(JSON.stringify({ error: 'synthesize request failed: ' + e }));
  }
})();
"####;

/// One synthesis result, as the Speak tab renders it.
#[derive(Clone, PartialEq, Default)]
struct Spoken {
    wav_base64: String,
    sample_rate: u32,
    audio_secs: f64,
    synth_ms: f64,
    xrt: f64,
    sentences: Vec<String>,
    phonemes: Vec<String>,
    sha256: String,
    /// `Some(true)` when a repeat synthesis produced identical samples.
    deterministic: Option<bool>,
    error: Option<String>,
}

/// One OCR result pane.
#[derive(Clone, PartialEq, Default)]
struct Reading {
    text: String,
    lines: usize,
    ms: f64,
    error: Option<String>,
}

/// What the Read tab shows after one image.
#[derive(Clone, PartialEq, Default)]
struct ReadOut {
    preview: String,
    width: u64,
    height: u64,
    /// "rendered" or "photographic" — the dispatch signal's own verdict.
    content: String,
    flatness: f64,
    crnn: Reading,
    parseq: Reading,
    error: Option<String>,
}

/// One stage of the Argus pipeline, as the timeline draws it.
#[derive(Clone, PartialEq, Default)]
struct Stage {
    name: String,
    ms: f64,
    what: String,
}

/// Everything the See tab shows after one image.
#[derive(Clone, PartialEq, Default)]
struct SeeOut {
    preview: String,
    caption: String,
    prompt: String,
    greedy: bool,
    width: u64,
    height: u64,
    /// The size the content path resized to before tiling — the step that
    /// explains where 17 tiles come from.
    resized_w: u64,
    resized_h: u64,
    rows: usize,
    cols: usize,
    tiles: usize,
    tile: usize,
    tok_image: u64,
    tok_text: u64,
    tok_prompt: u64,
    tok_generated: u64,
    tok_per_tile: u64,
    max_positions: u64,
    stages: Vec<Stage>,
    tower_per_tile_ms: Vec<f64>,
    step_ms: Vec<f64>,
    tokens_per_sec: f64,
    engine_ms: f64,
    wall_ms: f64,
    load_ms: f64,
    /// True when this call paid for the weight load — the reading is cold and
    /// says so rather than being averaged in with warm ones.
    cold: bool,
    ref_text: String,
    ref_ms: f64,
    ref_load_ms: f64,
    ref_error: Option<String>,
    ref_absent: Option<String>,
    error: Option<String>,
}

/// Stage colours, in execution order. Shared by the bar and its legend so the
/// two cannot disagree about which segment is which.
const STAGE_COLOURS: [(&str, &str); 7] = [
    ("decode", "#6e7681"),
    ("preprocess", "#d29922"),
    ("vision", "#f0883e"),
    ("assemble", "#a371f7"),
    ("prefill", "#1f6feb"),
    ("generate", "#3fb950"),
    ("detokenize", "#484f58"),
];

fn stage_colour(name: &str) -> &'static str {
    STAGE_COLOURS
        .iter()
        .find(|(n, _)| *n == name)
        .map_or("#6e7681", |(_, c)| *c)
}

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Listen,
    Speak,
    Read,
    See,
}

#[component]
fn App() -> Element {
    let mut tab = use_signal(|| Tab::Listen);
    rsx! {
        style { {CSS} }
        div { class: "wrap",
            header {
                h1 { "FFai — pure-Rust speech and text, side by side" }
                div { class: "tabs",
                    button {
                        class: if tab() == Tab::Listen { "tab on" } else { "tab" },
                        onclick: move |_| tab.set(Tab::Listen),
                        "Listen · ASR vs whisper.cpp"
                    }
                    button {
                        class: if tab() == Tab::Speak { "tab on" } else { "tab" },
                        onclick: move |_| tab.set(Tab::Speak),
                        "Speak · TTS vs piper"
                    }
                    button {
                        class: if tab() == Tab::Read { "tab on" } else { "tab" },
                        onclick: move |_| tab.set(Tab::Read),
                        "Read · OCR, crnn vs parseq"
                    }
                    button {
                        class: if tab() == Tab::See { "tab on" } else { "tab" },
                        onclick: move |_| tab.set(Tab::See),
                        "See · VLM vs PyTorch"
                    }
                }
            }
            // Both views stay MOUNTED, hidden by CSS rather than unmounted, so
            // switching tabs mid-session does not tear down the recorder or
            // discard a transcript.
            div { class: if tab() == Tab::Listen { "view" } else { "view hidden" }, Listen {} }
            div { class: if tab() == Tab::Speak { "view" } else { "view hidden" }, Speak {} }
            div { class: if tab() == Tab::Read { "view" } else { "view hidden" }, Read {} }
            div { class: if tab() == Tab::See { "view" } else { "view hidden" }, See {} }
        }
    }
}

/// Argus's tab: one image, and where every millisecond of describing it went.
///
/// # Why this tab is a pipeline diagram and not just two panes
///
/// Listen and Read are races — two implementations, identical input, read the
/// answers. This is a race too (Argus against PyTorch on the same checkpoint,
/// the same file, the same pinned decode config), but the race is the *less*
/// interesting half.
///
/// The interesting half is that a VLM's cost is not where anyone expects. Ask
/// someone why captioning took four seconds and they will say "the language
/// model". Almost always it is the picture: a still is cut into **seventeen**
/// 512x512 tiles, every one of them is a full `SigLIP` forward pass, and those
/// seventeen passes contribute **1088 image tokens** to a prompt that then has
/// to be prefilled in one go. The text the model writes back is the cheap part.
///
/// None of that is visible in a single number, so this tab draws it: the tile
/// grid over the image that produced it, the stage timeline, the token budget,
/// and the per-tile and per-token costs underneath.
#[component]
fn See() -> Element {
    let mut busy = use_signal(|| false);
    let mut prompt = use_signal(|| "What is written in this image?".to_string());
    let mut max_tokens = use_signal(|| 64u32);
    let mut sampled = use_signal(|| false);
    let mut out = use_signal(SeeOut::default);

    let pick = move |_| {
        if busy() {
            return;
        }
        busy.set(true);
        spawn(async move {
            // Percent-encode the prompt: it is a sentence, and sentences carry
            // spaces, `&` and `?`. `encodeURIComponent` is not available on the
            // Rust side, so the few characters that matter are escaped here.
            let q = format!(
                "prompt={}&max={}{}",
                url_escape(&prompt()),
                max_tokens(),
                if sampled() { "&seed=42&temp=0.7" } else { "" }
            );
            let js = SEE_JS.replace("__QUERY__", &q);
            let mut eval = document::eval(&js);
            if let Ok(msg) = eval.recv::<String>().await {
                let v: serde_json::Value = serde_json::from_str(&msg).unwrap_or_default();
                let f = |k: &str| v[k].as_f64().unwrap_or(0.0);
                let tok = |k: &str| v["tokens"][k].as_u64().unwrap_or(0);
                let stages = v["stages"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .map(|st| Stage {
                                name: st["name"].as_str().unwrap_or_default().to_string(),
                                ms: st["ms"].as_f64().unwrap_or(0.0),
                                what: st["what"].as_str().unwrap_or_default().to_string(),
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let floats = |k: &str| -> Vec<f64> {
                    v[k].as_array()
                        .map(|a| a.iter().filter_map(serde_json::Value::as_f64).collect())
                        .unwrap_or_default()
                };
                let r = &v["reference"];
                out.set(SeeOut {
                    preview: v["preview"].as_str().unwrap_or_default().to_string(),
                    caption: v["caption"].as_str().unwrap_or_default().to_string(),
                    prompt: v["prompt"].as_str().unwrap_or_default().to_string(),
                    greedy: v["greedy"].as_bool().unwrap_or(true),
                    width: v["image"]["width"].as_u64().unwrap_or(0),
                    height: v["image"]["height"].as_u64().unwrap_or(0),
                    resized_w: v["image"]["resized"]["w"].as_u64().unwrap_or(0),
                    resized_h: v["image"]["resized"]["h"].as_u64().unwrap_or(0),
                    rows: v["grid"]["rows"].as_u64().unwrap_or(0) as usize,
                    cols: v["grid"]["cols"].as_u64().unwrap_or(0) as usize,
                    tiles: v["grid"]["tiles"].as_u64().unwrap_or(0) as usize,
                    tile: v["grid"]["tile"].as_u64().unwrap_or(0) as usize,
                    tok_image: tok("image"),
                    tok_text: tok("text"),
                    tok_prompt: tok("prompt"),
                    tok_generated: tok("generated"),
                    tok_per_tile: tok("per_tile"),
                    max_positions: tok("max_positions"),
                    stages,
                    tower_per_tile_ms: floats("tower_per_tile_ms"),
                    step_ms: floats("step_ms"),
                    tokens_per_sec: f("tokens_per_sec"),
                    engine_ms: f("engine_ms"),
                    wall_ms: f("wall_ms"),
                    load_ms: f("load_ms"),
                    cold: v["cold"].as_bool().unwrap_or(false),
                    ref_text: r["text"].as_str().unwrap_or_default().to_string(),
                    ref_ms: r["ms"].as_f64().unwrap_or(0.0),
                    ref_load_ms: r["load_ms"].as_f64().unwrap_or(0.0),
                    ref_error: r["error"].as_str().map(str::to_string),
                    ref_absent: r["absent"].as_str().map(str::to_string),
                    error: v["error"].as_str().map(str::to_string),
                });
            }
            busy.set(false);
        });
    };

    let r = out();
    let total: f64 = r.stages.iter().map(|s| s.ms).sum();
    let ours_ms = r.engine_ms;
    // Only meaningful when the reference actually ran.
    let ratio = if r.ref_ms > 0.0 && ours_ms > 0.0 {
        ours_ms / r.ref_ms
    } else {
        0.0
    };
    // One scale for both rows of the race: the slower arm defines full width.
    let scale = ours_ms.max(r.ref_ms).max(1.0);
    // Did the two implementations produce the same answer, and if not, how far
    // did they agree before parting? The prefix length is the informative part:
    // a port that is WRONG diverges at token one, and a port that merely broke
    // a near-tie differently agrees for a while and then splits.
    let ran_both = !r.caption.is_empty() && !r.ref_text.is_empty();
    let agree = ran_both && r.caption.trim() == r.ref_text.trim();
    let shared: usize = r
        .caption
        .chars()
        .zip(r.ref_text.chars())
        .take_while(|(a, b)| a == b)
        .count();
    let tower_peak = r.tower_per_tile_ms.iter().copied().fold(0.0f64, f64::max);
    let tower_min = r
        .tower_per_tile_ms
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let tower_mean = if r.tower_per_tile_ms.is_empty() {
        0.0
    } else {
        r.tower_per_tile_ms.iter().sum::<f64>() / r.tower_per_tile_ms.len() as f64
    };
    let step_peak = r.step_ms.iter().copied().fold(0.0f64, f64::max);

    rsx! {
        p { class: "lede",
            "Choose an image — or paste one once the picker opens. Argus captions it in "
            "pure Rust on candle, and PyTorch captions the same staged file with the same "
            "checkpoint and the same pinned decode config, so the two panes are comparable. "
            "Everything below the panes is where the time actually went."
        }
        div { class: "row",
            button { class: "primary", disabled: busy(), onclick: pick,
                if busy() { "Describing…" } else { "Choose or paste an image" }
            }
            if !r.caption.is_empty() {
                span { class: "badge",
                    {format!("{}×{} · {} tiles · {} prompt tokens", r.width, r.height, r.tiles, r.tok_prompt)}
                }
            }
        }
        div { class: "knobs",
            label { class: "knob",
                span { "Prompt" }
                input {
                    class: "say", style: "margin:0;width:340px;",
                    value: "{prompt}",
                    oninput: move |e| prompt.set(e.value()),
                }
            }
            label { class: "knob",
                span { "Max tokens" }
                input {
                    r#type: "number", min: "1", max: "512",
                    value: "{max_tokens}",
                    oninput: move |e| { if let Ok(v) = e.value().parse::<u32>() { max_tokens.set(v.clamp(1, 512)); } },
                }
            }
            label { class: "knob",
                input {
                    r#type: "checkbox",
                    checked: sampled(),
                    onchange: move |e| sampled.set(e.checked()),
                }
                span { style: "min-width:0;", "sample (seed 42)" }
            }
        }

        if let Some(e) = r.error.clone() {
            div { class: "err", "{e}" }
        }

        if !r.caption.is_empty() {
            // ---- the two answers, side by side ----
            div { class: "panes",
                div { class: "pane won",
                    div { class: "pane-head",
                        strong { style: "color:#f0883e;", "Argus · pure Rust on candle" }
                        span { class: "ms", {format!("{ours_ms:.0} ms")} }
                    }
                    div { class: "note",
                        {format!("SmolVLM-256M-Instruct · {} · {} tokens generated · {:.1} tok/s",
                                 if r.greedy { "greedy" } else { "sampled, seed 42" },
                                 r.tok_generated, r.tokens_per_sec)}
                    }
                    pre { class: "ocr", "{r.caption}" }
                }
                div { class: "pane",
                    div { class: "pane-head",
                        strong { style: "color:#79c0ff;", "PyTorch · transformers" }
                        span { class: "ms",
                            if r.ref_ms > 0.0 { {format!("{:.0} ms", r.ref_ms)} } else { "—" }
                        }
                    }
                    div { class: "note", "the same checkpoint, greedy-64 / float32 / seed 0 — the arm the ledger's quality gate uses" }
                    if let Some(a) = r.ref_absent.clone() {
                        div { class: "warn", style: "margin:0;", "{a}" }
                    } else if let Some(e) = r.ref_error.clone() {
                        div { class: "err", "{e}" }
                    } else {
                        pre { class: "ocr", "{r.ref_text}" }
                    }
                }
            }
            if ran_both {
                div { class: if agree { "verdict same" } else { "verdict split" },
                    if agree {
                        b { "identical" }
                        span { " — both implementations produced the same answer, character for character." }
                    } else {
                        b { "diverged" }
                        span {
                            {format!(" — the two answers agree for {shared} characters, then split. \
                                      That is the expected shape, not a defect: our resampler is \
                                      bit-identical to PIL, but the reference's processor is \
                                      torchvision, and the two break a handful of coefficient ties \
                                      differently — about 20 pixels in 786,432. On a confident \
                                      output that changes nothing; on an uncertain one — illegible \
                                      handwriting, a blurred sign — the logits are near-tied and one \
                                      quantisation level decides the token. Measured on a pinned \
                                      50-image corpus: 49/50 answers byte-identical.")}
                        }
                    }
                }
            }
            if ratio > 0.0 {
                p { class: "note",
                    b { {format!("{ratio:.2}×")} }
                    {format!(" — Argus takes {ratio:.2} the reference's time on this image. \
                              Both warm: the model load ({:.1} s ours, {:.1} s theirs) happened \
                              before the clock started. The measured figure on a pinned 50-image \
                              corpus is 2.4× slower, with quality an exact tie and 49/50 answers \
                              byte-identical.", r.load_ms / 1000.0, r.ref_load_ms / 1000.0)}
                }
            }
            if r.cold {
                div { class: "warn",
                    {format!("This reading is COLD — it included {:.1} s of weight loading. \
                              Click again for a warm number.", r.load_ms / 1000.0)}
                }
            }

            // ---- where the time went ----
            //
            // Both rows share ONE scale — `scale` is the slower of the two — so
            // the bars can be compared by eye. Normalising each row to its own
            // width would draw two equal bars and quietly delete the result.
            h3 { class: "sect", "Where the time went" }
            div { class: "trow",
                span { class: "tlabel", "Argus" }
                div { class: "timeline",
                    for st in r.stages.iter().filter(|s| s.ms > 0.0) {
                        div {
                            class: "seg",
                            style: "width:{st.ms / scale * 100.0}%;background:{stage_colour(&st.name)};",
                            title: "{st.name}: {st.ms:.1} ms — {st.what}",
                        }
                    }
                }
                span { class: "ms", {format!("{ours_ms:.0} ms")} }
            }
            if r.ref_ms > 0.0 {
                div { class: "trow",
                    span { class: "tlabel", "PyTorch" }
                    div { class: "timeline",
                        div {
                            class: "seg",
                            style: "width:{r.ref_ms / scale * 100.0}%;background:#79c0ff;",
                            title: "transformers, end to end: {r.ref_ms:.0} ms",
                        }
                    }
                    span { class: "ms", {format!("{:.0} ms", r.ref_ms)} }
                }
                p { class: "note", style: "margin-top:6px;",
                    "The reference reports one number — it is a library call, not a pipeline we "
                    "instrumented — so its row is a single block. Ours is broken out below. The "
                    "two run one after the other, never at once: sharing the cores would make "
                    "both readings worse and neither true."
                }
            }
            div { class: "legend",
                for st in r.stages.iter() {
                    div { class: "leg",
                        i { style: "background:{stage_colour(&st.name)};" }
                        b { "{st.name}" }
                        span { class: "ms", {format!("{:.0} ms · {:.0}%", st.ms, st.ms / total * 100.0)} }
                        em { "{st.what}" }
                    }
                }
            }

            // ---- the tile grid: why vision costs what it does ----
            h3 { class: "sect",
                {format!("The picture becomes {} tiles", r.tiles)}
            }
            p { class: "note", style: "margin-top:0;",
                {format!("Longest edge to 2048, each edge rounded up to a multiple of {}, \
                          cut into a {}×{} grid — plus one global thumbnail of the whole \
                          image. {} tiles × {} tokens = {} image tokens. Every tile is its \
                          own SigLIP forward pass, which is why `vision` dominates the bar \
                          above.",
                         r.tile, r.rows, r.cols, r.tiles, r.tok_per_tile, r.tok_image)}
            }
            div { class: "tilewrap",
                img { class: "tileimg", src: "{r.preview}", alt: "the image being described" }
                if r.rows > 0 {
                    div {
                        class: "tilegrid",
                        style: "grid-template-columns:repeat({r.cols},1fr);grid-template-rows:repeat({r.rows},1fr);",
                        for i in 0..(r.rows * r.cols) {
                            div { class: "tilecell",
                                span { {format!("r{}c{}", i / r.cols + 1, i % r.cols + 1)} }
                            }
                        }
                    }
                }
            }
            div { class: "badge",
                {format!("source {}×{} → resized {}×{} → {} × {}px tiles + 1 thumbnail",
                         r.width, r.height, r.resized_w, r.resized_h, r.rows * r.cols, r.tile)}
            }

            // ---- per-tile cost ----
            if !r.tower_per_tile_ms.is_empty() {
                h3 { class: "sect", "Cost per tile" }
                div { class: "bars",
                    for (i, ms) in r.tower_per_tile_ms.iter().enumerate() {
                        div {
                            class: if i + 1 == r.tower_per_tile_ms.len() { "bar thumb" } else { "bar" },
                            style: "height:{ms / tower_peak * 100.0}%;",
                            title: {
                                if i + 1 == r.tower_per_tile_ms.len() {
                                    format!("global thumbnail: {ms:.0} ms")
                                } else {
                                    format!("r{}c{}: {ms:.0} ms", i / r.cols.max(1) + 1, i % r.cols.max(1) + 1)
                                }
                            },
                        }
                    }
                }
                p { class: "note", style: "margin-top:6px;",
                    {format!("One bar per tile, the last being the global thumbnail. \
                              {:.0}–{:.0} ms, mean {:.0} — near-flat, because every tile is the \
                              same {}×{} pixels regardless of what is in it. A VLM's vision cost \
                              is a function of the tile COUNT, not of how complicated the picture \
                              is. Doubling the resolution of this image would not make the model \
                              think harder; it would hand it more tiles.",
                             tower_min, tower_peak, tower_mean, r.tile, r.tile)}
                }
            }

            // ---- token budget ----
            h3 { class: "sect", "The prompt the decoder actually sees" }
            div { class: "tokbar",
                div {
                    class: "tokseg img",
                    style: "width:{r.tok_image as f64 / r.max_positions as f64 * 100.0}%;",
                    title: "{r.tok_image} image tokens",
                }
                div {
                    class: "tokseg txt2",
                    style: "width:{r.tok_text as f64 / r.max_positions as f64 * 100.0}%;",
                    title: "{r.tok_text} text tokens",
                }
            }
            div { class: "legend",
                div { class: "leg", i { style: "background:#f0883e;" } b { "image" }
                      span { class: "ms", {format!("{} tokens", r.tok_image)} }
                      em { {format!("{} tiles × {}", r.tiles, r.tok_per_tile)} } }
                div { class: "leg", i { style: "background:#a371f7;" } b { "text" }
                      span { class: "ms", {format!("{} tokens", r.tok_text)} }
                      em { "chat template + your question" } }
                div { class: "leg", i { style: "background:#21262d;" } b { "headroom" }
                      span { class: "ms", {format!("{} tokens", r.max_positions.saturating_sub(r.tok_prompt))} }
                      em { {format!("of the tower's {} positions", r.max_positions)} } }
            }
            p { class: "note",
                {format!("{:.0}% of this prompt is the picture. That is the number behind every \
                          other number on this page: prefill is one pass over all {} tokens, and \
                          it happens before a single word is written.",
                         r.tok_image as f64 / r.tok_prompt.max(1) as f64 * 100.0, r.tok_prompt)}
            }

            // ---- per-token generation ----
            if !r.step_ms.is_empty() {
                h3 { class: "sect", "Cost per generated token" }
                div { class: "bars",
                    for (i, ms) in r.step_ms.iter().enumerate() {
                        div {
                            class: "bar gen",
                            style: "height:{ms / step_peak * 100.0}%;",
                            title: "token {i + 1}: {ms:.1} ms",
                        }
                    }
                }
                p { class: "note", style: "margin-top:6px;",
                    {format!("{} tokens at {:.1}/s. Each bar is one forward pass with the KV cache \
                              already warm — flat and cheap next to the prefill, which is the point: \
                              a longer ANSWER costs little, a bigger PICTURE costs a lot.",
                             r.step_ms.len(), r.tokens_per_sec)}
                }
            }
        }
    }
}

/// Percent-escape the characters that would break a query value.
///
/// Deliberately small: this escapes what a prompt actually contains that a
/// query string cannot carry, rather than reimplementing RFC 3986. The server
/// decodes `%XX` and `+` and leaves everything else alone, so the two halves
/// agree by construction.
fn url_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Carmenta's tab: one image, BOTH OCR lineages, and the content class that
/// decides which one the pipeline would pick.
///
/// The side-by-side is the point. Neither engine is "the" OCR engine — the
/// measured sign-flip says `craft-crnn` wins clean screen text (frames
/// 1.602 % vs 5.034 % CER) and `craft-parseq` wins photographs (CORD
/// 21.70 % vs 27.42 %). A table asserts that; this lets you falsify it on
/// your own image in one click.
#[component]
fn Read() -> Element {
    let mut busy = use_signal(|| false);
    let mut out = use_signal(ReadOut::default);

    let pick = move |_| {
        if busy() {
            return;
        }
        busy.set(true);
        spawn(async move {
            let mut eval = document::eval(READ_JS);
            if let Ok(msg) = eval.recv::<String>().await {
                let v: serde_json::Value = serde_json::from_str(&msg).unwrap_or_default();
                let pane = |k: &str| {
                    let p = &v[k];
                    Reading {
                        text: p["text"].as_str().unwrap_or_default().to_string(),
                        lines: p["lines"].as_u64().unwrap_or(0) as usize,
                        ms: p["ms"].as_f64().unwrap_or(0.0),
                        error: p["error"].as_str().map(str::to_string),
                    }
                };
                out.set(ReadOut {
                    preview: v["preview"].as_str().unwrap_or_default().to_string(),
                    width: v["width"].as_u64().unwrap_or(0),
                    height: v["height"].as_u64().unwrap_or(0),
                    content: v["content"].as_str().unwrap_or_default().to_string(),
                    flatness: v["flatness"].as_f64().unwrap_or(0.0),
                    crnn: pane("crnn"),
                    parseq: pane("parseq"),
                    error: v["error"].as_str().map(str::to_string),
                });
            }
            busy.set(false);
        });
    };

    let r = out();
    let photo = r.content == "photographic";
    rsx! {
        p { class: "lede",
            "Choose an image — or paste one from the clipboard once the picker opens. "
            "Both OCR lineages read the identical pixels; the winner depends on what "
            "kind of image it is, which is why the pipeline dispatches instead of "
            "picking a favourite."
        }
        div { class: "row",
            button { class: "primary", disabled: busy(), onclick: pick,
                if busy() { "Reading…" } else { "Choose or paste an image" }
            }
            if !r.content.is_empty() {
                span { class: "badge",
                    {format!("{} · {}×{} · flatness {:.2}", r.content, r.width, r.height, r.flatness)}
                }
            }
        }
        if let Some(e) = r.error.clone() {
            div { class: "err", "{e}" }
        }
        if !r.preview.is_empty() {
            div { class: "shot", img { src: "{r.preview}", alt: "the image being read" } }
        }
        if !r.content.is_empty() {
            div { class: "panes",
                ReadPane {
                    name: "craft-crnn".to_string(),
                    note: if photo { "line-level CTC — weaker on photographs".to_string() }
                          else { "line-level CTC — the dispatch pick here".to_string() },
                    favoured: !photo,
                    pane: r.crnn.clone(),
                }
                ReadPane {
                    name: "craft-parseq".to_string(),
                    note: if photo { "word-level AR — the dispatch pick here".to_string() }
                          else { "word-level AR — beats PaddleOCR's recognizer on photo crops".to_string() },
                    favoured: photo,
                    pane: r.parseq.clone(),
                }
            }
        }
    }
}

#[component]
fn ReadPane(name: String, note: String, favoured: bool, pane: Reading) -> Element {
    rsx! {
        div { class: if favoured { "pane won" } else { "pane" },
            div { class: "pane-head",
                strong { "{name}" }
                span { class: "ms", {format!("{:.0} ms · {} lines", pane.ms, pane.lines)} }
            }
            div { class: "note", "{note}" }
            if let Some(e) = pane.error.clone() {
                div { class: "err", "{e}" }
            } else {
                pre { class: "ocr", "{pane.text}" }
            }
        }
    }
}

#[component]
fn Listen() -> Element {
    let mut running = use_signal(|| false);
    let mut mercury = use_signal(Vec::<Line>::new);
    let mut cpp = use_signal(Vec::<Line>::new);
    let mut status = use_signal(|| "idle".to_string());
    // Speaker labels ON by default — they are the capability worth showing.
    // But whisper.cpp does not diarize at all, so with this on the two panes
    // are timing different work, and the per-chunk numbers are NOT a
    // like-for-like speed comparison. Turning it off is what makes them one.
    let mut diarize = use_signal(|| true);

    let start = move |_| {
        if running() {
            return;
        }
        running.set(true);
        status.set("listening…".into());
        spawn(async move {
            // 1 s cadence, 10 s of context per pass. Nearly free: 1 s of audio
            // costs 213 ms and 10 s costs 311 ms, because the encoder pads to
            // 30 s either way.
            let js = RECORDER_JS
                .replace("__TICK__", "1")
                .replace("__WINDOW__", &WINDOW_SECS.to_string())
                .replace("__DIARIZE__", if diarize() { "1" } else { "0" });
            let mut eval = document::eval(&js);
            // One message per transcribed chunk until the recorder reports
            // done. Each carries BOTH engines' answers for the same audio.
            while let Ok(msg) = eval.recv::<String>().await {
                let Ok(v) = serde_json::from_str::<serde_json::Value>(&msg) else { continue };
                if v.get("done").is_some() {
                    break;
                }
                if let Some(f) = v.get("fatal").and_then(|f| f.as_str()) {
                    status.set(f.to_string());
                    break;
                }
                let pull = |key: &str| -> Line {
                    let o = v.get(key);
                    Line {
                        text: o
                            .and_then(|o| o.get("text"))
                            .and_then(|t| t.as_str())
                            .unwrap_or_default()
                            .to_string(),
                        ms: o.and_then(|o| o.get("ms")).and_then(|m| m.as_f64()).unwrap_or(0.0),
                        error: o
                            .and_then(|o| o.get("error"))
                            .and_then(|e| e.as_str())
                            .map(str::to_string),
                        committed: false,
                        turns: o
                            .and_then(|o| o.get("turns"))
                            .and_then(|t| t.as_array())
                            .map(|a| {
                                a.iter()
                                    .map(|t| Turn {
                                        speaker: t
                                            .get("speaker")
                                            .and_then(|s| s.as_str())
                                            .map(str::to_string),
                                        text: t
                                            .get("text")
                                            .and_then(|s| s.as_str())
                                            .unwrap_or_default()
                                            .to_string(),
                                        start: t
                                            .get("start")
                                            .and_then(serde_json::Value::as_f64)
                                            .unwrap_or(0.0),
                                        end: t
                                            .get("end")
                                            .and_then(serde_json::Value::as_f64)
                                            .unwrap_or(0.0),
                                    })
                                    .collect()
                            })
                            .unwrap_or_default(),
                    }
                };
                // Each pass REPLACES the live line; a commit freezes it and the
                // next pass starts a new one. That is what makes text grow and
                // self-correct while you speak, instead of arriving in blocks.
                let committing = v.get("commit").and_then(|c| c.as_bool()).unwrap_or(false);
                let mut put = |sig: &mut Signal<Vec<Line>>, line: Line| {
                    if line.text.is_empty() && line.error.is_none() {
                        return;
                    }
                    let mut w = sig.write();
                    match w.last_mut() {
                        Some(last) if !last.committed => *last = line,
                        _ => w.push(line),
                    }
                    if committing {
                        if let Some(last) = w.last_mut() {
                            last.committed = true;
                        }
                    }
                    // Keep the transcript bounded. One line per utterance is
                    // slow growth, but nothing pruned it and every line is a
                    // live DOM node — a long session grew the list and the
                    // render tree together until scrolling stuttered. A demo
                    // that degrades after twenty minutes is a demo that fails
                    // in front of someone.
                    const KEEP: usize = 200;
                    if w.len() > KEEP {
                        let drop_n = w.len() - KEEP;
                        w.drain(..drop_n);
                    }
                };
                put(&mut mercury, pull("mercury"));
                put(&mut cpp, pull("whispercpp"));
            }
            running.set(false);
            status.set("stopped".into());
        });
    };

    let stop = move |_| {
        document::eval(STOP_JS);
        status.set("finishing last chunk…".into());
    };

    let clear = move |_| {
        mercury.write().clear();
        cpp.write().clear();
    };

    rsx! {
        div {
            header {
                p { class: "sub",
                    "Both engines re-transcribe the same trailing 10 seconds of your \
                     microphone, once a second — same bytes, same greedy settings, same \
                     model size (tiny.en). Grey italic text is still being revised; it \
                     firms up when you pause."
                }
                p { class: "warn",
                    "The millisecond figures are NOT a speed comparison. Mercury runs warm \
                     in-process; whisper.cpp is a subprocess that reloads its model on \
                     every pass, so it is charged model load each time. Putting startup \
                     inside a timed run is the exact defect this project fixed at the \
                     benchmark level. For real throughput use "
                    code { "ffai bench asr" }
                    "."
                }
            }
            div { class: "bar",
                // Dynamic `class:` rather than a dynamic `style:` — a style
                // string is applied on mount only and will not re-apply when
                // the signal changes.
                button {
                    class: if running() { "btn on" } else { "btn" },
                    disabled: running(),
                    onclick: start,
                    "▶ Start"
                }
                button {
                    class: "btn",
                    disabled: !running(),
                    onclick: stop,
                    "■ Stop"
                }
                button { class: "btn ghost", onclick: clear, "Clear" }
                // Locked while running: the recorder JS is templated with the
                // flag at spawn time, so flipping it mid-session would change
                // the label without changing what is measured — a control
                // that lies is worse than no control.
                button {
                    class: if diarize() { "btn on" } else { "btn ghost" },
                    disabled: running(),
                    onclick: move |_| diarize.set(!diarize()),
                    if diarize() { "Speakers: ON" } else { "Speakers: OFF" }
                }
                span { class: "status", "{status}" }
            }
            p { class: if diarize() { "warn" } else { "sub" },
                if diarize() {
                    "Speakers ON: Mercury is additionally running ECAPA-TDNN speaker \
                     embedding and cross-chunk identity matching — roughly four extra \
                     network passes per chunk — which whisper.cpp does not do at all. \
                     Measured at +621 ms on a 3 s chunk, 6.8× the ASR-only path. The two \
                     panes are NOT doing equal work; turn this off to compare speed."
                } else {
                    "Speakers OFF: both panes now do the same job — transcribe this audio. \
                     Mercury's ASR-only path measures 107 ms against whisper.cpp's 274 ms \
                     on a 3 s chunk, and silence costs Mercury ~0 ms because VAD drops it \
                     before the encoder, where whisper.cpp pays a full pass to print \
                     [BLANK_AUDIO]."
                }
            }
            div { class: "panes",
                Pane {
                    title: if diarize() { "Mercury (pure Rust) — ASR + speakers" } else { "Mercury (pure Rust) — ASR only" },
                    accent: "rust",
                    lines: mercury(),
                }
                Pane { title: "whisper.cpp (C++/ggml) — ASR only", accent: "cpp", lines: cpp() }
            }
        }
    }
}

#[component]
fn Speak() -> Element {
    let mut text = use_signal(|| {
        "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background."
            .to_string()
    });
    let mut speed = use_signal(|| 1.0f64);
    let mut noise_scale = use_signal(|| 0.667f64);
    let mut noise_w = use_signal(|| 0.8f64);
    let mut seed = use_signal(|| 42u64);
    let mut busy = use_signal(|| false);
    let mut result = use_signal(Spoken::default);

    // Signals are `Copy`, so the closure shadows them with local mutable
    // copies. That keeps `speak` itself `Fn`/`Copy` and therefore usable by
    // BOTH buttons; a closure that mutated its captures directly would be
    // `FnMut`, move into the first `onclick`, and refuse the second.
    let speak = move |verify: bool| {
        let (mut busy, mut result) = (busy, result);
        if busy() {
            return;
        }
        busy.set(true);
        let payload = serde_json::json!({
            "text": text(),
            "speed": speed(),
            "noise_scale": noise_scale(),
            "noise_w": noise_w(),
            "seed": seed(),
            "verify": verify,
        })
        .to_string();
        spawn(async move {
            let js = SPEAK_JS.replace("__PAYLOAD__", &format!("{}", serde_json::json!(payload)));
            let mut eval = document::eval(&js);
            if let Ok(msg) = eval.recv::<String>().await {
                match serde_json::from_str::<serde_json::Value>(&msg) {
                    Ok(v) => {
                        let strs = |k: &str| -> Vec<String> {
                            v.get(k)
                                .and_then(|a| a.as_array())
                                .map(|a| {
                                    a.iter()
                                        .filter_map(|s| s.as_str().map(str::to_string))
                                        .collect()
                                })
                                .unwrap_or_default()
                        };
                        result.set(Spoken {
                            wav_base64: v
                                .get("wav_base64")
                                .and_then(|s| s.as_str())
                                .unwrap_or_default()
                                .to_string(),
                            sample_rate: v
                                .get("sample_rate")
                                .and_then(|s| s.as_u64())
                                .unwrap_or(0) as u32,
                            audio_secs: v
                                .get("audio_secs")
                                .and_then(|s| s.as_f64())
                                .unwrap_or(0.0),
                            synth_ms: v.get("synth_ms").and_then(|s| s.as_f64()).unwrap_or(0.0),
                            xrt: v.get("xrt").and_then(|s| s.as_f64()).unwrap_or(0.0),
                            sentences: strs("sentences"),
                            phonemes: strs("phonemes"),
                            sha256: v
                                .get("sha256")
                                .and_then(|s| s.as_str())
                                .unwrap_or_default()
                                .to_string(),
                            deterministic: v.get("deterministic").and_then(|d| d.as_bool()),
                            error: v.get("error").and_then(|e| e.as_str()).map(str::to_string),
                        });
                    }
                    Err(e) => {
                        result.set(Spoken {
                            error: Some(format!("bad response: {e}")),
                            ..Spoken::default()
                        });
                    }
                }
            }
            busy.set(false);
        });
    };

    let r = result();
    rsx! {
        div {
            header {
                p { class: "sub",
                    "Type anything and Mercury speaks it — the full VITS stack on candle, \
                     running piper's own voice file (en_US-lessac-medium), phonemized by our \
                     own clean-room G2P with no espeak-ng and nothing GPL linked in."
                }
                p { class: "warn",
                    "The ×realtime figure is one warm call on a machine you are also using — \
                     it collapses under load (measured: 20× idle, under 2× with the cores \
                     saturated), so read it as a liveness check, not a benchmark. Mercury is \
                     currently BEHIND piper on synthesis throughput (19–20× against its \
                     25–32× on a quiet box) and that gate is reported as failing; use "
                    code { "ffai bench tts" }
                    " for the measured comparison. What this tab shows that a table cannot: \
                     the phonemes, the sentence split, and byte-identical output under a seed."
                }
            }
            div { class: "bar",
                button {
                    class: if busy() { "btn on" } else { "btn" },
                    disabled: busy(),
                    onclick: move |_| speak(false),
                    "🔊 Speak"
                }
                button {
                    class: "btn ghost",
                    disabled: busy(),
                    onclick: move |_| speak(true),
                    "Speak twice · prove determinism"
                }
                span { class: "status", if busy() { "synthesizing…" } else { "" } }
            }
            textarea {
                class: "say",
                rows: 3,
                value: "{text}",
                oninput: move |e| text.set(e.value()),
            }
            div { class: "knobs",
                Knob { label: "speed", value: speed(), min: 0.5, max: 2.0, step: 0.05,
                       oninput: move |v| speed.set(v) }
                Knob { label: "noise_scale", value: noise_scale(), min: 0.0, max: 1.5, step: 0.05,
                       oninput: move |v| noise_scale.set(v) }
                Knob { label: "noise_w", value: noise_w(), min: 0.0, max: 1.5, step: 0.05,
                       oninput: move |v| noise_w.set(v) }
                label { class: "knob",
                    span { "seed" }
                    input {
                        r#type: "number",
                        value: "{seed}",
                        min: "0",
                        oninput: move |e| { if let Ok(v) = e.value().parse::<u64>() { seed.set(v) } },
                    }
                }
            }

            if let Some(err) = &r.error {
                div { class: "pane", div { class: "body", span { class: "err", "{err}" } } }
            } else if !r.wav_base64.is_empty() {
                div { class: "pane rust",
                    h2 { "Audio" }
                    div { class: "body",
                        audio {
                            controls: true,
                            class: "player",
                            src: "data:audio/wav;base64,{r.wav_base64}",
                        }
                        div { class: "stats",
                            span { b { "{r.audio_secs:.2} s" } " audio" }
                            span { b { "{r.synth_ms:.0} ms" } " synthesis" }
                            span { b { "{r.xrt:.1}×" } " realtime" }
                            span { b { "{r.sample_rate} Hz" } }
                        }
                        div { class: "hash",
                            "sha256(samples) "
                            code { "{r.sha256}" }
                        }
                        match r.deterministic {
                            Some(true) => rsx! {
                                div { class: "ok",
                                    "✓ Synthesized twice with seed {seed} — the samples are \
                                     byte-identical. piper samples its noise inside the ONNX \
                                     graph and cannot reproduce its own output."
                                }
                            },
                            Some(false) => rsx! {
                                div { class: "err",
                                    "✗ The two runs differed. That is a bug — same input and \
                                     seed must give the same bytes."
                                }
                            },
                            None => rsx! {},
                        }
                    }
                }
                div { class: "pane",
                    h2 { "What the model was given" }
                    div { class: "body",
                        for (i, ipa) in r.phonemes.iter().enumerate() {
                            div { class: "line",
                                div { style: "flex:1",
                                    div { class: "sent",
                                        "{r.sentences.get(i).map(String::as_str).unwrap_or(\"\")}"
                                    }
                                    div { class: "ipa", "{ipa}" }
                                }
                            }
                        }
                        p { class: "note",
                            "Each sentence is synthesized on its own and joined with a silence \
                             gap — that is why the split is shown. The IPA is what our G2P \
                             emitted and what the voice actually received; it is gated against \
                             espeak-ng's output on a pinned corpus (83.6 % of holdout sentences \
                             match character-for-character), and fed through piper's own runtime \
                             it scores inside the 5 % round-trip band."
                        }
                    }
                }
            }
        }
    }
}

#[component]
fn Knob(
    label: String,
    value: f64,
    min: f64,
    max: f64,
    step: f64,
    oninput: EventHandler<f64>,
) -> Element {
    rsx! {
        label { class: "knob",
            span { "{label}" }
            input {
                r#type: "range",
                min: "{min}",
                max: "{max}",
                step: "{step}",
                value: "{value}",
                oninput: move |e| { if let Ok(v) = e.value().parse::<f64>() { oninput.call(v) } },
            }
            b { "{value:.2}" }
        }
    }
}

#[component]
fn Pane(title: String, accent: String, lines: Vec<Line>) -> Element {
    let total: f64 = lines.iter().map(|l| l.ms).sum();
    let avg = if lines.is_empty() { 0.0 } else { total / lines.len() as f64 };
    // Everyone heard so far, in label order rather than first-heard order: the
    // legend then keeps its positions as the session grows instead of
    // reshuffling every time a new voice arrives.
    let mut cast: Vec<String> = Vec::new();
    for who in lines
        .iter()
        .flat_map(|l| l.turns.iter())
        .filter_map(|t| t.speaker.as_ref())
    {
        if !cast.iter().any(|c| c == who) {
            cast.push(who.clone());
        }
    }
    cast.sort();
    rsx! {
        section { class: "pane {accent}",
            h2 { "{title}" }
            div { class: "meta",
                if lines.is_empty() {
                    "no chunks yet"
                } else {
                    "{lines.len()} chunks · {avg:.0} ms avg"
                }
                if !cast.is_empty() {
                    span { class: "cast",
                        for who in cast.iter() {
                            span {
                                class: "chip s{speaker_slot(who)}",
                                title: "{who}",
                                "{short_label(who)}"
                            }
                        }
                    }
                }
            }
            div { class: "body",
                for (i, line) in lines.iter().enumerate().rev() {
                    Chunk { key: "{i}", line: line.clone() }
                }
            }
        }
    }
}

/// One chunk's answer, laid out one utterance per row.
///
/// A component of its own because the layout needs plain Rust before any
/// markup: a row hides its speaker chip when the row above it had the same
/// speaker, and the ribbon needs the chunk's whole span before it can place
/// the first block — neither of which a `for` inside `rsx!` can look back or
/// forward to work out. Prepare the rows, then render them.
#[component]
fn Chunk(line: Line) -> Element {
    // No turns: whisper.cpp, which does not diarize, and Mercury with speakers
    // off. One paragraph is the right shape when there is nobody to attribute
    // it to, so this path is the previous rendering unchanged.
    if line.turns.is_empty() || line.error.is_some() {
        return rsx! {
            div { class: "line",
                if let Some(err) = &line.error {
                    span { class: "err", "{err}" }
                } else {
                    span {
                        class: if line.committed { "txt" } else { "txt live" },
                        "{line.text}"
                    }
                    span { class: "ms", "{line.ms:.0} ms" }
                }
            }
        };
    }

    /// A prepared row: everything the template needs, already decided.
    struct Row {
        /// Colour class — `s0`..`s7`, or `sx` for an unattributed segment.
        slot: String,
        /// Empty when the row above was the same speaker, so a monologue reads
        /// as one block instead of six copies of the same name.
        chip: String,
        title: String,
        at: String,
        text: String,
        /// Whisper's own marker for "nothing here". Dimmed rather than
        /// dropped: that the engine ran and found silence is a fact worth
        /// seeing, and it is what the VAD claim beside the panes is about.
        blank: bool,
        /// Position within the chunk's span, in percent, for the ribbon.
        left: f64,
        width: f64,
    }

    let t0 = line
        .turns
        .iter()
        .map(|t| t.start)
        .fold(f64::INFINITY, f64::min);

    let rows: Vec<Row> = line
        .turns
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let repeat = i > 0 && line.turns[i - 1].speaker == t.speaker;
            Row {
                slot: t
                    .speaker
                    .as_deref()
                    .map_or_else(|| "sx".to_string(), |w| format!("s{}", speaker_slot(w))),
                chip: match (&t.speaker, repeat) {
                    (Some(w), false) => short_label(w),
                    (Some(_), true) => String::new(),
                    (None, _) => "?".to_string(),
                },
                title: t.speaker.clone().unwrap_or_else(|| {
                    "no speaker turn covers this segment".to_string()
                }),
                at: clock(t.start),
                text: t.text.clone(),
                blank: t.text == "[BLANK_AUDIO]",
                // Against the window, not against this chunk's own span:
                // see `WINDOW_SECS`. A chunk holding two seconds of speech
                // draws two seconds of ribbon.
                left: ((t.start - t0) / WINDOW_SECS * 100.0).clamp(0.0, 100.0),
                width: ((t.end - t.start) / WINDOW_SECS * 100.0).clamp(1.0, 100.0),
            }
        })
        .collect();

    rsx! {
        div { class: if line.committed { "chunk" } else { "chunk live" },
            // Who spoke when, to scale, across this chunk's own span. The rows
            // below say what was said; this says how it was shared out.
            div { class: "chunk-head",
                div { class: "ribbon",
                    for (i, r) in rows.iter().enumerate() {
                        div {
                            key: "{i}",
                            class: "blk {r.slot}",
                            style: "left:{r.left:.2}%;width:{r.width:.2}%",
                            title: "{r.title}",
                        }
                    }
                }
                span { class: "ms", "{line.ms:.0} ms" }
            }
            for (i, r) in rows.iter().enumerate() {
                div {
                    key: "{i}",
                    class: if r.blank { "turn {r.slot} blank" } else { "turn {r.slot}" },
                    span { class: "who", title: "{r.title}", "{r.chip}" }
                    span { class: "at", "{r.at}" }
                    span { class: "said", "{r.text}" }
                }
            }
        }
    }
}

const CSS: &str = r#"
* { box-sizing: border-box; }
body { margin:0; background:#0d1117; color:#e6edf3;
       font:15px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif; }
.wrap { max-width:1100px; margin:0 auto; padding:28px 20px 48px; }
h1 { font-size:22px; margin:0 0 4px; letter-spacing:-.01em; }
.sub { margin:0 0 10px; color:#8b949e; max-width:64ch; }
.warn { margin:0 0 20px; padding:10px 12px; max-width:64ch; font-size:13px;
        color:#d29922; background:#1c1a12; border:1px solid #3d2f11; border-radius:7px; }
.warn code { color:#e6edf3; background:#21262d; padding:1px 5px; border-radius:4px; }
.bar { display:flex; gap:10px; align-items:center; margin-bottom:20px; flex-wrap:wrap; }
.btn { background:#21262d; color:#e6edf3; border:1px solid #30363d; border-radius:7px;
       padding:9px 16px; font-size:14px; cursor:pointer; }
.btn:hover:not(:disabled) { background:#30363d; }
.btn:disabled { opacity:.4; cursor:not-allowed; }
.btn.on { background:#1f6feb; border-color:#1f6feb; }
.btn.ghost { background:transparent; }
.status { color:#8b949e; font-size:13px; margin-left:4px; }
.panes { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
@media (max-width:820px){ .panes { grid-template-columns:1fr; } }
.pane { border:1px solid #30363d; border-radius:10px; overflow:hidden; background:#161b22; }
.pane h2 { font-size:14px; margin:0; padding:12px 14px 4px; }
.pane.rust h2 { color:#f0883e; }
.pane.cpp h2 { color:#79c0ff; }
.meta { padding:0 14px 10px; color:#8b949e; font-size:12px;
        border-bottom:1px solid #30363d; }
.body { padding:6px 14px 14px; max-height:56vh; overflow-y:auto; }
.line { padding:9px 0; border-bottom:1px solid #21262d; display:flex; gap:10px;
        align-items:baseline; }
.line:last-child { border-bottom:0; }
.txt { flex:1; white-space:pre-wrap; }
.txt.live { color:#8b949e; font-style:italic; }
.ms { color:#6e7681; font-size:11px; font-variant-numeric:tabular-nums; white-space:nowrap; }
.err { color:#f85149; font-size:13px; }

/* ---- speaker turns ----
   One row per utterance, a lane colour per speaker, and a ribbon showing how
   the chunk's seconds were shared out. Eight colours cycle; the label's own
   number picks the slot, so SPEAKER_02 is the same colour in the legend, in
   the ribbon and on every row, for as long as the registry holds the
   identity. */
.s0 { --c:#58a6ff; } .s1 { --c:#3fb950; } .s2 { --c:#d29922; } .s3 { --c:#db61a2; }
.s4 { --c:#a371f7; } .s5 { --c:#39c5cf; } .s6 { --c:#ff9e64; } .s7 { --c:#bc8cff; }
.sx { --c:#6e7681; }
.chunk { padding:9px 0; border-bottom:1px solid #21262d; }
.chunk:last-child { border-bottom:0; }
.chunk-head { display:flex; align-items:center; gap:10px; margin-bottom:6px; }
.ribbon { position:relative; flex:1; height:6px; border-radius:3px;
          background:#21262d; overflow:hidden; }
.ribbon .blk { position:absolute; top:0; bottom:0; border-radius:3px;
               background:var(--c); }
.turn { display:flex; gap:9px; align-items:baseline; padding:3px 0 3px 9px;
        border-left:3px solid var(--c); }
.turn + .turn { margin-top:2px; }
.turn .who { flex:0 0 26px; font-size:11px; font-weight:600; color:var(--c);
             font-variant-numeric:tabular-nums; }
.turn .at { flex:0 0 32px; font-size:11px; color:#6e7681;
            font-variant-numeric:tabular-nums; }
.turn .said { flex:1; }
.turn.blank .said { color:#6e7681; font-style:italic; }
.chunk.live .said { color:#8b949e; font-style:italic; }
.cast { display:inline-flex; gap:5px; margin-left:8px; vertical-align:middle; }
.chip { border:1px solid var(--c); color:var(--c); border-radius:999px;
        padding:0 7px; font-size:11px; font-weight:600; line-height:16px;
        font-variant-numeric:tabular-nums; }

/* ---- read tab ---- */
.shot { margin:14px 0; border:1px solid #30363d; border-radius:8px; overflow:hidden; background:#0d1117; }
.shot img { display:block; max-width:100%; max-height:340px; margin:0 auto; }
.panes { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
.pane { border:1px solid #30363d; border-radius:8px; padding:12px; background:#0d1117; }
.pane.won { border-color:#f0883e; }
.pane-head { display:flex; justify-content:space-between; align-items:baseline; gap:8px; }
.note { color:#8b949e; font-size:12px; margin:4px 0 8px; }
.ocr { white-space:pre-wrap; word-break:break-word; font-size:13px; line-height:1.5;
       margin:0; max-height:320px; overflow:auto; color:#e6edf3; }
.badge { color:#8b949e; font-size:12px; font-variant-numeric:tabular-nums; }
@media (max-width:820px) { .panes { grid-template-columns:1fr; } }

/* ---- tabs ---- */
.tabs { display:flex; gap:6px; margin:14px 0 18px; border-bottom:1px solid #30363d; }
.tab { background:transparent; color:#8b949e; border:0; border-bottom:2px solid transparent;
       padding:8px 14px; font-size:14px; cursor:pointer; }
.tab:hover { color:#e6edf3; }
.tab.on { color:#e6edf3; border-bottom-color:#f0883e; }
.view.hidden { display:none; }

/* ---- speak tab ---- */
.say { width:100%; background:#0d1117; color:#e6edf3; border:1px solid #30363d;
       border-radius:8px; padding:11px 13px; font:inherit; resize:vertical;
       margin-bottom:14px; }
.say:focus { outline:0; border-color:#1f6feb; }
.knobs { display:flex; gap:18px; flex-wrap:wrap; margin-bottom:18px; }
.knob { display:flex; align-items:center; gap:8px; font-size:12px; color:#8b949e; }
.knob span { min-width:74px; }
.knob b { color:#e6edf3; font-variant-numeric:tabular-nums; min-width:34px; }
.knob input[type=range] { width:110px; accent-color:#f0883e; }
.knob input[type=number] { width:88px; background:#0d1117; color:#e6edf3;
       border:1px solid #30363d; border-radius:6px; padding:4px 7px; font:inherit; }
.pane + .pane { margin-top:14px; }
.player { width:100%; margin-bottom:12px; }
.stats { display:flex; gap:18px; flex-wrap:wrap; font-size:12px; color:#8b949e;
         padding-bottom:10px; }
.stats b { color:#e6edf3; font-variant-numeric:tabular-nums; }
.hash { font-size:11px; color:#6e7681; word-break:break-all; padding-bottom:10px; }
.hash code { color:#8b949e; }
.ok { color:#3fb950; font-size:13px; background:#0f1c12; border:1px solid #1f4227;
      border-radius:7px; padding:9px 11px; }
.sent { color:#e6edf3; margin-bottom:3px; }
.ipa { color:#f0883e; font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
       font-size:13px; word-break:break-word; }
.note { color:#8b949e; font-size:12px; margin:12px 0 0; max-width:72ch; }

/* ---- see tab (Argus) ---- */
.sect { font-size:14px; margin:22px 0 8px; color:#e6edf3; letter-spacing:-.01em; }
.trow { display:flex; align-items:center; gap:10px; margin-bottom:6px; }
.tlabel { width:64px; flex:none; font-size:12px; color:#8b949e; text-align:right; }
.trow .ms { width:74px; flex:none; text-align:right; }
.trow .timeline { flex:1; }
.timeline { display:flex; height:26px; border-radius:6px; overflow:hidden;
            border:1px solid #30363d; background:#0d1117; }
.timeline .seg { min-width:2px; transition:filter .15s; }
.timeline .seg:hover { filter:brightness(1.35); }
.legend { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
          gap:4px 18px; margin-top:10px; }
.leg { display:flex; align-items:baseline; gap:7px; font-size:12px; color:#8b949e; }
.leg i { width:9px; height:9px; border-radius:2px; flex:none; align-self:center; }
.leg b { color:#e6edf3; min-width:74px; }
.leg .ms { min-width:96px; }
.leg em { font-style:normal; color:#6e7681; overflow:hidden; text-overflow:ellipsis;
          white-space:nowrap; }
.tilewrap { position:relative; display:inline-block; max-width:100%; margin:4px 0 8px;
            border:1px solid #30363d; border-radius:8px; overflow:hidden; line-height:0; }
.tileimg { display:block; width:auto; height:auto; min-width:min(420px,100%);
            max-width:100%; max-height:400px; image-rendering:pixelated; }
.tilegrid { position:absolute; inset:0; display:grid; pointer-events:none; }
.tilecell { border:1px solid rgba(240,136,62,.55); display:flex; align-items:flex-start;
            justify-content:flex-start; }
.tilecell span { font:10px/1 ui-monospace,SFMono-Regular,Menlo,monospace;
                 color:#0d1117; background:rgba(240,136,62,.85); padding:2px 3px;
                 border-radius:0 0 3px 0; }
.bars { display:flex; align-items:flex-end; gap:3px; height:74px; padding:6px 8px;
        border:1px solid #30363d; border-radius:8px; background:#0d1117; }
.bar { flex:1; min-width:3px; background:#f0883e; border-radius:2px 2px 0 0; min-height:2px; }
.bar.thumb { background:#a371f7; }
.bar.gen { background:#3fb950; }
.bar:hover { filter:brightness(1.4); }
.tokbar { display:flex; height:22px; border-radius:6px; overflow:hidden;
          border:1px solid #30363d; background:#21262d; }
.tokseg.img { background:#f0883e; }
.tokseg.txt2 { background:#a371f7; }
.row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin:8px 0 4px; }
.primary { background:#1f6feb; color:#fff; border:1px solid #1f6feb; border-radius:7px;
           padding:9px 16px; font-size:14px; cursor:pointer; }
.primary:disabled { opacity:.5; cursor:not-allowed; }
.lede { color:#8b949e; max-width:78ch; margin:0 0 14px; }
.verdict { margin:12px 0 0; padding:9px 12px; border-radius:7px; font-size:13px;
           max-width:82ch; }
.verdict b { text-transform:uppercase; letter-spacing:.04em; font-size:11px; }
.verdict.same { color:#3fb950; background:#0f1c12; border:1px solid #1f4227; }
.verdict.split { color:#d29922; background:#1c1a12; border:1px solid #3d2f11; }
.verdict.split span, .verdict.same span { color:#8b949e; }
"#;
