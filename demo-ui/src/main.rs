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
  node.onaudioprocess = (e) => {
    if (!S.stop) pending.push(new Float32Array(e.inputBuffer.getChannelData(0)));
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
        const res = await fetch('/transcribe', { method: 'POST', body: wav });
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

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Listen,
    Speak,
    Read,
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
                }
            }
            // Both views stay MOUNTED, hidden by CSS rather than unmounted, so
            // switching tabs mid-session does not tear down the recorder or
            // discard a transcript.
            div { class: if tab() == Tab::Listen { "view" } else { "view hidden" }, Listen {} }
            div { class: if tab() == Tab::Speak { "view" } else { "view hidden" }, Speak {} }
            div { class: if tab() == Tab::Read { "view" } else { "view hidden" }, Read {} }
        }
    }
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
            let js = RECORDER_JS.replace("__TICK__", "1").replace("__WINDOW__", "10");
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
                span { class: "status", "{status}" }
            }
            div { class: "panes",
                Pane { title: "Mercury (pure Rust)", accent: "rust", lines: mercury() }
                Pane { title: "whisper.cpp (C++/ggml)", accent: "cpp", lines: cpp() }
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
    rsx! {
        section { class: "pane {accent}",
            h2 { "{title}" }
            div { class: "meta",
                if lines.is_empty() {
                    "no chunks yet"
                } else {
                    "{lines.len()} chunks · {avg:.0} ms avg"
                }
            }
            div { class: "body",
                for line in lines.iter().rev() {
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
.txt { flex:1; }
.txt.live { color:#8b949e; font-style:italic; }
.ms { color:#6e7681; font-size:11px; font-variant-numeric:tabular-nums; white-space:nowrap; }
.err { color:#f85149; font-size:13px; }

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
"#;
