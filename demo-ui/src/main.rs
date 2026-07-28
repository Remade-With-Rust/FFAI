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
  const CHUNK = __CHUNK__;

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

  async function flush() {
    const have = pending.reduce((a, b) => a + b.length, 0);
    // Below ~0.4 s there is not enough audio to be worth a round trip.
    if (have < ctx.sampleRate * 0.4) return;
    const pcm = flatten(pending); pending = [];
    const wav = encodeWav(resample(pcm, ctx.sampleRate, 16000), 16000);
    try {
      const res = await fetch('/transcribe', { method: 'POST', body: wav });
      dioxus.send(await res.text());
    } catch (e) {
      dioxus.send(JSON.stringify({ fatal: 'transcribe request failed: ' + e }));
    }
  }

  while (!S.stop) {
    await new Promise((r) => setTimeout(r, 200));
    const have = pending.reduce((a, b) => a + b.length, 0);
    if (have >= ctx.sampleRate * CHUNK) await flush();
  }
  // Stop was pressed: send whatever is left rather than dropping the last
  // sentence mid-word.
  await flush();
  stream.getTracks().forEach((t) => t.stop());
  ctx.close();
  dioxus.send(JSON.stringify({ done: true }));
})();
"####;

const STOP_JS: &str = "window.__ffai = window.__ffai || {}; window.__ffai.stop = true;";

#[component]
fn App() -> Element {
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
            let js = RECORDER_JS.replace("__CHUNK__", "5");
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
                    }
                };
                let (m, c) = (pull("mercury"), pull("whispercpp"));
                if !m.text.is_empty() || m.error.is_some() {
                    mercury.write().push(m);
                }
                if !c.text.is_empty() || c.error.is_some() {
                    cpp.write().push(c);
                }
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
        style { {CSS} }
        div { class: "wrap",
            header {
                h1 { "Mercury vs whisper.cpp" }
                p { class: "sub",
                    "Both engines transcribe the same 5-second chunk of your microphone. \
                     Same audio, same greedy decode settings, same model size (tiny.en)."
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
                            span { class: "txt", "{line.text}" }
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
.ms { color:#6e7681; font-size:11px; font-variant-numeric:tabular-nums; white-space:nowrap; }
.err { color:#f85149; font-size:13px; }
"#;
