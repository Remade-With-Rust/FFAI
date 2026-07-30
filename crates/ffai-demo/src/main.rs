//! Side-by-side demo server: Mercury and whisper.cpp on the SAME audio.
//!
//! The browser records from the microphone, sends 16 kHz mono WAV here, and
//! this serves both transcripts back so they can be read next to each other.
//!
//! **Why both engines get a file rather than a buffer.** The point of the demo
//! is that the two implementations see *identical* input. Writing the posted
//! bytes to one temp file and handing that same path to both removes the whole
//! class of "did they actually get the same audio" doubt — which is the
//! mistake this project has made before at the benchmark level (a reference
//! invoked with `-nt` was doing 23 % less work for months). The file costs a
//! few milliseconds against a transcription and buys certainty.
//!
//! No storage, no state: the temp file is deleted before the response returns.
//!
//! ```sh
//! cargo run --release -p ffai-demo
//! # then open http://127.0.0.1:8787
//! ```

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ffai_carmenta::engine::CraftCrnn;
use ffai_core::engine::{AsrEngine, AsrOptions, OcrEngine, OcrOptions, TtsEngine, TtsOptions};
use ffai_mercury::asr::WhisperCandle;
use ffai_mercury::tts::PiperCandle;
use serde_json::json;

const ADDR: &str = "127.0.0.1:8787";
/// Cap the posted body. A microphone chunk is a few hundred KB; anything
/// wildly larger is a mistake or an attack, and an unbounded read on a public
/// socket is how a demo becomes an OOM.
const MAX_BODY: usize = 64 * 1024 * 1024;

struct Engines {
    mercury: WhisperCandle,
    whisper_cli: Option<PathBuf>,
    model: PathBuf,
}

/// The TTS engine lives behind its OWN lock, not inside `Engines`.
///
/// Sharing one mutex would serialise synthesis behind transcription: the ASR
/// demo holds its guard for the length of a whisper.cpp subprocess, so a
/// `Synthesize` click during a live session would block on it for a second or
/// more and look like the TTS engine was slow. Two locks, two independent
/// paths, and the speak tab stays responsive while the listen tab runs.
struct Speaker {
    piper: PiperCandle,
}

/// The OCR pair, behind its own lock for the same reason `Speaker` is: a
/// read must not queue behind a live transcription.
///
/// BOTH engines are held because the Read tab's whole point is the measured
/// content sign-flip — `craft-crnn` wins clean screen text (frames 1.602 %
/// vs 5.034 % CER), `craft-parseq` wins photographs (CORD 21.70 % vs
/// 27.42 %). Neither is "the OCR engine"; showing them side by side on the
/// user's OWN image is the honest way to present that.
struct Reader {
    crnn: CraftCrnn,
    parseq: CraftCrnn,
}

fn main() {
    let whisper_cli = [".whispercpp/whisper-cli.exe", ".whispercpp/whisper-cli"]
        .iter()
        .map(PathBuf::from)
        .find(|p| p.exists());
    let model = PathBuf::from(".whispercpp/ggml-tiny.en.bin");

    if whisper_cli.is_none() || !model.exists() {
        eprintln!(
            "note: whisper.cpp not found (.whispercpp/whisper-cli[.exe] + ggml-tiny.en.bin).\n\
             The demo still runs — Mercury transcribes and the whisper.cpp pane says so."
        );
    }

    let engines = Arc::new(Mutex::new(Engines {
        mercury: WhisperCandle::new(),
        whisper_cli,
        model,
    }));
    let speaker = Arc::new(Mutex::new(Speaker { piper: PiperCandle::new() }));
    // Both OCR lineages, constructed lazily — weights load on first read, so
    // starting the demo does not pay for a tab nobody opens.
    let reader = Arc::new(Mutex::new(Reader {
        crnn: CraftCrnn::new(),
        parseq: CraftCrnn::new_parseq(),
    }));

    // Warm Mercury before serving so the first click is not paying model load.
    // The same courtesy the bench harness extends to every implementation.
    eprintln!("loading Mercury (whisper-tiny-en) ...");
    if let Ok(g) = engines.lock() {
        let silence = ffai_core::types::AudioBuffer {
            samples: vec![0.0; 16000],
            sample_rate: 16000,
            channels: 1,
        };
        let _ = g.mercury.transcribe(&silence, &AsrOptions::default());
    }

    // Warm the voice too, for the same reason — and report the failure here
    // rather than inside the first click, since an unconverted voice is a
    // setup problem with a one-command fix.
    eprintln!("loading the voice (piper-vits-lessac-medium) ...");
    if let Ok(g) = speaker.lock() {
        if let Err(e) = g.piper.synthesize("Warm up.", &TtsOptions::default()) {
            eprintln!(
                "note: TTS unavailable — {e}\n      The Speak tab will show this message; \
                 the Listen tab is unaffected."
            );
        }
    }

    let listener = match TcpListener::bind(ADDR) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("cannot bind {ADDR}: {e}");
            std::process::exit(1);
        }
    };
    println!("\n  FFai side-by-side demo → http://{ADDR}\n");

    for stream in listener.incoming().flatten() {
        let engines = engines.clone();
        let speaker = speaker.clone();
        let reader = reader.clone();
        std::thread::spawn(move || {
            if let Err(e) = handle(stream, &engines, &speaker, &reader) {
                eprintln!("connection error: {e}");
            }
        });
    }
}

fn handle(
    mut stream: TcpStream,
    engines: &Arc<Mutex<Engines>>,
    speaker: &Arc<Mutex<Speaker>>,
    reader: &Arc<Mutex<Reader>>,
) -> std::io::Result<()> {
    // Read headers first: everything up to the blank line.
    let mut buf = Vec::new();
    let mut byte = [0u8; 1];
    while !buf.ends_with(b"\r\n\r\n") {
        if stream.read(&mut byte)? == 0 {
            return Ok(());
        }
        buf.push(byte[0]);
        if buf.len() > 16 * 1024 {
            return respond(&mut stream, 431, "text/plain", b"headers too large");
        }
    }
    let head = String::from_utf8_lossy(&buf).into_owned();
    let mut lines = head.lines();
    let request = lines.next().unwrap_or_default();
    let mut parts = request.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let path = parts.next().unwrap_or("/").to_string();
    let len: usize = head
        .lines()
        .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
        .and_then(|l| l.split(':').nth(1))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0);

    if len > MAX_BODY {
        return respond(&mut stream, 413, "text/plain", b"body too large");
    }
    let mut body = vec![0u8; len];
    if len > 0 {
        stream.read_exact(&mut body)?;
    }

    match (method.as_str(), path.as_str()) {
        ("POST", "/read") => {
            let json = read_image(&body, reader);
            respond(&mut stream, 200, "application/json", json.as_bytes())
        }
        ("POST", "/transcribe") => {
            let json = transcribe_both(&body, engines);
            respond(&mut stream, 200, "application/json", json.as_bytes())
        }
        ("POST", "/synthesize") => {
            let json = synthesize(&body, speaker);
            respond(&mut stream, 200, "application/json", json.as_bytes())
        }
        ("GET", p) => serve_static(&mut stream, p),
        _ => respond(&mut stream, 405, "text/plain", b"method not allowed"),
    }
}

/// Run both engines over the same bytes and report each one's text and wall
/// time. A failure in one is reported in its own pane rather than failing the
/// request — the whole point is to see them side by side, including when one
/// of them cannot answer.
fn transcribe_both(wav: &[u8], engines: &Arc<Mutex<Engines>>) -> String {
    if wav.len() < 44 {
        return json_error("posted body is not a WAV (too short)");
    }
    // Deleted on EVERY path, including the early returns below. Without this
    // the `fs::write` failure and the poisoned-lock paths both leak a WAV into
    // the temp directory — once per request, forever, on exactly the paths
    // that fire when something is already going wrong.
    //
    // FFAI_DEMO_KEEP_AUDIO retains the chunk instead, for the case where the
    // two panes disagree and the only way to find out why is to replay the
    // exact audio through both engines offline. Off by default: the demo
    // promises no storage, and a debug switch should have to be asked for.
    struct TempWav(PathBuf);
    impl Drop for TempWav {
        fn drop(&mut self) {
            if std::env::var_os("FFAI_DEMO_KEEP_AUDIO").is_some() {
                return;
            }
            std::fs::remove_file(&self.0).ok();
        }
    }

    let tmp = TempWav(std::env::temp_dir().join(format!(
        "ffai-demo-{}.wav",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    )));
    if let Err(e) = std::fs::write(&tmp.0, wav) {
        return json_error(&format!("could not stage audio: {e}"));
    }
    let tmp = &tmp.0;

    let guard = match engines.lock() {
        Ok(g) => g,
        Err(_) => return json_error("engine lock poisoned"),
    };

    // ---- Mercury, in process ----
    let t0 = Instant::now();
    let (mercury_text, mercury_err) = match ffai_media::load_audio(tmp) {
        // The full Mercury-X layer, live:
        //
        //   VAD          on by default — a sliding window over a microphone is
        //                mostly silence, and without segmentation every silent
        //                tick costs an encoder pass to produce nothing. One
        //                captured session ran 7 chunks for 2 lines of text.
        //   diarize      speaker turns per chunk.
        //   persist      the label survives into the NEXT chunk. Without it
        //                each call names its clusters afresh, so the same
        //                person is SPEAKER_00 one tick and SPEAKER_01 the
        //                next — measured at 53.58% DER against 5.68% with it.
        //                That is why diarization was not in this demo before.
        Ok(audio) => match guard.mercury.transcribe(
            &audio,
            &AsrOptions { diarize: true, persist_speakers: true, ..AsrOptions::default() },
        ) {
            Ok(t) => (label_speakers(&t), None),
            Err(e) => (String::new(), Some(e.to_string())),
        },
        Err(e) => (String::new(), Some(format!("decode failed: {e}"))),
    };
    let mercury_ms = t0.elapsed().as_secs_f64() * 1e3;

    // ---- whisper.cpp, as a subprocess on the SAME file ----
    let t1 = Instant::now();
    let (cpp_text, cpp_err) = match (&guard.whisper_cli, guard.model.exists()) {
        (Some(bin), true) => run_whisper_cpp(bin, &guard.model, tmp),
        _ => (
            String::new(),
            Some("whisper.cpp not installed (see docs/benchmarking.md)".to_string()),
        ),
    };
    let cpp_ms = t1.elapsed().as_secs_f64() * 1e3;
    drop(guard);

    // Correlating a disagreement to its audio needs all three on one line:
    // the file, and what each engine made of it.
    if std::env::var_os("FFAI_DEMO_KEEP_AUDIO").is_some() {
        eprintln!(
            "kept {}\n     mercury:     {mercury_text:?}\n     whisper.cpp: {cpp_text:?}",
            tmp.display()
        );
    }

    serde_json::json!({
        "mercury": { "text": mercury_text, "ms": mercury_ms, "error": mercury_err },
        "whispercpp": { "text": cpp_text, "ms": cpp_ms, "error": cpp_err },
    })
    .to_string()
}

/// Synthesize posted text and report everything worth *seeing* about it.
///
/// The response carries four things a waveform alone cannot show:
///
/// - **the phonemes** our clean-room G2P produced, per sentence — pronunciation
///   bugs are invisible in audio and obvious in IPA;
/// - **the sentence split**, because long-form input is synthesized per
///   sentence and joined, and where the cuts land is a real decision;
/// - **synthesis time and ×realtime**, measured around the engine call only;
/// - **a SHA-256 of the samples**, and on request a second synthesis of the
///   same input so the two hashes can be compared in the UI. That is the
///   determinism claim made checkable rather than asserted — piper samples
///   noise inside its ONNX graph and cannot reproduce its own output.
fn synthesize(body: &[u8], speaker: &Arc<Mutex<Speaker>>) -> String {
    let req: serde_json::Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => return json!({ "error": format!("bad request: {e}") }).to_string(),
    };
    let text = req.get("text").and_then(|t| t.as_str()).unwrap_or("").trim().to_string();
    if text.is_empty() {
        return json!({ "error": "nothing to say — type some text" }).to_string();
    }
    // Cap the utterance: this is a demo on a shared socket, and synthesis time
    // is linear in characters.
    const MAX_CHARS: usize = 2000;
    if text.chars().count() > MAX_CHARS {
        return json!({ "error": format!("text longer than {MAX_CHARS} characters") }).to_string();
    }

    let f32_opt = |k: &str| req.get(k).and_then(|v| v.as_f64()).map(|v| v as f32);
    let opts = TtsOptions {
        voice: None,
        speed: f32_opt("speed").unwrap_or(1.0).clamp(0.25, 4.0),
        // `null` means "the voice's own default" all the way down to the
        // engine, so the demo's knobs and the library's defaults cannot drift.
        noise_scale: f32_opt("noise_scale").map(|v| v.clamp(0.0, 2.0)),
        noise_w: f32_opt("noise_w").map(|v| v.clamp(0.0, 2.0)),
        seed: req.get("seed").and_then(|v| v.as_u64()).unwrap_or(0),
        sentence_silence_s: f32_opt("sentence_silence").unwrap_or(0.2).clamp(0.0, 2.0),
    };
    let verify = req.get("verify").and_then(|v| v.as_bool()).unwrap_or(false);

    let guard = match speaker.lock() {
        Ok(g) => g,
        Err(_) => return json!({ "error": "voice lock poisoned" }).to_string(),
    };

    let phonemes = guard.piper.phonemes(&text).unwrap_or_default();

    let t0 = Instant::now();
    let audio = match guard.piper.synthesize(&text, &opts) {
        Ok(a) => a,
        Err(e) => return json!({ "error": e.to_string() }).to_string(),
    };
    let synth_ms = t0.elapsed().as_secs_f64() * 1e3;

    // The determinism check runs the SAME call again and hashes both. Only on
    // request: it doubles the work, and a demo should not spend that silently.
    let repeat_hash = if verify {
        guard.piper.synthesize(&text, &opts).ok().map(|a| sha256_hex(samples_bytes(&a.samples)))
    } else {
        None
    };
    drop(guard);

    let hash = sha256_hex(samples_bytes(&audio.samples));
    let audio_secs = audio.duration_secs();
    let wav = wav_bytes(&audio);

    json!({
        "wav_base64": base64(&wav),
        "sample_rate": audio.sample_rate,
        "audio_secs": audio_secs,
        "synth_ms": synth_ms,
        "xrt": if synth_ms > 0.0 { audio_secs / (synth_ms / 1e3) } else { 0.0 },
        "sentences": ffai_mercury::tts::chunk::sentences(&text),
        "phonemes": phonemes,
        "sha256": hash,
        "sha256_repeat": repeat_hash,
        "deterministic": repeat_hash.map(|r| r == hash),
        "error": serde_json::Value::Null,
    })
    .to_string()
}

/// 16-bit PCM WAV, which is what every browser will play from a data URL.
fn wav_bytes(audio: &ffai_core::types::AudioBuffer) -> Vec<u8> {
    let n = audio.samples.len();
    let rate = audio.sample_rate;
    let mut out = Vec::with_capacity(44 + n * 2);
    let data_len = (n * 2) as u32;
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 2).to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &s in &audio.samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Hash the SAMPLES, not the WAV: the header carries a length that would make
/// two identical renderings hash alike for a trivial reason. The samples are
/// the engine's actual output, which is the thing being claimed stable.
fn samples_bytes(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 4);
    for s in samples {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    bytes
}

fn sha256_hex(bytes: Vec<u8>) -> String {
    // Reuse the hash the corpus manifests are pinned with, so "identical" here
    // means the same thing it means in the ledger.
    ffai_bench::corpus::file_sha256(&bytes)
}

/// Standard base64. Hand-rolled to keep the demo dependency-free — it is 20
/// lines and the alternative is a crate in the tree for one data URL.
fn base64(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((bytes.len() + 2) / 3 * 4);
    for chunk in bytes.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}

/// Matched settings to the benchmark harness: greedy (`-bs 1 -bo 1`), and
/// deliberately NOT `-nt`. That flag suppresses timestamp *generation*, not
/// just printing, and handing the reference 23 % less work is exactly the
/// defect the mission plan spent a milestone finding (§6.17).
fn run_whisper_cpp(bin: &Path, model: &Path, wav: &Path) -> (String, Option<String>) {
    let out = Command::new(bin)
        .args(["-m", &model.to_string_lossy(), "-t", "24", "-bs", "1", "-bo", "1", "-np"])
        .arg(wav)
        .output();
    match out {
        Ok(o) if o.status.success() => {
            let text = String::from_utf8_lossy(&o.stdout);
            // whisper-cli prints "[start --> end]  text" unless -np; with -np
            // it prints bare text lines. Strip any timestamp prefix that
            // survives so the two panes hold comparable text.
            let cleaned: String = text
                .lines()
                .map(|l| l.split_once(']').map(|(_, t)| t).unwrap_or(l).trim())
                .filter(|l| !l.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            (cleaned, None)
        }
        Ok(o) => (
            String::new(),
            Some(format!(
                "whisper-cli exited {}: {}",
                o.status,
                String::from_utf8_lossy(&o.stderr).lines().last().unwrap_or("")
            )),
        ),
        Err(e) => (String::new(), Some(format!("could not run whisper-cli: {e}"))),
    }
}

/// Prefix each line with whoever said it.
///
/// Speaker turns are a separate timeline from segments — one segment can span
/// a speaker change and one turn can cover several segments — so this matches
/// by the turn containing the segment's START rather than pretending the two
/// align. Where a turn cannot be found the text is emitted unlabelled rather
/// than guessed at.
fn label_speakers(t: &ffai_core::types::Transcript) -> String {
    let turns = match &t.speakers {
        Some(s) if !s.is_empty() => s,
        // Diarization off, or nothing found: unchanged behaviour.
        _ => return t.text().trim().to_string(),
    };
    let mut out = Vec::new();
    for seg in &t.segments {
        let text = seg.value.trim();
        if text.is_empty() {
            continue;
        }
        match turns.iter().find(|w| seg.start >= w.start - 0.25 && seg.start < w.end + 0.25) {
            Some(w) => out.push(format!("{}: {text}", w.value)),
            None => out.push(text.to_string()),
        }
    }
    out.join("
")
}

/// Read one image with BOTH OCR lineages and report each one's text, wall
/// time and line count, plus the content class the pipeline detected.
///
/// Same discipline as `transcribe_both`: identical bytes to both engines, a
/// failure in one reported in its own pane rather than failing the request.
/// The engines decode the PNG themselves from the same buffer, so neither
/// gets a head start on image decoding.
fn read_image(body: &[u8], reader: &Arc<Mutex<Reader>>) -> String {
    const PNG_MAGIC: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    if body.len() < 8 || body[..8] != PNG_MAGIC {
        return json_error("posted body is not a PNG (the demo decodes PNG only until the rff image decoders land)");
    }
    // ffai-media reads from a path, so the bytes land in a temp file that is
    // deleted on EVERY path below — the demo promises no storage.
    let path = std::env::temp_dir().join(format!("ffai-demo-{}.png", std::process::id()));
    if let Err(e) = std::fs::write(&path, body) {
        return json_error(&format!("could not stage the image: {e}"));
    }
    let img = match ffai_media::load_image(&path) {
        Ok(i) => i,
        Err(e) => {
            std::fs::remove_file(&path).ok();
            return json_error(&format!("could not decode the image: {e}"));
        }
    };
    std::fs::remove_file(&path).ok();

    let kind = match ffai_carmenta::content::classify(&img) {
        ffai_carmenta::content::ContentKind::Rendered => "rendered",
        ffai_carmenta::content::ContentKind::Photographic => "photographic",
    };
    let flatness = ffai_carmenta::content::flatness(&img);

    let guard = match reader.lock() {
        Ok(g) => g,
        Err(_) => return json_error("reader lock poisoned"),
    };
    let opts = OcrOptions::default();
    let run = |engine: &CraftCrnn| {
        let t0 = Instant::now();
        match engine.recognize(&img, &opts) {
            Ok(out) => {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                let lines: Vec<String> = out.lines().map(|l| l.text.clone()).collect();
                json!({ "text": out.text(), "lines": lines.len(), "ms": ms })
            }
            Err(e) => json!({ "error": e.to_string() }),
        }
    };
    let crnn = run(&guard.crnn);
    let parseq = run(&guard.parseq);
    json!({
        "width": img.width,
        "height": img.height,
        "content": kind,
        "flatness": flatness,
        "crnn": crnn,
        "parseq": parseq,
    })
    .to_string()
}

fn json_error(msg: &str) -> String {
    serde_json::json!({
        "mercury": { "text": "", "ms": 0.0, "error": msg },
        "whispercpp": { "text": "", "ms": 0.0, "error": msg },
    })
    .to_string()
}

/// Serve the built Dioxus app. `dx build` writes into a nested target dir; the
/// candidates below cover a release build and a plain `public/` copy.
fn serve_static(stream: &mut TcpStream, path: &str) -> std::io::Result<()> {
    let rel = if path == "/" { "index.html" } else { path.trim_start_matches('/') };
    // Refuse traversal before touching the filesystem.
    if rel.contains("..") {
        return respond(stream, 400, "text/plain", b"bad path");
    }
    let roots = [
        PathBuf::from("demo-ui/target/dx/demo-ui/release/web/public"),
        PathBuf::from("demo-ui/dist"),
        PathBuf::from("crates/ffai-demo/static"),
    ];
    for root in &roots {
        let candidate = root.join(rel);
        if let Ok(bytes) = std::fs::read(&candidate) {
            let ct = match candidate.extension().and_then(|e| e.to_str()) {
                Some("html") => "text/html; charset=utf-8",
                Some("js") => "text/javascript",
                Some("css") => "text/css",
                Some("wasm") => "application/wasm",
                Some("json") => "application/json",
                _ => "application/octet-stream",
            };
            return respond(stream, 200, ct, &bytes);
        }
    }
    respond(
        stream,
        404,
        "text/html; charset=utf-8",
        b"<h1>UI not built</h1><p>Run <code>cd demo-ui &amp;&amp; dx build --platform web --release</code>, then reload.</p>",
    )
}

fn respond(
    stream: &mut TcpStream,
    code: u16,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    let reason = match code {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        431 => "Request Header Fields Too Large",
        _ => "Error",
    };
    let head = format!(
        "HTTP/1.1 {code} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\
         Cache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}
