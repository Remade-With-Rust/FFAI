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

use ffai_core::engine::{AsrEngine, AsrOptions};
use ffai_mercury::asr::WhisperCandle;

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
        std::thread::spawn(move || {
            if let Err(e) = handle(stream, &engines) {
                eprintln!("connection error: {e}");
            }
        });
    }
}

fn handle(mut stream: TcpStream, engines: &Arc<Mutex<Engines>>) -> std::io::Result<()> {
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
        ("POST", "/transcribe") => {
            let json = transcribe_both(&body, engines);
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
        // VAD comes on by default now, which is what this demo wanted anyway:
        // a sliding window over a live microphone is mostly silence, and
        // without segmentation every silent tick costs a full encoder pass to
        // produce nothing. One captured session ran 7 chunks for 2 lines.
        Ok(audio) => match guard.mercury.transcribe(&audio, &AsrOptions::default()) {
            Ok(t) => (t.text().trim().to_string(), None),
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
