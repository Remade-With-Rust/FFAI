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

use ffai_argus::SmolVlm;
use ffai_carmenta::engine::CraftCrnn;
use ffai_core::engine::{
    AsrEngine, AsrOptions, Decoding, OcrEngine, OcrOptions, TtsEngine, TtsOptions, VlmOptions,
};
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

/// Argus, and the reference it is measured against.
///
/// Behind its own lock for the same reason `Speaker` and `Reader` are, and
/// more so: it is the heaviest pair here — ~1 GB of `f32` safetensors on our
/// side, another copy inside the Python worker — so a `See` click must not be
/// able to stall a live transcription and vice versa.
struct Seer {
    argus: SmolVlm,
    /// `PyTorch` + `transformers` on the SAME checkpoint, held open.
    ///
    /// This is the demo's equivalent of `whisper-cli`, and it is the same arm
    /// the ledger's quality gate uses — `corpora/refs/smolvlm_hf_ref.py`, the
    /// file that pins greedy / 64 tokens / float32 / seed 0. Pointing the demo
    /// at the reference the benchmark already uses is what stops the demo and
    /// the bench quietly disagreeing about what "the reference" means.
    ///
    /// **Held open, not spawned per click.** Each spawn reloads a gigabyte of
    /// weights; a per-click process would put ~15 s of model load inside every
    /// reference reading and make `PyTorch` look absurdly slow for a reason that
    /// has nothing to do with `PyTorch`. The demo already warms Mercury before
    /// serving so the first click is not paying load — this extends the same
    /// courtesy to the side being compared against, which is the only way the
    /// two numbers mean the same thing.
    reference: Option<RefWorker>,
}

/// A live `smolvlm_hf_ref.py --serve`: one request line in, one result line out.
struct RefWorker {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    stdout: std::io::BufReader<std::process::ChildStdout>,
    load_secs: f64,
}

impl RefWorker {
    /// Start the worker and wait for its `{"load_secs":...}` handshake.
    ///
    /// Returning `None` rather than erroring: an absent Python stack is a
    /// setup state, not a failure — the demo runs, and the reference pane says
    /// what is missing, exactly as the whisper.cpp pane does.
    fn start() -> Option<Self> {
        let py = [".venv-argus/Scripts/python.exe", ".venv-argus/bin/python"]
            .iter()
            .map(PathBuf::from)
            .find(|p| p.exists())?;
        let script = PathBuf::from("corpora/refs/smolvlm_hf_ref.py");
        if !script.exists() {
            return None;
        }
        let mut child = Command::new(py)
            .args([
                script.to_str()?,
                "--serve",
                "--model",
                "HuggingFaceTB/SmolVLM-256M-Instruct",
                "--dtype",
                "float32",
                "--seed",
                "0",
            ])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .ok()?;
        let stdin = child.stdin.take()?;
        let mut stdout = std::io::BufReader::new(child.stdout.take()?);
        let mut line = String::new();
        // The handshake doubles as the readiness signal: it does not arrive
        // until the weights are resident, so a request sent after it is warm.
        if std::io::BufRead::read_line(&mut stdout, &mut line).ok()? == 0 {
            return None;
        }
        let v: serde_json::Value = serde_json::from_str(line.trim()).ok()?;
        Some(Self {
            child,
            stdin,
            stdout,
            load_secs: v["load_secs"].as_f64().unwrap_or(0.0),
        })
    }

    /// One caption. `Err` is a message for the pane, never a panic.
    fn caption(
        &mut self,
        path: &Path,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<(String, f64), String> {
        let req = json!({
            "path": path.to_string_lossy(),
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
        });
        writeln!(self.stdin, "{req}").map_err(|e| format!("reference worker gone: {e}"))?;
        self.stdin
            .flush()
            .map_err(|e| format!("reference worker gone: {e}"))?;
        let mut line = String::new();
        match std::io::BufRead::read_line(&mut self.stdout, &mut line) {
            Ok(0) => return Err("reference worker exited".into()),
            Ok(_) => {}
            Err(e) => return Err(format!("reference worker read failed: {e}")),
        }
        let v: serde_json::Value =
            serde_json::from_str(line.trim()).map_err(|e| format!("reference said: {e}"))?;
        if let Some(e) = v["error"].as_str() {
            return Err(e.to_string());
        }
        Ok((
            v["text"].as_str().unwrap_or_default().to_string(),
            v["secs"].as_f64().unwrap_or(0.0) * 1e3,
        ))
    }
}

impl Drop for RefWorker {
    fn drop(&mut self) {
        // Close stdin first: the worker's loop ends on EOF, so this is a clean
        // shutdown. Kill only if it does not take the hint.
        let _ = self.stdin.write_all(b"\n");
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
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
    let speaker = Arc::new(Mutex::new(Speaker {
        piper: PiperCandle::new(),
    }));
    // Both OCR lineages, constructed lazily — weights load on first read, so
    // starting the demo does not pay for a tab nobody opens.
    let reader = Arc::new(Mutex::new(Reader {
        crnn: CraftCrnn::new(),
        parseq: CraftCrnn::new_parseq(),
    }));
    let seer = Arc::new(Mutex::new(Seer {
        argus: SmolVlm::with_manifest_dir(PathBuf::from("models")),
        reference: None,
    }));

    // Argus warms in the BACKGROUND, unlike the engines above.
    //
    // It is ~1 GB of weights, so warming it in line would hold the demo off
    // the network for twenty seconds for the sake of a tab nobody may open.
    // Warming it lazily instead would put that twenty seconds inside the first
    // click — in a tab whose entire subject is latency, which would be the
    // worst place in the program to hide it.
    //
    // So: serve immediately, load behind. A click that lands early still gets
    // an honest number, because the response reports model load as its own
    // line rather than folding it into the caption time.
    {
        let seer = seer.clone();
        std::thread::spawn(move || {
            eprintln!("loading Argus + the PyTorch reference (smolvlm-256m) in the background ...");
            let t = Instant::now();
            // Started OUTSIDE the lock: the reference takes ~15 s to load a
            // gigabyte of weights, and holding the mutex through that would
            // block the first See click on work it does not need yet.
            let reference = RefWorker::start();
            match &reference {
                Some(r) => eprintln!(
                    "  reference (PyTorch/transformers) ready, weights loaded in {:.1}s",
                    r.load_secs
                ),
                None => eprintln!(
                    "note: PyTorch reference unavailable (needs .venv-argus with torch, \
                     transformers, pillow).\n      The See tab still runs Argus and says \
                     the reference pane is absent."
                ),
            }
            let msg = match seer.lock() {
                Ok(mut g) => {
                    g.reference = reference;
                    match g.argus.warm() {
                        Ok(()) => format!("Argus ready in {:.1}s", t.elapsed().as_secs_f64()),
                        Err(e) => format!(
                            "note: Argus unavailable - {e}\n      The See tab will show this; \
                             the other tabs are unaffected."
                        ),
                    }
                }
                Err(_) => "note: Argus lock poisoned during warm-up".to_string(),
            };
            eprintln!("{msg}");
        });
    }

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
    if let Ok(g) = speaker.lock()
        && let Err(e) = g.piper.synthesize("Warm up.", &TtsOptions::default())
    {
        eprintln!(
            "note: TTS unavailable — {e}\n      The Speak tab will show this message; \
                 the Listen tab is unaffected."
        );
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
        let seer = seer.clone();
        std::thread::spawn(move || {
            if let Err(e) = handle(stream, &engines, &speaker, &reader, &seer) {
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
    seer: &Arc<Mutex<Seer>>,
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
        ("POST", p) if p.starts_with("/describe") => {
            let json = describe_image(&body, p, seer);
            respond(&mut stream, 200, "application/json", json.as_bytes())
        }
        ("POST", "/read") => {
            let json = read_image(&body, reader);
            respond(&mut stream, 200, "application/json", json.as_bytes())
        }
        ("POST", p) if p.starts_with("/transcribe") => {
            // `?diarize=0` turns the speaker layer off. It is a query rather
            // than a body field because the body is raw WAV bytes.
            //
            // This exists because the two panes were not doing the same work
            // and the UI invited the wrong conclusion: Mercury ran
            // diarize+persist while whisper.cpp ran plain transcription, so
            // a per-chunk average read "Mercury 2x slower" when the ASR-only
            // paths are 107 ms against whisper.cpp's 274 ms. Measured, the
            // speaker layer is +621 ms on a 3 s chunk — 6.8x the ASR path.
            let diarize = !p.contains("diarize=0");
            // The browser reports where its trailing window starts; absent, 0.0
            // simply disables the grid alignment rather than misplacing it.
            let offset_secs = p
                .split_once("offset=")
                .and_then(|(_, v)| v.split('&').next())
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(0.0);
            let json = transcribe_both(&body, engines, diarize, offset_secs);
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
fn transcribe_both(
    wav: &[u8],
    engines: &Arc<Mutex<Engines>>,
    diarize: bool,
    offset_secs: f64,
) -> String {
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
            .map_or(0, |d| d.as_nanos())
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
    let (mercury_text, mercury_turns, mercury_err) = match ffai_media::load_audio(tmp) {
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
        //
        // `diarize` is a per-request toggle (`?diarize=0`) so the panes can
        // be put on EQUAL work: whisper.cpp does not diarize at all, and
        // with the speaker layer on, Mercury is running a whole second
        // network (ECAPA-TDNN, ~4 forwards per 3 s chunk at 1.5 s windows /
        // 0.75 s hop) that the other pane never runs.
        Ok(audio) => match guard.mercury.transcribe(
            &audio,
            &AsrOptions {
                diarize,
                persist_speakers: diarize,
                // Where this chunk sits in the session. The browser sends a
                // SLIDING trailing window, so without this the diarizer's
                // window grid is anchored to the buffer and moves with it —
                // identical audio gets re-cut at new offsets every tick and
                // every speaker embedding is recomputed. Supplying it took
                // the live pattern from 1.87x to 3.16x.
                stream_offset_secs: offset_secs,
                ..AsrOptions::default()
            },
        ) {
            Ok(t) => {
                let us = utterances(&t, offset_secs);
                (label_speakers(&us), turns_json(&us), None)
            }
            Err(e) => (String::new(), Vec::new(), Some(e.to_string())),
        },
        Err(e) => (
            String::new(),
            Vec::new(),
            Some(format!("decode failed: {e}")),
        ),
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

    // `diarize` rides back so the UI can label what was actually measured.
    // A timing whose configuration is not on screen beside it is how the
    // "2x slower" reading happened in the first place.
    serde_json::json!({
        "mercury": {
            "text": mercury_text,
            "ms": mercury_ms,
            "error": mercury_err,
            // One entry per segment, in session time. The pane lays these out;
            // `text` is the same content flattened, kept for a client that
            // does not.
            "turns": mercury_turns,
        },
        "whispercpp": { "text": cpp_text, "ms": cpp_ms, "error": cpp_err },
        "diarize": diarize,
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
    let text = req
        .get("text")
        .and_then(|t| t.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if text.is_empty() {
        return json!({ "error": "nothing to say — type some text" }).to_string();
    }
    // Cap the utterance: this is a demo on a shared socket, and synthesis time
    // is linear in characters.
    const MAX_CHARS: usize = 2000;
    if text.chars().count() > MAX_CHARS {
        return json!({ "error": format!("text longer than {MAX_CHARS} characters") }).to_string();
    }

    // JSON numbers are f64; TtsOptions takes f32. Narrowing a speed or pitch
    // knob costs precision no listener can hear, and the value is clamped by the
    // synthesiser anyway.
    #[allow(clippy::cast_possible_truncation)]
    let f32_opt = |k: &str| {
        req.get(k)
            .and_then(serde_json::Value::as_f64)
            .map(|v| v as f32)
    };
    let opts = TtsOptions {
        voice: None,
        speed: f32_opt("speed").unwrap_or(1.0).clamp(0.25, 4.0),
        // `null` means "the voice's own default" all the way down to the
        // engine, so the demo's knobs and the library's defaults cannot drift.
        noise_scale: f32_opt("noise_scale").map(|v| v.clamp(0.0, 2.0)),
        noise_w: f32_opt("noise_w").map(|v| v.clamp(0.0, 2.0)),
        seed: req
            .get("seed")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0),
        sentence_silence_s: f32_opt("sentence_silence").unwrap_or(0.2).clamp(0.0, 2.0),
    };
    let verify = req
        .get("verify")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);

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
        guard
            .piper
            .synthesize(&text, &opts)
            .ok()
            .map(|a| sha256_hex(samples_bytes(&a.samples)))
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
    // WAV headers carry a u32 byte count, so this narrowing is the format's,
    // not ours. The `as i16` below is preceded by `.clamp(-1.0, 1.0)`, which is
    // what makes it exact.
    #[allow(clippy::cast_possible_truncation)]
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
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            T[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            T[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// Matched settings to the benchmark harness: greedy (`-bs 1 -bo 1`), and
/// deliberately NOT `-nt`. That flag suppresses timestamp *generation*, not
/// just printing, and handing the reference 23 % less work is exactly the
/// defect the mission plan spent a milestone finding (§6.17).
fn run_whisper_cpp(bin: &Path, model: &Path, wav: &Path) -> (String, Option<String>) {
    let out = Command::new(bin)
        .args([
            "-m",
            &model.to_string_lossy(),
            "-t",
            "24",
            "-bs",
            "1",
            "-bo",
            "1",
            "-np",
        ])
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
                .map(|l| l.split_once(']').map_or(l, |(_, t)| t).trim())
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
                String::from_utf8_lossy(&o.stderr)
                    .lines()
                    .last()
                    .unwrap_or("")
            )),
        ),
        Err(e) => (
            String::new(),
            Some(format!("could not run whisper-cli: {e}")),
        ),
    }
}

/// One segment of transcript with whoever said it and where it sits in the
/// session.
struct Utterance {
    /// `None` when diarization is off, or when no turn covers this segment.
    speaker: Option<String>,
    text: String,
    /// Seconds from the START OF THE SESSION, not from this chunk.
    start: f64,
    end: f64,
}

/// Pair each segment with whoever said it, on the session clock.
///
/// Speaker turns are a separate timeline from segments — one segment can span
/// a speaker change and one turn can cover several segments — so this matches
/// by the turn containing the segment's START rather than pretending the two
/// align. Where a turn cannot be found the segment is emitted UNATTRIBUTED
/// rather than guessed at.
///
/// `offset_secs` is where this chunk begins in the session. The browser posts
/// a SLIDING trailing window, so a time measured from the buffer moves every
/// tick and cannot be laid out; adding the offset here is what makes the
/// numbers comparable between chunks.
fn utterances(t: &ffai_core::types::Transcript, offset_secs: f64) -> Vec<Utterance> {
    let turns: &[ffai_core::types::TimedSegment<String>] = match &t.speakers {
        Some(s) if !s.is_empty() => s,
        // Diarization off, or nothing found: segments, unattributed.
        _ => &[],
    };
    t.segments
        .iter()
        .filter(|seg| !seg.value.trim().is_empty())
        .map(|seg| Utterance {
            speaker: turns
                .iter()
                .find(|w| seg.start >= w.start - 0.25 && seg.start < w.end + 0.25)
                .map(|w| w.value.clone()),
            text: seg.value.trim().to_string(),
            start: offset_secs + seg.start,
            end: offset_secs + seg.end,
        })
        .collect()
}

/// The same utterances as JSON, for a UI that wants to LAY THEM OUT rather
/// than print them.
///
/// The flat `text` field keeps its `SPEAKER_00: ...` prefixes for logs and for
/// a client that only wants a string, but a prefix is a lossy encoding: it
/// throws away the timing, and it is indistinguishable from someone actually
/// saying "speaker zero zero". The structured field is what the panes render.
fn turns_json(us: &[Utterance]) -> Vec<serde_json::Value> {
    us.iter()
        .map(|u| {
            json!({
                "speaker": u.speaker,
                "text": u.text,
                "start": u.start,
                "end": u.end,
            })
        })
        .collect()
}

/// Prefix each line with whoever said it — the string form, for logs and for
/// the UI's fallback when a response carries no structured turns.
fn label_speakers(us: &[Utterance]) -> String {
    us.iter()
        .map(|u| match &u.speaker {
            Some(who) => format!("{who}: {}", u.text),
            None => u.text.clone(),
        })
        .collect::<Vec<_>>()
        .join("\n")
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
        return json_error(
            "posted body is not a PNG (the demo decodes PNG only until the rff image decoders land)",
        );
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

/// Percent-decode one query value.
///
/// Written out because the prompt arrives in the query string and prompts
/// contain spaces, punctuation and `?`. `+` is a space here — the browser's
/// `URLSearchParams` encodes it that way — and a malformed `%` escape is left
/// literal rather than dropped, because a mangled prompt the user can SEE
/// beats a silently shortened one.
fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < b.len() => {
                let hex = std::str::from_utf8(&b[i + 1..i + 3]).ok();
                match hex.and_then(|h| u8::from_str_radix(h, 16).ok()) {
                    Some(v) => {
                        out.push(v);
                        i += 3;
                    }
                    None => {
                        out.push(b[i]);
                        i += 1;
                    }
                }
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn query<'a>(path: &'a str, key: &str) -> Option<&'a str> {
    let (_, q) = path.split_once('?')?;
    q.split('&')
        .filter_map(|kv| kv.split_once('='))
        .find(|(k, _)| *k == key)
        .map(|(_, v)| v)
}

/// Argus: one image in, a caption out, and the stage-by-stage cost of getting
/// there.
///
/// **It races the same reference the benchmark does.** Listen puts Mercury
/// beside `whisper-cli`; this puts Argus beside `PyTorch` + `transformers`
/// running the identical `SmolVLM-256M-Instruct` checkpoint, through
/// `corpora/refs/smolvlm_hf_ref.py` — the file that pins the decode config the
/// ledger's quality gate is measured under. Both sides get the SAME staged
/// file, for the same reason both ASR engines do: it removes the question of
/// whether they actually saw identical input.
///
/// What is NOT claimed here is a quality verdict. One image is an anecdote;
/// the measured result is 49/50 answers byte-identical on a pinned corpus, and
/// that number belongs to the ledger, not to whatever the user just pasted.
fn describe_image(body: &[u8], path: &str, seer: &Arc<Mutex<Seer>>) -> String {
    let prompt = query(path, "prompt")
        .map(percent_decode)
        .filter(|p| !p.trim().is_empty())
        .unwrap_or_else(|| "What is written in this image?".to_string());
    let max_new_tokens = query(path, "max")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(64)
        .clamp(1, 512);
    let seed = query(path, "seed").and_then(|v| v.parse::<u64>().ok());
    let temperature = query(path, "temp")
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.7);

    // `load_image` reads a path, so the bytes land in a temp file that is
    // removed on EVERY path below — the demo promises no storage.
    let staged = std::env::temp_dir().join(format!(
        "ffai-demo-see-{}.bin",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos())
    ));
    if let Err(e) = std::fs::write(&staged, body) {
        return json!({ "error": format!("could not stage the image: {e}") }).to_string();
    }
    let t_decode = Instant::now();
    let img = match ffai_media::load_image(&staged) {
        Ok(i) => i,
        Err(e) => {
            std::fs::remove_file(&staged).ok();
            return json!({ "error": format!("could not decode the image: {e}") }).to_string();
        }
    };
    let decode_ms = t_decode.elapsed().as_secs_f64() * 1e3;
    // NOT removed yet: the reference reads the same staged file from its own
    // process, which is the whole point — identical bytes, not the same bytes
    // decoded twice. Removed on every path below.
    struct Staged(PathBuf);
    impl Drop for Staged {
        fn drop(&mut self) {
            std::fs::remove_file(&self.0).ok();
        }
    }
    let staged = Staged(staged);
    let staged = &staged.0;

    let mut guard = match seer.lock() {
        Ok(g) => g,
        Err(_) => return json!({ "error": "seer lock poisoned" }).to_string(),
    };

    // Whether the weights were resident BEFORE this call decides whether the
    // wall time below is a warm number or a cold one. Reported either way; a
    // latency demo that quietly includes a one-off 20 s load in its first
    // reading is telling the reader something false about every later one.
    let was_loaded = guard.argus.is_loaded();
    let t_load = Instant::now();
    if !was_loaded && guard.argus.warm().is_err() {
        // Fall through: the describe call below reports the real error.
    }
    let load_ms = if was_loaded {
        0.0
    } else {
        t_load.elapsed().as_secs_f64() * 1e3
    };

    let opts = VlmOptions {
        prompt: Some(prompt.clone()),
        max_new_tokens: Some(max_new_tokens),
        decoding: match seed {
            Some(seed) => Decoding::Sampled {
                temperature,
                top_p: None,
                top_k: None,
                seed,
            },
            None => Decoding::Greedy,
        },
        ..VlmOptions::default()
    };

    let t_wall = Instant::now();
    let (caption, tr) = match guard.argus.describe_image_traced(&img, &opts) {
        Ok(v) => v,
        Err(e) => return json!({ "error": e.to_string() }).to_string(),
    };
    let wall_ms = t_wall.elapsed().as_secs_f64() * 1e3;

    // ---- the reference arm, on the same file ----
    let reference = match guard.reference.as_mut() {
        Some(w) => {
            let load = w.load_secs * 1e3;
            match w.caption(staged, &prompt, max_new_tokens) {
                Ok((text, ms)) => json!({
                    "text": text, "ms": ms, "load_ms": load,
                    "engine": "PyTorch + transformers", "config": "greedy-64 / float32 / seed 0",
                }),
                Err(e) => json!({ "error": e, "load_ms": load }),
            }
        }
        None => json!({
            "absent": "PyTorch reference not running - needs .venv-argus with torch, \
                       transformers and pillow, plus corpora/refs/smolvlm_hf_ref.py",
        }),
    };

    // The stage list is ORDERED because the UI draws it as a timeline, and a
    // timeline whose segments are not in execution order is a lie told with a
    // chart.
    let stages = json!([
        { "name": "decode",      "ms": decode_ms,       "what": "PNG/JPEG -> RGB8, rusty_png / rusty_jpeg" },
        { "name": "preprocess",  "ms": tr.preprocess_ms, "what": "two Lanczos resizes, tile cut, normalise" },
        { "name": "vision",      "ms": tr.tower_ms,      "what": format!("SigLIP + connector, {}x", tr.tiles) },
        { "name": "assemble",    "ms": tr.assemble_ms,   "what": "chat template, tokenize, embed, splice" },
        { "name": "prefill",     "ms": tr.prefill_ms,    "what": format!("one pass over {} tokens", tr.prompt_tokens) },
        { "name": "generate",    "ms": tr.decode_ms(),   "what": format!("{} tokens, one pass each", tr.step_ms.len()) },
        { "name": "detokenize",  "ms": tr.detokenize_ms, "what": "ids -> text" },
    ]);

    json!({
        "caption": caption,
        "prompt": prompt,
        "greedy": seed.is_none(),
        "image": {
            "width": img.width,
            "height": img.height,
            "resized": tr.resized_to.first().map_or_else(
                || json!(null), |(w, h)| json!({ "w": w, "h": h })),
        },
        "grid": { "rows": tr.rows, "cols": tr.cols, "tiles": tr.tiles, "tile": tr.tile },
        "tokens": {
            "image": tr.image_tokens,
            "text": tr.text_tokens,
            "prompt": tr.prompt_tokens,
            "generated": tr.step_ms.len(),
            "per_tile": tr.image_tokens.checked_div(tr.tiles).unwrap_or(0),
            "max_positions": tr.max_positions,
        },
        "stages": stages,
        "tower_per_tile_ms": tr.tower_per_tile_ms,
        "step_ms": tr.step_ms,
        "tokens_per_sec": tr.tokens_per_sec(),
        "engine_ms": tr.total_ms(),
        "wall_ms": wall_ms,
        "reference": reference,
        "decode_ms": decode_ms,
        "load_ms": load_ms,
        "cold": !was_loaded,
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
    let rel = if path == "/" {
        "index.html"
    } else {
        path.trim_start_matches('/')
    };
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
#[cfg(test)]
mod tests {
    use super::{label_speakers, turns_json, utterances};
    use ffai_core::types::{TimedSegment, Transcript};

    fn seg(start: f64, end: f64, value: &str) -> TimedSegment<String> {
        TimedSegment {
            start,
            end,
            value: value.to_string(),
            confidence: None,
        }
    }

    fn transcript(
        segments: Vec<TimedSegment<String>>,
        speakers: Option<Vec<TimedSegment<String>>>,
    ) -> Transcript {
        Transcript {
            language: None,
            segments,
            words: None,
            speakers,
        }
    }

    /// A turn covering a segment's START owns it, even where the segment runs
    /// past the turn's end — the two timelines do not align, and the pane
    /// draws whichever answer this gives.
    #[test]
    fn segment_takes_the_turn_that_covers_its_start() {
        let t = transcript(
            vec![seg(0.0, 2.5, "hello there"), seg(2.6, 4.0, "hi")],
            Some(vec![
                seg(0.0, 2.0, "SPEAKER_00"),
                seg(2.5, 4.0, "SPEAKER_01"),
            ]),
        );
        let us = utterances(&t, 0.0);
        assert_eq!(us[0].speaker.as_deref(), Some("SPEAKER_00"));
        assert_eq!(us[1].speaker.as_deref(), Some("SPEAKER_01"));
    }

    /// The whole point of threading the offset through: the browser posts a
    /// sliding window, so a time measured from the buffer moves under the
    /// same audio every tick and cannot be laid out against anything.
    #[test]
    fn times_come_back_on_the_session_clock() {
        let t = transcript(vec![seg(1.0, 2.0, "later")], None);
        let us = utterances(&t, 30.0);
        assert!((us[0].start - 31.0).abs() < 1e-9);
        assert!((us[0].end - 32.0).abs() < 1e-9);
    }

    /// A segment no turn covers is emitted unattributed rather than given to
    /// whoever spoke nearest.
    #[test]
    fn an_uncovered_segment_stays_unattributed() {
        let t = transcript(
            vec![seg(9.0, 9.5, "who said that")],
            Some(vec![seg(0.0, 2.0, "SPEAKER_00")]),
        );
        let us = utterances(&t, 0.0);
        assert_eq!(us[0].speaker, None);
        assert_eq!(label_speakers(&us), "who said that");
    }

    /// Speakers off: segments, in order, exactly as before diarization
    /// existed. Empty segments are dropped rather than rendered as blank rows.
    #[test]
    fn no_speakers_is_the_plain_transcript() {
        let t = transcript(
            vec![
                seg(0.0, 1.0, "one"),
                seg(1.0, 2.0, "  "),
                seg(2.0, 3.0, "two"),
            ],
            None,
        );
        let us = utterances(&t, 0.0);
        assert_eq!(us.len(), 2);
        assert_eq!(label_speakers(&us), "one\ntwo");
    }

    /// What the pane actually parses. `speaker` is null, not absent, when
    /// nobody owns the segment — an absent field and a null one read the same
    /// in the UI, but only one of them is a promise.
    #[test]
    fn json_carries_speaker_text_and_both_times() {
        let t = transcript(
            vec![seg(0.0, 1.0, "mine"), seg(5.0, 6.0, "nobody's")],
            Some(vec![seg(0.0, 1.0, "SPEAKER_03")]),
        );
        let j = turns_json(&utterances(&t, 10.0));
        assert_eq!(j[0]["speaker"], "SPEAKER_03");
        assert_eq!(j[0]["text"], "mine");
        assert_eq!(j[0]["start"], 10.0);
        assert_eq!(j[0]["end"], 11.0);
        assert!(j[1]["speaker"].is_null());
    }
}
