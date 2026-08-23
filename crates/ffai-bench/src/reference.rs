//! World-standard reference adapters — the "oracle" seat from Prometheus's
//! trial stage (`prom-trial::oracle`), generalized to external AI tools.
//!
//! # Why batch mode is the primary contract
//!
//! The naive design — invoke the reference once per clip and time it — is
//! *wrong* for AI tooling, and wrong in the direction that flatters us. A
//! Python reference spends seconds on interpreter startup and model load;
//! transcribing a 5-second clip takes a fraction of that. Timing
//! per-invocation would measure Python's startup, report our Rust as
//! spectacularly faster, and the claim would be indefensible.
//!
//! So an ASR reference is invoked **once for the whole corpus** with a file
//! list, and reports per-clip transcription time itself. That yields two
//! honest numbers, both recorded:
//!
//! - **warm RTF** — steady-state throughput, model already loaded. This is
//!   what implementations publish, and what a server-side user experiences.
//! - **end-to-end RTF** — total wall clock for the batch, including the one
//!   model load, amortized over the corpus. This is what a CLI user
//!   experiences.
//!
//! Neither alone is the truth; quoting only the flattering one is how
//! benchmarks lie. Single-file mode (`command`) remains for simple tools
//! (tesseract) where startup is negligible.

use ffai_core::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;

/// One external reference implementation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferenceSpec {
    /// Display name, e.g. "faster-whisper-tiny".
    pub name: String,
    /// Task this reference applies to: "asr", "tts", "ocr", "vlm".
    pub task: String,
    /// **Preferred.** Argv where `{filelist}` is replaced by a temp file
    /// holding one audio path per line. The adapter must emit JSONL on
    /// stdout: `{"path": ..., "text": ..., "transcribe_secs": ...}` per clip,
    /// plus an optional `{"load_secs": ...}` line.
    pub batch_command: Option<Vec<String>>,
    /// Fallback: argv where `{input}` is replaced by one media path; stdout
    /// is taken as the output text. Startup cost lands in the timing.
    pub command: Option<Vec<String>>,
    /// Optional argv printing a version string (recorded in the ledger).
    pub version_command: Option<Vec<String>>,
    /// What this reference is *configured as* — e.g. `"tiny.en/greedy"`.
    ///
    /// **Declared, never inferred.** The quality gate needs to know which
    /// references are comparable to the engine, and deriving that by looking
    /// for "tiny" or "greedy" in `name` would make a benchmark's meaning
    /// depend on a naming convention nobody is enforcing.
    ///
    /// Without it the gate answers only "is our output as good as the best
    /// ASR available?", by picking the lowest WER of everything that ran —
    /// which for a 39M greedy engine meant being failed against
    /// `openai-whisper-base`, a 74M beam-search model. That is a real
    /// question, but it is not "is our implementation good", and the two were
    /// being reported under one label. References that leave this unset are
    /// scored in the open comparison only.
    #[serde(default)]
    pub config: Option<String>,
}

/// One clip's result from a batch run.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipResult {
    pub path: String,
    pub text: String,
    /// Transcription-only seconds as reported by the adapter (excludes model
    /// load). `None` if the adapter didn't report it.
    pub transcribe_secs: Option<f64>,
}

/// The parsed output of one batch invocation.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BatchResult {
    pub clips: Vec<ClipResult>,
    /// Model load seconds as reported by the adapter.
    pub load_secs: Option<f64>,
    /// Total processing seconds for the whole batch, for tools that report
    /// only an aggregate (whisper.cpp prints one timing block per *run*, not
    /// per file). Used when per-clip timings are absent — reporting the
    /// aggregate honestly beats splitting it into per-clip numbers the tool
    /// never measured.
    pub batch_transcribe_secs: Option<f64>,
    /// Median resident memory across the run — what the tree sits at while
    /// working, as opposed to its worst instant.
    pub steady_bytes: Option<u64>,
    /// Peak working set of the reference's own process, in bytes.
    ///
    /// Measured after `wait()` while the `Child` still owns its handle — see
    /// [`crate::footprint`]. `None` where the platform has no implementation,
    /// so the footprint gate can skip honestly rather than invent a number.
    pub peak_bytes: Option<u64>,
}

impl BatchResult {
    /// Adapter-reported processing time for the batch: the sum of per-clip
    /// timings when available, otherwise the reported aggregate.
    #[must_use]
    pub fn transcribe_secs(&self) -> Option<f64> {
        let sum: f64 = self.clips.iter().filter_map(|c| c.transcribe_secs).sum();
        if sum > 0.0 {
            Some(sum)
        } else {
            self.batch_transcribe_secs
        }
    }

    /// Look up a clip's text by the path the adapter echoed back.
    #[must_use]
    pub fn text_for(&self, path: &Path) -> Option<&str> {
        let want = path.to_string_lossy().replace('\\', "/");
        self.clips
            .iter()
            .find(|c| c.path.replace('\\', "/") == want)
            .map(|c| c.text.as_str())
    }
}

/// A named, executable scorer — declared HERE and not in a corpus.
///
/// # Why this lives in the references file
///
/// `references.toml` has always been executable input: every entry names an
/// argv this crate spawns, and it is read and reviewed as such. A
/// `corpora/*.toml` was pure data.
///
/// When VLM scoring landed, the corpus grew a `[scorer]` block carrying an
/// argv, and that moved the trust boundary: a file shaped like data could
/// suddenly run code. The risk was never that execution is *possible* — it is
/// that execution became **invisible**, buried on line 20 of a thousand lines
/// of hashes and prompts that a reviewer will skim.
///
/// So the argv moved back here, and the corpus now merely *selects* a scorer
/// by name. **Data selects from a set; code defines the set.** The corpus
/// keeps `metric` and `scale`, which are facts about the benchmark's numbers
/// rather than instructions to execute anything.
#[derive(Debug, Clone, Deserialize)]
pub struct NamedScorer {
    /// The name a corpus refers to.
    pub name: String,
    /// Argv; `{predictions}` is replaced with the predictions JSONL path.
    pub command: Vec<String>,
    /// Optional argv printing a version string, recorded in the ledger.
    #[serde(default)]
    pub version_command: Option<Vec<String>>,
}

impl NamedScorer {
    /// The argv as a single line, for the ledger and for printing before the
    /// spawn — an executed command should be visible, not merely permitted.
    #[must_use]
    pub fn command_line(&self) -> String {
        self.command.join(" ")
    }
}

/// The declaration file: references, and the scorers they may be paired with.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ReferenceFile {
    #[serde(default, rename = "reference")]
    pub references: Vec<ReferenceSpec>,
    /// Named scorers a VLM corpus may select. See [`NamedScorer`].
    #[serde(default, rename = "scorer")]
    pub scorers: Vec<NamedScorer>,
}

impl ReferenceFile {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        toml::from_str(&text).map_err(|e| Error::Other(format!("bad references file: {e}")))
    }

    pub fn for_task<'a>(&'a self, task: &str) -> impl Iterator<Item = &'a ReferenceSpec> {
        self.references.iter().filter(move |r| r.task == task)
    }

    /// Look up a scorer a corpus asked for by name.
    #[must_use]
    pub fn scorer(&self, name: &str) -> Option<&NamedScorer> {
        self.scorers.iter().find(|s| s.name == name)
    }
}

impl ReferenceSpec {
    #[must_use]
    pub fn supports_batch(&self) -> bool {
        self.batch_command.is_some()
    }

    /// Run the whole corpus in one invocation (see module docs).
    pub fn run_batch(&self, inputs: &[PathBuf]) -> Result<BatchResult> {
        self.run_batch_subst(inputs, &[])
    }

    /// `run_batch` with extra `{placeholder}` -> value substitutions.
    ///
    /// Exists for VLM references, which need the corpus manifest as well as
    /// the file list: a VLM item is an (image, question) pair, and the
    /// question lives in the manifest — pinned there so it falls inside the
    /// corpus fingerprint. Without `{corpus}` every VLM reference would have
    /// to be re-declared per dataset just to vary one argument.
    ///
    /// Substitution is literal and applied to each argv element, the same way
    /// `{filelist}` already is.
    pub fn run_batch_subst(
        &self,
        inputs: &[PathBuf],
        extra: &[(&str, &str)],
    ) -> Result<BatchResult> {
        let argv = self.batch_command.as_ref().ok_or_else(|| {
            Error::Other(format!("reference `{}` has no batch_command", self.name))
        })?;

        let listing: String = inputs
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join("\n");
        let list_path = std::env::temp_dir().join(format!("ffai-bench-{}.filelist", self.name));
        std::fs::write(&list_path, listing)?;

        let argv: Vec<String> = argv
            .iter()
            .map(|a| {
                let mut s = a.replace("{filelist}", &list_path.to_string_lossy());
                for (k, v) in extra {
                    s = s.replace(k, v);
                }
                s
            })
            .collect();
        let result = self.exec_measured(&argv);
        std::fs::remove_file(&list_path).ok();
        let (stdout, peak, steady) = result?;
        let mut parsed = parse_batch_output(&stdout, &self.name)?;
        parsed.peak_bytes = peak;
        parsed.steady_bytes = steady;
        Ok(parsed)
    }

    /// Run on one input file, returning stdout as text.
    pub fn run(&self, input: &Path) -> Result<String> {
        let argv = self
            .command
            .as_ref()
            .ok_or_else(|| Error::Other(format!("reference `{}` has no command", self.name)))?;
        let argv: Vec<String> = argv
            .iter()
            .map(|a| a.replace("{input}", &input.to_string_lossy()))
            .collect();
        self.exec(&argv)
    }

    fn exec(&self, argv: &[String]) -> Result<String> {
        self.exec_measured(argv).map(|(stdout, _, _)| stdout)
    }

    /// Run the reference and also report its peak working set.
    ///
    /// `Command::output()` cannot be used here: it consumes the `Child` and
    /// drops the process handle, and the handle is exactly what the memory
    /// counters are read through. So this spawns, drains both pipes on their
    /// own threads (a child that fills a pipe buffer while nobody reads it
    /// deadlocks), waits, and only then queries — while the `Child` is still
    /// alive and owns the handle.
    fn exec_measured(&self, argv: &[String]) -> Result<(String, Option<u64>, Option<u64>)> {
        use std::io::Read;
        use std::process::Stdio;

        let (prog, args) = argv.split_first().ok_or_else(|| {
            Error::Other(format!("reference `{}` has an empty command", self.name))
        })?;
        let mut child = Command::new(prog)
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                Error::Other(format!(
                    "reference `{}` failed to launch (`{prog}`): {e} — is it installed and on PATH?",
                    self.name
                ))
            })?;

        // Scope the measurement to the whole TREE, assigned immediately so the
        // launcher has not yet forked the process that does the work.
        //
        // Most references are two processes deep — this one runs
        // `python.exe adapter.py --bin whisper-cli.exe`, so measuring the
        // direct child measures the Python launcher. That reported **5 MiB for
        // a reference that loads a 77.7 MB model**, and a 127x ratio against
        // us. Both impossible, both plausible-looking in a table.
        let job = std::sync::Arc::new(crate::footprint::Job::create());
        if let Some(j) = job.as_ref() {
            // The return value MATTERS and was being dropped. This job object is
            // the whole reason the measurement is trustworthy: without it we
            // measured the Python launcher instead of the process doing the
            // work, and reported 5 MiB for a reference that loads a 77.7 MB
            // model. If the assignment fails and nobody says so, the harness
            // silently reverts to producing exactly that number - impossible,
            // and plausible-looking in a table.
            //
            // `create()` returns None on non-Windows, so reaching here means a
            // real job object exists and the failure is worth shouting about.
            if !j.assign(&child) {
                eprintln!(
                    "WARNING: could not assign the reference process to its job object;                      footprint numbers from this run measure the direct child only and                      must not be compared against ours"
                );
            }
        }
        // Sample the tree's resident memory while it runs and keep the maximum.
        // Sampling is not optional here: a process's counters die with it, and
        // the process that matters (the grandchild doing the inference) exits
        // before we could look.
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let peak_seen = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        // Samples are kept, not just the maximum: peak is set by the worst
        // instant (usually model load), while the MEDIAN is what the process
        // actually sits at while working. Reporting only the peak would
        // compare our load spike against their load spike and call it
        // footprint.
        let samples = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u64>::new()));
        let sampler = {
            let (job, done, peak_seen, samples) = (
                job.clone(),
                done.clone(),
                peak_seen.clone(),
                samples.clone(),
            );
            std::thread::spawn(move || {
                use std::sync::atomic::Ordering;
                while !done.load(Ordering::Relaxed) {
                    if let Some(ws) = job.as_ref().as_ref().and_then(|j| j.working_set_now()) {
                        peak_seen.fetch_max(ws, Ordering::Relaxed);
                        if ws > 0
                            && let Ok(mut v) = samples.lock()
                        {
                            v.push(ws);
                        }
                    }
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
            })
        };

        let mut out_pipe = child.stdout.take().expect("stdout piped above");
        let mut err_pipe = child.stderr.take().expect("stderr piped above");
        let out_thread = std::thread::spawn(move || {
            let mut buf = Vec::new();
            out_pipe.read_to_end(&mut buf).ok();
            buf
        });
        let err_thread = std::thread::spawn(move || {
            let mut buf = Vec::new();
            err_pipe.read_to_end(&mut buf).ok();
            buf
        });

        let status = child.wait().map_err(|e| {
            Error::Other(format!(
                "reference `{}` could not be waited on: {e}",
                self.name
            ))
        })?;
        done.store(true, std::sync::atomic::Ordering::Relaxed);
        sampler.join().ok();
        // The sampled tree maximum; the direct child's own peak is the
        // fallback when no job could be created, and is explicitly weaker
        // because it misses whatever the launcher spawned.
        let peak = Some(peak_seen.load(std::sync::atomic::Ordering::Relaxed))
            .filter(|b| *b > 0)
            .or_else(|| crate::footprint::peak_child(&child).map(|p| p.0));
        let steady = samples.lock().ok().and_then(|mut v| {
            if v.is_empty() {
                return None;
            }
            v.sort_unstable();
            Some(v[v.len() / 2])
        });

        let stdout = out_thread.join().unwrap_or_default();
        let stderr = err_thread.join().unwrap_or_default();
        if !status.success() {
            return Err(Error::Other(format!(
                "reference `{}` exited with {}: {}",
                self.name,
                status,
                String::from_utf8_lossy(&stderr).trim()
            )));
        }
        Ok((String::from_utf8_lossy(&stdout).into_owned(), peak, steady))
    }

    /// The argv this reference invokes, with placeholders left intact — the
    /// decode configuration as recorded in the ledger.
    #[must_use]
    pub fn command_line(&self) -> String {
        self.batch_command
            .as_ref()
            .or(self.command.as_ref())
            .map(|argv| argv.join(" "))
            .unwrap_or_default()
    }

    /// Best-effort version string (first line of `version_command` output).
    #[must_use]
    pub fn version(&self) -> Option<String> {
        let argv = self.version_command.as_ref()?;
        let (prog, args) = argv.split_first()?;
        let out = Command::new(prog).args(args).output().ok()?;
        let text = if out.stdout.trim_ascii().is_empty() {
            out.stderr
        } else {
            out.stdout
        };
        String::from_utf8_lossy(&text)
            .lines()
            .next()
            .map(|s| s.trim().to_string())
    }
}

/// Parse an adapter's JSONL stdout. Lines that aren't JSON objects are
/// ignored (adapters sometimes leak a progress line); a line with `text` is
/// a clip, a line with `load_secs` is metadata.
fn parse_batch_output(stdout: &str, name: &str) -> Result<BatchResult> {
    let mut out = BatchResult::default();
    for line in stdout.lines() {
        let line = line.trim();
        if !line.starts_with('{') {
            continue;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if let Some(load) = value.get("load_secs").and_then(|v| v.as_f64()) {
            out.load_secs = Some(load);
        }
        if let Some(total) = value.get("batch_transcribe_secs").and_then(|v| v.as_f64()) {
            out.batch_transcribe_secs = Some(total);
        }
        if let Some(text) = value.get("text").and_then(|v| v.as_str()) {
            out.clips.push(ClipResult {
                path: value
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
                text: text.to_string(),
                transcribe_secs: value.get("transcribe_secs").and_then(|v| v.as_f64()),
            });
        }
    }
    if out.clips.is_empty() {
        return Err(Error::Other(format!(
            "reference `{name}` produced no parseable JSONL clip results — check the adapter \
             contract in crates/ffai-bench/src/reference.rs"
        )));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// TTS batch mode
// ---------------------------------------------------------------------------

/// One utterance's result from a TTS batch run: the adapter read a text file
/// and wrote a WAV.
#[derive(Debug, Clone, PartialEq)]
pub struct TtsClipResult {
    /// The input text path the adapter echoed back.
    pub path: String,
    /// The generated WAV's path.
    pub wav: PathBuf,
    /// Synthesis-only seconds (model loaded, WAV writing excluded).
    pub synth_secs: Option<f64>,
    /// Time-to-first-audio: `synthesize()` call to first chunk. For a
    /// single-sentence input a non-streaming engine reports ttfa == synth.
    pub ttfa_secs: Option<f64>,
}

/// The parsed output of one TTS batch invocation.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TtsBatchResult {
    pub clips: Vec<TtsClipResult>,
    pub load_secs: Option<f64>,
    /// Adapter-reported metadata worth carrying into the ledger notes —
    /// voice name + sha256 and the effective synthesis knobs. The voice file
    /// is not corpus-pinned, so its hash rides in the record instead.
    pub meta: Vec<String>,
    pub steady_bytes: Option<u64>,
    pub peak_bytes: Option<u64>,
}

impl TtsBatchResult {
    /// Look up an utterance's result by input text path.
    #[must_use]
    pub fn clip_for(&self, path: &Path) -> Option<&TtsClipResult> {
        let want = path.to_string_lossy().replace('\\', "/");
        self.clips
            .iter()
            .find(|c| c.path.replace('\\', "/") == want)
    }

    /// Adapter-reported synthesis time for the whole batch.
    #[must_use]
    pub fn synth_secs(&self) -> Option<f64> {
        let sum: f64 = self.clips.iter().filter_map(|c| c.synth_secs).sum();
        if sum > 0.0 { Some(sum) } else { None }
    }
}

impl ReferenceSpec {
    /// Run a whole TTS corpus in one invocation. `{filelist}` is replaced by
    /// a temp file of text paths, `{outdir}` by the directory the adapter
    /// must write WAVs into. JSONL contract: a `{"load_secs": ...}` line,
    /// optional metadata lines (any object with a `"voice"` key), and one
    /// `{"path": ..., "wav": ..., "synth_secs": ..., "ttfa_secs": ...}` per
    /// utterance. See `corpora/refs/piper_ref.py` for the working example.
    pub fn run_batch_tts(&self, inputs: &[PathBuf], outdir: &Path) -> Result<TtsBatchResult> {
        let argv = self.batch_command.as_ref().ok_or_else(|| {
            Error::Other(format!("reference `{}` has no batch_command", self.name))
        })?;

        std::fs::create_dir_all(outdir)?;
        let listing: String = inputs
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join("\n");
        let list_path = std::env::temp_dir().join(format!("ffai-bench-{}.filelist", self.name));
        std::fs::write(&list_path, listing)?;

        let argv: Vec<String> = argv
            .iter()
            .map(|a| {
                a.replace("{filelist}", &list_path.to_string_lossy())
                    .replace("{outdir}", &outdir.to_string_lossy())
            })
            .collect();
        let result = self.exec_measured(&argv);
        std::fs::remove_file(&list_path).ok();
        let (stdout, peak, steady) = result?;
        let mut parsed = parse_tts_batch_output(&stdout, &self.name)?;
        parsed.peak_bytes = peak;
        parsed.steady_bytes = steady;
        Ok(parsed)
    }
}

/// Parse a TTS adapter's JSONL stdout. Same tolerance rules as the ASR
/// parser: non-JSON lines are ignored, a line with `wav` is an utterance, a
/// line with `load_secs` is timing metadata, a line with `voice` is carried
/// into the ledger notes verbatim.
fn parse_tts_batch_output(stdout: &str, name: &str) -> Result<TtsBatchResult> {
    let mut out = TtsBatchResult::default();
    for line in stdout.lines() {
        let line = line.trim();
        if !line.starts_with('{') {
            continue;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if let Some(load) = value.get("load_secs").and_then(|v| v.as_f64()) {
            out.load_secs = Some(load);
        }
        if value.get("voice").is_some() {
            out.meta.push(line.to_string());
        }
        if let Some(wav) = value.get("wav").and_then(|v| v.as_str()) {
            out.clips.push(TtsClipResult {
                path: value
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
                wav: PathBuf::from(wav),
                synth_secs: value.get("synth_secs").and_then(|v| v.as_f64()),
                ttfa_secs: value.get("ttfa_secs").and_then(|v| v.as_f64()),
            });
        }
    }
    if out.clips.is_empty() {
        return Err(Error::Other(format!(
            "reference `{name}` produced no parseable JSONL utterance results — check the TTS \
             adapter contract in crates/ffai-bench/src/reference.rs"
        )));
    }
    Ok(out)
}

#[cfg(test)]
mod tts_tests {
    use super::*;

    #[test]
    fn parses_tts_jsonl_and_carries_voice_metadata() {
        let stdout = concat!(
            "some progress noise\n",
            "{\"load_secs\": 2.1}\n",
            "{\"voice\": \"en_US-lessac-medium\", \"voice_sha256\": \"abc\"}\n",
            "{\"path\": \"a.txt\", \"wav\": \"out/a.wav\", \"synth_secs\": 0.07, \"ttfa_secs\": 0.07}\n",
            "{\"path\": \"b.txt\", \"wav\": \"out/b.wav\", \"synth_secs\": 0.05}\n",
        );
        let r = parse_tts_batch_output(stdout, "piper").unwrap();
        assert_eq!(r.clips.len(), 2);
        assert_eq!(r.load_secs, Some(2.1));
        assert_eq!(r.meta.len(), 1);
        assert!(r.meta[0].contains("voice_sha256"));
        assert!((r.synth_secs().unwrap() - 0.12).abs() < 1e-9);
        assert_eq!(
            r.clip_for(Path::new("b.txt")).unwrap().wav,
            PathBuf::from("out/b.wav")
        );
        // Windows paths from the adapter still match POSIX-style queries.
        let win = parse_tts_batch_output(
            "{\"path\": \"corpora\\\\texts\\\\a.txt\", \"wav\": \"o.wav\"}\n",
            "p",
        )
        .unwrap();
        assert!(win.clip_for(Path::new("corpora/texts/a.txt")).is_some());
    }

    #[test]
    fn tts_empty_output_is_an_error_not_a_silent_zero() {
        assert!(parse_tts_batch_output("nothing\n", "piper").is_err());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_reference_file_with_both_modes() {
        let f: ReferenceFile = toml::from_str(
            r#"
            [[reference]]
            name = "faster-whisper-tiny"
            task = "asr"
            batch_command = ["python", "ref.py", "--batch", "{filelist}"]
            version_command = ["python", "-c", "print(1)"]

            [[reference]]
            name = "tesseract"
            task = "ocr"
            command = ["tesseract", "{input}", "stdout"]
            "#,
        )
        .unwrap();
        assert_eq!(f.references.len(), 2);
        assert!(f.for_task("asr").next().unwrap().supports_batch());
        assert!(!f.for_task("ocr").next().unwrap().supports_batch());
    }

    #[test]
    fn parses_jsonl_batch_output_and_ignores_noise() {
        let stdout = concat!(
            "loading model...\n",
            "{\"load_secs\": 1.5}\n",
            "{\"path\": \"a.wav\", \"text\": \"hello\", \"transcribe_secs\": 0.25}\n",
            "{\"path\": \"b.wav\", \"text\": \"world\", \"transcribe_secs\": 0.75}\n",
        );
        let r = parse_batch_output(stdout, "test").unwrap();
        assert_eq!(r.clips.len(), 2);
        assert_eq!(r.load_secs, Some(1.5));
        assert_eq!(r.transcribe_secs(), Some(1.0));

        // Aggregate-only adapters (whisper.cpp) are supported too.
        let agg = parse_batch_output(
            "{\"batch_transcribe_secs\": 4.0}
{\"path\": \"a.wav\", \"text\": \"hi\"}
",
            "agg",
        )
        .unwrap();
        assert_eq!(agg.transcribe_secs(), Some(4.0));
        assert_eq!(r.text_for(Path::new("b.wav")), Some("world"));
    }

    #[test]
    fn empty_output_is_an_error_not_a_silent_zero() {
        assert!(parse_batch_output("nothing here\n", "test").is_err());
    }
}
