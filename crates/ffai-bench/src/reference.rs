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
}

impl BatchResult {
    /// Sum of adapter-reported per-clip transcription time.
    pub fn transcribe_secs(&self) -> Option<f64> {
        let sum: f64 = self.clips.iter().filter_map(|c| c.transcribe_secs).sum();
        (sum > 0.0).then_some(sum)
    }

    /// Look up a clip's text by the path the adapter echoed back.
    pub fn text_for(&self, path: &Path) -> Option<&str> {
        let want = path.to_string_lossy().replace('\\', "/");
        self.clips
            .iter()
            .find(|c| c.path.replace('\\', "/") == want)
            .map(|c| c.text.as_str())
    }
}

/// The declaration file: a list of references.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ReferenceFile {
    #[serde(default, rename = "reference")]
    pub references: Vec<ReferenceSpec>,
}

impl ReferenceFile {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        toml::from_str(&text).map_err(|e| Error::Other(format!("bad references file: {e}")))
    }

    pub fn for_task<'a>(&'a self, task: &str) -> impl Iterator<Item = &'a ReferenceSpec> {
        self.references.iter().filter(move |r| r.task == task)
    }
}

impl ReferenceSpec {
    pub fn supports_batch(&self) -> bool {
        self.batch_command.is_some()
    }

    /// Run the whole corpus in one invocation (see module docs).
    pub fn run_batch(&self, inputs: &[PathBuf]) -> Result<BatchResult> {
        let argv = self
            .batch_command
            .as_ref()
            .ok_or_else(|| Error::Other(format!("reference `{}` has no batch_command", self.name)))?;

        let listing: String = inputs
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join("\n");
        let list_path = std::env::temp_dir().join(format!("ffai-bench-{}.filelist", self.name));
        std::fs::write(&list_path, listing)?;

        let argv: Vec<String> = argv
            .iter()
            .map(|a| a.replace("{filelist}", &list_path.to_string_lossy()))
            .collect();
        let result = self.exec(&argv);
        std::fs::remove_file(&list_path).ok();
        parse_batch_output(&result?, &self.name)
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
        let (prog, args) = argv
            .split_first()
            .ok_or_else(|| Error::Other(format!("reference `{}` has an empty command", self.name)))?;
        let out = Command::new(prog).args(args).output().map_err(|e| {
            Error::Other(format!(
                "reference `{}` failed to launch (`{prog}`): {e} — is it installed and on PATH?",
                self.name
            ))
        })?;
        if !out.status.success() {
            return Err(Error::Other(format!(
                "reference `{}` exited with {}: {}",
                self.name,
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }

    /// The argv this reference invokes, with placeholders left intact — the
    /// decode configuration as recorded in the ledger.
    pub fn command_line(&self) -> String {
        self.batch_command
            .as_ref()
            .or(self.command.as_ref())
            .map(|argv| argv.join(" "))
            .unwrap_or_default()
    }

    /// Best-effort version string (first line of `version_command` output).
    pub fn version(&self) -> Option<String> {
        let argv = self.version_command.as_ref()?;
        let (prog, args) = argv.split_first()?;
        let out = Command::new(prog).args(args).output().ok()?;
        let text = if out.stdout.trim_ascii().is_empty() { out.stderr } else { out.stdout };
        String::from_utf8_lossy(&text).lines().next().map(|s| s.trim().to_string())
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
        if let Some(text) = value.get("text").and_then(|v| v.as_str()) {
            out.clips.push(ClipResult {
                path: value.get("path").and_then(|v| v.as_str()).unwrap_or_default().to_string(),
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
        assert_eq!(r.text_for(Path::new("b.wav")), Some("world"));
    }

    #[test]
    fn empty_output_is_an_error_not_a_silent_zero() {
        assert!(parse_batch_output("nothing here\n", "test").is_err());
    }
}
