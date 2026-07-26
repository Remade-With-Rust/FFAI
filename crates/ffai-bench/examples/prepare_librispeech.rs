//! Build a hash-pinned FFai corpus manifest from the canonical LibriSpeech
//! release. Reproducible by anyone: same archive + same arguments produces a
//! byte-identical manifest, so a published benchmark can be re-derived from
//! the public source rather than trusted.
//!
//! ```text
//! curl -LO https://www.openslr.org/resources/12/test-clean.tar.gz
//! cargo run -p ffai-bench --example prepare_librispeech -- \
//!     --archive test-clean.tar.gz \
//!     --out corpora/clips/librispeech-test-clean \
//!     --manifest corpora/librispeech-test-clean-v1.toml \
//!     --count 16
//! ```
//!
//! Prerequisites: `tar` (Windows 10+, macOS, Linux) and `ffmpeg` on PATH.
//! ffmpeg is a *preparation* tool only — it converts LibriSpeech's FLAC to
//! the 16 kHz mono WAV every implementation reads, so no codec difference
//! sits inside the measured path. (When rff's FLAC decoder lands, this step
//! moves in-house.)
//!
//! LibriSpeech is CC BY 4.0 (Panayotov et al., 2015). The audio is NOT
//! committed to the repo — only the manifest that pins it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse()?;

    // Extract outside the repo: the archive expands to ~400 MB and only the
    // selected clips belong under `out`.
    let extracted = args.work.clone();
    if extracted.exists() {
        println!("reusing existing extraction at {}", extracted.display());
    } else {
        println!("extracting {} ...", args.archive.display());
        std::fs::create_dir_all(&extracted)?;
        let status = Command::new("tar")
            .arg("-xzf")
            .arg(&args.archive)
            .arg("-C")
            .arg(&extracted)
            .status()?;
        if !status.success() {
            return Err(format!("tar failed with {status}").into());
        }
    }

    // LibriSpeech/test-clean/<speaker>/<chapter>/{*.flac, *.trans.txt}
    let root = find_split_root(&extracted)?;
    println!("split root: {}", root.display());

    let mut utterances: Vec<Utterance> = Vec::new();
    for speaker in sorted_dirs(&root)? {
        for chapter in sorted_dirs(&speaker)? {
            let transcripts = read_transcripts(&chapter)?;
            for (id, text) in transcripts {
                let flac = chapter.join(format!("{id}.flac"));
                if flac.exists() {
                    utterances.push(Utterance { id, flac, text });
                }
            }
        }
    }
    utterances.sort_by(|a, b| a.id.cmp(&b.id));
    println!("found {} utterances", utterances.len());
    if utterances.is_empty() {
        return Err("no utterances found — is this a LibriSpeech archive?".into());
    }

    // Deterministic, speaker-spread selection: stride through the sorted list
    // so the sample isn't all one voice.
    let want = args.count.min(utterances.len());
    let stride = utterances.len() / want;
    let selected: Vec<&Utterance> = (0..want).map(|i| &utterances[i * stride]).collect();

    std::fs::create_dir_all(args.out.join("audio"))?;
    std::fs::create_dir_all(args.out.join("truth"))?;

    let mut rows = String::new();
    let mut total_secs = 0.0f64;
    for (i, utt) in selected.iter().enumerate() {
        let wav_rel = PathBuf::from("audio").join(format!("{}.wav", utt.id));
        let txt_rel = PathBuf::from("truth").join(format!("{}.txt", utt.id));
        let wav_abs = args.out.join(&wav_rel);
        let txt_abs = args.out.join(&txt_rel);

        // 16 kHz mono signed-16 WAV — the universal ASR input format.
        let status = Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-i"])
            .arg(&utt.flac)
            .args(["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"])
            .arg(&wav_abs)
            .status()?;
        if !status.success() {
            return Err(format!("ffmpeg failed on {}", utt.flac.display()).into());
        }
        std::fs::write(&txt_abs, &utt.text)?;

        let bytes = std::fs::read(&wav_abs)?;
        let sha = ffai_bench::corpus::file_sha256(&bytes);
        let secs = ffai_media::load_audio(&wav_abs)
            .map(|a| a.duration_secs())
            .unwrap_or(0.0);
        total_secs += secs;

        // Every 3rd clip is train, the rest holdout: claims are measured on
        // holdout only, and a train split exists so tuning has somewhere legal
        // to happen.
        let split = if i % 3 == 2 { "train" } else { "holdout" };

        rows.push_str(&format!(
            "\n[[clips]]\n\
             id = \"{}\"\n\
             path = \"{}\"\n\
             ground_truth = \"{}\"\n\
             class = \"clean_speech\"\n\
             split = \"{split}\"\n\
             license = \"CC-BY-4.0\"\n\
             sha256 = \"{sha}\"\n",
            utt.id,
            rel_from_manifest(&args.manifest, &args.out.join(&wav_rel)),
            rel_from_manifest(&args.manifest, &args.out.join(&txt_rel)),
        ));
        println!("  {:<28} {:>6.2}s  {split}", utt.id, secs);
    }

    let manifest = format!(
        // ASCII-only header: generated manifests should survive any tool's
        // encoding assumptions.
        "# LibriSpeech test-clean subset - generated by\n\
         #   cargo run -p ffai-bench --example prepare_librispeech\n\
         # Source: https://www.openslr.org/resources/12/test-clean.tar.gz\n\
         # LibriSpeech (Panayotov et al., 2015), CC BY 4.0.\n\
         # Audio is NOT committed; re-derive it with the command above.\n\
         # {} clips, {:.1}s total audio.\n\n\
         name = \"librispeech-test-clean\"\n\
         version = 1\n\
         task = \"asr\"\n{rows}",
        selected.len(),
        total_secs,
    );
    if let Some(parent) = args.manifest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.manifest, manifest)?;
    println!(
        "\nwrote {} ({} clips, {:.1}s audio)",
        args.manifest.display(),
        selected.len(),
        total_secs
    );
    Ok(())
}

struct Utterance {
    id: String,
    flac: PathBuf,
    text: String,
}

struct Args {
    archive: PathBuf,
    out: PathBuf,
    manifest: PathBuf,
    count: usize,
    /// Scratch directory for the extracted archive (outside the repo).
    work: PathBuf,
}

impl Args {
    fn parse() -> Result<Self, Box<dyn std::error::Error>> {
        let mut map = BTreeMap::new();
        let argv: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0;
        while i + 1 < argv.len() {
            if let Some(key) = argv[i].strip_prefix("--") {
                map.insert(key.to_string(), argv[i + 1].clone());
                i += 2;
            } else {
                i += 1;
            }
        }
        let get = |k: &str| -> Result<String, String> {
            map.get(k).cloned().ok_or_else(|| format!("missing --{k}"))
        };
        Ok(Args {
            archive: PathBuf::from(get("archive")?),
            out: PathBuf::from(
                map.get("out")
                    .cloned()
                    .unwrap_or_else(|| "corpora/clips/librispeech-test-clean".into()),
            ),
            manifest: PathBuf::from(
                map.get("manifest")
                    .cloned()
                    .unwrap_or_else(|| "corpora/librispeech-test-clean-v1.toml".into()),
            ),
            count: map
                .get("count")
                .map(|c| c.parse())
                .transpose()?
                .unwrap_or(16),
            work: map
                .get("work")
                .map(PathBuf::from)
                .unwrap_or_else(|| std::env::temp_dir().join("ffai-librispeech")),
        })
    }
}

/// Find the directory holding `<speaker>/<chapter>` — i.e. `.../test-clean`.
fn find_split_root(extracted: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let ls = extracted.join("LibriSpeech");
    let base = if ls.exists() { ls } else { extracted.to_path_buf() };
    for dir in sorted_dirs(&base)? {
        let name = dir.file_name().unwrap_or_default().to_string_lossy().to_string();
        if name.contains("clean") || name.contains("other") {
            return Ok(dir);
        }
    }
    Ok(base)
}

fn sorted_dirs(dir: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    out.sort();
    Ok(out)
}

/// LibriSpeech transcripts: one `<id> <UPPERCASE TEXT>` per line.
fn read_transcripts(chapter: &Path) -> Result<Vec<(String, String)>, Box<dyn std::error::Error>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(chapter)? {
        let path = entry?.path();
        if path.to_string_lossy().ends_with(".trans.txt") {
            for line in std::fs::read_to_string(&path)?.lines() {
                if let Some((id, text)) = line.split_once(' ') {
                    out.push((id.to_string(), text.trim().to_string()));
                }
            }
        }
    }
    out.sort();
    Ok(out)
}

/// Clip paths are stored relative to the manifest's directory.
fn rel_from_manifest(manifest: &Path, target: &Path) -> String {
    let base = manifest.parent().unwrap_or(Path::new("."));
    let rel = pathdiff(base, target);
    rel.to_string_lossy().replace('\\', "/")
}

fn pathdiff(base: &Path, target: &Path) -> PathBuf {
    let base = base.components().collect::<Vec<_>>();
    let target_components = target.components().collect::<Vec<_>>();
    let common = base
        .iter()
        .zip(target_components.iter())
        .take_while(|(a, b)| a == b)
        .count();
    let mut out = PathBuf::new();
    for _ in common..base.len() {
        out.push("..");
    }
    for c in &target_components[common..] {
        out.push(c.as_os_str());
    }
    out
}
