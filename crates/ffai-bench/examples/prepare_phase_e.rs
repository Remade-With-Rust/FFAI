//! Phase E corpora: the two the project has been missing.
//!
//! Every corpus FFai has is continuous clean read speech. That blind spot has
//! cost real defects — Whisper hallucinating `you` on silence, and a cough
//! transcribed as `Hah!` — both found by a browser demo in ten seconds of
//! microphone audio, after 268 pinned clips saw neither. It also means
//! `--diarize` has no gate at all: LibriSpeech has one speaker per clip.
//!
//! Two corpora are built here.
//!
//! **`silence-and-nonspeech`** — audio containing no words. Ground truth is
//! the empty string. Digital silence, room tone at several levels, a tone, a
//! click, and a noise burst. This is the corpus that would have caught the
//! `you` bug on day one, and it costs nothing to carry.
//!
//! **`librispeech-diarization`** — multi-speaker audio assembled from
//! LibriSpeech clips whose speaker IDs are known, so the ground truth RTTM is
//! **exact by construction** rather than annotated. That is the honest way to
//! bootstrap a diarization gate without a licence-encumbered conversational
//! corpus.
//!
//! Its limits, stated here rather than discovered later: speakers never
//! overlap, turns are clean cuts rather than natural interruptions, and the
//! audio is studio-quality read speech. A system can score well here and do
//! badly on a real meeting. It gates *regression*, not *readiness* — which is
//! still infinitely more than no gate.
//!
//! ```sh
//! cargo run --release -p ffai-bench --example prepare_phase_e
//! ```

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Deterministic noise, so a rebuilt corpus has the same hashes.
struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f32 / (1u64 << 53) as f32 - 0.5
    }
}

const SR: u32 = 16_000;

fn write_wav(path: &Path, samples: &[f32]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let data_len = samples.len() * 2;
    let mut out = Vec::with_capacity(44 + data_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&SR.to_le_bytes());
    out.extend_from_slice(&(SR * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    for s in samples {
        out.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    fs::write(path, out)
}

fn read_wav(path: &Path) -> std::io::Result<Vec<f32>> {
    let bytes = fs::read(path)?;
    if bytes.len() < 44 {
        return Ok(Vec::new());
    }
    Ok(bytes[44..]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect())
}

fn sha256_of(path: &Path) -> std::io::Result<String> {
    Ok(ffai_bench::corpus::file_sha256(&fs::read(path)?))
}

fn secs(n: f64) -> usize {
    (n * SR as f64) as usize
}

fn main() -> std::io::Result<()> {
    build_silence_corpus()?;
    build_diarization_corpus()?;
    Ok(())
}

/// Audio with no words in it. Ground truth: the empty string.
fn build_silence_corpus() -> std::io::Result<()> {
    let root = PathBuf::from("corpora/clips/silence-and-nonspeech");
    let mut clips = Vec::new();
    let mut rng = Rng(0x5EED_1234_5678_9ABC);

    let mut add = |id: &str, samples: Vec<f32>, class: &str, clips: &mut Vec<String>| -> std::io::Result<()> {
        let audio = root.join("audio").join(format!("{id}.wav"));
        let truth = root.join("truth").join(format!("{id}.txt"));
        write_wav(&audio, &samples)?;
        fs::create_dir_all(truth.parent().expect("has parent"))?;
        // Empty ground truth: anything the model emits here is an error.
        fs::write(&truth, "")?;
        clips.push(format!(
            "[[clips]]\nid = \"{id}\"\npath = \"clips/silence-and-nonspeech/audio/{id}.wav\"\n\
             ground_truth = \"clips/silence-and-nonspeech/truth/{id}.txt\"\n\
             class = \"{class}\"\nsplit = \"holdout\"\nlicense = \"CC0-1.0\"\n\
             sha256 = \"{}\"\n",
            sha256_of(&audio)?
        ));
        Ok(())
    };

    add("digital-silence-5s", vec![0.0; secs(5.0)], "other", &mut clips)?;
    add("digital-silence-30s", vec![0.0; secs(30.0)], "other", &mut clips)?;

    // Room tone at three levels. The quietest is near the noise floor of a
    // good microphone; the loudest is a noisy laptop fan.
    for (name, amp) in [("room-tone-quiet", 0.0005f32), ("room-tone-mid", 0.004), ("room-tone-loud", 0.02)] {
        let s: Vec<f32> = (0..secs(8.0)).map(|_| rng.next_f32() * amp * 2.0).collect();
        add(name, s, "other", &mut clips)?;
    }

    // A steady tone: not speech, and nothing a VAD energy threshold should
    // pass on its own.
    let tone: Vec<f32> = (0..secs(6.0))
        .map(|i| 0.15 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SR as f32).sin())
        .collect();
    add("tone-440hz", tone, "other", &mut clips)?;

    // A click and a short burst — transients that are loud but wordless.
    let mut click = vec![0.0f32; secs(4.0)];
    click[secs(2.0)] = 0.9;
    click[secs(2.0) + 1] = -0.7;
    add("click", click, "other", &mut clips)?;

    let mut burst = vec![0.0f32; secs(4.0)];
    for i in 0..secs(0.25) {
        let env = (-6.0 * i as f32 / secs(0.25) as f32).exp();
        burst[secs(1.5) + i] = rng.next_f32() * 1.6 * env;
    }
    add("noise-burst", burst, "other", &mut clips)?;

    let manifest = format!(
        "# Audio containing NO WORDS. Ground truth is the empty string, so any\n\
         # token an engine emits here is an error.\n\
         #\n\
         # This corpus exists because every other corpus in the project is\n\
         # continuous clean read speech, and that blind spot shipped two real\n\
         # defects: Whisper hallucinating \"you\" on silence, and a cough coming\n\
         # out as \"Hah!\". Both were found by a browser demo in ten seconds of\n\
         # microphone audio after 268 pinned clips saw neither.\n\
         #\n\
         # Synthesised deterministically by\n\
         #   cargo run --release -p ffai-bench --example prepare_phase_e\n\
         # so the hashes are reproducible. Synthetic audio is banned from a\n\
         # QUALITY verdict (see codec-tune-quality); it is entirely valid for a\n\
         # CORRECTNESS one, which is what this measures: did the engine invent\n\
         # words that were never spoken.\n\n\
         name = \"silence-and-nonspeech\"\nversion = 1\ntask = \"asr\"\n\n{}",
        clips.join("\n")
    );
    fs::write("corpora/silence-and-nonspeech.toml", manifest)?;
    println!("wrote corpora/silence-and-nonspeech.toml ({} clips)", clips.len());
    Ok(())
}

/// Multi-speaker audio with an exact RTTM, assembled from known speakers.
fn build_diarization_corpus() -> std::io::Result<()> {
    let src = PathBuf::from("corpora/clips/librispeech-test-clean/audio");
    if !src.exists() {
        eprintln!(
            "skipping diarization corpus: {} not found — run prepare_librispeech first",
            src.display()
        );
        return Ok(());
    }

    // Group the source clips by LibriSpeech speaker id (the leading field of
    // `<speaker>-<chapter>-<utterance>.wav`).
    let mut by_speaker: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    for entry in fs::read_dir(&src)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("wav") {
            continue;
        }
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or_default().to_string();
        if let Some(id) = stem.split('-').next() {
            by_speaker.entry(id.to_string()).or_default().push(path.clone());
        }
    }
    let mut speakers: Vec<(String, Vec<PathBuf>)> = by_speaker.into_iter().collect();
    speakers.retain(|(_, v)| v.len() >= 2);
    for (_, v) in speakers.iter_mut() {
        v.sort();
    }
    if speakers.len() < 4 {
        eprintln!("skipping diarization corpus: only {} usable speakers", speakers.len());
        return Ok(());
    }

    let root = PathBuf::from("corpora/clips/librispeech-diarization");
    let gap = vec![0.0f32; secs(0.4)];
    let mut clips = Vec::new();

    // Six conversations: 2, 2, 3, 3, 4, 4 speakers, each speaker taking two
    // turns so a system must RE-IDENTIFY a returning voice rather than merely
    // cut on silence. That distinction is the one worth gating.
    let plans: [&[usize]; 6] = [
        &[0, 1, 0, 1],
        &[2, 3, 2, 3],
        &[0, 2, 4, 0, 2, 4],
        &[1, 3, 5, 1, 3, 5],
        &[0, 1, 2, 3, 0, 1, 2, 3],
        &[2, 3, 4, 5, 2, 3, 4, 5],
    ];

    for (n, plan) in plans.iter().enumerate() {
        if plan.iter().copied().max().unwrap_or(0) >= speakers.len() {
            continue;
        }
        let id = format!("conv-{:02}", n + 1);
        let mut samples: Vec<f32> = Vec::new();
        let mut rttm = String::new();
        let mut taken: BTreeMap<usize, usize> = BTreeMap::new();

        for &si in plan.iter() {
            let (speaker_id, files) = &speakers[si];
            let which = taken.entry(si).or_insert(0);
            let file = &files[*which % files.len()];
            *which += 1;

            let audio = read_wav(file)?;
            if audio.is_empty() {
                continue;
            }
            let start = samples.len() as f64 / SR as f64;
            samples.extend_from_slice(&audio);
            let end = samples.len() as f64 / SR as f64;
            samples.extend_from_slice(&gap);

            // RTTM carries DURATION, not end time — the classic misreading.
            rttm.push_str(&format!(
                "SPEAKER {id} 1 {:.3} {:.3} <NA> <NA> {speaker_id} <NA> <NA>\n",
                start,
                end - start
            ));
        }

        let audio_path = root.join("audio").join(format!("{id}.wav"));
        let truth_path = root.join("truth").join(format!("{id}.rttm"));
        write_wav(&audio_path, &samples)?;
        fs::create_dir_all(truth_path.parent().expect("has parent"))?;
        fs::write(&truth_path, &rttm)?;

        let distinct: std::collections::BTreeSet<usize> = plan.iter().copied().collect();
        clips.push(format!(
            "# {} speakers, {:.1}s\n[[clips]]\nid = \"{id}\"\n\
             path = \"clips/librispeech-diarization/audio/{id}.wav\"\n\
             ground_truth = \"clips/librispeech-diarization/truth/{id}.rttm\"\n\
             class = \"clean_speech\"\nsplit = \"holdout\"\nlicense = \"CC-BY-4.0\"\n\
             sha256 = \"{}\"\n",
            distinct.len(),
            samples.len() as f64 / SR as f64,
            sha256_of(&audio_path)?
        ));
    }

    let manifest = format!(
        "# Multi-speaker audio with an EXACT ground truth.\n\
         #\n\
         # Assembled from LibriSpeech clips whose speaker ids are known, so the\n\
         # RTTM is exact by construction rather than annotated. That is how a\n\
         # diarization gate gets bootstrapped without a licence-encumbered\n\
         # conversational corpus.\n\
         #\n\
         # Every speaker takes at least two turns, so a system must RE-IDENTIFY\n\
         # a returning voice rather than merely cut on silence. Cutting on\n\
         # silence alone scores badly here, which is the point.\n\
         #\n\
         # LIMITS, stated here rather than discovered later: speakers never\n\
         # overlap, turns are clean cuts rather than natural interruptions, and\n\
         # the audio is studio-quality read speech with 0.4s of digital silence\n\
         # between turns. A system can score well here and badly on a real\n\
         # meeting. This gates REGRESSION, not READINESS.\n\
         #\n\
         # Rebuild with\n\
         #   cargo run --release -p ffai-bench --example prepare_phase_e\n\n\
         name = \"librispeech-diarization\"\nversion = 1\ntask = \"asr\"\n\n{}",
        clips.join("\n")
    );
    fs::write("corpora/librispeech-diarization.toml", manifest)?;
    println!("wrote corpora/librispeech-diarization.toml ({} conversations)", clips.len());
    Ok(())
}
