"""piper1-gpl batch TTS adapter for `ffai bench tts`.

Contract (crates/ffai-bench/src/reference.rs, TTS batch mode): invoked ONCE
for the whole corpus with a {filelist} of text-file paths and an {outdir} for
generated WAVs. Emits JSONL on stdout:

    {"load_secs": 0.42}
    {"voice": "en_US-lessac-medium", "voice_sha256": "5efe09..."}
    {"path": "<input .txt>", "wav": "<output .wav>", "synth_secs": 0.31,
     "ttfa_secs": 0.11, "audio_secs": 2.48}

- `synth_secs` is synthesis only: consuming piper's chunk generator, model
  already loaded. WAV writing and file I/O are excluded (the harness's wall
  clock covers them).
- `ttfa_secs` is time-to-first-audio: from synthesize() call to the first
  chunk's arrival — the latency a streaming caller experiences.
- Synthesis uses the VOICE CONFIG'S OWN defaults (noise_scale, length_scale,
  noise_w) — piper's shipped behaviour, which is the thing being baselined.
  The effective values are echoed in the voice metadata line so the ledger
  can hold them.

NOTE piper samples noise inside the ONNX graph with no seed control, so two
runs produce different audio. The harness scores the audio of the run it
kept; run-to-run WER variance is piper's own property and is recorded as
such (mission plan section 8).
"""

import argparse
import hashlib
import json
import sys
import time
import wave
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, help="file with one text path per line")
    parser.add_argument("--model", required=True, help="voice .onnx path")
    parser.add_argument("--outdir", required=True, help="directory for generated WAVs")
    args = parser.parse_args()

    text_paths = [
        Path(line.strip())
        for line in Path(args.batch).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    from piper import PiperVoice  # import inside the timed region: it drags onnxruntime in

    voice = PiperVoice.load(args.model)
    # Untimed warm-up: onnxruntime's first inference carries session/kernel
    # setup (~3.3 s measured against ~0.07 s steady). Folding that into the
    # first clip's synth_secs would misreport warm throughput — the same
    # unfairness the ASR harness fixes with an untimed engine warm-up pass.
    # It is one-time startup cost, so it belongs in load_secs.
    for _ in voice.synthesize("Warm up."):
        pass
    load_secs = time.perf_counter() - t0
    print(json.dumps({"load_secs": round(load_secs, 4)}), flush=True)

    sha = hashlib.sha256(Path(args.model).read_bytes()).hexdigest()
    cfg = voice.config
    print(
        json.dumps(
            {
                "voice": Path(args.model).stem,
                "voice_sha256": sha,
                "sample_rate": cfg.sample_rate,
                "noise_scale": cfg.noise_scale,
                "length_scale": cfg.length_scale,
                "noise_w": cfg.noise_w_scale,
                "num_speakers": cfg.num_speakers,
            }
        ),
        flush=True,
    )

    for text_path in text_paths:
        text = text_path.read_text(encoding="utf-8").strip()
        wav_path = outdir / (text_path.stem + ".wav")

        chunks = []
        ttfa = None
        start = time.perf_counter()
        for chunk in voice.synthesize(text):
            if ttfa is None:
                ttfa = time.perf_counter() - start
            chunks.append(chunk)
        synth_secs = time.perf_counter() - start

        if not chunks:
            print(json.dumps({"path": str(text_path), "error": "no audio produced"}), flush=True)
            continue

        sample_rate = chunks[0].sample_rate
        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(chunks[0].sample_channels)
            wav.setsampwidth(chunks[0].sample_width)
            wav.setframerate(sample_rate)
            for chunk in chunks:
                wav.writeframes(chunk.audio_int16_bytes)

        frames = sum(len(c.audio_int16_bytes) // (c.sample_width * c.sample_channels) for c in chunks)
        print(
            json.dumps(
                {
                    "path": str(text_path),
                    "wav": str(wav_path.resolve()),
                    "synth_secs": round(synth_secs, 4),
                    "ttfa_secs": round(ttfa, 4) if ttfa is not None else None,
                    "audio_secs": round(frames / sample_rate, 4),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())
