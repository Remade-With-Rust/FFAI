"""M-T1 substitution-gate synthesizer: renders TWO phoneme arms through
piper's own runtime (docs/mercury-tts-mission.md §6.2).

Arm A ("espeak"): the pinned espeak fixture phonemes — what piper would say.
Arm B ("ours"):   Mercury's pure-Rust G2P output for the same sentences.

Same voice, same runtime, same knobs — the ONLY difference is the phoneme
input, so the judged WER difference prices the phonemizer and nothing else.
Both arms run at noise_scale = noise_w = 0: deterministic, so the gate is
reproducible and no noise sample can flatter either arm.

    piper_substitute.py --fixtures <espeak.jsonl> --ours <ours.jsonl> \
        --model <voice.onnx> --outdir <dir>

Reads ids from --ours (the holdout); emits JSONL per wav:
    {"id": ..., "arm": "espeak"|"ours", "wav": ..., "synth_secs": ...}
"""

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True, help="espeak phoneme fixture jsonl")
    ap.add_argument("--ours", required=True, help="our phoneme jsonl: {id, ipa}")
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    from piper import PiperVoice, SynthesisConfig

    voice = PiperVoice.load(args.model)
    rate = voice.config.sample_rate
    syn = SynthesisConfig(noise_scale=0.0, noise_w_scale=0.0, length_scale=1.0)

    ours = {}
    for line in Path(args.ours).read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            ours[obj["id"]] = obj["ipa"]

    espeak = {}
    for line in Path(args.fixtures).read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            if obj["id"] in ours:
                espeak[obj["id"]] = obj["phonemes"]  # list of sentences

    missing = sorted(set(ours) - set(espeak))
    if missing:
        print(json.dumps({"error": f"ids missing from fixtures: {missing[:5]}..."}))
        sys.exit(1)

    for arm, source in (("espeak", espeak), ("ours", ours)):
        outdir = Path(args.outdir) / arm
        outdir.mkdir(parents=True, exist_ok=True)
        for sid in sorted(ours):
            if arm == "espeak":
                sentences = source[sid]
            else:
                # Our IPA string, split into espeak's codepoint phonemes —
                # exactly what voice.phonemize() would have produced.
                sentences = [list(source[sid])]
            t0 = time.perf_counter()
            chunks = []
            for phonemes in sentences:
                ids = voice.phonemes_to_ids(phonemes)
                chunks.append(voice.phoneme_ids_to_audio(ids, syn_config=syn))
            audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
            synth = time.perf_counter() - t0
            wav = outdir / f"{sid}.wav"
            write_wav(wav, audio, rate)
            print(
                json.dumps(
                    {"id": sid, "arm": arm, "wav": str(wav.resolve()), "synth_secs": round(synth, 4)}
                ),
                flush=True,
            )


if __name__ == "__main__":
    sys.exit(main())
