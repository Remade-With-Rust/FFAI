"""Dump espeak-ng's phonemization of a text corpus, via piper's own embedded
espeak — the EXACT phonemizer the piper voices were trained against, which is
what makes this the oracle for Mercury's pure-Rust G2P (mission plan M-T1).

espeak-ng is GPL; it runs here as an out-of-process fixture generator only —
the same seat openai-whisper's Python holds for the ASR mel oracle. Nothing
GPL is linked into any shipped crate.

    .venv-bench/Scripts/python corpora/refs/dump_espeak_phonemes.py \
        --batch <filelist> --model .piper-voices/en_US-lessac-medium.onnx \
        > corpora/fixtures/harvard-espeak-phonemes-v1.jsonl

Output: one JSON object per input text file:
    {"id": "hvd-01-01", "text": "...", "phonemes": [["ð", "ə", ...], ...],
     "phoneme_ids": [[1, 41, ...], ...]}

`phonemes` is a list of sentences, each a list of espeak IPA phoneme strings
(one codepoint-cluster per entry, exactly what `phoneme_id_map` keys on).
`phoneme_ids` is the voice's id mapping of each sentence — BOS/EOS/pad
interleaving included — i.e. the literal model input. M-T1's substitution
gate feeds OUR ids through piper's runtime against these.
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True, help="file with one text path per line")
    parser.add_argument("--model", required=True, help="voice .onnx path")
    args = parser.parse_args()

    from piper import PiperVoice

    voice = PiperVoice.load(args.model)
    for line in Path(args.batch).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        text_path = Path(line.strip())
        text = text_path.read_text(encoding="utf-8").strip()
        sentences = voice.phonemize(text)
        ids = [voice.phonemes_to_ids(s) for s in sentences]
        print(
            json.dumps(
                {
                    "id": text_path.stem,
                    "text": text,
                    "phonemes": sentences,
                    "phoneme_ids": ids,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())
