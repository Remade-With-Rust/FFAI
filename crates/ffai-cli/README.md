# ffai-cli

The `ffai` binary — a thin shell over the [FFai](https://github.com/Remade-With-Rust/FFAI) library crates, the way the `ffmpeg` binary is a thin shell over libav*.

```sh
cargo install ffai-cli
```

**It contains no logic.** Every capability lives in a crate you can embed; this only parses arguments and prints. If the CLI can do it, your application can do it without the CLI.

## Speech

```sh
ffai asr -i talk.wav                                   # text to stdout
ffai asr -i talk.wav -o talk.srt                       # SubRip
ffai asr -i talk.wav -o talk.vtt --word-timestamps     # WebVTT, inline word timing
ffai asr -i talk.wav -o talk.json --word-timestamps    # JSON with per-word times
ffai asr -i meeting.wav --diarize --max-speakers 3     # who spoke when
ffai asr -i talk.wav --no-vad                          # raw fixed 30 s grid
```

Output format follows the extension, as `ffmpeg -o` does.

## Discovery

```sh
ffai engines                    # every engine + status, like `ffmpeg -codecs`
ffai models                     # manifests, licenses, cache status
ffai models --fetch whisper-tiny-en    # download BEFORE a timed run
```

`ffai engines` prints `stub` / `experimental` / `stable` honestly — a stub is registered and selectable and will tell you it is a stub rather than failing obscurely.

## Benchmarking

```sh
ffai bench asr --corpus corpora/librispeech-test-clean-v2.toml
```

Runs the engine and every declared reference over the same holdout clips, scores them with the same metric code, and appends a record to `bench/ledger.jsonl`.

## Flags that mean what they say

Conflicting flags error rather than silently picking one: `--no-vad` with `--diarize` is refused, because diarization needs speech segmentation and quietly degrading would be worse. Out-of-range thresholds are rejected with the reason. A flag that implies another says so on stderr.

## License

MIT OR Apache-2.0.
