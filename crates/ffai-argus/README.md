# ffai-argus

**Argus** — [FFai](https://github.com/Remade-With-Rust/FFAI)'s VLM component. Named for Argus Panoptes, the all-seeing watchman.

## Status: registered stub

This crate exists, registers with the engine registry, and appears in `ffai engines` marked `stub`. Calling it returns `Error::NotImplemented` with a message saying so.

That is deliberate. FFai registers its planned engines from the start so the shape of the toolkit is visible and a missing capability fails loudly and early, rather than looking like a crash or an obscure error at the wrong layer.

```sh
ffai engines --task vlm     # see what is here and what it says about itself
```

## What it will be

`VlmEngine` implementations to describe images and understand video, engine-selectable by name the way codecs are in ffmpeg. Sequencing is in the [roadmap](https://github.com/Remade-With-Rust/FFAI/blob/master/ROADMAP.md); the mission plan for this component is in [docs/](https://github.com/Remade-With-Rust/FFAI/tree/master/docs).

Mercury (`ffai-mercury`) is the component that is live today, and it is the template: pure Rust, oracle-gated against reference implementations, every claim traceable to a ledger line.

## License

MIT OR Apache-2.0.
