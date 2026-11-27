# ffai-minerva-web: Minerva in the browser

The whole BitNet b1.58 model running in a browser tab: WebAssembly (simd128), in a Web Worker,
behind wasm-bindgen. Nothing leaves the page.

- `src/lib.rs`: `ModelBuffer` (the model file streamed straight into the module's memory) and
  `Engine` (`complete` with streamed pieces, `apply_chat_template`, `tokenize`, `detokenize`,
  `reset`, and `full_head_fnv`, the check against native).
- `www/worker.js`: the engine in a worker (load, chat, check messages).
- `www/index.html`: a chat page. `www/test.html`: the browser check.
- `serve.py`: a local server for `www/`, with `/model.gguf` streamed from a model file.

## Build

The `wasm-bindgen` crate is pinned to the installed `wasm-bindgen-cli` (0.2.128); the two must
match exactly.

```sh
# from crates/ffai-minerva
RUSTFLAGS="-C target-feature=+simd128" cargo build -p ffai-minerva-web --release --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir web/www/pkg target/wasm32-unknown-unknown/release/ffai_minerva_web.wasm
python web/serve.py MODEL-i2_s.gguf --prompts PROMPTS_DIR --port 8099   # then open http://127.0.0.1:8099/
```

Threads (`threads` feature: rayon on Web Workers over shared memory). The page must be
cross-origin isolated (`serve.py` sends `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp`); the worker falls back to `pkg/` when it is not.
Recent nightlies no longer make the memory shared on their own, so the linker flags are explicit,
and the rayon helper's bundler-style `import('../../..')` is pointed at the module file so plain
static hosting works:

```sh
RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+simd128 -C link-arg=--shared-memory -C link-arg=--max-memory=4294967296 -C link-arg=--import-memory -C link-arg=--export=__wasm_init_tls -C link-arg=--export=__tls_size -C link-arg=--export=__tls_align -C link-arg=--export=__tls_base"   CARGO_TARGET_DIR=target/wasm-threads cargo +nightly build -p ffai-minerva-web --release   --target wasm32-unknown-unknown --features threads -Z build-std=panic_abort,std
wasm-bindgen --target web --out-dir web/www/pkg-threads target/wasm-threads/wasm32-unknown-unknown/release/ffai_minerva_web.wasm
sed -i "s#await import('../../..');#await import('../../../ffai_minerva_web.js');#" web/www/pkg-threads/snippets/wasm-bindgen-rayon-*/src/workerHelpers.js
```

## Checked

In headless Chrome 158 (Windows, simd128, one thread), `test.html` gave the native logits hash for
p64 + 8 tokens through the full head (`1c2856725481ae14`, the same as x86, the Pixel and
wasmtime). It also held a greedy two-turn chat with the native replies ("The capital of France is
Paris." / "The Seine River runs through Paris."). Under load:

| | one thread | 8 threads |
|---|---|---|
| decode | ~1.2 tok/s | ~8.0 tok/s |
| first token (14-token chat prompt) | 7.3 s | 1.1 s |
| p64 + 8 through the full head | 23.6 s | 3.8 s |

Same hash and same replies with 8 threads (`test.html?threads=8`).
