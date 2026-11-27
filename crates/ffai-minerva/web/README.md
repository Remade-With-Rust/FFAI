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

## Checked

In headless Chrome 158 (Windows, simd128, one thread), `test.html` gave the native logits hash for
p64 + 8 tokens through the full head (`1c2856725481ae14`, the same as x86, the Pixel and
wasmtime). It also held a greedy two-turn chat with the native replies ("The capital of France is
Paris." / "The Seine River runs through Paris."). Decode ran at about 1.2 tok/s on one thread,
under load.
