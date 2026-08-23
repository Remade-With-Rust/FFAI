// Step 2 A/B: what is SIMD128 worth on OUR kernel, in a wasm runtime?
//
// craft-crnn with readLine — detection skipped, so this times the CRNN, which
// is the only recognizer that reaches `conv3x3`. (SVTR has no convolutions at
// all, so the mobiledet-svtr pair cannot benefit from the kernel; that path is
// candle's scalar gemm end to end and is gated on the upstream +simd128 fix.)
//
//   node crnn_ab.mjs <pkg-dir> <craft.safetensors> <crnn.safetensors> <img.rgba> <w> <h> <reps>
import { readFileSync } from 'node:fs';
const [pkg, craft, crnn, rgbaPath, w, h, reps] = process.argv.slice(2);
const { Reader, allocator } = await import(`./${pkg}/ffai_carmenta_wasm.js`);
console.log(`pkg: ${pkg}   allocator: ${allocator()}`);
const r = Reader.craftCrnn(new Uint8Array(readFileSync(craft)), new Uint8Array(readFileSync(crnn)));
const rgba = new Uint8Array(readFileSync(rgbaPath));
r.readLine(rgba, Number(w), Number(h));      // untimed warm-up
let best = Infinity, lines = [];
for (let i = 0; i < Number(reps); i++) {
  const t = performance.now();
  lines = r.readLine(rgba, Number(w), Number(h));
  best = Math.min(best, performance.now() - t);
}
console.log(`readLine: ${best.toFixed(0)} ms (min of ${reps})`);
console.log(`text: ${JSON.stringify(lines.map(l => l.text).join(' '))}`);
