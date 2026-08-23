// Does Carmenta actually RUN on wasm? — the Step 1 gate of
// docs/plans/carmenta-wasm-plan.md.
//
// "Compiles for wasm" is not "works in a browser": three panics sat on the
// default path, every one of them defaulted ON by the ABSENCE of an
// environment (`FFAI_PROFILE`, `FFAI_REC_SERIAL`, `FFAI_CONV3X3` are all
// unreadable in a browser, so every escape hatch was shut). Nothing but
// execution proves they are gone.
//
// Node rather than a browser because it runs headless and takes the same
// module; the browser story is demo.html. RGBA comes in pre-decoded, exactly
// as `getImageData` would hand it over.
//
//   node node_smoke.mjs <pkg> <det> <crnn> <img.rgba> <w> <h> [reps]
//
// Defaults to the mobiledet-crnn pair — the one that measured fastest in a
// wasm runtime, which is NOT the one the native ranking predicts.

import { readFileSync } from 'node:fs';

const [pkg, det, crnn, rgbaPath, w, h, reps = '3'] = process.argv.slice(2);
if (!h) {
  console.error('usage: node node_smoke.mjs <pkg> <det> <crnn> <img.rgba> <w> <h> [reps]');
  process.exit(2);
}

const { Reader, allocator } = await import(`./${pkg}/ffai_carmenta_wasm.js`);

const width = Number(w);
const height = Number(h);
const rgba = new Uint8Array(readFileSync(rgbaPath));
const want = width * height * 4;
if (rgba.length !== want) {
  console.error(`RGBA is ${rgba.length} bytes, expected ${want} for ${width}x${height}`);
  process.exit(2);
}

console.log(`allocator: ${allocator()}`);

const detBytes = new Uint8Array(readFileSync(det));
const crnnBytes = new Uint8Array(readFileSync(crnn));
console.log(
  `weights: det ${(detBytes.length / 1e6).toFixed(1)} MB, rec ${(crnnBytes.length / 1e6).toFixed(1)} MB`,
);

let t = performance.now();
const reader = Reader.mobiledetCrnn(detBytes, crnnBytes);
console.log(`model load: ${(performance.now() - t).toFixed(0)} ms`);

// Untimed first pass: lazy allocation and first-touch are not what we are
// measuring, and putting them in the number is the defect §6.1 fixed at the
// benchmark level.
reader.read(rgba, width, height);

let best = Infinity;
let lines = [];
for (let i = 0; i < Number(reps); i++) {
  t = performance.now();
  lines = reader.read(rgba, width, height);
  const ms = performance.now() - t;
  if (ms < best) best = ms;
}

console.log(`read (detect + recognize): ${best.toFixed(0)} ms (min of ${reps}, single-threaded wasm)`);
console.log(`${lines.length} lines:`);
for (const l of lines) {
  console.log(
    `  ${JSON.stringify(l.text)}  @ ${l.x.toFixed(0)},${l.y.toFixed(0)} ` +
      `${l.width.toFixed(0)}x${l.height.toFixed(0)}  conf ${l.confidence.toFixed(2)}`,
  );
}
