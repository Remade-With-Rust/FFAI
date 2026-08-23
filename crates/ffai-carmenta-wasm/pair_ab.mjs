// Which recognizer should a browser ship? Interleaved, same crop, same module.
//
// `readLine` on both, so detection is out of it and this is purely the
// recognizer — the choice that decides what a wasm build downloads and how
// long a read takes. Interleaved because this machine has shown 2.4x spread on
// identical configurations minutes apart; min-of-rounds is the estimator that
// survives that.
//
//   node pair_ab.mjs <pkg> <craft> <crnn> <det> <rec> <charset> <img.rgba> <w> <h> <rounds>
import { readFileSync } from 'node:fs';

const [pkg, craft, crnn, det, rec, cs, rgbaPath, w, h, rounds] = process.argv.slice(2);
const { Reader } = await import(`./${pkg}/ffai_carmenta_wasm.js`);
const rgba = new Uint8Array(readFileSync(rgbaPath));
const W = Number(w);
const H = Number(h);

const arms = {
  'craft-crnn': Reader.craftCrnn(
    new Uint8Array(readFileSync(craft)),
    new Uint8Array(readFileSync(crnn)),
  ),
  'mobiledet-svtr': Reader.mobiledetSvtr(
    new Uint8Array(readFileSync(det)),
    new Uint8Array(readFileSync(rec)),
    readFileSync(cs, 'utf8'),
  ),
};

const best = {};
const text = {};
for (const k of Object.keys(arms)) {
  arms[k].readLine(rgba, W, H); // untimed warm-up
  best[k] = Infinity;
}

for (let r = 0; r < Number(rounds); r++) {
  for (const [name, reader] of Object.entries(arms)) {
    const t = performance.now();
    const lines = reader.readLine(rgba, W, H);
    const ms = performance.now() - t;
    if (ms < best[name]) {
      best[name] = ms;
      text[name] = lines.map((l) => l.text).join(' ');
    }
    console.log(`round ${r + 1}  ${name.padEnd(15)} ${ms.toFixed(0)} ms`);
  }
}

console.log(`\nmin-of-${rounds}, readLine, wasm single-threaded:`);
for (const k of Object.keys(arms)) {
  console.log(`  ${k.padEnd(15)} ${best[k].toFixed(0)} ms   ${JSON.stringify(text[k])}`);
}
