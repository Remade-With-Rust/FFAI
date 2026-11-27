// Minerva's engine in a Web Worker, so generation never blocks the page.
// Messages in:  {type:'load', url, ctx} | {type:'chat', messages, n_predict, temperature, top_k,
//               top_p, min_p, seed} | {type:'fnv', ids, n} | {type:'reset'}
// Messages out: {type:'progress', loaded, total} | {type:'ready', isa, ctx} | {type:'piece', text}
//               | {type:'done', stats} | {type:'fnv', result} | {type:'error', message}
import init, { ModelBuffer, Engine } from './pkg/ffai_minerva_web.js';

let engine = null;

async function load(url, ctx) {
  const wasm = await init();
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetching the model: HTTP ${res.status}`);
  const total = Number(res.headers.get('content-length'));
  if (!total) throw new Error('the server did not send the model size (content-length)');
  // The file streams straight into the module's memory: no second copy in JavaScript.
  const buf = new ModelBuffer(total);
  const ptr = buf.ptr();
  const reader = res.body.getReader();
  let off = 0;
  let last = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (off + value.length > total) throw new Error('the model is larger than its content-length');
    // A fresh view each time: memory.buffer is replaced when the memory grows.
    new Uint8Array(wasm.memory.buffer, ptr + off, value.length).set(value);
    off += value.length;
    if (off - last > 16 << 20 || off === total) {
      last = off;
      postMessage({ type: 'progress', loaded: off, total });
    }
  }
  if (off !== total) throw new Error(`the model arrived short: ${off} of ${total} bytes`);
  engine = new Engine(buf, ctx || 4096);
  postMessage({ type: 'ready', isa: engine.isa(), ctx: engine.ctx() });
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'load') {
      await load(m.url, m.ctx);
    } else if (m.type === 'chat') {
      const prompt = engine.apply_chat_template(JSON.stringify(m.messages));
      const stats = engine.complete(
        prompt, m.n_predict ?? 512, m.temperature ?? 0.7, m.top_k ?? 40, m.top_p ?? 0.95,
        m.min_p ?? 0.05, m.seed ?? Math.floor(Math.random() * 2 ** 32),
        (text) => postMessage({ type: 'piece', text }),
      );
      postMessage({ type: 'done', stats: JSON.parse(stats) });
    } else if (m.type === 'fnv') {
      postMessage({ type: 'fnv', result: JSON.parse(engine.full_head_fnv(new Uint32Array(m.ids), m.n)) });
    } else if (m.type === 'reset') {
      engine.reset();
    }
  } catch (err) {
    postMessage({ type: 'error', message: String((err && err.message) || err) });
  }
};
