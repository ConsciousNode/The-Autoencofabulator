// Finite-difference check of ScoreNet.backward, run on the classes lifted out
// of index.html (float64, tiny step). Zero dependencies: `node tests/gradcheck.js`
const fs = require('fs'), path = require('path'), vm = require('vm');
const src = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
const cut = (a, b) => { const i = src.indexOf(a), j = src.indexOf(b, i); if (i < 0 || j < 0) throw new Error('marker ' + a); return src.slice(i, j); };
const code = [cut('const CFG = {', 'const FORMAT ='), cut('const VOCAB_WORDS', '\nfunction tagToIds'), cut('function randn()', 'function yieldToMain'), cut('function clamp(', '\nfunction esc('),
  cut('class Linear {', 'class Adam {'), cut('class Adam {', 'class ScoreNet {'), cut('class ScoreNet {', '\n// ═'), 'this.N = ScoreNet;'].join('\n');
const ctx = { Math, Float32Array: Float64Array, Array, Object, console: { log() {} } }; vm.createContext(ctx); vm.runInContext(code, ctx);
let seed = 3; ctx.Math.random = Math.random = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
const net = new ctx.N();
const x = Float64Array.from({ length: 256 }, () => Math.random() * 2 - 1), tgt = Float64Array.from({ length: 256 }, () => Math.random() * 2 - 1);
const args = [x, 37, 1, [3, 9], 0.2, 0.5];
const loss = () => { const o = net.forward(...args); let l = 0; for (let i = 0; i < o.length; i++) l += (o[i] - tgt[i]) ** 2; return l / o.length; };
net.zeroGrad();
const o = net.forward(...args), go = new Float64Array(256); for (let i = 0; i < 256; i++) go[i] = 2 * (o[i] - tgt[i]) / 256;
net.backward(go);
const eps = 1e-6; let fails = 0;
function chk(name, W, G, idx) {
  let dot = 0, na = 0, nn = 0;
  for (const i of idx) { const o0 = W[i]; W[i] = o0 + eps; const a = loss(); W[i] = o0 - eps; const b = loss(); W[i] = o0; const num = (a - b) / (2 * eps); dot += G[i] * num; na += G[i] * G[i]; nn += num * num; }
  const cos = dot / Math.sqrt(na * nn + 1e-300), ratio = Math.sqrt(na / (nn + 1e-300)), ok = cos > 0.999 && Math.abs(ratio - 1) < 0.01; if (!ok) fails++;
  console.info(`${name.padEnd(12)} cos=${cos.toFixed(4)} ratio=${ratio.toFixed(4)} ${ok ? 'ok' : 'FAIL'}`);
}
const sample = n => Array.from({ length: 30 }, () => Math.floor(Math.random() * n));
for (const n of ['l4', 'l3', 'l2', 'l1', 'adaShift1', 'adaScale1', 'adaShift2', 'adaScale2']) chk(n, net[n].weight, net[n].gradW, sample(net[n].weight.length));
for (const n of ['ln1', 'ln2', 'ln3', 'ln4']) chk(n + '.w', net[n].weight, net[n].gradW, sample(net[n].weight.length));
const S = 32, F = 16;
chk('semEmb', net.semEmb, net.gSem, [...Array(S).keys()].map(i => 3 * S + i).concat([...Array(S).keys()].map(i => 9 * S + i)));
chk('fmtEmb', net.fmtEmb, net.gFmt, [...Array(F).keys()].map(i => 1 * F + i));
console.info(fails ? `${fails} FAILED` : 'ALL OK');
process.exit(fails ? 1 : 0);
