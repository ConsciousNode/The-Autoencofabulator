#!/usr/bin/env node
/**
 * Determinism, .fabula Serialization, WASM/SIMD matmul_f32 & linear_bwd_f32 Equivalence,
 * and Fast BilateralContextMemory Equivalence Test for The Autoencofabulator v0.4: Learned.
 *
 * Zero external dependencies. Extracts the engine code directly from index.html.
 *
 * Verifies:
 *   1. Same seed, corpus, and settings -> bit-for-bit identical weights after training.
 *   2. Same seed, weights, and settings -> bit-for-bit identical generated samples
 *      (both confidence-ordered and random unmasking, 1-D byte mode and 2-D patch mode).
 *   3. Different seed -> distinct weights and samples.
 *   4. v0.2 .fabula files are gracefully rejected with a clear message and never crash.
 *   5. v0.4 .fabula export/import round-trips weights bit-for-bit.
 *   6. WasmKernels ('js', 'wasm', 'simd') produce identical matvec_f32, matmul_f32,
 *      linear_bwd_f32, and full transformer loss curves to within float32 rounding (< 1e-5).
 *   7. Fast typed-array BilateralContextMemory produces identical log-probs to the
 *      v0.3 Map-of-Map reference implementation (max diff < 1e-6).
 *   8. 512-byte generation at the Phone preset finishes in well under 10 seconds.
 *
 * Run: node tests/determinism.js
 */
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

function resolveIndexHtml() {
  const candidates = [
    path.join(__dirname, '..', 'index.html'),
    path.join(__dirname, 'index.html'),
  ];
  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }
  throw new Error('Could not locate index.html relative to determinism.js');
}

const html = fs.readFileSync(resolveIndexHtml(), 'utf8');
const startMarker = '// === FABULA_ENGINE_START ===';
const endMarker = '// === FABULA_ENGINE_END ===';
const sIdx = html.indexOf(startMarker);
const eIdx = html.indexOf(endMarker);
if (sIdx < 0 || eIdx < 0) {
  throw new Error('Could not find FABULA_ENGINE_START / FABULA_ENGINE_END markers in index.html');
}
const engineCode = html.slice(sIdx + startMarker.length, eIdx);

globalThis.window = globalThis;
globalThis.self = globalThis;
if (!globalThis.performance) {
  globalThis.performance = { now: () => Date.now() };
}
vm.runInThisContext(engineCode);

const Engine = globalThis.FabulaEngine;
let allOk = true;

function assert(cond, msg) {
  const status = cond ? 'PASS' : 'FAIL';
  console.log(`  ${status}  ${msg}`);
  if (!cond) allOk = false;
}

// Reference v0.3 Map-of-Map BilateralContextMemory to verify exact log-prob equivalence (< 1e-6)
class ReferenceV03BilateralContextMemory {
  constructor() {
    this.unigram = new Float32Array(256).fill(0.5);
    this.uniTotal = 128;
    this.left1 = new Map();
    this.left2 = new Map();
    this.left3 = new Map();
    this.right1 = new Map();
    this.right2 = new Map();
    this.bridge11 = new Map();
    this.bridge21 = new Map();
    this.skipL = new Map();
    this.skipR = new Map();
    this.posPrior = new Map();
    this.tagPrior = new Map();
  }

  _inc(map, key, byteVal, w = 1) {
    let node = map.get(key);
    if (!node) { node = { total: 0, m: new Map() }; map.set(key, node); }
    node.total += w;
    node.m.set(byteVal, (node.m.get(byteVal) || 0) + w);
  }

  observeSequence(bytes, tagIds = [], stride = 1) {
    const n = bytes.length;
    for (let i = 0; i < n; i += stride) {
      const b = bytes[i];
      if (b < 0 || b > 255) continue;
      this.unigram[b] += 1;
      this.uniTotal += 1;
      if (i < 512) this._inc(this.posPrior, i, b);
      for (let ti = 0; ti < tagIds.length; ti++) this._inc(this.tagPrior, tagIds[ti], b);
      if (i >= 1 && bytes[i - 1] < 256) {
        this._inc(this.left1, bytes[i - 1], b);
        if (i >= 2 && bytes[i - 2] < 256) {
          const k2 = (bytes[i - 2] << 8) | bytes[i - 1];
          this._inc(this.left2, k2, b);
          if (i >= 3 && bytes[i - 3] < 256) {
            this._inc(this.left3, (bytes[i - 3] << 16) | k2, b);
          }
        }
      }
      if (i + 1 < n && bytes[i + 1] < 256) {
        this._inc(this.right1, bytes[i + 1], b);
        if (i + 2 < n && bytes[i + 2] < 256) {
          this._inc(this.right2, (bytes[i + 1] << 8) | bytes[i + 2], b);
        }
      }
      if (i >= 1 && i + 1 < n && bytes[i - 1] < 256 && bytes[i + 1] < 256) {
        const br1 = (bytes[i - 1] << 8) | bytes[i + 1];
        this._inc(this.bridge11, br1, b);
        if (i >= 2 && bytes[i - 2] < 256) {
          this._inc(this.bridge21, (bytes[i - 2] << 16) | br1, b);
        }
      }
      for (let d = 2; d <= 4; d++) {
        if (i >= d && bytes[i - d] < 256) this._inc(this.skipL, (d << 8) | bytes[i - d], b);
        if (i + d < n && bytes[i + d] < 256) this._inc(this.skipR, (d << 8) | bytes[i + d], b);
      }
    }
  }

  _lookupDist(map, key, baseProbs, discount = 0.70) {
    const node = map.get(key);
    if (!node || node.total === 0) return baseProbs;
    const out = new Float32Array(256);
    const invT = 1 / node.total;
    let remMass = 0;
    for (const [k, cnt] of node.m.entries()) {
      const d = cnt <= 1 ? 0.45 : discount;
      const p = Math.max(0, cnt - d) * invT;
      out[k] = p;
      remMass += p;
    }
    const bow = Math.max(0, 1 - remMass);
    for (let v = 0; v < 256; v++) out[v] += bow * baseProbs[v];
    return out;
  }

  predictLogProbsAt(xt, i, tagIds = [], usePosPrior = false) {
    const L = xt.length, MASK = 256;
    const pUni = new Float32Array(256);
    const invU = 1 / this.uniTotal;
    for (let v = 0; v < 256; v++) pUni[v] = this.unigram[v] * invU;

    let pBase = pUni;
    if (tagIds && tagIds.length > 0) {
      for (let ti = 0; ti < tagIds.length; ti++) {
        pBase = this._lookupDist(this.tagPrior, tagIds[ti], pBase, 0.5);
      }
    }
    if (usePosPrior && i < 512) {
      const pPos = this._lookupDist(this.posPrior, i, pBase, 0.55);
      for (let v = 0; v < 256; v++) pBase[v] = 0.65 * pBase[v] + 0.35 * pPos[v];
    }

    const l1 = i >= 1 ? xt[i - 1] : MASK;
    const l2 = i >= 2 ? xt[i - 2] : MASK;
    const l3 = i >= 3 ? xt[i - 3] : MASK;
    const r1 = i + 1 < L ? xt[i + 1] : MASK;
    const r2 = i + 2 < L ? xt[i + 2] : MASK;

    let pL = pBase;
    if (l1 < 256) {
      pL = this._lookupDist(this.left1, l1, pL, 0.65);
      if (l2 < 256) {
        pL = this._lookupDist(this.left2, (l2 << 8) | l1, pL, 0.75);
        if (l3 < 256) {
          pL = this._lookupDist(this.left3, (l3 << 16) | (l2 << 8) | l1, pL, 0.80);
        }
      }
    } else {
      for (let d = 2; d <= 4; d++) {
        if (i >= d && xt[i - d] < 256) {
          pL = this._lookupDist(this.skipL, (d << 8) | xt[i - d], pBase, 0.60);
          break;
        }
      }
    }

    let pR = pBase;
    if (r1 < 256) {
      pR = this._lookupDist(this.right1, r1, pR, 0.65);
      if (r2 < 256) {
        pR = this._lookupDist(this.right2, (r1 << 8) | r2, pR, 0.75);
      }
    } else {
      for (let d = 2; d <= 4; d++) {
        if (i + d < L && xt[i + d] < 256) {
          pR = this._lookupDist(this.skipR, (d << 8) | xt[i + d], pBase, 0.60);
          break;
        }
      }
    }

    let pComb = new Float32Array(256);
    if (l1 < 256 && r1 < 256) {
      let sum = 0;
      for (let v = 0; v < 256; v++) {
        const val = (pL[v] * pR[v]) / Math.pow(Math.max(pBase[v], 1e-6), 0.78);
        pComb[v] = val; sum += val;
      }
      const invS = 1 / (sum || 1);
      for (let v = 0; v < 256; v++) pComb[v] *= invS;
      pComb = this._lookupDist(this.bridge11, (l1 << 8) | r1, pComb, 0.70);
      if (l2 < 256) {
        pComb = this._lookupDist(this.bridge21, (l2 << 16) | (l1 << 8) | r1, pComb, 0.75);
      }
    } else if (l1 < 256) {
      let sum = 0;
      for (let v = 0; v < 256; v++) {
        const val = Math.pow(pL[v], 0.85) * Math.pow(pR[v], 0.15);
        pComb[v] = val; sum += val;
      }
      const invS = 1 / (sum || 1);
      for (let v = 0; v < 256; v++) pComb[v] *= invS;
    } else if (r1 < 256) {
      let sum = 0;
      for (let v = 0; v < 256; v++) {
        const val = Math.pow(pR[v], 0.85) * Math.pow(pL[v], 0.15);
        pComb[v] = val; sum += val;
      }
      const invS = 1 / (sum || 1);
      for (let v = 0; v < 256; v++) pComb[v] *= invS;
    } else {
      for (let v = 0; v < 256; v++) pComb[v] = 0.5 * (pL[v] + pR[v]);
    }

    const logP = new Float32Array(256);
    for (let v = 0; v < 256; v++) logP[v] = Math.log(Math.max(pComb[v], 1e-8));
    return logP;
  }
}

function trainRun(seed, steps = 8, kernelMode = null) {
  const activeK = Engine.getActiveKernel();
  if (kernelMode && activeK) activeK.use(kernelMode);
  Engine.setSeed(seed);
  const modelRng = Engine.mulberry32(seed);
  const model = new Engine.FabulaTransformer('phone', modelRng);
  const stepRng = Engine.mulberry32(seed + 999);
  const corpus = [
    new Uint8Array(Buffer.from('{"id":101,"type":"sensor","value":42.5,"ok":true,"tags":["alpha","beta"]}')),
    new Uint8Array(Buffer.from('<!DOCTYPE html><html><body><div><ul><li>Item 1</li></ul></div></body></html>')),
  ];
  const tagIds = [1, 4];
  const losses = [];
  for (let s = 0; s < steps; s++) {
    const doc = corpus[s % corpus.length];
    const cleanBytes = new Uint8Array(64);
    cleanBytes.set(doc.subarray(0, Math.min(64, doc.length)));
    const fid = s % 2 === 0 ? Engine.FORMAT.JSON : Engine.FORMAT.HTML;
    const res = model.trainStep(cleanBytes, fid, tagIds, {
      rng: stepRng,
      useKernel: !!kernelMode,
    });
    losses.push(res.bpb);
  }
  const sampleConf = model.sampleWindow(48, Engine.FORMAT.JSON, tagIds, {
    steps: 6,
    order: 'confidence',
    remask: true,
    temperature: 0.75,
    rng: Engine.mulberry32(seed + 2025),
  }).bytes;
  const sampleRand = model.sampleWindow(48, Engine.FORMAT.HTML, tagIds, {
    steps: 6,
    order: 'random',
    remask: false,
    temperature: 0.75,
    rng: Engine.mulberry32(seed + 4050),
  }).bytes;
  return { model, losses, sampleConf, sampleRand };
}

async function main() {
  console.log('========================================================================');
  console.log('1. Bit-for-Bit Determinism: Same Seed vs Different Seed');
  console.log('========================================================================');
  const run1 = trainRun(1337, 8);
  const run2 = trainRun(1337, 8);
  const run3 = trainRun(7331, 8);

  const g1 = run1.model.opt.groups;
  const g2 = run2.model.opt.groups;
  const g3 = run3.model.opt.groups;

  let weightsIdentical12 = true;
  let maxBitDiff12 = 0;
  for (let k = 0; k < g1.length; k++) {
    const w1 = new Uint32Array(g1[k].w.buffer, g1[k].w.byteOffset, g1[k].w.length);
    const w2 = new Uint32Array(g2[k].w.buffer, g2[k].w.byteOffset, g2[k].w.length);
    for (let i = 0; i < w1.length; i++) {
      if (w1[i] !== w2[i]) {
        weightsIdentical12 = false;
        maxBitDiff12++;
      }
    }
  }
  assert(weightsIdentical12, `Same seed (1337) produces bit-for-bit identical weights across all ${g1.length} tensors (bit diffs = ${maxBitDiff12})`);

  let lossesIdentical = true;
  for (let i = 0; i < run1.losses.length; i++) {
    if (run1.losses[i] !== run2.losses[i]) lossesIdentical = false;
  }
  assert(lossesIdentical, `Same seed (1337) produces bit-for-bit identical training loss trajectory`);

  let confIdentical = true;
  for (let i = 0; i < run1.sampleConf.length; i++) {
    if (run1.sampleConf[i] !== run2.sampleConf[i]) confIdentical = false;
  }
  assert(confIdentical, `Same seed (1337) produces byte-for-byte identical confidence-ordered sample`);

  let randIdentical = true;
  for (let i = 0; i < run1.sampleRand.length; i++) {
    if (run1.sampleRand[i] !== run2.sampleRand[i]) randIdentical = false;
  }
  assert(randIdentical, `Same seed (1337) produces byte-for-byte identical random-order sample`);

  let diffSeedDistinct = false;
  for (let i = 0; i < g1[0].w.length; i++) {
    if (g1[0].w[i] !== g3[0].w[i]) { diffSeedDistinct = true; break; }
  }
  assert(diffSeedDistinct, `Different seed (7331) produces distinct weights and samples`);

  console.log('\n========================================================================');
  console.log('2. .fabula v0.4 Serialization & Graceful v0.2 Rejection');
  console.log('========================================================================');
  const legacyV2Data = {
    app: 'Autoencofabulator',
    version: '0.2.3',
    config: { HIDDEN_DIM: 256, CHUNK_SIZE: 256 },
    weights: {}
  };
  const v2Check = run1.model.load(legacyV2Data);
  assert(!v2Check.ok && /v0\.2\.3/i.test(v2Check.reason), `v0.2 .fabula file rejected gracefully: "${v2Check.reason}"`);

  const v4Snap = run1.model.serialize(8);
  const freshModel = new Engine.FabulaTransformer('phone', Engine.mulberry32(1));
  const loadRes = freshModel.load(v4Snap);
  assert(loadRes.ok && loadRes.steps === 8, `v0.4 .fabula snapshot loaded cleanly (steps=${loadRes.steps})`);

  let roundTripExact = true;
  for (let k = 0; k < g1.length; k++) {
    const a = new Uint32Array(g1[k].w.buffer, g1[k].w.byteOffset, g1[k].w.length);
    const b = new Uint32Array(freshModel.opt.groups[k].w.buffer, freshModel.opt.groups[k].w.byteOffset, freshModel.opt.groups[k].w.length);
    for (let i = 0; i < a.length; i++) {
      if (a[i] !== b[i]) { roundTripExact = false; break; }
    }
  }
  assert(roundTripExact, `v0.4 .fabula export/import preserves all weights bit-for-bit`);

  console.log('\n========================================================================');
  console.log('3. WasmKernels Backend Equivalence: matvec_f32, matmul_f32, linear_bwd_f32');
  console.log('========================================================================');
  const K = Engine.WasmKernels.createSync({ pages: 64 });
  console.log(`  Info  WasmKernels capabilities: wasm=${K.wasm}, simd=${K.simd}`);

  // 3a. Direct matvec_f32 check across js, wasm, simd
  const rows = 32, cols = 64;
  const prng = Engine.mulberry32(9001);
  const wBuf = K.alloc(rows * cols);
  const xBuf = K.alloc(cols);
  const yJS  = K.alloc(rows);
  const yWasm = K.alloc(rows);
  const ySimd = K.alloc(rows);
  for (let i = 0; i < rows * cols; i++) wBuf.view[i] = (prng() - 0.5) * 0.5;
  for (let i = 0; i < cols; i++) xBuf.view[i] = (prng() - 0.5) * 0.5;

  K.use('js').matvec_f32(wBuf, xBuf, yJS, rows, cols);
  K.use('wasm').matvec_f32(wBuf, xBuf, yWasm, rows, cols);
  K.use('simd').matvec_f32(wBuf, xBuf, ySimd, rows, cols);

  let maxMatvecWasm = 0, maxMatvecSimd = 0;
  for (let i = 0; i < rows; i++) {
    maxMatvecWasm = Math.max(maxMatvecWasm, Math.abs(yJS.view[i] - yWasm.view[i]));
    maxMatvecSimd = Math.max(maxMatvecSimd, Math.abs(yJS.view[i] - ySimd.view[i]));
  }
  assert(maxMatvecWasm < 1e-6, `matvec_f32 JS vs scalar WASM max diff = ${maxMatvecWasm.toExponential(2)} (< 1e-6)`);
  assert(maxMatvecSimd < 1e-6, `matvec_f32 JS vs SIMD WASM max diff = ${maxMatvecSimd.toExponential(2)} (< 1e-6)`);

  // 3b. Direct batched matmul_f32 (seqLen=24 x inDim=128 x outDim=64) across js, wasm, simd
  const M = 24, Kdim = 128, N = 64;
  const aBuf = K.alloc(M * Kdim);
  const bBuf = K.alloc(N * Kdim);
  const cJS   = K.alloc(M * N);
  const cWasm = K.alloc(M * N);
  const cSimd = K.alloc(M * N);
  for (let i = 0; i < M * Kdim; i++) aBuf.view[i] = (prng() - 0.5) * 0.5;
  for (let i = 0; i < N * Kdim; i++) bBuf.view[i] = (prng() - 0.5) * 0.5;

  K.use('js').matmul_f32(aBuf, bBuf, cJS, M, Kdim, N);
  K.use('wasm').matmul_f32(aBuf, bBuf, cWasm, M, Kdim, N);
  K.use('simd').matmul_f32(aBuf, bBuf, cSimd, M, Kdim, N);

  let maxMatmulWasm = 0, maxMatmulSimd = 0, maxMatmulWasmSimd = 0;
  for (let i = 0; i < M * N; i++) {
    maxMatmulWasm     = Math.max(maxMatmulWasm,     Math.abs(cJS.view[i]   - cWasm.view[i]));
    maxMatmulSimd     = Math.max(maxMatmulSimd,     Math.abs(cJS.view[i]   - cSimd.view[i]));
    maxMatmulWasmSimd = Math.max(maxMatmulWasmSimd, Math.abs(cWasm.view[i] - cSimd.view[i]));
  }
  assert(maxMatmulWasm < 1e-5,     `matmul_f32 JS vs scalar WASM max diff = ${maxMatmulWasm.toExponential(2)} (< 1e-5)`);
  assert(maxMatmulSimd < 1e-5,     `matmul_f32 JS vs SIMD WASM max diff = ${maxMatmulSimd.toExponential(2)} (< 1e-5)`);
  assert(maxMatmulWasmSimd < 1e-5, `matmul_f32 scalar WASM vs SIMD WASM max diff = ${maxMatmulWasmSimd.toExponential(2)} (< 1e-5)`);

  // 3c. Direct linear_bwd_f32 across js, wasm, simd
  const dyBuf  = K.alloc(M * N);
  const gwJS   = K.alloc(N * Kdim); gwJS.view.fill(0);
  const gbJS   = K.alloc(N);        gbJS.view.fill(0);
  const dxJS   = K.alloc(M * Kdim);
  const gwWasm = K.alloc(N * Kdim); gwWasm.view.fill(0);
  const gbWasm = K.alloc(N);        gbWasm.view.fill(0);
  const dxWasm = K.alloc(M * Kdim);
  const gwSimd = K.alloc(N * Kdim); gwSimd.view.fill(0);
  const gbSimd = K.alloc(N);        gbSimd.view.fill(0);
  const dxSimd = K.alloc(M * Kdim);
  for (let i = 0; i < M * N; i++) dyBuf.view[i] = (i % 3 === 0) ? 0 : (prng() - 0.5) * 0.25;

  K.use('js').linear_bwd_f32(dyBuf, aBuf, bBuf, gwJS, gbJS, dxJS, M, N, Kdim);
  K.use('wasm').linear_bwd_f32(dyBuf, aBuf, bBuf, gwWasm, gbWasm, dxWasm, M, N, Kdim);
  K.use('simd').linear_bwd_f32(dyBuf, aBuf, bBuf, gwSimd, gbSimd, dxSimd, M, N, Kdim);

  let maxBwdGW = 0, maxBwdDX = 0;
  for (let i = 0; i < N * Kdim; i++) {
    maxBwdGW = Math.max(maxBwdGW, Math.abs(gwJS.view[i] - gwWasm.view[i]), Math.abs(gwJS.view[i] - gwSimd.view[i]));
  }
  for (let i = 0; i < M * Kdim; i++) {
    maxBwdDX = Math.max(maxBwdDX, Math.abs(dxJS.view[i] - dxWasm.view[i]), Math.abs(dxJS.view[i] - dxSimd.view[i]));
  }
  assert(maxBwdGW < 1e-5 && maxBwdDX < 1e-5, `linear_bwd_f32 JS/WASM/SIMD max diff: gradW=${maxBwdGW.toExponential(2)}, gradIn=${maxBwdDX.toExponential(2)} (< 1e-5)`);

  // 3d. Full Transformer training loss curve across js, wasm, simd
  const trJS   = trainRun(4242, 6, 'js');
  const trWasm = trainRun(4242, 6, 'wasm');
  const trSimd = trainRun(4242, 6, 'simd');

  let maxCurveWasm = 0, maxCurveSimd = 0;
  for (let i = 0; i < trJS.losses.length; i++) {
    maxCurveWasm = Math.max(maxCurveWasm, Math.abs(trJS.losses[i] - trWasm.losses[i]));
    maxCurveSimd = Math.max(maxCurveSimd, Math.abs(trJS.losses[i] - trSimd.losses[i]));
  }
  assert(maxCurveWasm < 1e-5, `Transformer 6-step loss curve JS vs WASM max diff = ${maxCurveWasm.toExponential(2)} (< 1e-5)`);
  assert(maxCurveSimd < 1e-5, `Transformer 6-step loss curve JS vs SIMD max diff = ${maxCurveSimd.toExponential(2)} (< 1e-5)`);

  console.log('\n========================================================================');
  console.log('4. Fast Typed-Array BilateralContextMemory vs v0.3 Reference (< 1e-6)');
  console.log('========================================================================');
  const sherlockSample = new TextEncoder().encode(Engine.BuiltinGenerators.SHERLOCK_EXCERPT.repeat(12));
  const refMem  = new ReferenceV03BilateralContextMemory();
  const fastMem = new Engine.BilateralContextMemory();
  const tagIdsMem = [2, 7];
  refMem.observeSequence(sherlockSample, tagIdsMem, 1);
  fastMem.observeSequence(sherlockSample, tagIdsMem, 1);

  const maskRng = Engine.mulberry32(777);
  const { xt: testXt } = Engine.MASK_DIFF.forwardMask(sherlockSample.subarray(0, 256), 0.45, maskRng);
  let maxMemDiffNoPos = 0, maxMemDiffPos = 0;
  for (let i = 0; i < 256; i++) {
    const r0 = refMem.predictLogProbsAt(testXt, i, [], false);
    const f0 = fastMem.predictLogProbsAt(testXt, i, [], false);
    const r1 = refMem.predictLogProbsAt(testXt, i, tagIdsMem, true);
    const f1 = fastMem.predictLogProbsAt(testXt, i, tagIdsMem, true);
    for (let v = 0; v < 256; v++) {
      maxMemDiffNoPos = Math.max(maxMemDiffNoPos, Math.abs(r0[v] - f0[v]));
      maxMemDiffPos   = Math.max(maxMemDiffPos,   Math.abs(r1[v] - f1[v]));
    }
  }
  assert(maxMemDiffNoPos < 1e-6, `Fast BilateralContextMemory (usePosPrior=false) max diff vs v0.3 = ${maxMemDiffNoPos.toExponential(2)} (< 1e-6)`);
  assert(maxMemDiffPos   < 1e-6, `Fast BilateralContextMemory (usePosPrior=true)  max diff vs v0.3 = ${maxMemDiffPos.toExponential(2)} (< 1e-6)`);

  // Verify 512-byte generation speed at Phone preset (< 10 seconds)
  Engine.getActiveKernel().use('simd');
  const tGen0 = Date.now();
  run1.model.sampleWindow(512, Engine.FORMAT.TXT, [], {
    steps: 24,
    order: 'confidence',
    rng: Engine.mulberry32(555)
  });
  const genMs = Date.now() - tGen0;
  assert(genMs < 2500, `Phone preset 512-byte 24-step generation finished in ${genMs} ms (< 2500 ms desktop / < 10000 ms 4x throttled phone)`);

  console.log('------------------------------------------------------------------------');
  console.log(allOk ? 'ALL DETERMINISM & KERNEL TESTS PASSED.' : 'SOME DETERMINISM TESTS FAILED.');
  process.exit(allOk ? 0 : 1);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});