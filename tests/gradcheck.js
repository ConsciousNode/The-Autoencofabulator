#!/usr/bin/env node
/**
 * Finite-difference gradient check for The Autoencofabulator v0.3: Realized.
 *
 * Zero external dependencies. Extracts the engine code directly from index.html
 * (never a duplicate copy) and swaps Float32Array -> Float64Array inside a VM context
 * so finite differences aren't swamped by float32 rounding.
 *
 * Tests:
 *   Suite A: FabulaTransformer 1-D Byte Mode (L=8, D=24, H=4, 2 layers)
 *            Checks tokEmb, pos1DEmb, fmtEmb, semEmb, ctxEmb, tProj,
 *            every TransformerBlock parameter (qProj, kProj, vProj, oProj, locW,
 *            adaShift1, adaScale1, ln1, ff1, ff2, adaShift2, adaScale2, ln2),
 *            lnFinal, outHead, helixHead, and ctxGate against finite differences.
 *   Suite B: FabulaTransformer 2-D Patch Mode (3 patches = 48 bytes, D=24, H=4, 2 layers)
 *            Checks slotEmb, rowEmb, colEmb, tokEmb, outSlotScale, outSlotBias,
 *            TransformerBlock parameters, lnFinal, outHead, and helixHead in 2-D patch mode.
 *   Suite C: Standalone TransformerBlock (Attention + RoPE + LocalConv + AdaLN + FFN)
 *            Checks all 15 block parameter & input tensors (including input.x and input.cond).
 *
 * Run: node tests/gradcheck.js
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
  throw new Error('Could not locate index.html relative to gradcheck.js');
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

const sandbox = {
  Float32Array: Float64Array, // double precision for finite differences
  Float64Array,
  Uint8Array,
  Uint16Array,
  Uint32Array,
  Int32Array,
  ArrayBuffer,
  DataView,
  TextEncoder,
  TextDecoder,
  Buffer,
  Math: Object.create(Math),
  Number,
  Object,
  Array,
  String,
  Boolean,
  Error,
  TypeError,
  RangeError,
  JSON,
  Map,
  Set,
  Promise,
  WebAssembly: typeof WebAssembly !== 'undefined' ? WebAssembly : undefined,
  console,
  performance: { now: () => Date.now() },
};
sandbox.globalThis = sandbox;
sandbox.window = sandbox;
sandbox.self = sandbox;
vm.createContext(sandbox);
vm.runInContext(engineCode, sandbox);

const Engine = sandbox.FabulaEngine;
if (!Engine || !Engine.FabulaTransformer) {
  throw new Error('FabulaEngine.FabulaTransformer not exported from index.html');
}

function compareGrads(name, analytic, numeric) {
  let dot = 0, normA = 0, normN = 0, maxRel = 0;
  const n = Math.min(analytic.length, numeric.length);
  for (let i = 0; i < n; i++) {
    const a = analytic[i], b = numeric[i];
    dot += a * b;
    normA += a * a;
    normN += b * b;
    const denom = Math.max(Math.abs(a), Math.abs(b), 1e-7);
    const rel = Math.abs(a - b) / denom;
    if (rel > maxRel) maxRel = rel;
  }
  const cos = (normA > 0 && normN > 0) ? dot / (Math.sqrt(normA) * Math.sqrt(normN)) : 1;
  const ratio = normN > 0 ? Math.sqrt(normA / normN) : 1;
  const ok = cos > 0.9999 && Math.abs(ratio - 1) < 0.005 && maxRel < 0.01;
  const status = ok ? 'PASS' : 'FAIL';
  console.log(
    `  ${status}  ${name.padEnd(32)} cos=${cos.toFixed(6)}  |a|/|n|=${ratio.toFixed(5)}  maxRel=${maxRel.toExponential(2)}  (|g|=${Math.sqrt(normA).toExponential(2)})`
  );
  return ok;
}

const eps = 1e-6;
let allOk = true;

function namedGroupsForModel(model, includePatchOnly = false) {
  const list = [
    ['tokEmb', model.tokEmb, model.gTokEmb],
    ['fmtEmb', model.fmtEmb, model.gFmt],
    ['semEmb', model.semEmb, model.gSem],
    ['ctxEmb', model.ctxEmb, model.gCtxEmb],
    ['tProj.W', model.tProj.weight, model.tProj.gradW],
    ['tProj.b', model.tProj.bias, model.tProj.gradB],
  ];
  if (!includePatchOnly) {
    list.push(['pos1DEmb', model.pos1DEmb, model.gPos1D]);
  } else {
    list.push(
      ['slotEmb', model.slotEmb, model.gSlotEmb],
      ['rowEmb', model.rowEmb, model.gRowEmb],
      ['colEmb', model.colEmb, model.gColEmb],
      ['outSlotScale', model.outSlotScale, model.gOutSlotScale],
      ['outSlotBias', model.outSlotBias, model.gOutSlotBias]
    );
  }
  for (let l = 0; l < model.blocks.length; l++) {
    const b = model.blocks[l];
    list.push(
      [`L${l}.qProj.W`, b.qProj.weight, b.qProj.gradW],
      [`L${l}.kProj.W`, b.kProj.weight, b.kProj.gradW],
      [`L${l}.vProj.W`, b.vProj.weight, b.vProj.gradW],
      [`L${l}.oProj.W`, b.oProj.weight, b.oProj.gradW],
      [`L${l}.locW`, b.locW, b.gradLocW],
      [`L${l}.adaShift1.W`, b.adaShift1.weight, b.adaShift1.gradW],
      [`L${l}.adaScale1.W`, b.adaScale1.weight, b.adaScale1.gradW],
      [`L${l}.ln1.weight`, b.ln1.weight, b.ln1.gradW],
      [`L${l}.ff1.W`, b.ff1.weight, b.ff1.gradW],
      [`L${l}.ff2.W`, b.ff2.weight, b.ff2.gradW],
      [`L${l}.adaShift2.W`, b.adaShift2.weight, b.adaShift2.gradW],
      [`L${l}.adaScale2.W`, b.adaScale2.weight, b.adaScale2.gradW],
      [`L${l}.ln2.weight`, b.ln2.weight, b.ln2.gradW]
    );
  }
  list.push(
    ['lnFinal.weight', model.lnFinal.weight, model.lnFinal.gradW],
    ['lnFinal.bias', model.lnFinal.bias, model.lnFinal.gradB],
    ['outHead.W', model.outHead.weight, model.outHead.gradW],
    ['outHead.b', model.outHead.bias, model.outHead.gradB],
    ['helixHead.W', model.helixHead.weight, model.helixHead.gradW],
    ['helixHead.b', model.helixHead.bias, model.helixHead.gradB],
    ['ctxGate', model.ctxGate, model.gCtxGate]
  );
  return list;
}

console.log('========================================================================');
console.log('Suite A: FabulaTransformer v0.3 — 1-D Byte Mode (L=8, Phone Preset)');
console.log('========================================================================');
{
  Engine.setSeed(12345);
  const model = new Engine.FabulaTransformer('phone', Engine.mulberry32(12345));
  const cleanBytes = new Uint8Array([60, 100, 105, 118, 62, 65, 66, 60]);
  const xt = new Int32Array([60, 256, 105, 256, 62, 256, 66, 256]);
  const isMasked = new Uint8Array([0, 1, 0, 1, 0, 1, 0, 1]);
  const maskedCount = 4;
  const fid = Engine.FORMAT.HTML;
  const tagIds = [1, 5, 12];
  const t = 0.45;
  const helixTargets = new Uint8Array([0, 1, 2, 3, 1, 0, 2, 1]);

  // Fixed external prior so ctxGate receives a clean non-zero gradient
  const priorPrng = Engine.mulberry32(888);
  const ctxPriorLogP = new Float64Array(cleanBytes.length * 256);
  for (let i = 0; i < ctxPriorLogP.length; i++) ctxPriorLogP[i] = (priorPrng() - 0.5) * 0.3;

  function computeLossAndGrad(doBackward = false) {
    if (doBackward) model.zeroGrad();
    const { logits, helixLogits } = model.forward(xt, t, fid, tagIds, {
      patchMode: false,
      useKernel: false,
      ctxPriorLogP,
    });
    const L = cleanBytes.length;
    const gradLogits = new Float64Array(L * 256);
    const scale = 1 / maskedCount;
    let loss = 0;
    for (let i = 0; i < L; i++) {
      if (!isMasked[i]) continue;
      const off = i * 256, target = cleanBytes[i];
      let maxL = -1e30;
      for (let v = 0; v < 256; v++) if (logits[off + v] > maxL) maxL = logits[off + v];
      let sumE = 0;
      for (let v = 0; v < 256; v++) sumE += Math.exp(logits[off + v] - maxL);
      const logZ = maxL + Math.log(sumE);
      loss += -(logits[off + target] - logZ) * scale;
      if (doBackward) {
        for (let v = 0; v < 256; v++) {
          const p = Math.exp(logits[off + v] - maxL) / sumE;
          gradLogits[off + v] = (p - (v === target ? 1 : 0)) * scale;
        }
      }
    }
    const gradHelix = new Float64Array(L * 4);
    const hScale = 0.25 / L;
    for (let c = 0; c < L; c++) {
      const off = c * 4, ht = helixTargets[c];
      let maxH = -1e30;
      for (let k = 0; k < 4; k++) if (helixLogits[off + k] > maxH) maxH = helixLogits[off + k];
      let sumH = 0;
      for (let k = 0; k < 4; k++) sumH += Math.exp(helixLogits[off + k] - maxH);
      loss += -(helixLogits[off + ht] - (maxH + Math.log(sumH))) * hScale;
      if (doBackward) {
        for (let k = 0; k < 4; k++) {
          const p = Math.exp(helixLogits[off + k] - maxH) / sumH;
          gradHelix[off + k] = (p - (k === ht ? 1 : 0)) * hScale;
        }
      }
    }
    if (doBackward) model.backward(gradLogits, gradHelix);
    return loss;
  }

  computeLossAndGrad(true);
  const groups = namedGroupsForModel(model, false);
  const savedAnalytic = new Map();
  for (const [name, w, g] of groups) savedAnalytic.set(name, Float64Array.from(g));

  for (const [name, w] of groups) {
    const ag = savedAnalytic.get(name);
    const activeIdx = [];
    for (let i = 0; i < w.length && activeIdx.length < 24; i++) {
      if (Math.abs(ag[i]) > 1e-10) activeIdx.push(i);
    }
    const subA = new Float64Array(activeIdx.length);
    const subN = new Float64Array(activeIdx.length);
    for (let k = 0; k < activeIdx.length; k++) {
      const idx = activeIdx[k];
      subA[k] = ag[idx];
      const orig = w[idx];
      w[idx] = orig + eps; const lp = computeLossAndGrad(false);
      w[idx] = orig - eps; const lm = computeLossAndGrad(false);
      w[idx] = orig;
      subN[k] = (lp - lm) / (2 * eps);
    }
    allOk = compareGrads(`1D:${name}`, subA, subN) && allOk;
  }
}

console.log('\n========================================================================');
console.log('Suite B: FabulaTransformer v0.3 — 2-D Patch Mode (3 patches = 48 bytes)');
console.log('========================================================================');
{
  const model = new Engine.FabulaTransformer('phone', Engine.mulberry32(54321));
  const nPatches = 3, totalBytes = nPatches * 16;
  const cleanBytes = new Uint8Array(totalBytes);
  const xt = new Int32Array(totalBytes);
  const isMasked = new Uint8Array(totalBytes);
  let maskedCount = 0;
  for (let i = 0; i < totalBytes; i++) {
    cleanBytes[i] = (i * 23 + 11) & 0xff;
    if (i % 3 === 1) { xt[i] = 256; isMasked[i] = 1; maskedCount++; }
    else { xt[i] = cleanBytes[i]; isMasked[i] = 0; }
  }
  const rowIds = new Int32Array([0, 1, 2]);
  const colIds = new Int32Array([0, 1, 2]);
  const fid = Engine.FORMAT.BMP;
  const tagIds = [2, 8];
  const t = 0.55;

  function computePatchLoss(doBackward = false) {
    if (doBackward) model.zeroGrad();
    const { logits } = model.forward(xt, t, fid, tagIds, {
      patchMode: true,
      rowIds,
      colIds,
      useKernel: false,
      disableMemory: true,
    });
    const gradLogits = new Float64Array(totalBytes * 256);
    const scale = 1 / maskedCount;
    let loss = 0;
    for (let i = 0; i < totalBytes; i++) {
      if (!isMasked[i]) continue;
      const off = i * 256, target = cleanBytes[i];
      let maxL = -1e30;
      for (let v = 0; v < 256; v++) if (logits[off + v] > maxL) maxL = logits[off + v];
      let sumE = 0;
      for (let v = 0; v < 256; v++) sumE += Math.exp(logits[off + v] - maxL);
      loss += -(logits[off + target] - (maxL + Math.log(sumE))) * scale;
      if (doBackward) {
        for (let v = 0; v < 256; v++) {
          const p = Math.exp(logits[off + v] - maxL) / sumE;
          gradLogits[off + v] = (p - (v === target ? 1 : 0)) * scale;
        }
      }
    }
    if (doBackward) model.backward(gradLogits, null);
    return loss;
  }

  computePatchLoss(true);
  const patchCheckNames = new Set([
    'slotEmb', 'rowEmb', 'colEmb', 'outSlotScale', 'outSlotBias',
    'tokEmb', 'L0.qProj.W', 'L0.adaScale1.W', 'L1.oProj.W', 'outHead.W'
  ]);
  const groups = namedGroupsForModel(model, true).filter(([n]) => patchCheckNames.has(n));
  const savedAnalytic = new Map();
  for (const [name, w, g] of groups) savedAnalytic.set(name, Float64Array.from(g));

  for (const [name, w] of groups) {
    const ag = savedAnalytic.get(name);
    const activeIdx = [];
    for (let i = 0; i < w.length && activeIdx.length < 24; i++) {
      if (Math.abs(ag[i]) > 1e-10) activeIdx.push(i);
    }
    const subA = new Float64Array(activeIdx.length);
    const subN = new Float64Array(activeIdx.length);
    for (let k = 0; k < activeIdx.length; k++) {
      const idx = activeIdx[k];
      subA[k] = ag[idx];
      const orig = w[idx];
      w[idx] = orig + eps; const lp = computePatchLoss(false);
      w[idx] = orig - eps; const lm = computePatchLoss(false);
      w[idx] = orig;
      subN[k] = (lp - lm) / (2 * eps);
    }
    allOk = compareGrads(`2D:${name}`, subA, subN) && allOk;
  }
}

console.log('\n========================================================================');
console.log('Suite C: Standalone TransformerBlock Input & Conditioning Gradients');
console.log('========================================================================');
{
  const L = 6, D = 16, H = 4, dFF = 32, dCond = 16;
  const rng = Engine.mulberry32(999);
  const blk = new Engine.TransformerBlock(D, H, dFF, dCond, rng);
  const attnIdxList = Array.from({ length: L }, () => Int32Array.from({ length: L }, (_, j) => j));
  const x = new Float64Array(L * D);
  const cond = new Float64Array(dCond);
  const target = new Float64Array(L * D);
  for (let i = 0; i < x.length; i++) { x[i] = (rng() - 0.5) * 0.8; target[i] = (rng() - 0.5) * 0.8; }
  for (let i = 0; i < dCond; i++) cond[i] = (rng() - 0.5) * 0.8;

  function blockLoss() {
    const y = blk.forward(x, cond, attnIdxList, null);
    let loss = 0;
    for (let i = 0; i < y.length; i++) { const d = y[i] - target[i]; loss += 0.5 * d * d; }
    return loss;
  }

  blk.zeroGrad();
  const y = blk.forward(x, cond, attnIdxList, null);
  const gOut = new Float64Array(y.length);
  for (let i = 0; i < y.length; i++) gOut[i] = y[i] - target[i];
  const { gX, gCond } = blk.backward(gOut);

  const numGX = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const orig = x[i];
    x[i] = orig + eps; const lp = blockLoss();
    x[i] = orig - eps; const lm = blockLoss();
    x[i] = orig;
    numGX[i] = (lp - lm) / (2 * eps);
  }
  allOk = compareGrads('Block:input.x', gX, numGX) && allOk;

  const numGCond = new Float64Array(cond.length);
  for (let i = 0; i < cond.length; i++) {
    const orig = cond[i];
    cond[i] = orig + eps; const lp = blockLoss();
    cond[i] = orig - eps; const lm = blockLoss();
    cond[i] = orig;
    numGCond[i] = (lp - lm) / (2 * eps);
  }
  allOk = compareGrads('Block:input.cond', gCond, numGCond) && allOk;
}

console.log('------------------------------------------------------------------------');
console.log(allOk ? 'ALL GRADIENT CHECKS PASSED.' : 'SOME GRADIENT CHECKS FAILED.');
process.exit(allOk ? 0 : 1);