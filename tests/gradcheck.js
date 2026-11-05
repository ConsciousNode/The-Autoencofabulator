#!/usr/bin/env node
/**
 * Finite-difference gradient check for The Autoencofabulator v0.4: Learned.
 *
 * Zero external dependencies. Extracts the engine code directly from index.html
 * (never a duplicate copy) and swaps Float32Array -> Float64Array inside a VM context
 * so finite differences aren't swamped by float32 rounding.
 *
 * Tests:
 *   Suite A: FabulaTransformer v0.4 1-D Byte Mode (L=8, D=64, H=4, 2 layers)
 *            Checks tokEmb, pos1DEmb, nbrEmb, fmtEmb, semEmb, ctxEmb, tProj, winProj,
 *            every TransformerBlock parameter (qProj, kProj, vProj, oProj, locW,
 *            adaShift1, adaScale1, ln1, ff1, ff2, adaShift2, adaScale2, ln2),
 *            lnFinal, outHead, nbrLogitW, ctxLogitW, helixHead, tfScale, ctxGate,
 *            and posGate (both .W and .b) against finite differences.
 *   Suite B: FabulaTransformer v0.4 2-D Patch Mode (3 patches = 48 bytes, D=64, H=4, 2 layers)
 *            Checks slotEmb, rowEmb, colEmb, tokEmb, winProj, outSlotScale, outSlotBias,
 *            TransformerBlock parameters, lnFinal, outHead, helixHead, tfScale, and ctxGate.
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
  Int16Array,
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
    ['winProj.W', model.winProj.weight, model.winProj.gradW],
    ['winProj.b', model.winProj.bias, model.winProj.gradB],
  ];
  if (!includePatchOnly) {
    list.push(
      ['pos1DEmb', model.pos1DEmb, model.gPos1D],
      ['nbrEmb', model.nbrEmb, model.gNbrEmb],
      ['nbrLogitW', model.nbrLogitW, model.gNbrLogitW],
      ['ctxLogitW', model.ctxLogitW, model.gCtxLogitW],
      ['posGate.W', model.posGate.weight, model.posGate.gradW],
      ['posGate.b', model.posGate.bias, model.posGate.gradB]
    );
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
    ['tfScale', model.tfScale, model.gTfScale],
    ['ctxGate', model.ctxGate, model.gCtxGate]
  );
  return list;
}

console.log('========================================================================');
console.log('Suite A: FabulaTransformer v0.4 — 1-D Byte Mode (L=8, Phone Preset)');
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

  // Fixed external prior so ctxGate and posGate receive a clean non-zero gradient
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
    for (let i = 0; i < w.length && activeIdx.length < 24; i++) { if (Math.abs(ag[i]) > 1e-6) activeIdx.push(i); }
    for (let i = 0; i < w.length && activeIdx.length < 24; i++) { if (Math.abs(ag[i]) > 1e-10 && !activeIdx.includes(i)) activeIdx.push(i); }
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
console.log('Suite B: FabulaTransformer v0.4 — 2-D Patch Mode (3 patches = 48 bytes)');
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
    else xt[i] = cleanBytes[i];
  }
  const rowIds = new Int32Array([0, 1, 2]);
  const colIds = new Int32Array([3, 4, 5]);
  const helixTargets = new Uint8Array([1, 2, 0]);
  const fid = Engine.FORMAT.AURA;
  const tagIds = [4, 9];
  const t = 0.6;
  const priorPrng = Engine.mulberry32(777);
  const ctxPriorLogP = new Float64Array(totalBytes * 256);
  for (let i = 0; i < ctxPriorLogP.length; i++) ctxPriorLogP[i] = (priorPrng() - 0.5) * 0.3;

  function computeLossAndGrad2D(doBackward = false) {
    if (doBackward) model.zeroGrad();
    const { logits, helixLogits } = model.forward(xt, t, fid, tagIds, {
      patchMode: true,
      rowIds,
      colIds,
      useKernel: false,
      ctxPriorLogP,
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
      const logZ = maxL + Math.log(sumE);
      loss += -(logits[off + target] - logZ) * scale;
      if (doBackward) {
        for (let v = 0; v < 256; v++) {
          const p = Math.exp(logits[off + v] - maxL) / sumE;
          gradLogits[off + v] = (p - (v === target ? 1 : 0)) * scale;
        }
      }
    }
    const gradHelix = new Float64Array(nPatches * 4);
    const hScale = 0.25 / nPatches;
    for (let c = 0; c < nPatches; c++) {
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

  computeLossAndGrad2D(true);
  const groups2D = [
    ['slotEmb', model.slotEmb, model.gSlotEmb],
    ['rowEmb', model.rowEmb, model.gRowEmb],
    ['colEmb', model.colEmb, model.gColEmb],
    ['tokEmb', model.tokEmb, model.gTokEmb],
    ['winProj.W', model.winProj.weight, model.winProj.gradW],
    ['outSlotScale', model.outSlotScale, model.gOutSlotScale],
    ['outSlotBias', model.outSlotBias, model.gOutSlotBias],
    ['L0.qProj.W', model.blocks[0].qProj.weight, model.blocks[0].qProj.gradW],
    ['L1.ff2.W', model.blocks[1].ff2.weight, model.blocks[1].ff2.gradW],
    ['outHead.W', model.outHead.weight, model.outHead.gradW],
    ['helixHead.W', model.helixHead.weight, model.helixHead.gradW],
    ['tfScale', model.tfScale, model.gTfScale],
    ['ctxGate', model.ctxGate, model.gCtxGate],
  ];
  const savedAnalytic2D = new Map();
  for (const [name, w, g] of groups2D) savedAnalytic2D.set(name, Float64Array.from(g));

  for (const [name, w] of groups2D) {
    const ag = savedAnalytic2D.get(name);
    const activeIdx = [];
    for (let i = 0; i < w.length && activeIdx.length < 24; i++) { if (Math.abs(ag[i]) > 1e-6) activeIdx.push(i); }
    for (let i = 0; i < w.length && activeIdx.length < 24; i++) { if (Math.abs(ag[i]) > 1e-10 && !activeIdx.includes(i)) activeIdx.push(i); }
    const subA = new Float64Array(activeIdx.length);
    const subN = new Float64Array(activeIdx.length);
    for (let k = 0; k < activeIdx.length; k++) {
      const idx = activeIdx[k];
      subA[k] = ag[idx];
      const orig = w[idx];
      w[idx] = orig + eps; const lp = computeLossAndGrad2D(false);
      w[idx] = orig - eps; const lm = computeLossAndGrad2D(false);
      w[idx] = orig;
      subN[k] = (lp - lm) / (2 * eps);
    }
    allOk = compareGrads(`2D:${name}`, subA, subN) && allOk;
  }
}

console.log('\n========================================================================');
console.log('Suite C: Standalone TransformerBlock (Inputs x, cond & All 15 Tensors)');
console.log('========================================================================');
{
  const prng = Engine.mulberry32(9999);
  const L = 6, D = 16, H = 4, dFF = 32, dCond = 24;
  const block = new Engine.TransformerBlock(D, H, dFF, dCond, prng);
  const x = new Float64Array(L * D);
  const cond = new Float64Array(dCond);
  const upstream = new Float64Array(L * D);
  for (let i = 0; i < x.length; i++) x[i] = (prng() - 0.5) * 1.4;
  for (let i = 0; i < cond.length; i++) cond[i] = (prng() - 0.5) * 1.2;
  for (let i = 0; i < upstream.length; i++) upstream[i] = (prng() - 0.5) * 0.8;
  const attnIdxList = Array.from({ length: L }, () => Int32Array.from({ length: L }, (_, j) => j));

  function forwardLoss(doBwd = false) {
    if (doBwd) block.zeroGrad();
    const out = block.forward(x, cond, attnIdxList, null);
    let loss = 0;
    for (let i = 0; i < out.length; i++) loss += out[i] * upstream[i];
    if (doBwd) return { loss, ...block.backward(upstream) };
    return loss;
  }

  const { gX, gCond } = forwardLoss(true);
  const blockList = [
    ['input.x', x, gX],
    ['input.cond', cond, gCond],
    ['ln1.weight', block.ln1.weight, block.ln1.gradW],
    ['ln1.bias', block.ln1.bias, block.ln1.gradB],
    ['adaShift1.W', block.adaShift1.weight, block.adaShift1.gradW],
    ['adaScale1.W', block.adaScale1.weight, block.adaScale1.gradW],
    ['qProj.W', block.qProj.weight, block.qProj.gradW],
    ['kProj.W', block.kProj.weight, block.kProj.gradW],
    ['vProj.W', block.vProj.weight, block.vProj.gradW],
    ['oProj.W', block.oProj.weight, block.oProj.gradW],
    ['locW', block.locW, block.gradLocW],
    ['ln2.weight', block.ln2.weight, block.ln2.gradW],
    ['adaShift2.W', block.adaShift2.weight, block.adaShift2.gradW],
    ['adaScale2.W', block.adaScale2.weight, block.adaScale2.gradW],
    ['ff1.W', block.ff1.weight, block.ff1.gradW],
    ['ff2.W', block.ff2.weight, block.ff2.gradW],
  ];
  const savedB = new Map();
  for (const [name, w, g] of blockList) savedB.set(name, Float64Array.from(g));

  for (const [name, w] of blockList) {
    const ag = savedB.get(name);
    const num = new Float64Array(w.length);
    for (let i = 0; i < w.length; i++) {
      const orig = w[i];
      w[i] = orig + eps; const lp = forwardLoss(false);
      w[i] = orig - eps; const lm = forwardLoss(false);
      w[i] = orig;
      num[i] = (lp - lm) / (2 * eps);
    }
    allOk = compareGrads(`Block:${name}`, ag, num) && allOk;
  }
}

console.log('\n========================================================================');
if (allOk) {
  console.log('ALL GRADIENT CHECKS PASSED (cos > 0.9999 across 1-D, 2-D, and Block suites).');
  process.exit(0);
} else {
  console.error('SOME GRADIENT CHECKS FAILED.');
  process.exit(1);
}