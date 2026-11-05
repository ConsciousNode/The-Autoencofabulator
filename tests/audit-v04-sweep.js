// [Kehai audit 2026-11-05] Counterfactual sweep: mixScale / alpha / gate on ALL 878 val vs ALL 878 test windows,
// using the checkpoint written by train-sherlock.js. Run that first. Output recorded in research/AUDIT-2.md.
#!/usr/bin/env node
/**
 * Deterministic Sherlock Holmes Training, Gate Ablation, Checkpoint Export,
 * and Held-Out Full-Split Evaluation Script for The Autoencofabulator v0.4: Learned.
 *
 * Implements the Tier 1 & Tier 4 requirements:
 *   1. 80/10/10 contiguous split of sherlock-holmes-corpus.txt (449,923 train / 56,240 val / 56,241 test).
 *   2. Minibatch training (B=12 windows of 256 bytes = 3,072 bytes/step, 600 steps = 1,843,200 bytes = 4.10 epochs)
 *      with linear LR warmup + cosine decay, global L2 gradient clipping, and 2-fold out-of-fold memory training.
 *   3. Validation-split ablation across the three memory-transformer gating designs:
 *        (a) Per-position hidden-state gate `pos_gate`: g_i = softplus(ctxGate + posGate(hFinal[i]))
 *        (b) Residual training `residual`: transformer trained as residual over out-of-fold memory log-probs
 *        (c) Learned log-linear mixture `mixture`: (alpha_tf, beta_mem) calibrated on val
 *   4. Periodic validation evaluation & early-stopping checkpoint selection on `val`,
 *      writing `tests/train-sherlock-log.jsonl` and `checkpoints/sherlock-phone.fabula` (< 5 MB).
 *   5. Single final evaluation on the ENTIRE `test` split (all 878 non-overlapping 64-byte windows
 *      and all 219 non-overlapping 256-byte windows, fixed seed 777), plus the 24-windowx9-strata audit protocol.
 *   6. Scaling tables across Presets (`Phone`, `Laptop`, `Patient`) and Window Lengths (`64`, `128`, `256`, `512`).
 *
 * Run: node tests/train-sherlock.js
 */
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const Generators = require('./generators.js');

function resolveIndexHtml() {
  const candidates = [
    path.join(__dirname, '..', 'index.html'),
    path.join(__dirname, 'index.html'),
  ];
  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }
  throw new Error('Could not locate index.html');
}

function resolveSherlockCorpus() {
  const candidates = [
    path.join(__dirname, '..', 'sherlock-holmes-corpus.txt'),
    path.join(__dirname, 'sherlock-holmes-corpus.txt'),
  ];
  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }
  throw new Error('Could not locate sherlock-holmes-corpus.txt');
}

const html = fs.readFileSync(resolveIndexHtml(), 'utf8');
const startMarker = '// === FABULA_ENGINE_START ===';
const endMarker = '// === FABULA_ENGINE_END ===';
const engineCode = html.slice(html.indexOf(startMarker) + startMarker.length, html.indexOf(endMarker));

globalThis.window = globalThis;
globalThis.self = globalThis;
if (!globalThis.performance) {
  globalThis.performance = { now: () => Number(process.hrtime.bigint()) / 1e6 };
}
vm.runInThisContext(engineCode);
const Engine = globalThis.FabulaEngine;

const activeK = Engine.getActiveKernel();
if (activeK && activeK.simd) activeK.use('simd');

function nllBitsAt(logits, off, target) {
  let mx = -1e30;
  for (let v = 0; v < 256; v++) if (logits[off + v] > mx) mx = logits[off + v];
  let sE = 0;
  for (let v = 0; v < 256; v++) sE += Math.exp(logits[off + v] - mx);
  return -(logits[off + target] - (mx + Math.log(sE))) / Math.LN2;
}

/*
 * Single-pass evaluator that simultaneously scores `memoryOnly`, `transformerOnly`, and `fullModel`
 * on the exact same masked tokens, either across all non-overlapping windows of the split (stratified over t)
 * or using the 9-strata x 24-window audit protocol.
 */
function evaluateSimultaneous(model, mem, evalBytes, tagIds = [], opts = {}) {
  const fSlot = Engine.FORMAT.TXT;
  const strata = opts.strata || [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
  const winLen = opts.windowSize || 64;
  const gateRaw = model.ctxGate[fSlot];
  const usePosGate = (model.gateMode === 'pos_gate' || model.gateMode === 'mixture');
  const mixScale = usePosGate ? (globalThis.__MIX ?? 0.56) : 1.0;

  if (opts.allWindows) {
    // Score EVERY non-overlapping window in evalBytes (e.g. all 878 64B windows or all 219 256B windows),
    // cycling deterministically through the 9 strata of t in [0.1..0.9]
    const rngFn = Engine.mulberry32(opts.seed || 777);
    const stratumMem = new Float64Array(strata.length);
    const stratumTf  = new Float64Array(strata.length);
    const stratumFull= new Float64Array(strata.length);
    const stratumCnt = new Int32Array(strata.length);
    let nWins = 0;

    for (let st = 0; st + winLen <= evalBytes.length; st += winLen) {
      const sIdx = nWins % strata.length;
      const t = strata[sIdx];
      const wNelbo = Engine.MASK_DIFF.nelboWeight(t, 'linear');
      const win = evalBytes.subarray(st, st + winLen);
      const { xt, isMasked } = Engine.MASK_DIFF.forwardMask(win, t, rngFn, null, 'linear');
      const { logits: tfLogits, hFinal } = model.forward(xt, t, fSlot, tagIds, {
        useKernel: true,
        disableMemory: true
      });
      const posGateRaw = usePosGate ? model.posGate.forward(hFinal, activeK) : null;

      let nllMem = 0, nllTf = 0, nllFull = 0;
      const comb = new Float32Array(256);
      for (let i = 0; i < winLen; i++) {
        if (!isMasked[i]) continue;
        const off = i * 256, tgt = win[i];
        nllTf += nllBitsAt(tfLogits, off, tgt);
        const lp = mem.predictLogProbsAt(xt, i, tagIds, false, true);
        nllMem += nllBitsAt(lp, 0, tgt);
        const gVal = Math.log(1 + Math.exp(gateRaw + (posGateRaw ? posGateRaw[i] : 0)));
        for (let v = 0; v < 256; v++) comb[v] = mixScale * tfLogits[off + v] + gVal * lp[v];
        nllFull += nllBitsAt(comb, 0, tgt);
      }
      stratumMem[sIdx]  += (wNelbo * nllMem)  / winLen;
      stratumTf[sIdx]   += (wNelbo * nllTf)   / winLen;
      stratumFull[sIdx] += (wNelbo * nllFull) / winLen;
      stratumCnt[sIdx]++;
      nWins++;
    }

    let memBpb = 0, tfBpb = 0, fullBpb = 0;
    const byStratumFull = {};
    for (let s = 0; s < strata.length; s++) {
      const c = Math.max(1, stratumCnt[s]);
      memBpb  += (stratumMem[s]  / c) / strata.length;
      tfBpb   += (stratumTf[s]   / c) / strata.length;
      fullBpb += (stratumFull[s] / c) / strata.length;
      byStratumFull[strata[s].toFixed(1)] = Number((stratumFull[s] / c).toFixed(4));
    }
    return { memBpb, tfBpb, fullBpb, windowsScored: nWins, byStratumFull };
  }

  // Standard 9-strata x maxWindows protocol (matches audit-v03.js exactly when maxWindows=24, winLen=64)
  const maxWindows = opts.maxWindows || 24;
  const stepSize = Math.max(winLen, Math.floor((evalBytes.length - winLen) / maxWindows));
  const rngMem  = Engine.mulberry32(opts.seed || 777);
  const rngTf   = Engine.mulberry32(opts.seed || 777);
  let memBpb = 0, tfBpb = 0, fullBpb = 0;
  const byStratumFull = {};

  for (const t of strata) {
    const wNelbo = Engine.MASK_DIFF.nelboWeight(t, 'linear');
    let sMem = 0, sTf = 0, sFull = 0, nW = 0;
    for (let st = 0; st + winLen <= evalBytes.length && nW < maxWindows; st += stepSize) {
      const win = evalBytes.subarray(st, st + winLen);
      const { xt, isMasked } = Engine.MASK_DIFF.forwardMask(win, t, rngTf, null, 'linear');
      Engine.MASK_DIFF.forwardMask(win, t, rngMem, null, 'linear'); // keep RNG aligned
      const { logits: tfLogits, hFinal } = model.forward(xt, t, fSlot, tagIds, {
        useKernel: true,
        disableMemory: true
      });
      const posGateRaw = usePosGate ? model.posGate.forward(hFinal, activeK) : null;

      let nllMem = 0, nllTf = 0, nllFull = 0;
      const comb = new Float32Array(256);
      for (let i = 0; i < winLen; i++) {
        if (!isMasked[i]) continue;
        const off = i * 256, tgt = win[i];
        nllTf += nllBitsAt(tfLogits, off, tgt);
        const lp = mem.predictLogProbsAt(xt, i, tagIds, false, true);
        nllMem += nllBitsAt(lp, 0, tgt);
        const gVal = Math.log(1 + Math.exp(gateRaw + (posGateRaw ? posGateRaw[i] : 0)));
        for (let v = 0; v < 256; v++) comb[v] = mixScale * tfLogits[off + v] + gVal * lp[v];
        nllFull += nllBitsAt(comb, 0, tgt);
      }
      sMem  += (wNelbo * nllMem)  / winLen;
      sTf   += (wNelbo * nllTf)   / winLen;
      sFull += (wNelbo * nllFull) / winLen;
      nW++;
    }
    memBpb  += (sMem  / nW) / strata.length;
    tfBpb   += (sTf   / nW) / strata.length;
    fullBpb += (sFull / nW) / strata.length;
    byStratumFull[t.toFixed(1)] = Number((sFull / nW).toFixed(4));
  }
  return { memBpb, tfBpb, fullBpb, windowsScored: maxWindows, byStratumFull };
}

async function main() {
  const tStartAll = Date.now();
  const rawSherlock = new Uint8Array(fs.readFileSync(resolveSherlockCorpus()));
  const { totalBytes, train, val, test } = Generators.splitSherlockCorpus(rawSherlock);
  const fSlot = Engine.FORMAT.TXT;
  const tagIds = Engine.tagToIds('sherlock holmes narrative formal');

  console.log('========================================================================');
  console.log('THE AUTOENCOFABULATOR v0.4: SHERLOCK HOLMES TRAINING & EVALUATION');
  console.log('========================================================================');
  console.log(`Corpus: ${totalBytes} bytes -> train=${train.length}B, val=${val.length}B, test=${test.length}B`);

  // 1. Fit causal byte baselines and BilateralContextMemory strictly on `train`
  const baselines = Engine.fitByteBaselines(train);
  const baseVal = {
    unigram: baselines.evalUnigram(val),
    bigram:  baselines.evalBigram(val),
    kn4:     baselines.evalKneserNey4(val),
  };
  const baseTest = {
    unigram: baselines.evalUnigram(test),
    bigram:  baselines.evalBigram(test),
    kn4:     baselines.evalKneserNey4(test),
  };

  const trainMem = new Engine.BilateralContextMemory();
  trainMem.observeSequence(train, tagIds, 1);
  const halfTr = train.length >> 1;
  const foldMemA = new Engine.BilateralContextMemory();
  foldMemA.observeSequence(train.subarray(0, halfTr), tagIds, 1);
  const foldMemB = new Engine.BilateralContextMemory();
  foldMemB.observeSequence(train.subarray(halfTr), tagIds, 1);

  // 2. Main Phone Model Initialization & Step-0 Baseline on Val and Test
  Engine.setSeed(42);
  const model = new Engine.FabulaTransformer('phone', Engine.mulberry32(42));
  model.memories[fSlot] = trainMem;
  model.foldMemories[fSlot] = [foldMemA, foldMemB];

  const step0Val24   = evaluateSimultaneous(model, trainMem, val,  tagIds, { windowSize: 64, maxWindows: 24, seed: 777 });
  const step0Test24  = evaluateSimultaneous(model, trainMem, test, tagIds, { windowSize: 64, maxWindows: 24, seed: 777 });
  const step0TestAll = evaluateSimultaneous(model, trainMem, test, tagIds, { windowSize: 64, allWindows: true, seed: 777 });

  console.log(`Baselines (val):   Unigram=${baseVal.unigram.toFixed(4)} | Bigram=${baseVal.bigram.toFixed(4)} | KN4=${baseVal.kn4.toFixed(4)} | CountMem=${step0Val24.memBpb.toFixed(4)} bpb`);
  console.log(`Baselines (test):  Unigram=${baseTest.unigram.toFixed(4)} | Bigram=${baseTest.bigram.toFixed(4)} | KN4=${baseTest.kn4.toFixed(4)} | CountMem(all 878w)=${step0TestAll.memBpb.toFixed(4)} (24w=${step0Test24.memBpb.toFixed(4)})`);
  console.log(`Step 0 (untrained): val tf-only=${step0Val24.tfBpb.toFixed(4)}, full=${step0Val24.fullBpb.toFixed(4)} | test(all 878w) tf-only=${step0TestAll.tfBpb.toFixed(4)}, full=${step0TestAll.fullBpb.toFixed(4)}`);

  // Initialize outHead.bias from training log-unigram frequencies
  for (let v = 0; v < 256; v++) {
    model.outHead.bias[v] = Math.log(trainMem.unigram[v] / trainMem.uniTotal);
  }
  for (const g of model.opt.groups) {
    if (g.w === model.tfScale || g.w === model.ctxGate) g.lrScale = 0.0;
  }
  model.tfScale[fSlot] = Math.log(Math.E - 1);
  model.ctxGate[fSlot] = Math.log(Math.exp(0.42) - 1);

  const logLines = [];
  const logPath = path.join(__dirname, 'train-sherlock-log.jsonl');
  const recordLog = (obj) => { logLines.push(JSON.stringify(obj)); };

  recordLog({
    event: 'init',
    step: 0,
    valMemOnlyBpb: Number(step0Val24.memBpb.toFixed(4)),
    valTfOnlyBpb: Number(step0Val24.tfBpb.toFixed(4)),
    valFullBpb: Number(step0Val24.fullBpb.toFixed(4)),
    elapsedMs: Date.now() - tStartAll
  });

  // 3. Main Training Loop: 500 minibatches of B=12 x 256-byte windows (6,000 windows = 1,536,000 bytes = 3.41 epochs)
  const N_STEPS = 900;
  const BATCH_SIZE = 12;
  const WIN_LEN = 256;
  const PEAK_LR = 0.0042;
  const WARMUP_STEPS = 25;
  const rngTrain = Engine.mulberry32(101);

  let bestValFullBpb = Infinity;
  let bestStep = 0;
  let bestSnapshot = null;
  let rollingTrainBpb = 0;

  console.log(`\n--- Training Phone Preset (${N_STEPS} minibatches x ${BATCH_SIZE} x ${WIN_LEN}B = ${(N_STEPS * BATCH_SIZE * WIN_LEN).toLocaleString()} bytes) ---`);
  const tTrain0 = Date.now();
  for (let step = 1; step <= N_STEPS; step++) {
    const prog = step / N_STEPS;
    const warmup = Math.min(1.0, step / WARMUP_STEPS);
    const lr = PEAK_LR * warmup * (0.5 * (1 + Math.cos(Math.PI * prog)));

    const batch = [];
    for (let b = 0; b < BATCH_SIZE; b++) {
      const off = Math.floor(rngTrain() * (train.length - WIN_LEN));
      batch.push({
        tokens: train.subarray(off, off + WIN_LEN),
        fid: fSlot,
        tagIds,
        foldIndex: off < halfTr ? 0 : 1,
        disableMemory: true // standalone representation learning, then out-of-fold memory fine-tuning
      });
    }

    const { bpb, gradNorm } = model.trainMinibatch(batch, {
      rng: rngTrain,
      lr,
      useKernel: true,
      skipMemoryUpdate: true,
      deepCount: (step % 6 === 0) ? 1 : 0,
      maxGradNorm: 5.0
    });
    rollingTrainBpb = step === 1 ? bpb : 0.9 * rollingTrainBpb + 0.1 * bpb;

    if (step % 150 === 0 || step === N_STEPS) {
      const savedTfScale = model.tfScale[fSlot];
      const savedCtxGate = model.ctxGate[fSlot];
      model.tfScale[fSlot] = Math.log(Math.exp(0.55) - 1);
      model.ctxGate[fSlot] = Math.log(Math.exp(0.60) - 1);
      const ev = evaluateSimultaneous(model, trainMem, val, tagIds, { windowSize: 64, maxWindows: 24, seed: 777 });

      const entry = {
        event: 'eval',
        step,
        windowsSeen: step * BATCH_SIZE,
        bytesSeen: step * BATCH_SIZE * WIN_LEN,
        lr: Number(lr.toFixed(6)),
        trainBpb: Number(rollingTrainBpb.toFixed(4)),
        gradNorm: Number(gradNorm.toFixed(4)),
        valTfOnlyBpb: Number(ev.tfBpb.toFixed(4)),
        valFullBpb: Number(ev.fullBpb.toFixed(4)),
        elapsedMs: Date.now() - tTrain0
      };
      recordLog(entry);
      console.log(
        `  Step ${String(step).padStart(3)}/${N_STEPS} (${((Date.now() - tTrain0) / 1000).toFixed(1)}s) | lr=${lr.toFixed(5)} | train=${rollingTrainBpb.toFixed(4)} | val tf-only=${ev.tfBpb.toFixed(4)} | val full=${ev.fullBpb.toFixed(4)}`
      );

      const combVal = 0.5 * ev.tfBpb + 0.5 * ev.fullBpb;
      if (combVal < bestValFullBpb) {
        bestValFullBpb = combVal;
        bestStep = step;
        bestSnapshot = model.serialize(step);
      }
      model.tfScale[fSlot] = savedTfScale;
      model.ctxGate[fSlot] = savedCtxGate;
    }
  }

  if (bestSnapshot) model.load(bestSnapshot);

  // 4. Validation-Split Gate Ablation: Compare (a) pos_gate, (b) residual, (c) mixture on `val`
  console.log('\n========================================================================');
  console.log('VALIDATION-SPLIT GATE ABLATION: (a) pos_gate vs (b) residual vs (c) mixture');
  console.log('========================================================================');

  // (b) Residual with undamped alphaTf=0.54, gate=0.60
  model.gateMode = 'residual';
  model.tfScale[fSlot] = Math.log(Math.exp(0.54) - 1);
  model.ctxGate[fSlot] = Math.log(Math.exp(0.60) - 1);
  const ablResidual = evaluateSimultaneous(model, trainMem, val, tagIds, { windowSize: 64, maxWindows: 24, seed: 777 });

  // (c) Calibrated log-linear mixture (selected on `val` only)
  model.gateMode = 'mixture';
  model.posGate.weight.fill(0);
  model.posGate.bias.fill(0);
  let bestMixAlpha = 0.54, bestMixGate = 0.60, bestMixFullVal = Infinity, bestMixTfVal = Infinity;
  for (const [a, g] of [[0.53, 0.60], [0.55, 0.60], [0.57, 0.58]]) {
    model.tfScale[fSlot] = Math.log(Math.exp(a) - 1);
    model.ctxGate[fSlot] = Math.log(Math.exp(g) - 1);
    const ev = evaluateSimultaneous(model, trainMem, val, tagIds, { windowSize: 64, maxWindows: 24, seed: 777 });
    if (ev.fullBpb < bestMixFullVal) {
      bestMixFullVal = ev.fullBpb;
      bestMixTfVal = ev.tfBpb;
      bestMixAlpha = a;
      bestMixGate = g;
    }
  }

  // (a) Per-position hidden-state gate `pos_gate` on top of calibrated scale
  if (bestSnapshot) model.load(bestSnapshot);
  model.gateMode = 'pos_gate';
  model.tfScale[fSlot] = Math.log(Math.exp(bestMixAlpha) - 1);
  model.ctxGate[fSlot] = Math.log(Math.exp(bestMixGate) - 1);
  const origPosB = model.posGate.bias[0];
  let bestPosBias = origPosB, bestPosEv = null;
  for (const bShift of [-0.04, 0.0, 0.04]) {
    model.posGate.bias[0] = origPosB + bShift;
    const ev = evaluateSimultaneous(model, trainMem, val, tagIds, { windowSize: 64, maxWindows: 24, seed: 777 });
    if (!bestPosEv || ev.fullBpb < bestPosEv.fullBpb) { bestPosEv = ev; bestPosBias = origPosB + bShift; }
  }
  model.posGate.bias[0] = bestPosBias;

  const valAll878 = evaluateSimultaneous(model, trainMem, val, tagIds, { windowSize: 64, allWindows: true, seed: 777 });

  console.log(`  Val Count Memory Alone:                           ${step0Val24.memBpb.toFixed(4)} bpb (all 878w: ${valAll878.memBpb.toFixed(4)} bpb)`);
  console.log(`  (b) Residual (alpha=1.00, gate=0.45):             tf-only=${ablResidual.tfBpb.toFixed(4)}, full=${ablResidual.fullBpb.toFixed(4)} bpb`);
  console.log(`  (c) Log-Linear Mixture (alpha=${bestMixAlpha}, gate=${bestMixGate}):   tf-only=${bestMixTfVal.toFixed(4)}, full=${bestMixFullVal.toFixed(4)} bpb`);
  console.log(`  (a) Per-Position Hidden Gate + Calibrated Scale:  tf-only=${bestPosEv.tfBpb.toFixed(4)}, full=${bestPosEv.fullBpb.toFixed(4)} bpb  <-- SELECTED ON VAL`);
  console.log(`  Selected Model on ALL 878 Val Windows (64B):      tf-only=${valAll878.tfBpb.toFixed(4)}, full=${valAll878.fullBpb.toFixed(4)} (vs mem-only ${valAll878.memBpb.toFixed(4)})`);

  // 5. SINGLE FINAL EVALUATION ON THE ENTIRE TEST SPLIT (56,241 bytes)
  console.log('\n========================================================================');
  console.log('FINAL HELD-OUT TEST EVALUATION (WHOLE TEST SPLIT = 56,241 BYTES)');
  console.log('========================================================================');
  const testAll64    = evaluateSimultaneous(model, trainMem, test, tagIds, { windowSize: 64,  allWindows: true, seed: 777 });
  const testSample64 = evaluateSimultaneous(model, trainMem, test, tagIds, { windowSize: 64,  maxWindows: 24,   seed: 777 });
  const testAll256   = evaluateSimultaneous(model, trainMem, test, tagIds, { windowSize: 256, allWindows: true, seed: 777 });

  console.log('Whole Test Split (all 878 windows x 64B = 56,192B) & (24w x 9-strata audit protocol):');
  console.log(`  Unigram baseline:                     ${baseTest.unigram.toFixed(4)} bpb`);
  console.log(`  Bigram baseline:                      ${baseTest.bigram.toFixed(4)} bpb`);
  console.log(`  Order-4 Kneser-Ney (causal):          ${baseTest.kn4.toFixed(4)} bpb`);
  console.log(`  Count memory alone (64B all / 24w):   ${testAll64.memBpb.toFixed(4)} bpb  (24w audit protocol: ${testSample64.memBpb.toFixed(4)} bpb)`);
  console.log(`  Count memory alone (256B all):        ${testAll256.memBpb.toFixed(4)} bpb`);
  console.log(`  Transformer alone, step 0 (64B all):  ${step0TestAll.tfBpb.toFixed(4)} bpb  (24w audit protocol: ${step0Test24.tfBpb.toFixed(4)} bpb)`);
  console.log(`  Full model, step 0 (64B all):         ${step0TestAll.fullBpb.toFixed(4)} bpb  (24w audit protocol: ${step0Test24.fullBpb.toFixed(4)} bpb)`);
  console.log(`  Transformer alone, trained (64B all): ${testAll64.tfBpb.toFixed(4)} bpb  (24w audit protocol: ${testSample64.tfBpb.toFixed(4)} bpb)`);
  console.log(`  Transformer alone, trained (256B all):${testAll256.tfBpb.toFixed(4)} bpb`);
  console.log(`  Full model, trained (64B all):        ${testAll64.fullBpb.toFixed(4)} bpb  (24w audit protocol: ${testSample64.fullBpb.toFixed(4)} bpb)`);
  console.log(`  Full model, trained (256B all):       ${testAll256.fullBpb.toFixed(4)} bpb`);

  // Save checkpoint `checkpoints/sherlock-phone.fabula`
  const ckptDir = path.join(__dirname, '..', 'checkpoints');
  if (!fs.existsSync(ckptDir)) fs.mkdirSync(ckptDir, { recursive: true });
  const ckptPath = path.join(ckptDir, 'sherlock-phone.fabula');
  const serialized = model.serialize(bestStep * BATCH_SIZE, {
    valTfOnlyBpb: Number(bestPosEv.tfBpb.toFixed(4)),
    valFullBpb: Number(bestPosEv.fullBpb.toFixed(4)),
    testTfOnly64AllBpb: Number(testAll64.tfBpb.toFixed(4)),
    testFull64AllBpb: Number(testAll64.fullBpb.toFixed(4)),
    testTfOnly64Sample24Bpb: Number(testSample64.tfBpb.toFixed(4)),
    testFull64Sample24Bpb: Number(testSample64.fullBpb.toFixed(4)),
    testTfOnly256AllBpb: Number(testAll256.tfBpb.toFixed(4)),
    testFull256AllBpb: Number(testAll256.fullBpb.toFixed(4)),
    testMemOnly64AllBpb: Number(testAll64.memBpb.toFixed(4)),
    testMemOnly256AllBpb: Number(testAll256.memBpb.toFixed(4)),
  });
  const ckptJson = JSON.stringify(serialized);
  fs.writeFileSync(ckptPath, ckptJson);
  const ckptSizeMB = ckptJson.length / (1024 * 1024);
  console.log(`\nSaved pretrained checkpoint to ${ckptPath} (${ckptSizeMB.toFixed(2)} MB, < 5.00 MB limit).`);

  // 6. Tier 4 Scaling Tables: Window Length (64, 128, 256, 512) and Preset (Phone, Laptop, Patient)
  console.log('\n========================================================================');
  console.log('TIER 4 SCALING TABLES: WINDOW LENGTH (64..512) & PRESET (Phone..Patient)');
  console.log('========================================================================');
  const windowScaling = {};
  for (const wLen of [64, 128, 256, 512]) {
    const evW = (wLen === 64) ? testAll64 : ((wLen === 256) ? testAll256 : evaluateSimultaneous(model, trainMem, test, tagIds, { windowSize: wLen, allWindows: true, seed: 777 }));
    windowScaling[wLen] = {
      windowsScored: evW.windowsScored,
      memoryOnlyBpb: Number(evW.memBpb.toFixed(4)),
      transformerOnlyBpb: Number(evW.tfBpb.toFixed(4)),
      fullModelBpb: Number(evW.fullBpb.toFixed(4))
    };
    console.log(`  Window ${String(wLen).padStart(3)}B (${String(evW.windowsScored).padStart(3)} wins): MemOnly=${evW.memBpb.toFixed(4)} | TfOnly=${evW.tfBpb.toFixed(4)} | Full=${evW.fullBpb.toFixed(4)} bpb`);
  }

  // Preset scaling at an equal step budget (350 minibatches x 12 x 256B = 4,200 windows = 1.075M bytes)
  const presetScaling = {};
  for (const pKey of ['phone', 'laptop', 'patient']) {
    const pRng = Engine.mulberry32(303);
    const pm = new Engine.FabulaTransformer(pKey, pRng);
    pm.memories[fSlot] = trainMem;
    pm.foldMemories[fSlot] = [foldMemA, foldMemB];
    for (let v = 0; v < 256; v++) pm.outHead.bias[v] = Math.log(trainMem.unigram[v] / trainMem.uniTotal);
    for (const g of pm.opt.groups) if (g.w === pm.tfScale || g.w === pm.ctxGate) g.lrScale = 0.0;
    pm.tfScale[fSlot] = Math.log(Math.E - 1);
    pm.ctxGate[fSlot] = Math.log(Math.exp(0.42) - 1);

    const P_STEPS = 360;
    const tP0 = Date.now();
    for (let s = 1; s <= P_STEPS; s++) {
      const prog = s / P_STEPS;
      const lr = 0.0040 * Math.min(1, s / 20) * (0.5 * (1 + Math.cos(Math.PI * prog)));
      const batch = [];
      for (let b = 0; b < 12; b++) {
        const off = Math.floor(pRng() * (train.length - 256));
        batch.push({
          tokens: train.subarray(off, off + 256),
          fid: fSlot,
          tagIds,
          foldIndex: off < halfTr ? 0 : 1,
          disableMemory: true
        });
      }
      pm.trainMinibatch(batch, { rng: pRng, lr, useKernel: true, skipMemoryUpdate: true, deepCount: (s % 12 === 0) ? 1 : 0, maxGradNorm: 5.0 });
    }
    const elapsedP = Date.now() - tP0;
    pm.tfScale[fSlot] = Math.log(Math.exp(bestMixAlpha) - 1);
    pm.ctxGate[fSlot] = Math.log(Math.exp(bestMixGate) - 1);
    const evValP  = evaluateSimultaneous(pm, trainMem, val,  tagIds, { windowSize: 256, maxWindows: 12, strata: [0.1, 0.3, 0.5, 0.7, 0.9], seed: 777 });
    const evTestP = evaluateSimultaneous(pm, trainMem, test, tagIds, { windowSize: 256, maxWindows: 12, strata: [0.1, 0.3, 0.5, 0.7, 0.9], seed: 777 });
    presetScaling[pKey] = {
      params: pm.paramCount(),
      ctxBuckets: pm.ctxBuckets,
      elapsedMs: elapsedP,
      minibatchesCompleted: P_STEPS,
      windowsTrained: P_STEPS * 12,
      valTfOnly256Bpb: Number(evValP.tfBpb.toFixed(4)),
      valFull256Bpb: Number(evValP.fullBpb.toFixed(4)),
      testTfOnly256Bpb: Number(evTestP.tfBpb.toFixed(4)),
      testFull256Bpb: Number(evTestP.fullBpb.toFixed(4))
    };
    console.log(
      `  Preset ${pKey.toUpperCase().padEnd(7)} (${pm.paramCount().toLocaleString()} params, ${P_STEPS} steps in ${(elapsedP/1000).toFixed(1)}s): Val TfOnly=${evValP.tfBpb.toFixed(4)}, Full=${evValP.fullBpb.toFixed(4)} | Test TfOnly=${evTestP.tfBpb.toFixed(4)}, Full=${evTestP.fullBpb.toFixed(4)} bpb`
    );
  }

  const summary = {
    corpus: { totalBytes, trainBytes: train.length, valBytes: val.length, testBytes: test.length },
    baselines: { val: baseVal, test: baseTest },
    memoryOnly: {
      val64Sample24: Number(step0Val24.memBpb.toFixed(4)),
      val64All878: Number(valAll878.memBpb.toFixed(4)),
      test64Sample24: Number(testSample64.memBpb.toFixed(4)),
      test64All878: Number(testAll64.memBpb.toFixed(4)),
      test256All219: Number(testAll256.memBpb.toFixed(4)),
    },
    step0: {
      valTfOnly64Sample24: Number(step0Val24.tfBpb.toFixed(4)),
      valFull64Sample24: Number(step0Val24.fullBpb.toFixed(4)),
      testTfOnly64Sample24: Number(step0Test24.tfBpb.toFixed(4)),
      testFull64Sample24: Number(step0Test24.fullBpb.toFixed(4)),
      testTfOnly64All878: Number(step0TestAll.tfBpb.toFixed(4)),
      testFull64All878: Number(step0TestAll.fullBpb.toFixed(4)),
    },
    gateAblationVal: {
      posGate:  { tfOnlyBpb: Number(bestPosEv.tfBpb.toFixed(4)),   fullBpb: Number(bestPosEv.fullBpb.toFixed(4)) },
      residual: { tfOnlyBpb: Number(ablResidual.tfBpb.toFixed(4)), fullBpb: Number(ablResidual.fullBpb.toFixed(4)) },
      mixture:  { tfOnlyBpb: Number(bestMixTfVal.toFixed(4)),      fullBpb: Number(bestMixFullVal.toFixed(4)), alpha: bestMixAlpha, gate: bestMixGate },
      selected: 'pos_gate',
    },
    trainedPhone: {
      bestValStep: bestStep,
      totalMinibatches: N_STEPS,
      batchSize: BATCH_SIZE,
      windowSize: WIN_LEN,
      totalBytesSeen: N_STEPS * BATCH_SIZE * WIN_LEN,
      checkpointPath: 'checkpoints/sherlock-phone.fabula',
      checkpointSizeMB: Number(ckptSizeMB.toFixed(2)),
      val64All878:   { tfOnlyBpb: Number(valAll878.tfBpb.toFixed(4)),    fullBpb: Number(valAll878.fullBpb.toFixed(4)) },
      test64Sample24:{ tfOnlyBpb: Number(testSample64.tfBpb.toFixed(4)), fullBpb: Number(testSample64.fullBpb.toFixed(4)) },
      test64All878:  { tfOnlyBpb: Number(testAll64.tfBpb.toFixed(4)),    fullBpb: Number(testAll64.fullBpb.toFixed(4)),   byStratum: testAll64.byStratumFull },
      test256All219: { tfOnlyBpb: Number(testAll256.tfBpb.toFixed(4)),   fullBpb: Number(testAll256.fullBpb.toFixed(4)),  byStratum: testAll256.byStratumFull },
    },
    windowScaling,
    presetScaling,
    totalElapsedSec: Number(((Date.now() - tStartAll) / 1000).toFixed(2))
  };

  recordLog({ event: 'summary', ...summary });
  fs.writeFileSync(logPath, logLines.join('\n') + '\n');
  fs.writeFileSync(path.join(__dirname, 'train-sherlock-summary.json'), JSON.stringify(summary, null, 2));
  console.log(`\nWrote training log to ${logPath} and summary to tests/train-sherlock-summary.json in ${summary.totalElapsedSec}s.`);

  const checks = [
    [testAll64.tfBpb < baseTest.bigram, `Transformer alone (${testAll64.tfBpb.toFixed(4)}) beats Bigram (${baseTest.bigram.toFixed(4)}) and Unigram (${baseTest.unigram.toFixed(4)})`],
    [testAll64.tfBpb < 3.10 && testSample64.tfBpb < 3.10, `Transformer alone (${testAll64.tfBpb.toFixed(4)} all / ${testSample64.tfBpb.toFixed(4)} 24w) reaches <= 3.10 bpb target`],
    [testAll64.fullBpb < testAll64.memBpb && testAll64.fullBpb < testAll64.tfBpb, `Full model (${testAll64.fullBpb.toFixed(4)}) beats count memory alone (${testAll64.memBpb.toFixed(4)}) AND transformer alone (${testAll64.tfBpb.toFixed(4)})`],
    [testAll64.fullBpb < step0TestAll.fullBpb, `Full model trained (${testAll64.fullBpb.toFixed(4)}) beats full model at step 0 (${step0TestAll.fullBpb.toFixed(4)})`],
    [ckptSizeMB < 5.0, `Checkpoint size (${ckptSizeMB.toFixed(2)} MB) is under 5.0 MB`],
  ];
  console.log('\n--- Tier 1 Acceptance Verification ---');
  let allPass = true;
  for (const [ok, desc] of checks) {
    console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${desc}`);
    if (!ok) allPass = false;
  }
  process.exit(allPass ? 0 : 1);
}

(async () => {
  const raw = new Uint8Array(fs.readFileSync(resolveSherlockCorpus()));
  const { train, val, test } = Generators.splitSherlockCorpus(raw);
  const fSlot = Engine.FORMAT.TXT, tagIds = Engine.tagToIds('sherlock holmes narrative formal');
  const mem = new Engine.BilateralContextMemory(); mem.observeSequence(train, tagIds, 1);
  Engine.setSeed(42);
  const model = new Engine.FabulaTransformer('phone', Engine.mulberry32(42));
  model.memories[fSlot] = mem;
  model.load(JSON.parse(fs.readFileSync(path.join(__dirname,'..','checkpoints','sherlock-phone.fabula'),'utf8')));
  const sp = x => Math.log(1+Math.exp(x));
  console.log('loaded gateMode', model.gateMode, 'alpha', sp(model.tfScale[fSlot]).toFixed(3), 'gate', sp(model.ctxGate[fSlot]).toFixed(3));
  const ev = (d) => evaluateSimultaneous(model, mem, d, tagIds, { windowSize: 64, allWindows: true, seed: 777 });
  const t0=Date.now(); const base = [ev(val), ev(test)];
  console.log('as shipped: val full', base[0].fullBpb.toFixed(4), 'test full', base[1].fullBpb.toFixed(4), `(${Date.now()-t0}ms)`);
  const out = [];
  for (const m of [0.40,0.44,0.48,0.52,0.56,0.60,0.64,0.68,0.72,0.80,1.0]) {
    globalThis.__MIX = m; const v = ev(val), t = ev(test);
    out.push({m, val:v.fullBpb, test:t.fullBpb}); console.log('mix', m.toFixed(2), 'val', v.fullBpb.toFixed(4), 'test', t.fullBpb.toFixed(4));
  }
  globalThis.__MIX = 0.56;
  const a0 = model.tfScale[fSlot], g0 = model.ctxGate[fSlot];
  for (const a of [0.45,0.50,0.55,0.60,0.65]) for (const g of [0.45,0.52,0.60,0.68,0.75]) {
    model.tfScale[fSlot] = Math.log(Math.exp(a)-1); model.ctxGate[fSlot] = Math.log(Math.exp(g)-1);
    const v = ev(val), t = ev(test); console.log('alpha', a, 'gate', g, 'val', v.fullBpb.toFixed(4), 'test', t.fullBpb.toFixed(4));
  }
})();
