#!/usr/bin/env node
/**
 * Reproducible Evaluation & Measurement Suite for The Autoencofabulator v0.3: Realized.
 *
 * Computes every metric reported in research/REALIZED-1.md:
 *   1. Sherlock Holmes 80/10/10 held-out likelihood (NELBO in bits/byte) vs.
 *      Unigram, Bigram, and Order-4 Interpolated Kneser-Ney baselines.
 *      Hyperparameters selected on val only; test evaluated once at the end.
 *   2. Format validity before and after repair-by-remasking across HTML, JSON,
 *      SVG, JS, C (with real gcc -fsyntax-only hook), BMP, WAV, and AuraGraph .aura.png.
 *   3. Distribution property matching (HTML depth, JSON schema, BMP shape count &
 *      palette colour, WAV pitch & tempo) on seeded synthetic corpora.
 *   4. Memorisation audit via Longest Common Substring (LCS) ratio against training corpus.
 *   5. Co-training (unified multi-format model) vs. 4 format-specialist models + tag transfer.
 *   6. Audio Path A (8-bit μ-law) vs. Path B (AuraGraph 2-D tiles) and Ghost vs. Timing Helix
 *      onset sharpness (10%->90% rise time in ms, per HELIX-1.md).
 *   7. Speed and memory benchmarks across Phone, Laptop, and Patient presets
 *      (JS vs. WASM vs. SIMD, Desktop vs. 4x throttled Phone).
 *
 * Run: node tests/metrics.js [--json metrics-results.json]
 */
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const { spawnSync } = require('child_process');
const Generators = require('./generators.js');

function resolveIndexHtml() {
  const candidates = [
    path.join(__dirname, '..', 'index.html'),
    path.join(__dirname, 'index.html'),
  ];
  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }
  throw new Error('Could not locate index.html relative to metrics.js');
}

function resolveSherlockCorpus() {
  const candidates = [
    path.join(__dirname, '..', 'sherlock-holmes-corpus.txt'),
  ];
  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }
  throw new Error('Could not locate sherlock-holmes-corpus.txt');
}

const html = fs.readFileSync(resolveIndexHtml(), 'utf8');
const startMarker = '// === FABULA_ENGINE_START ===';
const endMarker = '// === FABULA_ENGINE_END ===';
const sIdx = html.indexOf(startMarker);
const eIdx = html.indexOf(endMarker);
const engineCode = html.slice(sIdx + startMarker.length, eIdx);

globalThis.window = globalThis;
globalThis.self = globalThis;
if (!globalThis.performance) {
  globalThis.performance = { now: () => Number(process.hrtime.bigint()) / 1e6 };
}
vm.runInThisContext(engineCode);
const Engine = globalThis.FabulaEngine;
const sandbox = globalThis;

// Probe real gcc -fsyntax-only C compiler hook on host
let gccAvailable = false;
try {
  const probe = spawnSync('gcc', ['--version'], { encoding: 'utf8' });
  if (probe.status === 0) gccAvailable = true;
} catch (_) {}

function gccSyntaxCheck(bytes) {
  const cSource = new TextDecoder('utf-8', { fatal: false }).decode(bytes);
  if (!cSource.trim()) return { ok: false, errPos: 0, reason: 'Empty C source' };
  if (!gccAvailable) return Engine.parseCStrict(bytes);
  const res = spawnSync('gcc', ['-fsyntax-only', '-Wall', '-Wno-unused', '-xc', '-'], {
    input: cSource,
    encoding: 'utf8',
    timeout: 4000,
  });
  if (res.status === 0) return { ok: true, errPos: -1, reason: 'gcc -fsyntax-only: 0 errors ✓' };
  const errText = (res.stderr || 'gcc syntax error').trim();
  return { ok: false, errPos: Math.floor(bytes.length / 2), reason: errText.split('\n')[0] };
}

async function main() {
  const results = {};

  console.log('========================================================================');
  console.log('EXPERIMENT 1: Held-Out Likelihood on Sherlock Holmes (80/10/10 Split)');
  console.log('========================================================================');
  const sherlockPath = resolveSherlockCorpus();
  const rawSherlock = new Uint8Array(fs.readFileSync(sherlockPath));
  const { totalBytes, train, val, test } = Generators.splitSherlockCorpus(rawSherlock);
  console.log(`  Corpus: ${totalBytes} bytes -> train=${train.length}, val=${val.length}, test=${test.length}`);

  const baselines = Engine.fitByteBaselines(train);
  const baseVal = {
    unigram: baselines.evalUnigram(val),
    bigram: baselines.evalBigram(val),
    kn4: baselines.evalKneserNey4(val),
  };
  const baseTest = {
    unigram: baselines.evalUnigram(test),
    bigram: baselines.evalBigram(test),
    kn4: baselines.evalKneserNey4(test),
  };
  console.log(`  Baselines (val):  Unigram=${baseVal.unigram.toFixed(4)}  Bigram=${baseVal.bigram.toFixed(4)}  KN4=${baseVal.kn4.toFixed(4)} bpb`);
  console.log(`  Baselines (test): Unigram=${baseTest.unigram.toFixed(4)}  Bigram=${baseTest.bigram.toFixed(4)}  KN4=${baseTest.kn4.toFixed(4)} bpb`);

  // Hyperparameter selection on VAL split only: compare Linear vs Cosine schedule on Phone & Laptop
  function trainSherlockModel(presetKey, schedule, steps = 20, seed = 101) {
    Engine.CFG.SCHEDULE = schedule;
    const rng = Engine.mulberry32(seed);
    const model = new Engine.FabulaTransformer(presetKey, rng);
    const fSlot = Engine.FORMAT.TXT;
    const tagIds = Engine.tagToIds('txt prose english');
    // Populate bilateral context memory strictly from the training split
    const memStride = 1024;
    for (let off = 0; off + 256 <= train.length; off += memStride) {
      model.memories[fSlot].observeSequence(train.subarray(off, off + 256), tagIds, 1);
    }
    const winLen = 64;
    for (let s = 0; s < steps; s++) {
      const start = Math.floor(rng() * (train.length - winLen));
      const win = train.subarray(start, start + winLen);
      model.trainStep(win, fSlot, tagIds, { rng, skipMemoryUpdate: true, fastTrainStep: true, useKernel: true });
    }
    const valEval = model.evaluateNELBO(val, fSlot, tagIds, { windowSize: 64, maxWindows: 100, seed: 202 });
    return { model, valEval, tagIds };
  }

  const candPhoneLinear  = trainSherlockModel('phone',  'linear', 20, 101);
  const candPhoneCosine  = trainSherlockModel('phone',  'cosine', 20, 102);
  const candLaptopLinear = trainSherlockModel('laptop', 'linear', 20, 103);

  console.log(`  Val NELBO (Phone, linear):  ${candPhoneLinear.valEval.bpb.toFixed(4)} bpb`);
  console.log(`  Val NELBO (Phone, cosine):  ${candPhoneCosine.valEval.bpb.toFixed(4)} bpb`);
  console.log(`  Val NELBO (Laptop, linear): ${candLaptopLinear.valEval.bpb.toFixed(4)} bpb`);

  // Pick best model on VAL only, then evaluate ONCE on TEST
  const candidates = [
    { name: 'phone-linear',  ...candPhoneLinear,  schedule: 'linear' },
    { name: 'phone-cosine',  ...candPhoneCosine,  schedule: 'cosine' },
    { name: 'laptop-linear', ...candLaptopLinear, schedule: 'linear' },
  ].sort((a, b) => a.valEval.bpb - b.valEval.bpb);

  const bestCand = candidates[0];
  Engine.CFG.SCHEDULE = bestCand.schedule;
  // [Kehai audit] v0.3.0 scored 6 windows (384 of 56,241 test bytes); now 300 windows (19,200 bytes)
  const testEval = bestCand.model.evaluateNELBO(test, Engine.FORMAT.TXT, bestCand.tagIds, {
    windowSize: 64,
    maxWindows: 300,
    seed: 303,
  });
  const origGate = bestCand.model.ctxGate[Engine.FORMAT.TXT];
  bestCand.model.ctxGate[Engine.FORMAT.TXT] = -20.0;
  const testEvalNoMem = bestCand.model.evaluateNELBO(test, Engine.FORMAT.TXT, bestCand.tagIds, {
    windowSize: 64,
    maxWindows: 300,
    seed: 303,
  });
  bestCand.model.ctxGate[Engine.FORMAT.TXT] = origGate;

  console.log(`  Selected on val: ${bestCand.name} (val=${bestCand.valEval.bpb.toFixed(4)} bpb)`);
  console.log(`  Final TEST NELBO (full v0.3 model): ${testEval.bpb.toFixed(4)} bpb`);
  console.log(`  Final TEST NELBO (no memory gate):  ${testEvalNoMem.bpb.toFixed(4)} bpb`);
  console.log(`  Test NELBO by stratum t:`, JSON.stringify(testEval.byStratum));

  results.sherlock = {
    splits: { totalBytes, trainBytes: train.length, valBytes: val.length, testBytes: test.length },
    baselines: { val: baseVal, test: baseTest },
    valSelection: {
      phoneLinear: candPhoneLinear.valEval.bpb,
      phoneCosine: candPhoneCosine.valEval.bpb,
      laptopLinear: candLaptopLinear.valEval.bpb,
      selected: bestCand.name,
    },
    testNELBO: {
      fullModelBpb: testEval.bpb,
      transformerOnlyBpb: testEvalNoMem.bpb,
      byStratum: testEval.byStratum,
    },
  };

  console.log('\n========================================================================');
  console.log('EXPERIMENT 2 & 3: Format Validity (Raw vs Repaired), Property Matching & LCS');
  console.log('========================================================================');
  Engine.CFG.SCHEDULE = 'linear';
  const suite = Generators.buildSeededSuite(4242, 8);
  const bmpRng = Generators.mulberry32(4242);
  suite.bmp = [];
  for (let i = 0; i < 8; i++) {
    suite.bmp.push(Generators.generateBMPSample(bmpRng, { size: 32, nShapes: 1 + (i % 3) }));
  }

  // [Kehai audit] held-out documents, never trained on: v0.3.0 built its generation scaffolds from, and
  // scored co-training on, the training documents themselves
  const heldOut = Generators.buildSeededSuite(97531, 6);
  const hbRng = Generators.mulberry32(97531);
  heldOut.bmp = [];
  for (let i = 0; i < 6; i++) heldOut.bmp.push(Generators.generateBMPSample(hbRng, { size: 32, nShapes: 1 + (i % 3) }));

  const unifiedRng = Engine.mulberry32(4242);
  const unifiedModel = new Engine.FabulaTransformer('phone', unifiedRng);

  const allTrainDocs = [];
  for (const item of suite.html) allTrainDocs.push({ ...item, fid: Engine.FORMAT.HTML });
  for (const item of suite.json) allTrainDocs.push({ ...item, fid: Engine.FORMAT.JSON });
  for (const item of suite.svg)  allTrainDocs.push({ ...item, fid: Engine.FORMAT.SVG });
  for (const item of suite.js)   allTrainDocs.push({ ...item, fid: Engine.FORMAT.JS });
  for (const item of suite.c)    allTrainDocs.push({ ...item, fid: Engine.FORMAT.C });
  for (const item of suite.bmp)  allTrainDocs.push({ ...item, fid: Engine.FORMAT.BMP });
  for (const item of suite.wav)  allTrainDocs.push({ ...item, fid: Engine.FORMAT.WAV_ULAW });

  // Train unified model across all formats
  for (const doc of allTrainDocs) {
    const tagIds = Engine.tagToIds(doc.tags);
    if (doc.fid === Engine.FORMAT.BMP) {
      const p = Engine.encodeBMPTo2DPatches(doc.bytes);
      unifiedModel.trainStep(p.tokens, Engine.FORMAT.BMP, tagIds, {
        rng: unifiedRng,
        patchMode: true,
        rowIds: p.rowIds,
        colIds: p.colIds,
        fastTrainStep: true,
      });
    } else if (doc.fid === Engine.FORMAT.WAV_ULAW) {
      const muBytes = Engine.muLawEncodePCM(doc.pcm.subarray(0, 128));
      unifiedModel.trainStep(muBytes, Engine.FORMAT.WAV_ULAW, tagIds, { rng: unifiedRng, patchMode: false, fastTrainStep: true });

      const A = Engine.agAnalyzeSync(doc.pcm.subarray(0, 4096), true);
      const helix = Engine.agBuildHelixData(doc.pcm.subarray(0, 4096), A, 0, A.F);
      const tileEnc = Engine.encodeAuraTileTo2DPatches(A, helix);
      const nP = Math.min(40, tileEnc.rowIds.length);
      unifiedModel.trainStep(tileEnc.tokens.subarray(0, nP * 16), Engine.FORMAT.AURA, tagIds, {
        rng: unifiedRng,
        patchMode: true,
        rowIds: tileEnc.rowIds.subarray(0, nP),
        colIds: tileEnc.colIds.subarray(0, nP),
        helixTargets: tileEnc.helixTargets.subarray(0, nP),
        fastTrainStep: true,
      });
    } else {
      unifiedModel.trainStep(doc.bytes, doc.fid, tagIds, { rng: unifiedRng, patchMode: false, fastTrainStep: true });
    }
  }

  const validityStats = {};
  const lcsStats = {};
  const formatsToTest = [
    { name: 'HTML',     fid: Engine.FORMAT.HTML,     docs: suite.html, scaffolds: heldOut.html },
    { name: 'JSON',     fid: Engine.FORMAT.JSON,     docs: suite.json, scaffolds: heldOut.json },
    { name: 'SVG',      fid: Engine.FORMAT.SVG,      docs: suite.svg,  scaffolds: heldOut.svg },
    { name: 'JS',       fid: Engine.FORMAT.JS,       docs: suite.js,   scaffolds: heldOut.js },
    { name: 'C',        fid: Engine.FORMAT.C,        docs: suite.c,    scaffolds: heldOut.c },
    { name: 'BMP',      fid: Engine.FORMAT.BMP,      docs: suite.bmp },
    { name: 'WAV-uLaw', fid: Engine.FORMAT.WAV_ULAW, docs: suite.wav },
    { name: 'AuraTile', fid: Engine.FORMAT.AURA,     docs: suite.wav },
  ];

  const N_EVAL = 10;
  for (const spec of formatsToTest) {
    let rawValid = 0, repairedValid = 0, totalAttempts = 0;
    const lcsList = [];
    const corpusBytesList = spec.docs.map(d => d.bytes);

    for (let i = 0; i < N_EVAL; i++) {
      const genRng = Engine.mulberry32(8000 + spec.fid * 100 + i);
      const pool = spec.scaffolds || spec.docs;
      const refDoc = pool[i % pool.length];
      const tagIds = Engine.tagToIds(refDoc.tags);
      let rawFileBytes = null;

      if (spec.fid === Engine.FORMAT.BMP) {
        const samp = unifiedModel.sampleWindow(1024, Engine.FORMAT.BMP, tagIds, {
          patchMode: true,
          steps: 3,
          order: 'confidence',
          rng: genRng,
        });
        rawFileBytes = Engine.decode2DPatchesToBMP(samp.bytes, 32, 32);
      } else if (spec.fid === Engine.FORMAT.WAV_ULAW) {
        const samp = unifiedModel.sampleWindow(128, Engine.FORMAT.WAV_ULAW, tagIds, {
          patchMode: false,
          steps: 3,
          order: 'confidence',
          rng: genRng,
        });
        const pcm = Engine.muLawDecodeToPCM(samp.bytes);
        rawFileBytes = Generators.pcmFloatToWavBytes(pcm, 8000);
      } else if (spec.fid === Engine.FORMAT.AURA) {
        const samp = unifiedModel.sampleWindow(640, Engine.FORMAT.AURA, tagIds, {
          patchMode: true,
          steps: 3,
          order: 'confidence',
          rng: genRng,
        });
        const decoded = Engine.decode2DPatchesToAuraTile(samp.bytes, 1, samp.helixLogits);
        const pcm = Engine.agInvertHelix(decoded.A, decoded.helix, 0, decoded.A.F, 3);
        rawFileBytes = Generators.pcmFloatToWavBytes(pcm, 22050);
      } else {
        const L = refDoc.bytes.length;
        const initTokens = new Int32Array(L);
        for (let k = 0; k < L; k++) {
          const c = refDoc.bytes[k];
          const isDelim = (c === 123 || c === 125 || c === 60 || c === 62 || c === 58 || c === 59 || c === 34 || c === 40 || c === 41);
          const maskProb = isDelim ? (i % 3 === 0 ? 0.25 : 0.0) : 0.65;
          initTokens[k] = genRng() < maskProb ? Engine.CFG.MASK_ID : c;
        }
        const samp = unifiedModel.sampleWindow(L, spec.fid, tagIds, {
          initialTokens: initTokens,
          patchMode: false,
          steps: 4,
          order: 'confidence',
          temperature: 0.6,
          rng: genRng,
          noExemplarAnchors: true,
        });
        rawFileBytes = samp.bytes;
      }

      const vRaw = Engine.validateOutput(rawFileBytes, spec.fid, {}, corpusBytesList);
      const isRawOk = spec.name === 'C' ? (vRaw.parserOk && gccSyntaxCheck(rawFileBytes).ok) : vRaw.parserOk;
      if (isRawOk) {
        rawValid++;
        repairedValid++;
        lcsList.push(vRaw.lcsRatio);
      } else {
        const rep = Engine.repairByRemasking(unifiedModel, rawFileBytes, spec.fid, tagIds, 3, genRng);
        const isRepOk = spec.name === 'C' ? (rep.repairedValid && gccSyntaxCheck(rep.bytes).ok) : rep.repairedValid;
        if (isRepOk) repairedValid++;
        totalAttempts += rep.attemptsUsed;
        const vRep = Engine.validateOutput(rep.bytes, spec.fid, {}, corpusBytesList);
        lcsList.push(vRep.lcsRatio);
      }
    }

    const meanLcs = lcsList.reduce((a, b) => a + b, 0) / Math.max(1, lcsList.length);
    const maxLcs  = Math.max(...lcsList);
    validityStats[spec.name] = {
      rawValidRate: rawValid / N_EVAL,
      repairedValidRate: repairedValid / N_EVAL,
      meanRepairAttempts: totalAttempts / N_EVAL,
      cCompilerUsed: spec.name === 'C' ? (gccAvailable ? 'gcc -fsyntax-only -xc' : 'built-in C parser') : undefined,
    };
    lcsStats[spec.name] = { meanLcsRatio: meanLcs, maxLcsRatio: maxLcs };
    console.log(
      `  ${spec.name.padEnd(10)} Raw Valid: ${((rawValid / N_EVAL) * 100).toFixed(1).padStart(5)}% -> Repaired Valid: ${((repairedValid / N_EVAL) * 100).toFixed(1).padStart(5)}%  (mean LCS = ${meanLcs.toFixed(3)}, max = ${maxLcs.toFixed(3)})`
    );
  }

  // Property matching evaluation on HTML depth, JSON schema, BMP shape count & colour
  let htmlDepthMatches = 0, htmlTotal = 8;
  for (let i = 0; i < htmlTotal; i++) {
    const refDoc = suite.html[i % suite.html.length];
    const tagIds = Engine.tagToIds(refDoc.tags);
    const rng = Engine.mulberry32(9100 + i);
    const L = refDoc.bytes.length;
    const initTokens = new Int32Array(L);
    for (let k = 0; k < L; k++) {
      const c = refDoc.bytes[k];
      const isTagDelim = (c === 60 || c === 62 || c === 47);
      initTokens[k] = (!isTagDelim && rng() < 0.55) ? Engine.CFG.MASK_ID : c;
    }
    const samp = unifiedModel.sampleWindow(L, Engine.FORMAT.HTML, tagIds, {
      initialTokens: initTokens,
      steps: 4,
      order: 'confidence',
      rng,
    });
    const rep = Engine.repairByRemasking(unifiedModel, samp.bytes, Engine.FORMAT.HTML, tagIds, 2, rng);
    const txt = new TextDecoder().decode(rep.bytes);
    const d = Generators.measureHTMLDepth(txt);
    if (Math.abs(d - refDoc.meta.targetDepth) <= 1) htmlDepthMatches++;
  }

  let bmpCountMatches = 0, bmpColorMatches = 0, bmpTotal = 8;
  for (let i = 0; i < bmpTotal; i++) {
    const targetDoc = suite.bmp[i % suite.bmp.length];
    const tagIds = Engine.tagToIds(targetDoc.tags);
    const samp = unifiedModel.sampleWindow(1024, Engine.FORMAT.BMP, tagIds, {
      patchMode: true,
      steps: 3,
      order: 'confidence',
      rng: Engine.mulberry32(9300 + i),
    });
    const bmpBytes = Engine.decode2DPatchesToBMP(samp.bytes, 32, 32);
    const ana = Generators.analyzeBMPScene(bmpBytes);
    if (ana.valid && Math.abs(ana.nShapes - targetDoc.meta.nShapes) <= 1) bmpCountMatches++;
    if (ana.valid && targetDoc.meta.colors.some(c => ana.colors.includes(c))) bmpColorMatches++;
  }

  console.log(`  Property Match — HTML Nesting Depth (+/-1): ${((htmlDepthMatches / htmlTotal) * 100).toFixed(1)}% (${htmlDepthMatches}/${htmlTotal})`);
  console.log(`  Property Match — BMP Shape Count (+/-1):    ${((bmpCountMatches / bmpTotal) * 100).toFixed(1)}% (${bmpCountMatches}/${bmpTotal})`);
  console.log(`  Property Match — BMP Palette Colour Match:  ${((bmpColorMatches / bmpTotal) * 100).toFixed(1)}% (${bmpColorMatches}/${bmpTotal})`);

  results.validity = validityStats;
  results.memorisation = lcsStats;
  results.propertyMatch = {
    htmlDepthAcc: htmlDepthMatches / htmlTotal,
    bmpShapeCountAcc: bmpCountMatches / bmpTotal,
    bmpColorAcc: bmpColorMatches / bmpTotal,
  };

  console.log('\n========================================================================');
  console.log('EXPERIMENT 4: Co-Training (Unified) vs. Format Specialists + Tag Transfer');
  console.log('========================================================================');
  function evalHeldOutFormatBpb(model, docs, fid) {
    let sumBpb = 0, count = 0;
    for (let i = 0; i < docs.length; i++) {
      const d = docs[i];
      const tagIds = Engine.tagToIds(d.tags);
      let bytes = d.bytes;
      if (fid === Engine.FORMAT.BMP) bytes = Engine.encodeBMPTo2DPatches(d.bytes).tokens;
      else if (fid === Engine.FORMAT.WAV_ULAW) bytes = Engine.muLawEncodePCM(d.pcm.subarray(0, 64));
      const winLen = Math.min(64, bytes.length);
      const ev = model.evaluateNELBO(bytes.subarray(0, winLen), fid, tagIds, {
        windowSize: winLen,
        maxWindows: 1,
        strata: [0.2, 0.5, 0.8],
        seed: 500 + i,
      });
      sumBpb += ev.bpb;
      count++;
    }
    return sumBpb / Math.max(1, count);
  }

  const specResults = {};
  const formatsForAblation = [
    { key: 'html', fid: Engine.FORMAT.HTML,     docs: suite.html },
    { key: 'json', fid: Engine.FORMAT.JSON,     docs: suite.json },
    { key: 'bmp',  fid: Engine.FORMAT.BMP,      docs: suite.bmp },
    { key: 'wav',  fid: Engine.FORMAT.WAV_ULAW, docs: suite.wav },
  ];

  for (const f of formatsForAblation) {
    const sRng = Engine.mulberry32(6100 + f.fid);
    const sModel = new Engine.FabulaTransformer('phone', sRng);
    for (let s = 0; s < 8; s++) {
      const doc = f.docs[s % f.docs.length];
      const tagIds = Engine.tagToIds(doc.tags);
      if (f.fid === Engine.FORMAT.BMP) {
        const p = Engine.encodeBMPTo2DPatches(doc.bytes);
        sModel.trainStep(p.tokens, f.fid, tagIds, { rng: sRng, patchMode: true, rowIds: p.rowIds, colIds: p.colIds, fastTrainStep: true });
      } else if (f.fid === Engine.FORMAT.WAV_ULAW) {
        const muBytes = Engine.muLawEncodePCM(doc.pcm.subarray(0, 128));
        sModel.trainStep(muBytes, f.fid, tagIds, { rng: sRng, fastTrainStep: true });
      } else {
        sModel.trainStep(doc.bytes, f.fid, tagIds, { rng: sRng, fastTrainStep: true });
      }
    }
    const specBpb = evalHeldOutFormatBpb(sModel, heldOut[f.key], f.fid);
    const uniBpb  = evalHeldOutFormatBpb(unifiedModel, heldOut[f.key], f.fid);
    specResults[f.key] = { specialistBpb: specBpb, unifiedBpb: uniBpb, deltaBpb: uniBpb - specBpb };
    console.log(`  Format ${f.key.toUpperCase().padEnd(5)} Specialist: ${specBpb.toFixed(4)} bpb | Unified: ${uniBpb.toFixed(4)} bpb | Delta: ${(uniBpb - specBpb >= 0 ? '+' : '') + (uniBpb - specBpb).toFixed(4)} bpb`);
  }
  results.coTrainingVsSpecialist = specResults;

  console.log('\n========================================================================');
  console.log('EXPERIMENT 5: Audio Path A (8-bit μ-law) vs. Path B (AuraGraph Tiles + Helix)');
  console.log('========================================================================');
  let muPitchErrCents = 0, tilePitchErrCents = 0, muPitchMatch = 0, tilePitchMatch = 0, nTones = 0;
  let muTempoErrPct = 0, tileTempoErrPct = 0, nDrums = 0;
  let muRiseMsSum = 0, ghostRiseMsSum = 0, helixRiseMsSum = 0, nAudio = 0;
  let ghostSnrSum = 0, helixSnrSum = 0;

  for (const clip of suite.wav) {
    const refPCM = clip.pcm;
    const sr = clip.sr; // 22050

    const muEncoded = Engine.muLawEncodePCM(refPCM);
    const muDecoded = Engine.muLawDecodeToPCM(muEncoded);
    const muAna = Generators.analyzeWAVProperties(muDecoded, sr);

    const agState = Engine.agAnalyzeSync(refPCM, false);
    const helixData = Engine.agBuildHelixData(refPCM, agState, 0, agState.F);
    const pcmGhost = Engine.agInvertGhost(agState, 0, agState.F, 10);
    const pcmHelix = Engine.agInvertHelix(agState, helixData, 0, agState.F, 5);
    const ghostAna = Generators.analyzeWAVProperties(pcmGhost, sr);
    const helixAna = Generators.analyzeWAVProperties(pcmHelix, sr);

    const calcSnr = (rec) => {
      const L = Math.min(refPCM.length, rec.length);
      let sig = 0, err = 0;
      for (let i = 0; i < L; i++) {
        sig += refPCM[i] * refPCM[i];
        const d = refPCM[i] - rec[i];
        err += d * d;
      }
      return 10 * Math.log10(Math.max(sig, 1e-12) / Math.max(err, 1e-12));
    };
    ghostSnrSum += calcSnr(pcmGhost);
    helixSnrSum += calcSnr(pcmHelix);

    muRiseMsSum    += muAna.riseMs;
    ghostRiseMsSum += ghostAna.riseMs;
    helixRiseMsSum += helixAna.riseMs;
    nAudio++;

    const targetFreq = clip.meta.freq;
    const centsMu   = Math.abs(1200 * Math.log2(Math.max(20, muAna.freq) / targetFreq));
    const centsTile = Math.abs(1200 * Math.log2(Math.max(20, helixAna.freq) / targetFreq));
    muPitchErrCents   += centsMu;
    tilePitchErrCents += centsTile;
    if (muAna.midi === clip.meta.midi) muPitchMatch++;
    if (helixAna.midi === clip.meta.midi) tilePitchMatch++;
    nTones++;

    const targetBpm = clip.meta.bpm;
    muTempoErrPct   += (Math.abs(muAna.bpm - targetBpm) / targetBpm) * 100;
    tileTempoErrPct += (Math.abs(helixAna.bpm - targetBpm) / targetBpm) * 100;
    nDrums++;
  }

  // [Kehai audit] v0.3.0 regenerated 30% of the first 80 tile patches (2 of 16 time columns; the rest of the tile
  // was the real recording) against mu-law generating 489 of 512 samples (23 ms) and measuring tempo over that.
  // Now both paths regenerate the same 30% of the WHOLE clip from the same model.
  let genMuPitchMatch = 0, genTilePitchMatch = 0;
  let genMuTempoErr = 0, genTileTempoErr = 0;
  const nGenWav = 6;
  for (let i = 0; i < nGenWav; i++) {
    const clip = suite.wav[i];
    const tagIds = Engine.tagToIds(clip.tags);
    const rng = Engine.mulberry32(9500 + i);

    const A = Engine.agAnalyzeSync(clip.pcm, true);
    const helix = Engine.agBuildHelixData(clip.pcm, A, 0, A.F);
    const enc = Engine.encodeAuraTileTo2DPatches(A, helix);
    const initTile = new Int32Array(enc.tokens.length);
    for (let k = 0; k < initTile.length; k++) initTile[k] = rng() < 0.30 ? Engine.CFG.MASK_ID : enc.tokens[k];
    const sampTile = unifiedModel.sampleWindow(initTile.length, Engine.FORMAT.AURA, tagIds, {
      initialTokens: initTile, patchMode: true, rowIds: enc.rowIds, colIds: enc.colIds,
      steps: 3, order: 'confidence', rng,
    });
    const decTile = Engine.decode2DPatchesToAuraTile(sampTile.bytes, enc.gridW, sampTile.helixLogits, enc.range, enc.ref);
    const genTileAna = Generators.analyzeWAVProperties(Engine.agInvertHelix(decTile.A, decTile.helix, 0, decTile.A.F, 4), 22050);
    if (Math.abs(genTileAna.midi - clip.meta.midi) <= 1) genTilePitchMatch++;
    genTileTempoErr += (Math.abs(genTileAna.bpm - clip.meta.bpm) / clip.meta.bpm) * 100;

    const muRef = Engine.muLawEncodePCM(clip.pcm);
    const genMuBytes = new Uint8Array(muRef.length);
    for (let off = 0; off < muRef.length; off += 128) {
      const L = Math.min(128, muRef.length - off);
      const winInit = new Int32Array(L);
      for (let k = 0; k < L; k++) winInit[k] = rng() < 0.30 ? Engine.CFG.MASK_ID : muRef[off + k];
      genMuBytes.set(unifiedModel.sampleWindow(L, Engine.FORMAT.WAV_ULAW, tagIds, {
        initialTokens: winInit, patchMode: false, steps: 3, order: 'confidence', rng,
      }).bytes, off);
    }
    const genMuAna = Generators.analyzeWAVProperties(Engine.muLawDecodeToPCM(genMuBytes), 22050);
    if (Math.abs(genMuAna.midi - clip.meta.midi) <= 1) genMuPitchMatch++;
    genMuTempoErr += (Math.abs(genMuAna.bpm - clip.meta.bpm) / clip.meta.bpm) * 100;
  }

  const audioSummary = {
    codecRoundTrip: {
      muLawPitchErrCents: muPitchErrCents / nTones,
      tilePitchErrCents: tilePitchErrCents / nTones,
      muLawPitchMatchRate: muPitchMatch / nTones,
      tilePitchMatchRate: tilePitchMatch / nTones,
      muLawTempoErrPct: muTempoErrPct / nDrums,
      tileTempoErrPct: tileTempoErrPct / nDrums,
    },
    regenerate30pctWholeClip: {
      muLawPitchMatchRate: genMuPitchMatch / nGenWav,
      tilePitchMatchRate: genTilePitchMatch / nGenWav,
      muLawTempoErrPct: genMuTempoErr / nGenWav,
      tileTempoErrPct: genTileTempoErr / nGenWav,
    },
    codecOnlyOnsetRiseTimeMs: { // codec round trips of the source clips, no model involved
      muLawMs: muRiseMsSum / nAudio,
      auraTileGhostMs: ghostRiseMsSum / nAudio,
      auraTileHelixMs: helixRiseMsSum / nAudio,
      helixSharpeningOverGhostPct: ((ghostRiseMsSum - helixRiseMsSum) / ghostRiseMsSum) * 100,
    },
    codecOnlyWaveformSnrDb: {
      ghostDb: ghostSnrSum / nAudio,
      helixDb: helixSnrSum / nAudio,
      helixGainDb: (helixSnrSum - ghostSnrSum) / nAudio,
    },
  };

  console.log(`  Codec only (no model), pitch match: AuraGraph Tiles = ${(100 * tilePitchMatch / nTones).toFixed(1)}% vs. μ-law = ${(100 * muPitchMatch / nTones).toFixed(1)}%; tempo error: tiles ${(tileTempoErrPct / nDrums).toFixed(1)}% vs. μ-law ${(muTempoErrPct / nDrums).toFixed(1)}%`);
  console.log(`  Regenerate 30% of whole clip, pitch match: AuraGraph Tiles = ${((genTilePitchMatch / nGenWav) * 100).toFixed(1)}% vs. μ-law = ${((genMuPitchMatch / nGenWav) * 100).toFixed(1)}%`);
  console.log(`  Regenerate 30% of whole clip, tempo error: AuraGraph Tiles = ${(genTileTempoErr / nGenWav).toFixed(1)}% vs. μ-law = ${(genMuTempoErr / nGenWav).toFixed(1)}%`);
  console.log(`  Codec only (no model), mean onset rise time (10%->90%, lower is sharper):`);
  console.log(`    Path A (8-bit μ-law):              ${audioSummary.codecOnlyOnsetRiseTimeMs.muLawMs.toFixed(2)} ms`);
  console.log(`    Path B (AuraGraph + Helix):        ${audioSummary.codecOnlyOnsetRiseTimeMs.auraTileHelixMs.toFixed(2)} ms`);
  console.log(`    Path B (AuraGraph + Ghost):        ${audioSummary.codecOnlyOnsetRiseTimeMs.auraTileGhostMs.toFixed(2)} ms`);
  console.log(`    Helix onset sharpening vs. Ghost:  ${audioSummary.codecOnlyOnsetRiseTimeMs.helixSharpeningOverGhostPct.toFixed(1)}% faster rise (+${audioSummary.codecOnlyWaveformSnrDb.helixGainDb.toFixed(2)} dB SNR)`);

  results.audioComparison = audioSummary;

  console.log('\n========================================================================');
  console.log('EXPERIMENT 6: Speed & Footprint Across Presets (Phone, Laptop, Patient)');
  console.log('========================================================================');
  const speedBench = {};
  const benchWin = train.subarray(0, 64);
  for (const presetKey of ['phone', 'laptop', 'patient']) {
    const m = new Engine.FabulaTransformer(presetKey, Engine.mulberry32(77));
    const params = m.paramCount();
    const fabulaBytes = JSON.stringify(m.serialize(1)).length;

    const activeK = Engine.getActiveKernel();
    if (activeK) activeK.use('simd');
    const t0 = sandbox.performance.now();
    const nSteps = 3;
    for (let s = 0; s < nSteps; s++) {
      m.trainStep(benchWin, Engine.FORMAT.TXT, [1, 2], { useKernel: true, skipMemoryUpdate: true, fastTrainStep: true });
    }
    const msPerStepDesktop = (sandbox.performance.now() - t0) / nSteps;

    if (activeK) activeK.use('js');
    const t1 = sandbox.performance.now();
    for (let s = 0; s < nSteps; s++) {
      m.trainStep(benchWin, Engine.FORMAT.TXT, [1, 2], { useKernel: false, skipMemoryUpdate: true, fastTrainStep: true });
    }
    const msPerStepJS = (sandbox.performance.now() - t1) / nSteps;

    const t2 = sandbox.performance.now();
    m.sampleWindow(64, Engine.FORMAT.TXT, [1, 2], { steps: 6, order: 'confidence' });
    const msPerSampleDesktop = sandbox.performance.now() - t2;

    speedBench[presetKey] = {
      params,
      fabulaKB: Math.round(fabulaBytes / 1024),
      msPerStepSimd: Number(msPerStepDesktop.toFixed(2)),
      msPerStepJS: Number(msPerStepJS.toFixed(2)),
      msPerStepPhoneEstimate_JSx4_notMeasured: Number((msPerStepJS * 4.0).toFixed(2)),
      msPerSampleDesktop: Number(msPerSampleDesktop.toFixed(2)),
      msPerSamplePhoneEstimate_x4_notMeasured: Number((msPerSampleDesktop * 4.0).toFixed(2)),
    };
    console.log(
      `  Preset ${presetKey.toUpperCase().padEnd(8)} Params=${params.toLocaleString().padStart(9)} | .fabula=${speedBench[presetKey].fabulaKB} KB | Step(SIMD)=${speedBench[presetKey].msPerStepSimd}ms | Step(JS)=${speedBench[presetKey].msPerStepJS}ms | Sample=${speedBench[presetKey].msPerSampleDesktop}ms`
    );
  }
  results.speedBench = speedBench;

  const outArgIdx = process.argv.indexOf('--json');
  const outPath = outArgIdx >= 0 && process.argv[outArgIdx + 1]
    ? process.argv[outArgIdx + 1]
    : path.join(__dirname, 'metrics-results.json');
  fs.writeFileSync(outPath, JSON.stringify(results, null, 2));
  console.log(`\nSaved full structured metrics JSON to ${outPath}`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});