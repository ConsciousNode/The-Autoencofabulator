#!/usr/bin/env node
/**
 * Reproducible Honest Evaluation & Measurement Suite for The Autoencofabulator v0.4: "Learned".
 *
 * Implements the complete v0.4 Honest Measurement Protocol (zero training-data copying,
 * `noExemplarAnchors: true` everywhere, model-only `repairByRemasking`, held-out evaluation):
 *
 *   1. Sherlock Holmes 80/10/10 Held-Out Likelihood (Continuous-Time NELBO in bits/byte)
 *      across the ENTIRE 56,241-byte `test` split (all 878 64B windows and all 219 256B windows)
 *      plus the 24-window x 9-strata audit protocol, comparing:
 *        - Unigram, Bigram, Order-4 Interpolated Kneser-Ney (causal)
 *        - BilateralContextMemory (count memory alone)
 *        - FabulaTransformer alone (step 0 vs trained, `disableMemory: true`)
 *        - Full v0.4 model (step 0 vs trained, `pos_gate` + calibrated mixture)
 *        - Validation-split gate ablation: (a) `pos_gate`, (b) `residual`, (c) `mixture`
 *        - Window-length scaling (64, 128, 256, 512) and Preset scaling (Phone, Laptop, Patient)
 *   2. Held-Out Multi-Format Generation & Model-Only Repair (`seed 97531`, never seen in training)
 *      under three regimes (all with `noExemplarAnchors: true`, `K = 5` max repair passes):
 *        - 30% non-delimiter masked scaffold infilling
 *        - 65% non-delimiter masked scaffold infilling
 *        - 100% from-scratch generation
 *       across HTML, JSON, SVG, JS, C (verified with `gcc -fsyntax-only -xc`), BMP, WAV, and AuraTile,
 *      plus Longest Common Substring (LCS) memorisation ratios against the training corpus.
 *   3. Distribution Property Matching on Held-Out Targets:
 *        - HTML nesting depth (+/-1 of target depth) vs. a Random-Valid-HTML Baseline generator
 *        - BMP shape count (+/-1) and palette colour accuracy (model-only, zero exemplar anchors, zero dither)
 *   4. Co-Training (Unified Multi-Format Model) vs. 4 Format Specialists on Held-Out Documents (`seed 97531`).
 *   5. Fair Audio Rematch on Held-Out Clips (`seed 97531`):
 *        - Codec-only round-trip baseline (8-bit u-law vs. AuraGraph Ghost vs. AuraGraph Timing Helix)
 *        - 30% whole-clip infilling (both paths mask & regenerate 30% of the ENTIRE clip)
 *        - 100% from-scratch tag-conditioned audio generation
 *   6. Speed & Footprint Benchmarks across Phone, Laptop, and Patient Presets:
 *        - JS vs. Scalar WASM vs. SIMD WASM resident-weight `matmul_f32` + `linear_bwd_f32` throughput
 *        - 512-byte 24-step Phone generation latency (< 2.5s desktop / < 10s 4x throttled phone)
 *        - Fast typed-array `BilateralContextMemory` lookup throughput
 *
 * Run: node tests/metrics.js [--json tests/metrics-results.json]
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
const sandbox = globalThis;

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
  const activeK = Engine.getActiveKernel();
  if (activeK && activeK.simd) activeK.use('simd');

  console.log('========================================================================');
  console.log('EXPERIMENT 1: Sherlock Holmes 80/10/10 Held-Out Likelihood & Scaling');
  console.log('========================================================================');
  const rawSherlock = new Uint8Array(fs.readFileSync(resolveSherlockCorpus()));
  const { totalBytes, train, val, test } = Generators.splitSherlockCorpus(rawSherlock);
  console.log(`  Corpus: ${totalBytes} bytes -> train=${train.length}B, val=${val.length}B, test=${test.length}B`);

  const summaryPath = path.join(__dirname, 'train-sherlock-summary.json');
  if (!fs.existsSync(summaryPath)) {
    throw new Error('Missing tests/train-sherlock-summary.json; run `node tests/train-sherlock.js` first.');
  }
  const sherlockSummary = JSON.parse(fs.readFileSync(summaryPath, 'utf8'));
  console.log(`  Unigram (test):                       ${sherlockSummary.baselines.test.unigram.toFixed(4)} bpb`);
  console.log(`  Bigram (test):                        ${sherlockSummary.baselines.test.bigram.toFixed(4)} bpb`);
  console.log(`  Order-4 Kneser-Ney (causal, test):    ${sherlockSummary.baselines.test.kn4.toFixed(4)} bpb`);
  console.log(`  Count memory alone (test 64B all):    ${sherlockSummary.memoryOnly.test64All878.toFixed(4)} bpb (24w: ${sherlockSummary.memoryOnly.test64Sample24.toFixed(4)})`);
  console.log(`  Transformer alone, step 0 (64B all):  ${sherlockSummary.step0.testTfOnly64All878.toFixed(4)} bpb (24w: ${sherlockSummary.step0.testTfOnly64Sample24.toFixed(4)})`);
  console.log(`  Full model, step 0 (64B all):         ${sherlockSummary.step0.testFull64All878.toFixed(4)} bpb (24w: ${sherlockSummary.step0.testFull64Sample24.toFixed(4)})`);
  console.log(`  Transformer alone, trained (64B all): ${sherlockSummary.trainedPhone.test64All878.tfOnlyBpb.toFixed(4)} bpb (24w: ${sherlockSummary.trainedPhone.test64Sample24.tfOnlyBpb.toFixed(4)})`);
  console.log(`  Transformer alone, trained (256B all):${sherlockSummary.trainedPhone.test256All219.tfOnlyBpb.toFixed(4)} bpb`);
  console.log(`  Full model, trained (64B all):        ${sherlockSummary.trainedPhone.test64All878.fullBpb.toFixed(4)} bpb (24w: ${sherlockSummary.trainedPhone.test64Sample24.fullBpb.toFixed(4)})`);
  console.log(`  Full model, trained (256B all):       ${sherlockSummary.trainedPhone.test256All219.fullBpb.toFixed(4)} bpb`);
  console.log(`  Pretrained checkpoint:                ${sherlockSummary.trainedPhone.checkpointPath} (${sherlockSummary.trainedPhone.checkpointSizeMB} MB)`);

  results.sherlock = sherlockSummary;

  console.log('\n========================================================================');
  console.log('EXPERIMENT 2 & 3: Held-Out Multi-Format Validity, Repair, Property Match & LCS');
  console.log('========================================================================');
  Engine.CFG.SCHEDULE = 'linear';
  const N_TRAIN_PER_FMT = 16;
  const N_EVAL = 10;

  const suite = Generators.buildSeededSuite(4242, N_TRAIN_PER_FMT);
  const bmpRng = Generators.mulberry32(4242);
  suite.bmp = [];
  for (let i = 0; i < N_TRAIN_PER_FMT; i++) {
    suite.bmp.push(Generators.generateBMPSample(bmpRng, { size: 32, nShapes: 1 + (i % 3) }));
  }

  // Strictly held-out test documents generated with distinct seed 97531
  const heldOut = Generators.buildSeededSuite(97531, N_EVAL);
  const hbRng = Generators.mulberry32(97531);
  heldOut.bmp = [];
  for (let i = 0; i < N_EVAL; i++) {
    heldOut.bmp.push(Generators.generateBMPSample(hbRng, { size: 32, nShapes: 1 + (i % 3) }));
  }

  const unifiedRng = Engine.mulberry32(4242);
  const unifiedModel = new Engine.FabulaTransformer('phone', unifiedRng);

  // Populate format memories & train unified model across all formats on training suite only
  const fmtEntries = [
    { key: 'html', fid: Engine.FORMAT.HTML,     docs: suite.html },
    { key: 'json', fid: Engine.FORMAT.JSON,     docs: suite.json },
    { key: 'svg',  fid: Engine.FORMAT.SVG,      docs: suite.svg },
    { key: 'js',   fid: Engine.FORMAT.JS,       docs: suite.js },
    { key: 'c',    fid: Engine.FORMAT.C,        docs: suite.c },
    { key: 'bmp',  fid: Engine.FORMAT.BMP,      docs: suite.bmp },
    { key: 'wav',  fid: Engine.FORMAT.WAV_ULAW, docs: suite.wav },
  ];

  for (const f of fmtEntries) {
    for (let ep = 0; ep < 2; ep++) {
      for (const doc of f.docs) {
        const tagIds = Engine.tagToIds(doc.tags);
        if (f.fid === Engine.FORMAT.BMP) {
          const p = Engine.encodeBMPTo2DPatches(doc.bytes);
          unifiedModel.trainStep(p.tokens, Engine.FORMAT.BMP, tagIds, {
            rng: unifiedRng,
            patchMode: true,
            rowIds: p.rowIds,
            colIds: p.colIds,
            skipMemoryUpdate: ep > 0,
            fastTrainStep: true,
          });
        } else if (f.fid === Engine.FORMAT.WAV_ULAW) {
          const muBytes = Engine.muLawEncodePCM(doc.pcm.subarray(0, 256));
          unifiedModel.trainStep(muBytes, Engine.FORMAT.WAV_ULAW, tagIds, {
            rng: unifiedRng,
            patchMode: false,
            skipMemoryUpdate: ep > 0,
            fastTrainStep: true,
          });
          if (ep === 0) {
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
          }
        } else {
          unifiedModel.trainStep(doc.bytes, f.fid, tagIds, {
            rng: unifiedRng,
            patchMode: false,
            skipMemoryUpdate: ep > 0,
            fastTrainStep: true,
          });
        }
      }
    }
  }

  function buildHeldOutScaffold(docBytes, maskRate, idx, rngFn) {
    const L = docBytes.length;
    const initTokens = new Int32Array(L);
    if (maskRate >= 0.999) {
      initTokens.fill(Engine.CFG.MASK_ID);
      return initTokens;
    }
    for (let k = 0; k < L; k++) {
      const c = docBytes[k];
      const isDelim = (c === 123 || c === 125 || c === 60 || c === 62 || c === 58 || c === 59 || c === 34 || c === 40 || c === 41);
      const pMask = isDelim ? (idx % 3 === 0 ? 0.25 * (maskRate / 0.65) : 0.0) : maskRate;
      initTokens[k] = rngFn() < pMask ? Engine.CFG.MASK_ID : c;
    }
    return initTokens;
  }

  const textFormats = [
    { name: 'HTML', fid: Engine.FORMAT.HTML, trainDocs: suite.html, heldOutDocs: heldOut.html },
    { name: 'JSON', fid: Engine.FORMAT.JSON, trainDocs: suite.json, heldOutDocs: heldOut.json },
    { name: 'SVG',  fid: Engine.FORMAT.SVG,  trainDocs: suite.svg,  heldOutDocs: heldOut.svg },
    { name: 'JS',   fid: Engine.FORMAT.JS,   trainDocs: suite.js,   heldOutDocs: heldOut.js },
    { name: 'C',    fid: Engine.FORMAT.C,    trainDocs: suite.c,    heldOutDocs: heldOut.c },
  ];

  const validityByRegime = {
    infill30pct: {},
    infill65pct: {},
    fromScratch100pct: {},
  };

  for (const [regimeKey, maskRate, label] of [
    ['infill30pct', 0.30, 'Held-Out 30% Non-Delimiter Masked Infill'],
    ['infill65pct', 0.65, 'Held-Out 65% Non-Delimiter Masked Infill'],
    ['fromScratch100pct', 1.00, '100% From-Scratch Generation (Tag-Conditioned)'],
  ]) {
    console.log(`\n  --- Regime: ${label} (N=${N_EVAL} per format, noExemplarAnchors=true, K=5 max repair) ---`);
    for (const spec of textFormats) {
      let rawValid = 0, repairedValid = 0, totalPasses = 0, repairedCount = 0;
      const lcsList = [];
      const trainBytesList = spec.trainDocs.map(d => d.bytes);

      for (let i = 0; i < N_EVAL; i++) {
        const genRng = Engine.mulberry32(8000 + Math.round(maskRate * 1000) + spec.fid * 100 + i);
        const refDoc = spec.heldOutDocs[i % spec.heldOutDocs.length];
        const tagIds = Engine.tagToIds(refDoc.tags);
        const L = refDoc.bytes.length;
        const initTokens = buildHeldOutScaffold(refDoc.bytes, maskRate, i, genRng);

        const samp = unifiedModel.sampleWindow(L, spec.fid, tagIds, {
          initialTokens: initTokens,
          patchMode: false,
          steps: maskRate >= 0.99 ? 16 : 6,
          order: 'confidence',
          temperature: 0.55,
          rng: genRng,
          noExemplarAnchors: true,
        });
        const rawBytes = samp.bytes;
        const vRaw = Engine.validateOutput(rawBytes, spec.fid, {}, trainBytesList);
        const isRawOk = spec.name === 'C' ? (vRaw.parserOk && gccSyntaxCheck(rawBytes).ok) : vRaw.parserOk;

        if (isRawOk) {
          rawValid++;
          repairedValid++;
          lcsList.push(vRaw.lcsRatio);
        } else {
          const rep = Engine.repairByRemasking(unifiedModel, rawBytes, spec.fid, tagIds, 5, genRng);
          const isRepOk = spec.name === 'C' ? (rep.repairedValid && gccSyntaxCheck(rep.bytes).ok) : rep.repairedValid;
          if (isRepOk) {
            repairedValid++;
            totalPasses += rep.attemptsUsed;
            repairedCount++;
          }
          const vRep = Engine.validateOutput(rep.bytes, spec.fid, {}, trainBytesList);
          lcsList.push(vRep.lcsRatio);
        }
      }

      const meanLcs = lcsList.reduce((a, b) => a + b, 0) / Math.max(1, lcsList.length);
      const maxLcs  = Math.max(...lcsList);
      const meanPasses = repairedCount > 0 ? totalPasses / repairedCount : 0;
      validityByRegime[regimeKey][spec.name] = {
        rawValidRate: rawValid / N_EVAL,
        repairedValidRate: repairedValid / N_EVAL,
        meanRepairPassesWhenRepaired: Number(meanPasses.toFixed(2)),
        meanLcsRatio: Number(meanLcs.toFixed(4)),
        maxLcsRatio: Number(maxLcs.toFixed(4)),
        cCompilerUsed: spec.name === 'C' ? (gccAvailable ? 'gcc -fsyntax-only -xc' : 'built-in C parser') : undefined,
      };
      console.log(
        `    ${spec.name.padEnd(6)} Raw: ${((rawValid / N_EVAL) * 100).toFixed(1).padStart(5)}% -> Repaired (K<=5): ${((repairedValid / N_EVAL) * 100).toFixed(1).padStart(5)}% (mean passes=${meanPasses.toFixed(2)}, mean LCS=${meanLcs.toFixed(3)}, max LCS=${maxLcs.toFixed(3)})`
      );
    }
  }

  // Binary / 2-D container formats from scratch (BMP, WAV-uLaw, AuraTile) with noExemplarAnchors: true
  console.log('\n  --- Binary & 2-D Container Formats (100% From-Scratch, noExemplarAnchors=true) ---');
  for (const binSpec of [
    { name: 'BMP',      fid: Engine.FORMAT.BMP,      trainDocs: suite.bmp },
    { name: 'WAV-uLaw', fid: Engine.FORMAT.WAV_ULAW, trainDocs: suite.wav },
    { name: 'AuraTile', fid: Engine.FORMAT.AURA,     trainDocs: suite.wav },
  ]) {
    let rawValid = 0;
    const lcsList = [];
    const trainBytesList = binSpec.trainDocs.map(d => d.bytes);
    for (let i = 0; i < N_EVAL; i++) {
      const genRng = Engine.mulberry32(8900 + binSpec.fid * 100 + i);
      const refDoc = binSpec.trainDocs[i % binSpec.trainDocs.length];
      const tagIds = Engine.tagToIds(refDoc.tags);
      let rawFileBytes = null;
      if (binSpec.fid === Engine.FORMAT.BMP) {
        const samp = unifiedModel.sampleWindow(1024, Engine.FORMAT.BMP, tagIds, {
          patchMode: true, steps: 4, order: 'confidence', rng: genRng, noExemplarAnchors: true
        });
        rawFileBytes = Engine.decode2DPatchesToBMP(samp.bytes, 32, 32);
      } else if (binSpec.fid === Engine.FORMAT.WAV_ULAW) {
        const samp = unifiedModel.sampleWindow(128, Engine.FORMAT.WAV_ULAW, tagIds, {
          patchMode: false, steps: 4, order: 'confidence', rng: genRng, noExemplarAnchors: true
        });
        rawFileBytes = Generators.pcmFloatToWavBytes(Engine.muLawDecodeToPCM(samp.bytes), 8000);
      } else {
        const samp = unifiedModel.sampleWindow(640, Engine.FORMAT.AURA, tagIds, {
          patchMode: true, steps: 4, order: 'confidence', rng: genRng, noExemplarAnchors: true
        });
        const dec = Engine.decode2DPatchesToAuraTile(samp.bytes, 1, samp.helixLogits);
        rawFileBytes = Generators.pcmFloatToWavBytes(Engine.agInvertHelix(dec.A, dec.helix, 0, dec.A.F, 3), 22050);
      }
      const vRaw = Engine.validateOutput(rawFileBytes, binSpec.fid, {}, trainBytesList);
      if (vRaw.parserOk) rawValid++;
      lcsList.push(vRaw.lcsRatio);
    }
    const meanLcs = lcsList.reduce((a, b) => a + b, 0) / lcsList.length;
    const maxLcs  = Math.max(...lcsList);
    validityByRegime.fromScratch100pct[binSpec.name] = {
      rawValidRate: rawValid / N_EVAL,
      repairedValidRate: rawValid / N_EVAL,
      meanLcsRatio: Number(meanLcs.toFixed(4)),
      maxLcsRatio: Number(maxLcs.toFixed(4)),
    };
    console.log(
      `    ${binSpec.name.padEnd(8)} Raw/Repaired: ${((rawValid / N_EVAL) * 100).toFixed(1).padStart(5)}% (mean LCS=${meanLcs.toFixed(3)}, max LCS=${maxLcs.toFixed(3)})`
    );
  }

  // Property Matching on Held-Out Targets vs Random-Valid Baseline
  let htmlDepthHeldOutMatch = 0, htmlDepthScratchMatch = 0, htmlDepthRandomBaselineMatch = 0;
  const htmlEvalCount = N_EVAL;
  const baseHtmlRng = Generators.mulberry32(77777);
  for (let i = 0; i < htmlEvalCount; i++) {
    const refDoc = heldOut.html[i % heldOut.html.length];
    const tagIds = Engine.tagToIds(refDoc.tags);
    const rng = Engine.mulberry32(9100 + i);
    const initTokens = buildHeldOutScaffold(refDoc.bytes, 0.55, i, rng);
    const samp = unifiedModel.sampleWindow(refDoc.bytes.length, Engine.FORMAT.HTML, tagIds, {
      initialTokens: initTokens, steps: 6, order: 'confidence', rng, noExemplarAnchors: true
    });
    const rep = Engine.repairByRemasking(unifiedModel, samp.bytes, Engine.FORMAT.HTML, tagIds, 4, rng);
    const dInfill = Generators.measureHTMLDepth(new TextDecoder().decode(rep.bytes));
    if (Math.abs(dInfill - refDoc.meta.targetDepth) <= 1) htmlDepthHeldOutMatch++;

    // From-scratch tag-conditioned HTML generation
    const scratchRng = Engine.mulberry32(9200 + i);
    const sampScratch = unifiedModel.sampleWindow(refDoc.bytes.length, Engine.FORMAT.HTML, tagIds, {
      steps: 16, order: 'confidence', rng: scratchRng, noExemplarAnchors: true
    });
    const repScratch = Engine.repairByRemasking(unifiedModel, sampScratch.bytes, Engine.FORMAT.HTML, tagIds, 5, scratchRng);
    const dScratch = Generators.measureHTMLDepth(new TextDecoder().decode(repScratch.bytes));
    if (Math.abs(dScratch - refDoc.meta.targetDepth) <= 1) htmlDepthScratchMatch++;

    // Unconditioned random-valid-HTML baseline (ignores tags, draws random depth in 1..5)
    const randDoc = Generators.generateHTMLSample(baseHtmlRng, { depth: 1 + Math.floor(baseHtmlRng() * 5) });
    const dRand = Generators.measureHTMLDepth( randDoc.text );
    if (Math.abs(dRand - refDoc.meta.targetDepth) <= 1) htmlDepthRandomBaselineMatch++;
  }

  let bmpCountMatches = 0, bmpColorMatches = 0;
  for (let i = 0; i < N_EVAL; i++) {
    const targetDoc = heldOut.bmp[i % heldOut.bmp.length];
    const tagIds = Engine.tagToIds(targetDoc.tags);
    const samp = unifiedModel.sampleWindow(1024, Engine.FORMAT.BMP, tagIds, {
      patchMode: true, steps: 4, order: 'confidence', rng: Engine.mulberry32(9300 + i), noExemplarAnchors: true
    });
    const bmpBytes = Engine.decode2DPatchesToBMP(samp.bytes, 32, 32);
    const ana = Generators.analyzeBMPScene(bmpBytes);
    if (ana.valid && Math.abs(ana.nShapes - targetDoc.meta.nShapes) <= 1) bmpCountMatches++;
    if (ana.valid && targetDoc.meta.colors.some(c => ana.colors.includes(c))) bmpColorMatches++;
  }

  console.log(`\n  Property Match (Held-Out Targets, N=${N_EVAL}, noExemplarAnchors=true):`);
  console.log(`    HTML Nesting Depth (+/-1) — Held-Out Infill:       ${((htmlDepthHeldOutMatch / htmlEvalCount) * 100).toFixed(1)}%`);
  console.log(`    HTML Nesting Depth (+/-1) — From-Scratch Model:    ${((htmlDepthScratchMatch / htmlEvalCount) * 100).toFixed(1)}%`);
  console.log(`    HTML Nesting Depth (+/-1) — Random Valid Baseline: ${((htmlDepthRandomBaselineMatch / htmlEvalCount) * 100).toFixed(1)}%`);
  console.log(`    BMP Shape Count (+/-1)    — Model-Only:            ${((bmpCountMatches / N_EVAL) * 100).toFixed(1)}%`);
  console.log(`    BMP Palette Colour Match  — Model-Only:            ${((bmpColorMatches / N_EVAL) * 100).toFixed(1)}%`);

  results.validityByRegime = validityByRegime;
  results.propertyMatch = {
    htmlDepthHeldOutInfillAcc: htmlDepthHeldOutMatch / htmlEvalCount,
    htmlDepthFromScratchAcc: htmlDepthScratchMatch / htmlEvalCount,
    htmlDepthRandomValidBaselineAcc: htmlDepthRandomBaselineMatch / htmlEvalCount,
    bmpShapeCountModelOnlyAcc: bmpCountMatches / N_EVAL,
    bmpColorModelOnlyAcc: bmpColorMatches / N_EVAL,
  };

  console.log('\n========================================================================');
  console.log('EXPERIMENT 4: Co-Training (Unified) vs. Format Specialists on Held-Out Docs');
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
        windowSize: winLen, maxWindows: 1, strata: [0.2, 0.5, 0.8], seed: 500 + i
      });
      sumBpb += ev.bpb;
      count++;
    }
    return sumBpb / Math.max(1, count);
  }

  const specResults = {};
  for (const f of [
    { key: 'html', fid: Engine.FORMAT.HTML,     docs: suite.html },
    { key: 'json', fid: Engine.FORMAT.JSON,     docs: suite.json },
    { key: 'bmp',  fid: Engine.FORMAT.BMP,      docs: suite.bmp },
    { key: 'wav',  fid: Engine.FORMAT.WAV_ULAW, docs: suite.wav },
  ]) {
    const sRng = Engine.mulberry32(6100 + f.fid);
    const sModel = new Engine.FabulaTransformer('phone', sRng);
    for (let ep = 0; ep < 2; ep++) {
      for (const doc of f.docs) {
        const tagIds = Engine.tagToIds(doc.tags);
        if (f.fid === Engine.FORMAT.BMP) {
          const p = Engine.encodeBMPTo2DPatches(doc.bytes);
          sModel.trainStep(p.tokens, f.fid, tagIds, { rng: sRng, patchMode: true, rowIds: p.rowIds, colIds: p.colIds, skipMemoryUpdate: ep > 0, fastTrainStep: true });
        } else if (f.fid === Engine.FORMAT.WAV_ULAW) {
          sModel.trainStep(Engine.muLawEncodePCM(doc.pcm.subarray(0, 256)), f.fid, tagIds, { rng: sRng, skipMemoryUpdate: ep > 0, fastTrainStep: true });
        } else {
          sModel.trainStep(doc.bytes, f.fid, tagIds, { rng: sRng, skipMemoryUpdate: ep > 0, fastTrainStep: true });
        }
      }
    }
    const specBpb = evalHeldOutFormatBpb(sModel, heldOut[f.key], f.fid);
    const uniBpb  = evalHeldOutFormatBpb(unifiedModel, heldOut[f.key], f.fid);
    specResults[f.key] = { specialistBpb: Number(specBpb.toFixed(4)), unifiedBpb: Number(uniBpb.toFixed(4)), deltaBpb: Number((uniBpb - specBpb).toFixed(4)) };
    console.log(`  Format ${f.key.toUpperCase().padEnd(5)} Held-Out Specialist: ${specBpb.toFixed(4)} bpb | Unified: ${uniBpb.toFixed(4)} bpb | Delta: ${(uniBpb - specBpb >= 0 ? '+' : '') + (uniBpb - specBpb).toFixed(4)} bpb`);
  }
  results.coTrainingVsSpecialist = specResults;

  console.log('\n========================================================================');
  console.log('EXPERIMENT 5: Fair Audio Rematch on Held-Out Clips (30% Whole-Clip & From-Scratch)');
  console.log('========================================================================');
  let muPitchErrCents = 0, tilePitchErrCents = 0, muPitchMatch = 0, tilePitchMatch = 0;
  let muTempoErrPct = 0, tileTempoErrPct = 0;
  let muRiseMsSum = 0, ghostRiseMsSum = 0, helixRiseMsSum = 0;
  let ghostSnrSum = 0, helixSnrSum = 0;
  const nAudio = heldOut.wav.length;

  for (const clip of heldOut.wav) {
    const refPCM = clip.pcm, sr = clip.sr;
    const muAna = Generators.analyzeWAVProperties(Engine.muLawDecodeToPCM(Engine.muLawEncodePCM(refPCM)), sr);
    const agState = Engine.agAnalyzeSync(refPCM, false);
    const helixData = Engine.agBuildHelixData(refPCM, agState, 0, agState.F);
    const pcmGhost = Engine.agInvertGhost(agState, 0, agState.F, 10);
    const pcmHelix = Engine.agInvertHelix(agState, helixData, 0, agState.F, 5);
    const ghostAna = Generators.analyzeWAVProperties(pcmGhost, sr);
    const helixAna = Generators.analyzeWAVProperties(pcmHelix, sr);

    const calcSnr = (rec) => {
      const L = Math.min(refPCM.length, rec.length);
      let sig = 0, err = 0;
      for (let i = 0; i < L; i++) { sig += refPCM[i] * refPCM[i]; const d = refPCM[i] - rec[i]; err += d * d; }
      return 10 * Math.log10(Math.max(sig, 1e-12) / Math.max(err, 1e-12));
    };
    ghostSnrSum += calcSnr(pcmGhost);
    helixSnrSum += calcSnr(pcmHelix);
    muRiseMsSum += muAna.riseMs;
    ghostRiseMsSum += ghostAna.riseMs;
    helixRiseMsSum += helixAna.riseMs;

    muPitchErrCents   += Math.abs(1200 * Math.log2(Math.max(20, muAna.freq) / clip.meta.freq));
    tilePitchErrCents += Math.abs(1200 * Math.log2(Math.max(20, helixAna.freq) / clip.meta.freq));
    if (Math.abs(muAna.midi - clip.meta.midi) <= 1) muPitchMatch++;
    if (Math.abs(helixAna.midi - clip.meta.midi) <= 1) tilePitchMatch++;
    muTempoErrPct   += (Math.abs(muAna.bpm - clip.meta.bpm) / clip.meta.bpm) * 100;
    tileTempoErrPct += (Math.abs(helixAna.bpm - clip.meta.bpm) / clip.meta.bpm) * 100;
  }

  // Fair 30% Whole-Clip Infilling AND 100% From-Scratch on Held-Out Clips
  let infill30MuPitch = 0, infill30TilePitch = 0, infill30MuTempoErr = 0, infill30TileTempoErr = 0;
  let scratchMuPitch = 0, scratchTilePitch = 0, scratchMuTempoErr = 0, scratchTileTempoErr = 0;
  const nGenWav = 6;

  for (let i = 0; i < nGenWav; i++) {
    const clip = heldOut.wav[i];
    const tagIds = Engine.tagToIds(clip.tags);
    const rng = Engine.mulberry32(9500 + i);

    const A = Engine.agAnalyzeSync(clip.pcm, true);
    const helix = Engine.agBuildHelixData(clip.pcm, A, 0, A.F);
    const enc = Engine.encodeAuraTileTo2DPatches(A, helix);

    // 1. 30% whole-clip masked (Path B: AuraGraph Tile)
    const initTile = new Int32Array(enc.tokens.length);
    for (let k = 0; k < initTile.length; k++) initTile[k] = rng() < 0.30 ? Engine.CFG.MASK_ID : enc.tokens[k];
    const sampTile30 = unifiedModel.sampleWindow(initTile.length, Engine.FORMAT.AURA, tagIds, {
      initialTokens: initTile, patchMode: true, rowIds: enc.rowIds, colIds: enc.colIds,
      steps: 3, order: 'confidence', rng, noExemplarAnchors: true
    });
    const decTile30 = Engine.decode2DPatchesToAuraTile(sampTile30.bytes, enc.gridW, sampTile30.helixLogits, enc.range, enc.ref);
    const anaTile30 = Generators.analyzeWAVProperties(Engine.agInvertHelix(decTile30.A, decTile30.helix, 0, decTile30.A.F, 4), 22050);
    if (Math.abs(anaTile30.midi - clip.meta.midi) <= 1) infill30TilePitch++;
    infill30TileTempoErr += (Math.abs(anaTile30.bpm - clip.meta.bpm) / clip.meta.bpm) * 100;

    // 2. 30% whole-clip masked (Path A: 8-bit u-law across entire clip)
    const muRef = Engine.muLawEncodePCM(clip.pcm);
    const genMu30 = new Uint8Array(muRef.length);
    for (let off = 0; off < muRef.length; off += 128) {
      const L = Math.min(128, muRef.length - off);
      const winInit = new Int32Array(L);
      for (let k = 0; k < L; k++) winInit[k] = rng() < 0.30 ? Engine.CFG.MASK_ID : muRef[off + k];
      genMu30.set(unifiedModel.sampleWindow(L, Engine.FORMAT.WAV_ULAW, tagIds, {
        initialTokens: winInit, patchMode: false, steps: 3, order: 'confidence', rng, noExemplarAnchors: true
      }).bytes, off);
    }
    const anaMu30 = Generators.analyzeWAVProperties(Engine.muLawDecodeToPCM(genMu30), 22050);
    if (Math.abs(anaMu30.midi - clip.meta.midi) <= 1) infill30MuPitch++;
    infill30MuTempoErr += (Math.abs(anaMu30.bpm - clip.meta.bpm) / clip.meta.bpm) * 100;

    // 3. 100% From-Scratch generation (Path B: AuraGraph Tile conditioned on tags only)
    const sampTileScratch = unifiedModel.sampleWindow(enc.tokens.length, Engine.FORMAT.AURA, tagIds, {
      patchMode: true, rowIds: enc.rowIds, colIds: enc.colIds,
      steps: 4, order: 'confidence', rng: Engine.mulberry32(9600 + i), noExemplarAnchors: true
    });
    const decTileSc = Engine.decode2DPatchesToAuraTile(sampTileScratch.bytes, enc.gridW, sampTileScratch.helixLogits);
    const anaTileSc = Generators.analyzeWAVProperties(Engine.agInvertHelix(decTileSc.A, decTileSc.helix, 0, decTileSc.A.F, 4), 22050);
    if (Math.abs(anaTileSc.midi - clip.meta.midi) <= 1) scratchTilePitch++;
    scratchTileTempoErr += (Math.abs(anaTileSc.bpm - clip.meta.bpm) / clip.meta.bpm) * 100;

    // 4. 100% From-Scratch generation (Path A: u-law conditioned on tags only)
    const sampMuScratch = unifiedModel.sampleWindow(512, Engine.FORMAT.WAV_ULAW, tagIds, {
      patchMode: false, steps: 4, order: 'confidence', rng: Engine.mulberry32(9700 + i), noExemplarAnchors: true
    });
    const anaMuSc = Generators.analyzeWAVProperties(Engine.muLawDecodeToPCM(sampMuScratch.bytes), 22050);
    if (Math.abs(anaMuSc.midi - clip.meta.midi) <= 1) scratchMuPitch++;
    scratchMuTempoErr += (Math.abs(anaMuSc.bpm - clip.meta.bpm) / clip.meta.bpm) * 100;
  }

  const audioSummary = {
    codecRoundTripHeldOut: {
      muLawPitchErrCents: Number((muPitchErrCents / nAudio).toFixed(2)),
      tilePitchErrCents: Number((tilePitchErrCents / nAudio).toFixed(2)),
      muLawPitchMatchRate: muPitchMatch / nAudio,
      tilePitchMatchRate: tilePitchMatch / nAudio,
      muLawTempoErrPct: Number((muTempoErrPct / nAudio).toFixed(2)),
      tileTempoErrPct: Number((tileTempoErrPct / nAudio).toFixed(2)),
    },
    infill30pctWholeClipHeldOut: {
      muLawPitchMatchRate: infill30MuPitch / nGenWav,
      tilePitchMatchRate: infill30TilePitch / nGenWav,
      muLawTempoErrPct: Number((infill30MuTempoErr / nGenWav).toFixed(2)),
      tileTempoErrPct: Number((infill30TileTempoErr / nGenWav).toFixed(2)),
    },
    fromScratch100pctHeldOut: {
      muLawPitchMatchRate: scratchMuPitch / nGenWav,
      tilePitchMatchRate: scratchTilePitch / nGenWav,
      muLawTempoErrPct: Number((scratchMuTempoErr / nGenWav).toFixed(2)),
      tileTempoErrPct: Number((scratchTileTempoErr / nGenWav).toFixed(2)),
    },
    codecOnlyOnsetRiseTimeMs: {
      muLawMs: Number((muRiseMsSum / nAudio).toFixed(2)),
      auraTileGhostMs: Number((ghostRiseMsSum / nAudio).toFixed(2)),
      auraTileHelixMs: Number((helixRiseMsSum / nAudio).toFixed(2)),
      helixSharpeningOverGhostPct: Number((((ghostRiseMsSum - helixRiseMsSum) / ghostRiseMsSum) * 100).toFixed(2)),
    },
    codecOnlyWaveformSnrDb: {
      ghostDb: Number((ghostSnrSum / nAudio).toFixed(2)),
      helixDb: Number((helixSnrSum / nAudio).toFixed(2)),
      helixGainDb: Number(((helixSnrSum - ghostSnrSum) / nAudio).toFixed(2)),
    },
  };

  console.log(`  Codec Round-Trip (Held-Out, N=${nAudio}):`);
  console.log(`    AuraGraph + Helix: Pitch Match=${(100 * tilePitchMatch / nAudio).toFixed(1)}% (${audioSummary.codecRoundTripHeldOut.tilePitchErrCents}c), Tempo Err=${audioSummary.codecRoundTripHeldOut.tileTempoErrPct}%`);
  console.log(`    8-bit u-law:       Pitch Match=${(100 * muPitchMatch / nAudio).toFixed(1)}% (${audioSummary.codecRoundTripHeldOut.muLawPitchErrCents}c), Tempo Err=${audioSummary.codecRoundTripHeldOut.muLawTempoErrPct}%`);
  console.log(`  30% Whole-Clip Infilling (Held-Out, N=${nGenWav}):`);
  console.log(`    AuraGraph + Helix: Pitch Match=${((infill30TilePitch / nGenWav) * 100).toFixed(1)}%, Tempo Err=${audioSummary.infill30pctWholeClipHeldOut.tileTempoErrPct}%`);
  console.log(`    8-bit u-law:       Pitch Match=${((infill30MuPitch / nGenWav) * 100).toFixed(1)}%, Tempo Err=${audioSummary.infill30pctWholeClipHeldOut.muLawTempoErrPct}%`);
  console.log(`  100% From-Scratch Generation (Held-Out Tags, N=${nGenWav}):`);
  console.log(`    AuraGraph + Helix: Pitch Match=${((scratchTilePitch / nGenWav) * 100).toFixed(1)}%, Tempo Err=${audioSummary.fromScratch100pctHeldOut.tileTempoErrPct}%`);
  console.log(`    8-bit u-law:       Pitch Match=${((scratchMuPitch / nGenWav) * 100).toFixed(1)}%, Tempo Err=${audioSummary.fromScratch100pctHeldOut.muLawTempoErrPct}%`);
  console.log(`  Onset Sharpness (10%->90% rise): Helix=${audioSummary.codecOnlyOnsetRiseTimeMs.auraTileHelixMs}ms vs Ghost=${audioSummary.codecOnlyOnsetRiseTimeMs.auraTileGhostMs}ms (${audioSummary.codecOnlyOnsetRiseTimeMs.helixSharpeningOverGhostPct}% faster, +${audioSummary.codecOnlyWaveformSnrDb.helixGainDb} dB SNR)`);

  results.audioComparison = audioSummary;

  console.log('\n========================================================================');
  console.log('EXPERIMENT 6: Speed & Memory Benchmarks (JS vs WASM vs SIMD across Presets)');
  console.log('========================================================================');
  const speedBench = {};
  const benchWin = train.subarray(0, 256);

  // Measure BilateralContextMemory lookup throughput
  const memBench = new Engine.BilateralContextMemory();
  memBench.observeSequence(train.subarray(0, 65536), [1, 2], 1);
  const tMem0 = sandbox.performance.now();
  const nLookups = 20000;
  for (let i = 0; i < nLookups; i++) {
    memBench.predictLogProbsAt(benchWin, (i & 255), [1, 2], false, true);
  }
  const memMs = sandbox.performance.now() - tMem0;
  const lookupsPerSec = Math.round((nLookups / memMs) * 1000);
  console.log(`  Fast Typed-Array BilateralContextMemory: ${lookupsPerSec.toLocaleString()} lookups/sec (${(memMs / nLookups * 1000).toFixed(2)} us/lookup)`);

  for (const presetKey of ['phone', 'laptop', 'patient']) {
    const m = new Engine.FabulaTransformer(presetKey, Engine.mulberry32(77));
    m.memories[Engine.FORMAT.TXT] = memBench;
    const params = m.paramCount();
    const fabulaBytes = JSON.stringify(m.serialize(1)).length;
    const nSteps = 4;

    const measureBackend = (mode, useK) => {
      if (activeK && mode !== 'js') activeK.use(mode);
      const t0 = sandbox.performance.now();
      for (let s = 0; s < nSteps; s++) {
        m.trainStep(benchWin, Engine.FORMAT.TXT, [1, 2], { useKernel: useK, skipMemoryUpdate: true, disableMemory: true });
      }
      return (sandbox.performance.now() - t0) / nSteps;
    };

    const msStepJS   = measureBackend('js', false);
    const msStepWasm = measureBackend('wasm', true);
    const msStepSimd = measureBackend('simd', true);

    if (activeK) activeK.use('simd');
    const tSamp0 = sandbox.performance.now();
    m.sampleWindow(512, Engine.FORMAT.TXT, [1, 2], { steps: 24, order: 'confidence', useKernel: true, rng: Engine.mulberry32(99) });
    const msSample512B24Step = sandbox.performance.now() - tSamp0;

    speedBench[presetKey] = {
      params,
      ctxBuckets: m.ctxBuckets,
      fabulaKB: Math.round(fabulaBytes / 1024),
      msPerStep256B_JS: Number(msStepJS.toFixed(2)),
      msPerStep256B_WASM: Number(msStepWasm.toFixed(2)),
      msPerStep256B_SIMD: Number(msStepSimd.toFixed(2)),
      tokensPerSecTrain_JS: Math.round((256 / msStepJS) * 1000),
      tokensPerSecTrain_WASM: Math.round((256 / msStepWasm) * 1000),
      tokensPerSecTrain_SIMD: Math.round((256 / msStepSimd) * 1000),
      simdSpeedupVsJS: Number((msStepJS / msStepSimd).toFixed(2)),
      msSample512B24StepDesktop: Number(msSample512B24Step.toFixed(1)),
      msSample512B24StepPhoneEst4x: Number((msSample512B24Step * 4.0).toFixed(1)),
    };
    console.log(
      `  Preset ${presetKey.toUpperCase().padEnd(7)} Params=${params.toLocaleString().padStart(9)} | Step(256B): JS=${msStepJS.toFixed(1)}ms, WASM=${msStepWasm.toFixed(1)}ms, SIMD=${msStepSimd.toFixed(1)}ms (${speedBench[presetKey].simdSpeedupVsJS}x) | 512B 24-step Sample=${msSample512B24Step.toFixed(0)}ms (4x phone=${(msSample512B24Step * 4 / 1000).toFixed(2)}s)`
    );
  }
  results.memoryLookupThroughputPerSec = lookupsPerSec;
  results.speedBench = speedBench;

  const outArgIdx = process.argv.indexOf('--json');
  const outPath = outArgIdx >= 0 && process.argv[outArgIdx + 1]
    ? process.argv[outArgIdx + 1]
    : path.join(__dirname, 'metrics-results.json');
  fs.writeFileSync(outPath, JSON.stringify(results, null, 2));
  console.log(`\nSaved full structured v0.4 metrics JSON to ${outPath}`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});