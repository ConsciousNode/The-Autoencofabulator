#!/usr/bin/env node
/**
 * Kehai's audit of The Autoencofabulator v0.3 "Realized" (2026-11-04).
 *
 * Re-runs each REALIZED-1 experiment next to a version that controls for the
 * problem found in the code, so both numbers sit side by side:
 *
 *   A. Sherlock: score far more test bytes (the report scored 384 of 56,241),
 *      and split the score into transformer-only / count-memory-only / both.
 *   B. Repair: how many "repaired" files are the model's own repair, a training
 *      document with re-rolled digits, or a training document byte for byte.
 *      Then the model-only repair rate, on scaffolds from documents it never saw.
 *   C. BMP: property match with and without the training-image anchors that
 *      sampleWindow pastes in by default (about 78% of patches).
 *   D. Co-training: unified vs specialist on held-out documents (the report
 *      scored the first 3 training documents).
 *   E. Audio: both paths regenerate the same fraction (30%) of the WHOLE clip
 *      (the report regenerated 30% of 8 tile frames, vs 489 of 512 mu-law
 *      samples measured over 23 ms).
 *
 * Run: node tests/audit-v03.js [--quick] [--engine path/to/index.html] [--json out.json]
 *   --engine lets you point it at the v0.3.0 index.html as delivered (git show 80b8688:index.html)
 */
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const Generators = require('./generators.js');

const QUICK = process.argv.includes('--quick');
const ei = process.argv.indexOf('--engine');
const html = fs.readFileSync(ei >= 0 ? process.argv[ei + 1] : path.join(__dirname, '..', 'index.html'), 'utf8');
const s0 = html.indexOf('// === FABULA_ENGINE_START ===');
const s1 = html.indexOf('// === FABULA_ENGINE_END ===');
globalThis.window = globalThis; globalThis.self = globalThis;
vm.runInThisContext(html.slice(s0 + 30, s1));
const E = globalThis.FabulaEngine;
const F = E.FORMAT;
const out = {};
const pct = (a, b) => (100 * a / b).toFixed(1) + '%';
const eqBytes = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
const skeleton = (u8) => Buffer.from(u8).toString('latin1').replace(/[0-9]/g, '#');

// ─── A. Sherlock ────────────────────────────────────────────────────────────
function logSoftmaxRow(row, off) {
  let m = -1e30; for (let v = 0; v < 256; v++) if (row[off + v] > m) m = row[off + v];
  let s = 0; for (let v = 0; v < 256; v++) s += Math.exp(row[off + v] - m);
  return m + Math.log(s);
}
function sherlock() {
  console.log('\n== A. Sherlock held-out likelihood ==');
  const raw = new Uint8Array(fs.readFileSync(path.join(__dirname, '..', 'sherlock-holmes-corpus.txt')));
  const { train, test } = Generators.splitSherlockCorpus(raw);
  const base = E.fitByteBaselines(train);
  const kn4 = base.evalKneserNey4(test);

  // Same model as metrics.js picks (phone-linear, seed 101, 20 steps), plus a 600-step one
  function build(steps) {
    E.CFG.SCHEDULE = 'linear';
    const rng = E.mulberry32(101);
    const m = new E.FabulaTransformer('phone', rng);
    const tag = E.tagToIds('txt prose english');
    for (let off = 0; off + 256 <= train.length; off += 1024)
      m.memories[F.TXT].observeSequence(train.subarray(off, off + 256), tag, 1);
    for (let s = 0; s < steps; s++) {
      const st = Math.floor(rng() * (train.length - 64));
      m.trainStep(train.subarray(st, st + 64), F.TXT, tag, { rng, skipMemoryUpdate: true, fastTrainStep: true, useKernel: true });
    }
    return { m, tag };
  }

  const strata = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
  function evalSplit(m, tag, nWin) {
    const W = 64, mem = m.memories[F.TXT];
    const gate = Math.log(1 + Math.exp(m.ctxGate[F.TXT]));
    const step = Math.floor((test.length - W) / nWin);
    const rng = E.mulberry32(303);
    const res = { full: {}, transformer: {}, memory: {} };
    for (const t of strata) {
      const w = E.MASK_DIFF.nelboWeight(t, 'linear');
      const acc = { full: 0, transformer: 0, memory: 0 };
      for (let k = 0; k < nWin; k++) {
        const win = test.subarray(k * step, k * step + W);
        const { xt, isMasked } = E.MASK_DIFF.forwardMask(win, t, rng, null, 'linear');
        const { logits } = m.forward(xt, t, F.TXT, tag, { useKernel: false, disableMemory: true });
        const comb = new Float64Array(256);
        for (let i = 0; i < W; i++) {
          if (!isMasked[i]) continue;
          const off = i * 256, y = win[i];
          const lp = mem.predictLogProbsAt(xt, i, tag, false);
          acc.transformer += -(logits[off + y] - logSoftmaxRow(logits, off)) / Math.LN2;
          acc.memory += -(lp[y] - logSoftmaxRow(lp, 0)) / Math.LN2;
          for (let v = 0; v < 256; v++) comb[v] = logits[off + v] + gate * lp[v];
          acc.full += -(comb[y] - logSoftmaxRow(comb, 0)) / Math.LN2;
        }
      }
      for (const key in acc) res[key][t.toFixed(1)] = w * acc[key] / (W * nWin);
    }
    for (const key in res) res[key].mean = strata.reduce((a, t) => a + res[key][t.toFixed(1)], 0) / strata.length;
    return res;
  }

  const r = {};
  for (const [label, steps, nWin] of [['as reported, 6 windows', 20, 6], ['20 steps', 20, QUICK ? 60 : 300], ['600 steps', 600, QUICK ? 60 : 300]]) {
    const { m, tag } = build(steps);
    r[label] = evalSplit(m, tag, nWin);
    r[label].bytesScored = nWin * 64;
    console.log(`  ${label.padEnd(24)} (${nWin * 64} test bytes) full=${r[label].full.mean.toFixed(4)}  transformer-only=${r[label].transformer.mean.toFixed(4)}  memory-only=${r[label].memory.mean.toFixed(4)}  | t=0.1 full=${r[label].full['0.1'].toFixed(3)} memory=${r[label].memory['0.1'].toFixed(3)}`);
  }
  console.log(`  KN4 on all ${test.length} test bytes: ${kn4.toFixed(4)}`);
  out.sherlock = { kn4Test: kn4, runs: r };
}

// ─── B. Repair provenance ──────────────────────────────────────────────────
function trainUnified(suite, seed = 4242) {
  E.CFG.SCHEDULE = 'linear';
  const rng = E.mulberry32(seed);
  const m = new E.FabulaTransformer('phone', rng);
  const plan = [['html', F.HTML], ['json', F.JSON], ['svg', F.SVG], ['js', F.JS], ['c', F.C], ['bmp', F.BMP], ['wav', F.WAV_ULAW]];
  for (const [k, fid] of plan) for (const d of suite[k]) {
    const tg = E.tagToIds(d.tags);
    if (fid === F.BMP) { const p = E.encodeBMPTo2DPatches(d.bytes); m.trainStep(p.tokens, fid, tg, { rng, patchMode: true, rowIds: p.rowIds, colIds: p.colIds, fastTrainStep: true }); }
    else if (fid === F.WAV_ULAW) {
      // same as metrics.js: one mu-law step and one AuraGraph-tile step per clip
      m.trainStep(E.muLawEncodePCM(d.pcm.subarray(0, 128)), fid, tg, { rng, patchMode: false, fastTrainStep: true });
      const A = E.agAnalyzeSync(d.pcm.subarray(0, 4096), true), hx = E.agBuildHelixData(d.pcm.subarray(0, 4096), A, 0, A.F);
      const te = E.encodeAuraTileTo2DPatches(A, hx), nP = Math.min(40, te.rowIds.length);
      m.trainStep(te.tokens.subarray(0, nP * 16), F.AURA, tg, { rng, patchMode: true, rowIds: te.rowIds.subarray(0, nP), colIds: te.colIds.subarray(0, nP), helixTargets: te.helixTargets.subarray(0, nP), fastTrainStep: true });
    }
    else m.trainStep(d.bytes, fid, tg, { rng, fastTrainStep: true });
  }
  return m;
}
// Argon's repair loop with the training-document fallback removed (lines 4229-4274 of index.html)
function modelOnlyRepair(model, raw, fid, tagIds, maxAttempts, rng) {
  let cur = Uint8Array.from(raw);
  for (let a = 1; a <= maxAttempts; a++) {
    const v = E.validateOutput(cur, fid);
    if (v.parserOk) return { ok: true, bytes: cur };
    const L = cur.length, c = v.errPos >= 0 ? v.errPos : L >> 1, rad = Math.min(L, 10 + a * 14);
    const init = new Int32Array(L);
    for (let i = 0; i < L; i++) {
      const bad = cur[i] < 9 || (cur[i] > 13 && cur[i] < 32) || cur[i] > 126;
      init[i] = (i >= c - rad && i <= c + rad) || bad ? E.CFG.MASK_ID : cur[i];
    }
    cur = model.sampleWindow(L, fid, tagIds, { initialTokens: init, steps: 12, order: 'confidence', remask: false, temperature: Math.max(0.25, 0.65 - a * 0.1), rng, noExemplarAnchors: true }).bytes;
  }
  return { ok: E.validateOutput(cur, fid).parserOk, bytes: cur };
}
function gcc(bytes) {
  const r = require('child_process').spawnSync('gcc', ['-fsyntax-only', '-xc', '-'], { input: Buffer.from(bytes), timeout: 4000 });
  return r.status === 0;
}
function scaffold(doc, i, rng) {
  const L = doc.bytes.length, init = new Int32Array(L);
  for (let k = 0; k < L; k++) {
    const c = doc.bytes[k];
    const delim = [123, 125, 60, 62, 58, 59, 34, 40, 41].includes(c);
    init[k] = rng() < (delim ? (i % 3 === 0 ? 0.25 : 0) : 0.65) ? E.CFG.MASK_ID : c;
  }
  return init;
}
function repair(train, heldOut) {
  console.log('\n== B. Repair-by-remasking: where the valid files come from ==');
  const model = trainUnified(train);
  const specs = [['HTML', F.HTML, 'html'], ['JSON', F.JSON, 'json'], ['SVG', F.SVG, 'svg'], ['JS', F.JS, 'js'], ['C', F.C, 'c']];
  const res = {};
  for (const [name, fid, key] of specs) {
    const trainBytes = train[key].map(d => d.bytes);
    const trainSkel = new Set(trainBytes.map(skeleton));
    const okFn = b => E.validateOutput(b, fid).parserOk && (name !== 'C' || gcc(b));
    const r = { raw: 0, argonValid: 0, verbatimTrainingDoc: 0, trainingDocDigitsRerolled: 0, modelOnlyValid_trainScaffold: 0, modelOnlyValid_heldOutScaffold: 0, n: 10 };
    for (let i = 0; i < 10; i++) {
      // as metrics.js: scaffold = a TRAINING doc with 65% of non-delimiters masked
      const rng = E.mulberry32(8000 + fid * 100 + i);
      const ref = train[key][i % train[key].length], tg = E.tagToIds(ref.tags);
      const samp = model.sampleWindow(ref.bytes.length, fid, tg, { initialTokens: scaffold(ref, i, rng), steps: 4, order: 'confidence', temperature: 0.6, rng, noExemplarAnchors: true }).bytes;
      if (okFn(samp)) { r.raw++; r.argonValid++; r.modelOnlyValid_trainScaffold++; }
      else {
        const rep = E.repairByRemasking(model, samp, fid, tg, 3, E.mulberry32(8000 + fid * 100 + i + 50));
        if (rep.repairedValid && okFn(rep.bytes)) {
          r.argonValid++;
          if (trainBytes.some(b => eqBytes(b, rep.bytes))) r.verbatimTrainingDoc++;
          else if (trainSkel.has(skeleton(rep.bytes))) r.trainingDocDigitsRerolled++;
        }
        const mo = modelOnlyRepair(model, samp, fid, tg, 3, E.mulberry32(8000 + fid * 100 + i + 50));
        if (mo.ok && okFn(mo.bytes)) r.modelOnlyValid_trainScaffold++;
      }
      // held-out scaffold: a document from a different seed, never trained on
      const ho = heldOut[key][i % heldOut[key].length], hrng = E.mulberry32(9800 + fid * 100 + i), htg = E.tagToIds(ho.tags);
      const hs = model.sampleWindow(ho.bytes.length, fid, htg, { initialTokens: scaffold(ho, i, hrng), steps: 4, order: 'confidence', temperature: 0.6, rng: hrng, noExemplarAnchors: true }).bytes;
      const hm = okFn(hs) ? { bytes: hs } : modelOnlyRepair(model, hs, fid, htg, 3, hrng);
      if (okFn(hm.bytes)) r.modelOnlyValid_heldOutScaffold++;
    }
    res[name] = r;
    console.log(`  ${name.padEnd(5)} raw ${pct(r.raw, 10).padStart(6)} | engine's repair ${pct(r.argonValid, 10).padStart(6)} (verbatim training doc ${r.verbatimTrainingDoc}, training doc w/ digits re-rolled ${r.trainingDocDigitsRerolled}) | model-only repair: train scaffolds ${pct(r.modelOnlyValid_trainScaffold, 10)}, held-out scaffolds ${pct(r.modelOnlyValid_heldOutScaffold, 10)}`);
  }
  out.repair = res;
  return model;
}

// ─── C. BMP anchors ─────────────────────────────────────────────────────────
function bmp(model, train) {
  console.log('\n== C. BMP property match, with and without pasted training patches ==');
  const r = {};
  for (const anchors of [true, false]) {
    let cnt = 0, col = 0, copied = 0;
    for (let i = 0; i < 8; i++) {
      const tgt = train.bmp[i % train.bmp.length], tg = E.tagToIds(tgt.tags);
      const s = model.sampleWindow(1024, F.BMP, tg, { patchMode: true, steps: 3, order: 'confidence', rng: E.mulberry32(9300 + i), noExemplarAnchors: !anchors });
      const a = Generators.analyzeBMPScene(E.decode2DPatchesToBMP(s.bytes, 32, 32));
      if (a.valid && Math.abs(a.nShapes - tgt.meta.nShapes) <= 1) cnt++;
      if (a.valid && tgt.meta.colors.some(c => a.colors.includes(c))) col++;
      // fraction of output bytes identical to the nearest training patch stream
      let best = 0;
      for (const d of train.bmp) { const p = E.encodeBMPTo2DPatches(d.bytes).tokens; if (p.length !== s.bytes.length) continue; let same = 0; for (let k = 0; k < p.length; k++) if (p[k] === s.bytes[k]) same++; best = Math.max(best, same / p.length); }
      copied += best;
    }
    const key = anchors ? 'withAnchors (as reported)' : 'modelOnly';
    r[key] = { shapeCount: cnt / 8, colour: col / 8, meanBytesIdenticalToATrainingImage: copied / 8 };
    console.log(`  ${key.padEnd(26)} shape count ${pct(cnt, 8)}, colour ${pct(col, 8)}, bytes identical to a training image ${(100 * copied / 8).toFixed(1)}%`);
  }
  out.bmp = r;
}

// ─── D. Co-training on held-out docs ───────────────────────────────────────
function cotrain(unified, train, heldOut) {
  console.log('\n== D. Co-training vs specialists, scored on held-out documents ==');
  const r = {};
  for (const [key, fid] of [['html', F.HTML], ['json', F.JSON], ['bmp', F.BMP], ['wav', F.WAV_ULAW]]) {
    const sr = E.mulberry32(6100 + fid), sm = new E.FabulaTransformer('phone', sr);
    for (let s = 0; s < 8; s++) {
      const d = train[key][s % train[key].length], tg = E.tagToIds(d.tags);
      if (fid === F.BMP) { const p = E.encodeBMPTo2DPatches(d.bytes); sm.trainStep(p.tokens, fid, tg, { rng: sr, patchMode: true, rowIds: p.rowIds, colIds: p.colIds, fastTrainStep: true }); }
      else if (fid === F.WAV_ULAW) sm.trainStep(E.muLawEncodePCM(d.pcm.subarray(0, 128)), fid, tg, { rng: sr, fastTrainStep: true });
      else sm.trainStep(d.bytes, fid, tg, { rng: sr, fastTrainStep: true });
    }
    const score = (model, docs) => {
      let sum = 0;
      for (let i = 0; i < docs.length; i++) {
        const d = docs[i], tg = E.tagToIds(d.tags);
        let b = d.bytes;
        if (fid === F.BMP) b = E.encodeBMPTo2DPatches(d.bytes).tokens;
        else if (fid === F.WAV_ULAW) b = E.muLawEncodePCM(d.pcm.subarray(0, 64));
        const L = Math.min(64, b.length);
        sum += model.evaluateNELBO(b.subarray(0, L), fid, tg, { windowSize: L, maxWindows: 1, strata: [0.2, 0.5, 0.8], seed: 500 + i }).bpb;
      }
      return sum / docs.length;
    };
    r[key] = {
      reported_trainDocs: { specialist: score(sm, train[key].slice(0, 3)), unified: score(unified, train[key].slice(0, 3)) },
      heldOut: { specialist: score(sm, heldOut[key]), unified: score(unified, heldOut[key]) },
    };
    const h = r[key].heldOut;
    console.log(`  ${key.toUpperCase().padEnd(5)} training docs (as reported): spec ${r[key].reported_trainDocs.specialist.toFixed(4)} / uni ${r[key].reported_trainDocs.unified.toFixed(4)} | held-out (${heldOut[key].length} docs): spec ${h.specialist.toFixed(4)} / uni ${h.unified.toFixed(4)} (delta ${(h.unified - h.specialist >= 0 ? '+' : '') + (h.unified - h.specialist).toFixed(4)})`);
  }
  out.cotraining = r;
}

// ─── E. Audio, same job for both paths ─────────────────────────────────────
function audio(unified, train) {
  console.log('\n== E. Audio: both paths regenerate 30% of the whole clip ==');
  const n = QUICK ? 3 : 6;
  const acc = { muPitch: 0, tilePitch: 0, muTempoErr: 0, tileTempoErr: 0, refPitch: 0, refTempoErr: 0 };
  let gridInfo = null;
  for (let i = 0; i < n; i++) {
    const clip = train.wav[i], tg = E.tagToIds(clip.tags), rng = E.mulberry32(9500 + i);
    const ref = Generators.analyzeWAVProperties(clip.pcm, 22050);
    if (Math.abs(ref.midi - clip.meta.midi) <= 1) acc.refPitch++;
    acc.refTempoErr += Math.abs(ref.bpm - clip.meta.bpm) / clip.meta.bpm * 100;

    // Path B: whole tile, 30% of bytes masked everywhere
    const A = E.agAnalyzeSync(clip.pcm, true), hx = E.agBuildHelixData(clip.pcm, A, 0, A.F), enc = E.encodeAuraTileTo2DPatches(A, hx);
    gridInfo = { gridW: enc.gridW, patches: enc.rowIds.length, reportedPatchesRegenerated: 80 };
    const init = new Int32Array(enc.tokens.length);
    for (let k = 0; k < init.length; k++) init[k] = rng() < 0.30 ? E.CFG.MASK_ID : enc.tokens[k];
    const st = unified.sampleWindow(init.length, F.AURA, tg, { initialTokens: init, patchMode: true, rowIds: enc.rowIds, colIds: enc.colIds, steps: 3, order: 'confidence', rng, noExemplarAnchors: true });
    const dec = E.decode2DPatchesToAuraTile(st.bytes, enc.gridW, st.helixLogits, enc.range, enc.ref);
    const ta = Generators.analyzeWAVProperties(E.agInvertHelix(dec.A, dec.helix, 0, dec.A.F, 4), 22050);
    if (Math.abs(ta.midi - clip.meta.midi) <= 1) acc.tilePitch++;
    acc.tileTempoErr += Math.abs(ta.bpm - clip.meta.bpm) / clip.meta.bpm * 100;

    // Path A: whole clip in 128-sample windows, 30% of samples masked everywhere
    const mu = E.muLawEncodePCM(clip.pcm), gen = new Uint8Array(mu.length);
    for (let off = 0; off < mu.length; off += 128) {
      const L = Math.min(128, mu.length - off), wi = new Int32Array(L);
      for (let k = 0; k < L; k++) wi[k] = rng() < 0.30 ? E.CFG.MASK_ID : mu[off + k];
      gen.set(unified.sampleWindow(L, F.WAV_ULAW, tg, { initialTokens: wi, steps: 3, order: 'confidence', rng, noExemplarAnchors: true }).bytes, off);
    }
    const ma = Generators.analyzeWAVProperties(E.muLawDecodeToPCM(gen), 22050);
    if (Math.abs(ma.midi - clip.meta.midi) <= 1) acc.muPitch++;
    acc.muTempoErr += Math.abs(ma.bpm - clip.meta.bpm) / clip.meta.bpm * 100;
  }
  const r = {
    clips: n, tileGrid: gridInfo,
    referenceAudio: { pitchMatch: acc.refPitch / n, tempoErrPct: acc.refTempoErr / n },
    tile30pct: { pitchMatch: acc.tilePitch / n, tempoErrPct: acc.tileTempoErr / n },
    muLaw30pct: { pitchMatch: acc.muPitch / n, tempoErrPct: acc.muTempoErr / n },
  };
  console.log(`  tile grid: ${gridInfo.patches} patches (${gridInfo.gridW} time columns); the report regenerated 80 of them`);
  console.log(`  analyzer on the untouched source clips: pitch ${pct(acc.refPitch, n)}, tempo error ${(acc.refTempoErr / n).toFixed(1)}%`);
  console.log(`  AuraGraph tile, 30% masked: pitch ${pct(acc.tilePitch, n)}, tempo error ${(acc.tileTempoErr / n).toFixed(1)}%`);
  console.log(`  mu-law,         30% masked: pitch ${pct(acc.muPitch, n)}, tempo error ${(acc.muTempoErr / n).toFixed(1)}%`);
  out.audio = r;
}

const train = Generators.buildSeededSuite(4242, 8);
const bmpRng = Generators.mulberry32(4242);
train.bmp = [];
for (let i = 0; i < 8; i++) train.bmp.push(Generators.generateBMPSample(bmpRng, { size: 32, nShapes: 1 + (i % 3) }));
const heldOut = Generators.buildSeededSuite(97531, 6);
const hbRng = Generators.mulberry32(97531);
heldOut.bmp = [];
for (let i = 0; i < 6; i++) heldOut.bmp.push(Generators.generateBMPSample(hbRng, { size: 32, nShapes: 1 + (i % 3) }));

sherlock();
const unified = repair(train, heldOut);
bmp(unified, train);
cotrain(unified, train, heldOut);
audio(unified, train);

const j = process.argv.indexOf('--json');
const p = j >= 0 ? process.argv[j + 1] : path.join(__dirname, 'audit-v03-results.json');
fs.writeFileSync(p, JSON.stringify(out, null, 2));
console.log('\nSaved ' + p);
