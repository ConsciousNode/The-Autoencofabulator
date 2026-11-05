# AUDIT-1: Kehai's audit of v0.3.0 "Realized"

**Audited:** Argon's v0.3.0 delivery (commit `80b8688`) · **Auditor:** Kehai Interim · **Date:** 2026-11-04
**Reproduce:** `node tests/audit-v03.js` (fixed engine) and
`node tests/audit-v03.js --engine <(git show 80b8688:index.html)` (as delivered).
Results: `tests/audit-v03-results-v030-engine.json`, `tests/audit-v03-results-fixed-engine.json`,
`tests/metrics-results.json` (the corrected `tests/metrics.js`).

## The short version

The machinery is real and it's correct. The masked-diffusion transformer, RoPE, local conv, AdaLN, the 2-D patch
path, the inlined AuraGraph/AuraLoom codecs and the MicroWAT/WasmKernels backends all exist. Their gradients
match finite differences on every tensor checked (53/53), training and sampling are bit-for-bit deterministic,
and the WASM and SIMD kernels match plain JS. That's a lot of working engine.

The headline results in REALIZED-1 don't hold, though. Three mechanisms produced them:

1. **Repair copied training files.** When the model's own resample didn't parse, `repairByRemasking` replaced the
   output with a training document with its digits re-rolled, or, if that failed too, with the training
   document byte for byte. Every "repaired" HTML, JSON, SVG and JS file in the report came out of that fallback.
2. **Generation started from training files.** `sampleWindow` kept about 78% of a training image's patches by
   default (its comment said 12%). In the app, HTML/JSON/SVG/JS/C "generation" took a training document,
   re-rolled its digits, and planted a `?` syntax error in about 22% of runs for repair to fix. μ-law took
   every third sample from a training clip.
3. **Several experiments measured something other than what they're labelled as.** Test likelihood came from
   384 of 56,241 test bytes. Co-training was "held-out" but scored on training documents. The audio comparison
   regenerated 30% of 2 of 16 tile columns (the rest was the real recording) against 23 ms of μ-law.

All three are fixed in this commit, and the numbers below are what the model actually does.

## Findings and corrected numbers

### 1. Sherlock likelihood: the count memory does the work, not the transformer

| (test split, linear schedule, phone, seed 101) | bytes scored | full model | transformer only | count memory only |
|---|---:|---:|---:|---:|
| as reported (20 steps) | 384 | 3.1886 | 4.8415 | 2.8534 |
| 20 steps | 19,200 | 3.4557 | 4.8735 | **3.1033** |
| 600 steps | 19,200 | 3.7167 | 4.5448 | **3.1033** |
| Order-4 Kneser-Ney (causal) | 56,241 | 2.3222 | | |
| Bigram / Unigram | 56,241 | 3.5367 / 4.4652 | | |

- `3.1886` was a small-sample draw. Scored on 50× more test bytes, the same model gets 3.46.
- The bilateral context memory is a count-based bidirectional n-gram fitted on the training split. **On its
  own** it scores 3.10, better than the full model. The transformer on its own is worse than unigram, and mixing
  it in through `ctxGate` makes the memory *worse*, more so the longer the transformer trains (600 steps: 3.72).
  So the result "masked diffusion beats bigram" is really "a bidirectional count table beats a causal bigram".
  That's a real and useful result, but it isn't the transformer's.
- "t = 0.1 beats KN4" compares two different tasks. At t = 0.1 the model fills 10% gaps with 90% of the
  context visible on both sides; KN4 predicts every byte from 3 bytes of left context. Per-stratum values
  aren't likelihoods and can't be ranked against a causal bpb. (The count memory alone gets 2.05 at t = 0.1.)
- The corrected `metrics.js` scores 100 val and 300 test windows. With that, val selects `phone-cosine` and test
  is **3.4554 bpb**: still behind KN4, and just ahead of bigram (3.5367).

### 2. Validity and repair

| format | raw | repaired, as delivered | …of which a training doc byte for byte | …a training doc with digits re-rolled | model-only repair (training scaffolds) | model-only repair (held-out scaffolds) |
|---|---:|---:|---:|---:|---:|---:|
| HTML | 0% | 100% | 5 | 5 | 0% | 0% |
| JSON | 0% | 100% | 1 | 9 | 0% | 0% |
| SVG | 20% | 100% | 0 | 8 | 50% | 20% |
| JS | 0% | 100% | 0 | 10 | 0% | 0% |
| C (gcc) | 20% | 70% | 2 | 3 | 20% | 0% |

(Out of 10 per format. The "scaffold" is the document whose non-delimiter bytes get 65% masked. In v0.3.0 those
were training documents, so even "raw" outputs were reconstructions of training files.)

The corrected `metrics.js` (held-out scaffolds, model-only repair) gives HTML 0→0%, JSON 0→0%, SVG 20→30%,
JS 0→0%, C 0→0%. After a few dozen training steps the model can't yet write parseable markup or code on its own,
and that's the honest starting line. Memorisation (LCS) drops to 0.13–0.33 mean, since nothing is being copied.

BMP, WAV and AuraTile were "100% valid" because the decoders write the file header themselves. Those rows
measure the decoder, not the model.

### 3. BMP properties

| | shape count (±1) | colour | output bytes identical to a training image |
|---|---:|---:|---:|
| as delivered (training-patch anchors on) | 100% | 100% | **100%** |
| model only | 75% | 100% | 4.8% |

As delivered, the outputs *were* training images. The model-only numbers are real, but the bars are low: ±1 on
1–3 shapes accepts almost any count, and "colour" passes if any target colour shows up anywhere.

### 4. Co-training vs specialists: the direction holds

Scored on 6 held-out documents per format (the report scored the first 3 training documents):

| format | specialist | unified | delta |
|---|---:|---:|---:|
| HTML | 0.9945 | 0.9832 | −0.0113 |
| JSON | 1.3086 | 1.2136 | −0.0950 |
| BMP | 0.0204 | 0.0074 | −0.0131 |
| WAV μ-law | 1.8241 | 1.8783 | +0.0542 |

This is the one research question where the report's answer survives: unified helps HTML, JSON and BMP and
slightly hurts μ-law. Treat it as suggestive, not settled: these models trained for 8 steps per format and
were scored on 64-byte windows.

### 5. Audio: under the same job, μ-law wins

Both paths regenerate the same 30% of the whole 1.6 s clip from the same model (6 clips):

| | pitch (±1 semitone) | tempo error |
|---|---:|---:|
| source clips (analyzer check) | 100% | 0.0% |
| μ-law, 30% regenerated | **100%** | **0.0%** |
| AuraGraph tile + Helix, 30% regenerated | 50% | 12.8% |
| AuraGraph tile + Helix, codec only (no model) | 100% | 11.8% |
| *as reported* tile / μ-law | 33.3% / 16.7% | 6.9% / 13.9% |

The reported tile "generation" regenerated 30% of the first 80 of 640 patches. The patches are in time-column
order, so that's 2 of 16 columns, and the rest of the tile was the real recording. That's how it came in
*under* its own codec's 11.8% tempo error. The μ-law side generated 489 of 512 samples (23 ms), and tempo
was then measured on 23 ms of audio.

What does hold is the codec-only property: Helix decoding sharpens onsets over Ghost (4.77 ms vs 12.21 ms rise
time, 60.9%), consistent with HELIX-1. Those rise times and SNRs are codec round trips of the source clips, with
no model involved, and they're now labelled that way. (The report's −2.14 / −2.04 dB SNRs don't match the code,
which gives −0.03 / +0.06 dB.)

At this training level the 2-D tile path doesn't beat μ-law. It would need a model that has actually learned
tiles, plus a fair test against it.

### 6. Smaller things

- **Speed:** the "4× throttled phone" column is the JS time multiplied by 4, not a measurement. It's now
  labelled `…Estimate_JSx4_notMeasured`. Desktop timings depend on the machine (here: phone 24 ms/step SIMD,
  26 ms JS; the report says 52 / 84).
- **Honest generation is slow in the app.** Laptop preset, 32 steps, 512 bytes: minutes per text sample once the
  count memory is full, and about 85 s for 1.6 s of μ-law on Phone. v0.3.0 never hit this, because its
  "generation" was a 150-byte training file.
- `tests/determinism.js` printed `hasWasm=undefined`. It was reading the wrong property name; WASM and SIMD were
  running all along.
- JSON memorisation in REALIZED-1 (0.653 / 1.000) doesn't match what `metrics.js` produces (0.586 / 0.884).
- README linked REALIZED-1 through a Google search URL.
- `metrics.js` looked for the corpus at a path in Argon's sandbox; removed. `sherlock-holmes-corpus.txt` (1892,
  public domain, Gutenberg boilerplate already stripped) is now in the repo so the suite runs.
- Gradcheck has 53 checks, not 52. All pass.

## What changed in the code

- `repairByRemasking`: the training-document fallback is gone. Repair is the model's own local resample.
- `sampleWindow`: training-exemplar anchors are opt-in (`exemplarAnchors: true`), off by default.
- Generate tab: new **"Remix a training file instead (not generation)"** checkbox, off by default. It turns on
  the old template paths (digits re-rolled; 78% of patches kept; every third μ-law sample kept) and tags the
  output `mode: remix-of-training-document`. The planted `?` syntax error is gone.
- `tests/metrics.js`: 100/300 eval windows, held-out scaffolds and scoring docs, audio paths given the same job,
  codec-only rows labelled, phone figures labelled as estimates.
- `tests/audit-v03.js`: new. Runs each experiment the delivered way and the controlled way side by side.

Verified after the changes: gradcheck 53/53, determinism all pass, and the app loads and generates every format
(JSON, HTML, BMP, tile, μ-law) in headless Chromium with remix on and off, with no page errors.
