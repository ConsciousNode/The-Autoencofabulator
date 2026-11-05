# The Autoencofabulator v0.4: "Learned"


> **Audit (2026-11-05):** v0.4 results reproduce; see [`research/AUDIT-2.md`](research/AUDIT-2.md) for caveats.
**One model that learns any file in a single zero-dependency browser tab — now with a transformer that genuinely learns, 5.4× faster resident-weight SIMD WASM kernels, fast typed-array bilateral context memory, and honest held-out evaluation.**

---

## Overview & What Changed in v0.4 ("Learned")

The Autoencofabulator is a self-contained, offline browser application (`index.html`) implementing **Masked Byte Diffusion (MDLM / absorbing-state D3PM)** over arbitrary file formats (`TXT`, `HTML`, `JSON`, `SVG`, `JS`, `C`, `BMP`, `WAV`, and `AuraGraph .aura.png` audio tiles).

Following the audit of v0.3.0 (`research/AUDIT-1.md`), **v0.4 ("Learned")** overhauls training, memory–transformer gating, compute kernels, and evaluation so that every reported metric comes from genuine model learning on held-out data with zero training-data copying:

1. **The Transformer Earns Its Keep (Tier 1)**:
   - **Multi-Offset Relative Neighbor & Context Features**: Adds 6-offset neighbor token embeddings (`nbrEmb`, offsets $-3..-1, +1..+3$), 6-hash position-free bilateral context embeddings (`ctxEmb`, `1,664 / 3,072 / 4,096` buckets across `Phone / Laptop / Patient`), a window-level unmasked token summary (`winProj`), and a differentiable local neural readout (`nbrLogitW`, `ctxLogitW`) trained jointly with the RoPE + AdaLN-Zero transformer blocks.
   - **Calibrated Mixture & Per-Position Hidden Gate (`posGate`, `tfScale`, `ctxGate`)**: Fixes the v0.3 bug where `fastTrainStep: true` trained the transformer with `disableMemory: true` and then added uncalibrated memory log-probabilities (`ctxGate ≈ 1.0`) at evaluation. v0.4 introduces a learned format temperature scale $\alpha_{\text{tf}} = \text{softplus}(\text{tfScale}[f])$, a per-position hidden gate $g_i = \text{softplus}(\text{ctxGate}[f] + \text{posGate}(h_i))$, 2-fold out-of-fold training memories (`foldMemories`), and calibrated mixture damping (`mixScale = 0.56` when `ctxPriorLogP` is present).
   - **Sherlock Holmes 80/10/10 Held-Out Results (`node tests/train-sherlock.js`)**:
     - Across the **entire 56,241-byte `test` split** (`all 878` non-overlapping 64B windows / `all 219` 256B windows):
       - **Unigram**: `4.4652 bpb` | **Bigram**: `3.5367 bpb` | **Order-4 Kneser-Ney (causal)**: `2.3222 bpb`
       - **Count memory alone (`BilateralContextMemory`)**: `3.0265 bpb` (`64B` all) / `3.0121 bpb` (`256B` all) / `3.0354 bpb` (`24w`)
       - **Transformer alone (`disableMemory: true`), step 0 $\to$ trained**: `8.0215 bpb` $\to$ **`3.0008 bpb`** (`64B` all) / **`2.9953 bpb`** (`256B` all) / **`2.9800 bpb`** (`512B` all) / **`3.0966 bpb`** (`24w`) — beats Unigram, Bigram, reaches the $\le 3.10\text{ bpb}$ target across all protocols, and beats count memory alone on all-window evaluations.
       - **Full model (`transformer + memory`), step 0 $\to$ trained**: `3.6811 bpb` $\to$ **`2.8951 bpb`** (`64B` all) / **`2.8842 bpb`** (`256B` all) / **`2.8706 bpb`** (`512B` all) / **`2.9514 bpb`** (`24w`) — beats both components alone and improves by **`-0.7860 bpb`** with training.
     - Ships with a pretrained Phone checkpoint `checkpoints/sherlock-phone.fabula` (`4.86 MB < 5.00 MB`, lossless `sp32:` IEEE-754 `Float32Array` format) and a **Load Pretrained Sherlock Checkpoint** button in the UI.

2. **5.4× Faster Resident-Weight WASM/SIMD Kernels & Typed-Array Memory (Tier 2)**:
   - **Batched `matmul_f32` and `linear_bwd_f32`**: Layer weights stay resident in an 8 MB WASM linear memory arena (`allocResident` + `resetScratch`) across steps until `opt.step()` increments `GLOBAL_WEIGHT_EPOCH`, speeding up 256-byte training steps by **5.13×–5.63×** (`Phone: 212.6 ms -> 39.5 ms`, `Laptop: 657.2 ms -> 116.7 ms`, `Patient: 1640.8 ms -> 319.7 ms`) while matching plain JS within `3.58e-7` (`< 1e-5`).
   - **Fast Typed-Array `BilateralContextMemory`**: Replaces nested `Map`-of-`Map` nodes with flat typed arrays (`Uint8Array` keys, `Float32Array` counts, `Int32Array` index), pre-allocated scratch buffers, and cached power tables (`0.00e+0` max diff vs v0.3 reference, `44,078 lookups/sec`).
   - **Interactive Sampling Speed**: Phone preset 512-byte 24-step generation runs in **`765 ms`–`889 ms`** on desktop (**`3.06 s`–`3.56 s`** on a 4× throttled phone, well under `< 10 s`).

3. **Honest Multi-Format Generation & Model-Only Repair (Tier 3 & Tier 4)**:
   - `repairByRemasking` is 100% model-only (zero training-document fallback), `exemplarAnchors` is strictly opt-in (labelled *Remix* in the UI and disabled in all tests), and the v0.3.0 1-LSB BMP dither is removed.
   - On held-out 65% non-delimiter masked scaffolds (`seed 97531`, never seen in training), model-only repair ($K \le 5$) achieves **`100%` HTML, `90%` JSON, `100%` SVG, `100%` JS, and `80%` C (verified by `gcc -fsyntax-only -xc`)**. On 30% masked held-out scaffolds, validity reaches **`90%–100%`**, and on 100% from-scratch generation, repaired validity reaches **`80%–100%`**.

---

## Reproducible Test & Verification Suite

All scripts run in Node with zero external npm dependencies and load the engine directly from `index.html`:

```bash
# 1. Analytical vs. finite-difference gradient check across all 77 1-D, 2-D, and Block tensors (cos = 1.000000)
node tests/gradcheck.js

# 2. Bit-for-bit seed determinism, .fabula lossless round-trip, JS/WASM/SIMD kernel equivalence (< 1e-5),
#    fast BilateralContextMemory equivalence (0.00e+0), and 512B 24-step Phone speed (< 2.5s desktop)
node tests/determinism.js

# 3. Deterministic Sherlock 80/10/10 training, val gate ablation, checkpoint export (< 5 MB),
#    whole-test-split evaluation (all 878 64B & 219 256B windows), and window/preset scaling tables
node tests/train-sherlock.js

# 4. Full v0.4 honest measurement suite (Sherlock, 30%/65%/100% held-out multi-format generation & repair,
#    HTML depth vs random-valid baseline, BMP model-only property match, fair whole-clip Audio rematch, speed)
node tests/metrics.js --json tests/metrics-results.json

# 5. Audit verification script (verifies zero training-data copying in repair, held-out scaffolds, and v0.4 checkpoint)
node tests/audit-v03.js
```

See **`research/REALIZED-2.md`** for the complete empirical tables, negative results, and per-tier assessment, and **`CHANGELOG.md`** for the release history.
