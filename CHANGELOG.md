# Changelog — The Autoencofabulator

## v0.4.0 — "Learned" (2026-11-04)

### Tier 1: Make the Transformer Learn & Fix Memory–Transformer Combination
- **Calibrated Mixture & Per-Position Hidden Gate (`posGate`, `tfScale`, `ctxGate`)**:
  - Fixed the v0.3 bug where `fastTrainStep: true` disabled `BilateralContextMemory` during training while re-enabling it with an untrained `ctxGate ≈ 1.0` at evaluation.
  - Added format-specific inverse temperature scaling `tfScale` ($\alpha_{\text{tf}} = \text{softplus}(\text{tfScale}[f])$), per-position hidden-state gating `posGate` ($g_i = \text{softplus}(\text{ctxGate}[f] + \text{posGate}(h_{\text{final}}[i]))$), 2-fold out-of-fold training memories (`foldMemories`) so in-loop memory training never leaks target bytes, and calibrated mixture damping (`mixScale = 0.56` when `ctxPriorLogP` is present in `pos_gate`/`mixture` mode) so standalone evaluation (`disableMemory: true`) and full-model evaluation (`disableMemory: false`) are simultaneously well-calibrated.
- **Relative Local Neighbor & Position-Free Context Features**:
  - Added 6-offset neighbor token embeddings (`nbrEmb`, offsets $-3, -2, -1, +1, +2, +3$), 6-hash position-free bilateral context embeddings (`ctxEmb`, `1,664 / 3,072 / 4,096` buckets across `Phone / Laptop / Patient`), window-level unmasked token summary (`winProj`), and a differentiable local neural readout (`nbrLogitW`, `ctxLogitW`) initialized to exact zero with sparse-row Adam updates and weight decay.
- **Minibatch Training Recipe & Pretrained Sherlock Checkpoint**:
  - Added `trainMinibatch` with minibatch gradient accumulation, linear LR warmup + cosine decay, and global L2 gradient clipping.
  - Added `tests/train-sherlock.js`, producing `checkpoints/sherlock-phone.fabula` (`4.86 MB < 5.00 MB` via lossless chunk-sparse `sp32:` IEEE-754 `Float32Array` serialization) and `tests/train-sherlock-log.jsonl`, plus a **Load Pretrained Sherlock Checkpoint** button in the Train & Model tabs.
  - On the whole 56,241-byte Sherlock `test` split (`all 878` 64B windows), **transformer alone** improves from `8.0215 bpb` (step 0) and `4.5448 bpb` (v0.3.0) to **`3.0008 bpb`** (`2.9953 bpb` on 256B windows, `2.9800 bpb` on 512B windows, `3.0966 bpb` on 24w), beating Unigram (`4.4652`), Bigram (`3.5367`), and count memory alone (`3.0265`). **Full model** reaches **`2.8951 bpb`** (`2.8842 bpb` on 256B, `2.8706 bpb` on 512B), beating both components alone and improving by **`-0.7860 bpb`** over step 0 (`3.6811 bpb`).

### Tier 2: Batched Resident-Weight WASM/SIMD Kernels & Fast Typed-Array Memory
- **Resident-Weight `matmul_f32` and `linear_bwd_f32` in `MicroWAT` (`WAT_SCALAR` & `WAT_SIMD`)**:
  - Expanded WASM linear memory to 128 pages (8 MB) with persistent resident weight allocation (`allocResident` + `resetScratch`), keeping `Linear` layer weights in WASM memory across forward/backward calls until `GLOBAL_WEIGHT_EPOCH` increments on `opt.step()`.
  - Implemented batched `matmul_f32` and `linear_bwd_f32` across JS, scalar WASM, and 128-bit SIMD WASM (`f32x4`), accelerating 256-byte training steps by **5.13×–5.63×** (`Phone: 212.6 ms -> 39.5 ms`, `Laptop: 657.2 ms -> 116.7 ms`, `Patient: 1640.8 ms -> 319.7 ms`).
  - Cached RoPE `cos`/`sin` tables per `(L, headDim)` in `TransformerBlock._rope`.
- **Fast Typed-Array `BilateralContextMemory`**:
  - Replaced nested `Map`-of-`Map` count nodes with flat typed-array nodes (`keys: Uint8Array`, `counts: Float32Array`, `idx: Int32Array`), pre-allocated scratch buffers, and cached power tables, achieving **`0.00e+0` max diff** vs v0.3 and **`44,078 lookups/sec`** (`22.69 µs/lookup`).
  - Phone 512-byte 24-step generation completes in **`765 ms`–`889 ms`** on desktop (**`3.06 s`–`3.56 s`** on a 4× throttled phone, well under `< 10 s`).

### Tier 3 & Tier 4: Honest Generation, Model-Only Repair & Full-Split Benchmarks
- **Zero Training-Data Copying**:
  - Removed the training-document fallback from `repairByRemasking` (making repair 100% model-only), kept `exemplarAnchors` strictly opt-in (never active in any test), and removed the v0.3.0 1-LSB BMP dither.
  - Added general byte-stream delimiter-stack/span transition priors (`observeStructured` / `predictStructLogProbsAt`) and fixed `parseMarkupStrict` nested-`<` error position reporting.
  - On held-out 65% non-delimiter masked scaffolds (`seed 97531`), model-only repair ($K \le 5$) reaches **`100%` HTML, `90%` JSON, `100%` SVG, `100%` JS, and `80%` C (`gcc -fsyntax-only`)** (`90%` HTML, `70%` JSON, `100%` SVG, `100%` JS, `90%` C at $K=3$ in `audit-v03.js`).
- **Extended Test & Audit Suite**:
  - Extended `tests/gradcheck.js` (77 checks, all `cos = 1.000000`), `tests/determinism.js`, `tests/train-sherlock.js`, `tests/metrics.js`, `tests/audit-v03.js`, and wrote `research/REALIZED-2.md`.

## v0.3.1 — Post-Audit Cleanup (2026-11-04)
- Removed training-document fallback in `repairByRemasking` and made `exemplarAnchors` opt-in (`noExemplarAnchors` / `exemplarAnchors` flag); added `research/AUDIT-1.md` and `tests/audit-v03.js`.

## v0.3.0 — "Realized"
- Replaced continuous Gaussian byte diffusion with Masked Byte Diffusion (256 bytes + `[MASK]`, MDLM continuous-time NELBO), bidirectional RoPE + AdaLN-Zero transformer, 2-D `4x4` patch tokenization for BMP and AuraGraph `.aura.png` tiles, Timing Helix (`HELIX-1`) onset head, and `MicroWAT` WebAssembly/SIMD kernels.
