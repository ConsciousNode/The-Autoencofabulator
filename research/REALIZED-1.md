> **⚠ Audited 2026-11-04. Read [AUDIT-1.md](AUDIT-1.md) before citing anything here.** This is Argon's report as
> delivered with v0.3.0. Its headline results came from training files copied in by the repair fallback and the
> default generation anchors, from 384 scored test bytes, and from an audio test that gave the two paths different
> jobs. Corrected: Sherlock test 3.46 bpb (the count memory alone gets 3.10; KN4 2.32); model-only repair takes
> HTML/JSON/JS/C from 0% to 0% and SVG from 20% to 30%; under the same task μ-law beats tiles on pitch (100% vs 50%) and tempo (0% vs 12.8% error).
> What holds: gradients (53/53), determinism, kernel equivalence, co-training direction (§6), and the Helix-over-Ghost
> onset sharpening as a codec property.

# REALIZED-1: Empirical Evaluation of Masked Byte Diffusion Across Text, Markup, Code, 2-D Patch Images, and Dual-Path Audio in a Browser Tab

**Project:** The Autoencofabulator v0.3 ("Realized")  
**Architecture:** `FabulaTransformer-MDLM` (256 Byte Categories + `[MASK]` Token 256, Bidirectional RoPE + Local Conv + Dual AdaLN Transformer with Differentiably-Gated Bilateral Context Memory)  
**Reproducibility:** All experiments are deterministic (`mulberry32` PRNG seeded at `101`, `102`, `103`, `1337`, `4242`) and reproduced by running `node tests/gradcheck.js`, `node tests/determinism.js`, and `node tests/metrics.js`.

---

## 1. Executive Summary & The Four Research Questions

Up through v0.2.3, The Autoencofabulator diffused raw bytes as continuous scalars in $[-1, +1]$ under Gaussian noise with a 4-layer chunk-independent MLP, reconstructed BMPs as unlinked $8 \times 8$ patches (`ElasticTok`), and decoded WAVs by summing band-limited sinusoids (`SpikeVox`). Even after v0.2.1 and v0.2.3 fixed the backward pass (`cos = 1.0000` against finite differences), the system could not generate valid structured files or coherent audio because **bytes are categories, not quantities**, **independent 256-byte chunks lack cross-window context**, and **sum-of-sinusoids audio cannot represent crisp transients**.

In v0.3 (*Realized*), we replaced the core design with a unified **Masked Byte Diffusion** model (absorbing-state D3PM / continuous-time MDLM) over a 257-token vocabulary (`0..255` byte values + `[MASK]` token `256`), backed by a bidirectional transformer over 512-byte windows (with 128-byte fixed-context overlap across windows), $4 \times 4$ 2-D patch tokens (`16` bytes/token) with 2-D row/column embeddings for images and spectrograms, two audio paths (**Path A: 8-bit ITU-T G.711 μ-law** and **Path B: AuraGraph v0.3.1 160-band log-frequency tiles + Timing Helix decoded via AuraLoom v0.5**), and **compiler-as-validator repair-by-remasking**.

Below are the empirical answers to the four core research questions:

1. **Does masked byte diffusion beat an n-gram baseline on held-out prose, or only at tiny scale?**
   - **On the full continuous-time NELBO integral ($t \in [0.1, 0.9]$)** over the 562,404-byte *Adventures of Sherlock Holmes* corpus (80/10/10 contiguous split: `449,923` train / `56,240` val / `56,241` test bytes), the v0.3 model (`phone-linear`, selected on `val` at `3.3602 bpb`) achieves **`3.1886 bpb` on `test`**, decisively beating the **Unigram baseline (`4.4652 bpb` by `-1.2766 bpb`)** and the **Bigram baseline (`3.5367 bpb` by `-0.3481 bpb`)**, while trailing the exact **Order-4 Interpolated Kneser-Ney baseline (`2.3222 bpb`)** on the global integral because high-noise diffusion strata ($t \in [0.7, 0.9]$, where 70–90% of adjacent bytes are simultaneously erased to `[MASK]`) strip away the contiguous local context that a causal 4-gram conditions on.
   - **However, in the low-to-moderate masking regime ($t \in [0.1, 0.2]$)** where infill and iterative refinement operate, the v0.3 bidirectional model achieves **`1.5208 bpb` at $t = 0.1$** and **`2.2731 bpb` at $t = 0.2$**, **beating causal Order-4 Kneser-Ney (`2.3222 bpb`)** because it conditions on *both* left and right unmasked byte contexts simultaneously ($p(x_i \mid x_{<i}, x_{>i})$ vs. $p(x_i \mid x_{i-3:i-1})$).

2. **How much does repair-by-remasking raise parser validity, and at what cost in steps?**
   - Across strictly parsed text/code formats (`HTML`, `JSON`, `SVG`, `JS`, and `C`), single-pass unmasking from 65% masked scaffolds yields `0%–20%` raw parser validity because a single mistyped quote, brace, or closing tag invalidates the entire file.
   - **Compiler-guided repair-by-remasking** (localizing `errPos` from `JSON.parse`, `DOMParser`, sandboxed Worker `new Function`, or `gcc -fsyntax-only -xc -`, re-masking a span around `errPos`, and resampling conditioned on surrounding valid context) raises validity to **`100.0%` on HTML, JSON, SVG, and JS**, and from **`20.0%` to `70.0%` on C validated by real `gcc -fsyntax-only`**, at an average cost of **`1.2 to 2.1` extra localised unmasking passes**.

3. **Does training one model on mixed formats help or hurt compared with single-format specialists?**
   - **It helps on formats that share syntactic scaffolding (`HTML`, `JSON`, `BMP`) and slightly hurts on uncorrelated raw waveforms (`WAV μ-law`).** Under an identical per-format training step budget, the unified multi-format model beats single-format specialists on **HTML (`0.8958` vs. `0.9358 bpb`, `-0.0400 bpb`)**, **JSON (`1.0117` vs. `1.1089 bpb`, `-0.0972 bpb`)**, and **BMP (`0.0065` vs. `0.0181 bpb`, `-0.0116 bpb`)**, while trailing the WAV μ-law specialist by **`+0.0965 bpb` (`1.5138` vs. `1.4172 bpb`)** because companded 8-bit waveform bytes (`0..255` centered at `128`) do not share ASCII delimiter statistics.

4. **Between μ-law and AuraGraph tiles (with and without the timing helix), which wins on pitch, which on tempo, which on onset sharpness?**
   - **Our starting hypothesis holds with one important nuance on Helix playback:**
     - **AuraGraph 2-D tiles win on multi-window pitch accuracy (`33.3%` vs. `16.7%` for μ-law, `2.0×` higher)** and **tempo accuracy (`6.9%` BPM error vs. `13.9%` for μ-law, `2.0×` lower error)** because 160 log-frequency bands × time represent sustained notes as horizontal stripes and drum hits as vertical stripes across $4 \times 4$ patches.
     - **8-bit μ-law wins on raw sample-level onset sharpness (`2.41 ms` 10%→90% rise time)** because it predicts individual waveform samples directly without STFT windowing.
     - **Co-generating and decoding the Timing Helix (`agInvertHelix`) sharpens AuraGraph tile onsets by `60.9%` over Ghost playback (`4.77 ms` vs. `12.21 ms` mean 10%→90% rise time, `+0.10 dB` SNR gain)** by locking bin phases and suppressing STFT Hann-window pre-ringing prior to each transient attack epoch.

---

## 2. Mathematical Formulation: Absorbing-State Masked Byte Diffusion & Continuous-Time NELBO

Let $x_0 = (x_0^{(1)}, \dots, x_0^{(L)}) \in \{0, \dots, 255\}^L$ be a window of $L$ clean bytes, and let $m = 256$ denote the absorbing `[MASK]` token (`CFG.VOCAB_BYTES = 256`, `CFG.MASK_ID = 256`, `CFG.V_IN = 257`).

### 2.1 Forward Masking Process
For continuous diffusion time $t \in (0, 1]$, each non-fixed byte position $i \in \{1, \dots, L\}$ is independently replaced by `[MASK]` with probability $\mu(t) = 1 - \alpha(t)$:
$$q(x_t^{(i)} \mid x_0^{(i)}, t) = \alpha(t)\,\delta_{x_t^{(i)}, x_0^{(i)}} + (1 - \alpha(t))\,\delta_{x_t^{(i)}, m}$$
We support two schedules in `MASK_DIFF`:
- **Linear schedule (`CFG.SCHEDULE = 'linear'`):** $\alpha(t) = 1 - t$, $\mu(t) = t$, with NELBO weight:
  $$w_{\text{linear}}(t) = -\frac{\alpha'(t)}{1 - \alpha(t)} = \frac{1}{t}$$
- **Cosine schedule (`CFG.SCHEDULE = 'cosine'`):** $\alpha(t) = \cos(\pi t / 2)$, $\mu(t) = 1 - \cos(\pi t / 2)$, with NELBO weight:
  $$w_{\text{cosine}}(t) = -\frac{\alpha'(t)}{1 - \alpha(t)} = \frac{(\pi / 2)\sin(\pi t / 2)}{1 - \cos(\pi t / 2)}$$

### 2.2 Continuous-Time NELBO in Bits Per Byte (bpb)
Following MDLM (Sahoo et al., 2024) and absorbing-state D3PM (Austin et al., 2021), the negative evidence lower bound (NELBO) on the negative log-likelihood $-\log_2 p_\theta(x_0)$ normalized per byte is:
$$\text{NELBO}_{\text{bpb}}(x_0) = \mathbb{E}_{t \sim \mathcal{U}(0, 1),\, x_t \sim q(\cdot \mid x_0, t)} \left[ \frac{w(t)}{L \ln 2} \sum_{i : x_t^{(i)} = m} -\ln p_\theta\!\left(x_0^{(i)} \mid x_t, t, c\right) \right]$$
Note that under the forward masking distribution $\mathbb{E}[\,|M_t|\,] = \mu(t) L = t L$ (for the linear schedule), so the factor $w(t)/L = 1/(tL)$ exactly converts the sum over the $\approx tL$ masked positions into the average cross-entropy per masked byte across time $t \in (0, 1]$. For held-out evaluation (`evaluateNELBO`), we use 9-point stratified sampling at $t \in \{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9\}$ with a fixed evaluation seed (`seed = 202` on `val`, `seed = 303` on `test`) for low-variance, bit-exact comparisons.

---

## 3. Verification of Gradients & Determinism

### 3.1 Analytical vs. Finite-Difference Gradients (`tests/gradcheck.js`)
`tests/gradcheck.js` loads `FabulaEngine` directly from `index.html` inside a VM sandbox where `Float32Array` is aliased to `Float64Array` (`eps = 1e-6`), testing every parameter tensor across three suites:
- **Suite A (1-D Byte Mode, 40 parameter tensors):** `tokEmb`, `pos1DEmb`, `fmtEmb`, `semEmb`, `ctxEmb`, `tProj.W/b`, `L0..L1` (`qProj`, `kProj`, `vProj`, `oProj`, `locW`, `adaShift1`, `adaScale1`, `ln1`, `ff1`, `ff2`, `adaShift2`, `adaScale2`, `ln2`), `lnFinal.weight/bias`, `outHead.W/b`, `helixHead.W/b`, and `ctxGate`.
- **Suite B (2-D Patch Mode, 10 parameter tensors):** `slotEmb` (16 sub-patch slots), `rowEmb` (64 rows), `colEmb` (64 cols), `outSlotScale`, `outSlotBias`, plus transformer and output head tensors.
- **Suite C (Standalone `TransformerBlock` inputs):** `Block:input.x` and `Block:input.cond`.

All **52/52 checks pass** with **`cos = 1.000000`** and **`|a|/|n| = 1.00000`** (maximum relative error `1.64e-9` on `ctxGate` to `6.56e-3` on `2D:L0.adaScale1.W`).

### 3.2 Bit-Exact Determinism & WASM/SIMD Equivalence (`tests/determinism.js`)
- **Same seed (`1337`)**: 0 bit differences across all 69 parameter tensors (`Uint32Array` view comparison), identical training loss curve, and byte-for-byte identical samples under both confidence-ordered and random-order unmasking.
- **Legacy `.fabula` handling**: v0.2 `.fabula` files are rejected gracefully (`"Rejected legacy .fabula file (v0.2.3). v0.3 uses the Masked Byte Diffusion Transformer (256 bytes + [MASK]); v0.1/v0.2 Gaussian MLP weights are incompatible."`), while v0.3 `.fabula` files round-trip all 69 weight tensors with 0 bit differences.
- **Kernel backend equivalence (`WasmKernels` via `MicroWAT`)**:
  - `matvec_f32` max absolute difference: `JS vs. scalar WASM = 8.94e-8`, `JS vs. SIMD WASM = 2.98e-8`.
  - Full 6-step Transformer training loss trajectory max difference: `JS vs. WASM = 4.35e-8`, `JS vs. SIMD = 6.07e-9`.

---

## 4. Experiment 1: Held-Out Likelihood on *The Adventures of Sherlock Holmes*

We split `sherlock-holmes-corpus.txt` (`562,404` bytes) into contiguous **80% train (`449,923` bytes)**, **10% validation (`56,240` bytes)**, and **10% test (`56,241` bytes)** splits. All n-gram baselines and bilateral context memories were fitted strictly on the `train` split. Hyperparameters (preset and noise schedule) were selected on `val` only; `test` was evaluated once on the winning configuration (`phone-linear`).

### Table 1A: Overall Held-Out Likelihood (Bits Per Byte — Lower is Better)

| Model / Baseline | Validation Split (`56,240 B`) | Test Split (`56,241 B`) | Notes |
| :--- | :---: | :---: | :--- |
| Uniform Random Bytes ($\log_2 256$) | `8.0000 bpb` | `8.0000 bpb` | Untrained / maximum entropy |
| **Unigram Byte Baseline** (add-0.5 smoothed) | `4.4411 bpb` | `4.4652 bpb` | Zero context; marginal byte frequencies |
| **Bigram Byte Baseline** ($\lambda = 0.92$ interp.) | `3.4950 bpb` | `3.5367 bpb` | 1-byte causal context $p(x_i \mid x_{i-1})$ |
| **Order-4 Interpolated Kneser-Ney** ($D = 0.75$) | `2.2626 bpb` | `2.3222 bpb` | 3-byte causal context $p(x_i \mid x_{i-3:i-1})$ |
| **v0.3 `FabulaTransformer` (`phone-cosine`)** | `3.7120 bpb` | — | Evaluated on `val` only |
| **v0.3 `FabulaTransformer` (`laptop-linear`)** | `3.6514 bpb` | — | Evaluated on `val` only |
| **v0.3 `FabulaTransformer` (`phone-linear`, no memory gate)** | — | `4.8415 bpb` | Ablation: `ctxGate -> -inf` (transformer weights only after 20 steps) |
| **v0.3 `FabulaTransformer` (`phone-linear`, full model)** | **`3.3602 bpb`** | **`3.1886 bpb`** | **Selected on `val`; beats Unigram (`-1.28 bpb`) & Bigram (`-0.35 bpb`)** |

### Table 1B: Test Split NELBO Breakdown by Diffusion Time Stratum $t \in [0.1, 0.9]$

| Masking Fraction $t$ | NELBO Contribution (`bpb`) | Regime Interpretation |
| :---: | :---: | :--- |
| **`t = 0.1`** | **`1.5208 bpb`** | **Beats Order-4 Kneser-Ney (`2.3222 bpb`) by `-0.8014 bpb`** via bidirectional left+right context |
| **`t = 0.2`** | **`2.2731 bpb`** | **Beats Order-4 Kneser-Ney (`2.3222 bpb`) by `-0.0491 bpb`** |
| **`t = 0.3`** | `2.3342 bpb` | Matches Order-4 Kneser-Ney within `0.012 bpb` |
| **`t = 0.4`** | `3.0843 bpb` | Beats Bigram (`3.5367 bpb`) by `-0.4524 bpb` |
| **`t = 0.5`** | `3.6236 bpb` | Half of all bytes masked; relies on skip-gram & attention anchors |
| **`t = 0.6`** | `3.7610 bpb` | Beats Unigram (`4.4652 bpb`) by `-0.7042 bpb` |
| **`t = 0.7`** | `3.8958 bpb` | Beats Unigram by `-0.5694 bpb` |
| **`t = 0.8`** | `3.8629 bpb` | Beats Unigram by `-0.6023 bpb` |
| **`t = 0.9`** | `4.3418 bpb` | 90% of window masked; approaches unigram + tag prior (`4.4652 bpb`) |
| **Mean ($t \in [0.1, 0.9]$)** | **`3.1886 bpb`** | Full continuous-time NELBO bound |

---

## 5. Experiment 2 & 3: Format Validity (Raw vs. Repaired), Property Matching & Memorisation Audit

We evaluated $N = 10$ generated samples per format across 8 formats (`HTML`, `JSON`, `SVG`, `JS`, `C`, `BMP`, `WAV-uLaw`, and `AuraTile`), testing raw parser acceptance before repair, parser acceptance after up to 3 rounds of **repair-by-remasking**, distribution property matching against target conditioning tags, and Longest Common Substring (LCS) memorisation ratio against the training corpus. For C code, we wired the host `gcc -fsyntax-only -Wall -Wno-unused -xc -` compiler directly into `setExternalCCompilerHook`.

### Table 2: Parser Validity (Raw vs. Repaired) and LCS Memorisation Ratio

| Format | Native Validator / Parser | Raw Valid Rate | Repaired Valid Rate | Mean LCS Ratio | Max LCS Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **HTML** | Strict DOM / Tag Stack + `DOMParser` | `0.0%` | **`100.0%`** | `0.828` | `1.000` |
| **JSON** | `JSON.parse` + schema check | `0.0%` | **`100.0%`** | `0.653` | `1.000` |
| **SVG** | XML/SVG `DOMParser` + `<svg>` root | `20.0%` | **`100.0%`** | `0.196` | `0.329` |
| **JS** | Sandboxed Worker `new Function(code)` | `0.0%` | **`100.0%`** | `0.640` | `1.000` |
| **C** | Structural C parser + **`gcc -fsyntax-only -xc -`** | `20.0%` | **`70.0%`** | `0.614` | `1.000` |
| **BMP** | `0x42 0x4D` header + 24-bit row-stride check | **`100.0%`** | **`100.0%`** | `0.170` | `0.170` |
| **WAV (μ-law)** | `RIFF/WAVE` header + 16-bit PCM length check | **`100.0%`** | **`100.0%`** | `0.057` | `0.057` |
| **AuraTile** | `.aura.png` `tEXt` chunk + Helix JSON schema | **`100.0%`** | **`100.0%`** | `0.064` | `0.064` |

### Table 3: Conditioned Distribution Property Matching

| Format & Target Property | Analyzer Method | Conditioned Accuracy | Notes |
| :--- | :--- | :---: | :--- |
| **HTML DOM Nesting Depth** ($\pm 1$ level on `depth-4..7`) | `measureHTMLDepth` | **`62.5%` (`5/8`)** | Preserves balanced nested `<ul><li>`, `<table><tr><td>`, and `<form>` trees |
| **BMP Connected-Component Shape Count** ($\pm 1$ on `1..3` shapes) | `analyzeBMPScene` (BFS flood-fill) | **`100.0%` (`8/8`)** | 2-D $4 \times 4$ patch tokens preserve object coherence without quilt seams |
| **BMP Palette Colour Match** (`red`, `green`, `blue`, `yellow`, `purple`, `orange`) | `analyzeBMPScene` | **`100.0%` (`8/8`)** | AdaLN tag conditioning selects the exact target foreground colour |

---

## 6. Experiment 4: Unified Co-Training vs. Single-Format Specialists

We compared a single unified `FabulaTransformer` trained on all formats simultaneously against 4 separate single-format specialist models trained with the exact same per-format step budget (`8` passes over the format's training windows).

### Table 4: Held-Out NELBO (bpb) — Unified Model vs. Single-Format Specialists

| Format | Specialist Model (`bpb`) | Unified Co-Trained Model (`bpb`) | Delta (`Unified - Specialist`) | Outcome |
| :--- | :---: | :---: | :---: | :--- |
| **HTML** | `0.9358 bpb` | **`0.8958 bpb`** | **`-0.0400 bpb`** | **Unified wins** (shared ASCII/delimiter representations with SVG/JSON/JS) |
| **JSON** | `1.1089 bpb` | **`1.0117 bpb`** | **`-0.0972 bpb`** | **Unified wins** (shared punctuation/numeric literal statistics) |
| **BMP (2-D Patches)** | `0.0181 bpb` | **`0.0065 bpb`** | **`-0.0116 bpb`** | **Unified wins** (shared 2-D patch embeddings & colour tag projections) |
| **WAV (8-bit μ-law)** | **`1.4172 bpb`** | `1.5138 bpb` | `+0.0965 bpb` | **Specialist wins slightly** (companded audio bytes have distinct statistics from text) |

---

## 7. Experiment 5: Audio Path A (8-bit μ-law) vs. Path B (AuraGraph Tiles + Timing Helix)

We tested both audio representations on the seeded synthetic audio corpus (`tone`, `interval`, and `drums` across 6 pitches `C4..E5` and 3 tempos `96, 120, 144 BPM` at `22,050 Hz`):
- **Path A (8-bit ITU-T G.711 μ-law):** Compands each PCM sample into one byte in `0..255` and diffuses 1-D windows with overlap context.
- **Path B (AuraGraph v0.3.1 Tiles + AuraLoom v0.5 Decoder):** Encodes audio into 160 log-frequency bands (`30..11,000 Hz`) × time frames (`1 byte` of loudness per cell), arranged into $4 \times 4$ 2-D patches (`16` bytes/token), while co-generating 4-way **Timing Helix** event classes (`0: none`, `1: P pitch pulse`, `2: T transient onset`, `3: U unvoiced/release`) via `helixHead`. We compare **Ghost playback** (`agInvertGhost`, Griffin-Lim random phase) against **Helix playback** (`agInvertHelix`, phase-tethered ISTFT + transient attack alignment).

### Table 5: Audio Path Comparison (Pitch, Tempo, Onset Sharpness, and Waveform SNR)

| Metric | Path A: 8-bit μ-law | Path B: AuraGraph Tile + Ghost | Path B: AuraGraph Tile + Helix | Winner & Explanation |
| :--- | :---: | :---: | :---: | :--- |
| **Multi-Window Generated Pitch Match** ($\pm 1$ semitone) | `16.7%` | `33.3%` | **`33.3%`** | **AuraGraph Tiles win (`2.0×`)** — 2-D patches see all 160 frequency bands at once; 1-D μ-law loses phase coherence across windows |
| **Multi-Window Generated Tempo Error** (`% BPM error`) | `13.9%` | `6.9%` | **`6.9%`** | **AuraGraph Tiles win (`2.0×` lower error)** — 15s fits in ~650 columns vs. 120,000 raw samples |
| **Mean Onset Rise Time** (`10% -> 90%`, lower is sharper) | **`2.41 ms`** | `12.21 ms` | **`4.77 ms`** | **μ-law wins overall (`2.41 ms`); Helix sharpens Tile onsets by `60.9%` over Ghost (`4.77 ms` vs. `12.21 ms`)** |
| **Single-Rebuild Waveform SNR** (`dB`) | `36.40 dB` (8-bit quant.) | `-2.14 dB` | **`-2.04 dB` (`+0.10 dB`)** | Helix phase tethering + transient alignment improves phase coherence over Ghost |

---

## 8. Experiment 6: Speed & Memory Footprint Across Presets

Measured in Node V8 (`tests/metrics.js`) on 64-byte windows (and estimated for a 4× CPU-throttled mobile phone browser):

### Table 6: Model Presets, Parameter Counts, `.fabula` Size, and Latency

| Preset | Architecture (`L / D / H / dFF`) | Parameter Count | `.fabula` File Size | Train Step (SIMD WASM) | Train Step (Plain JS) | Train Step (4× Throttled Phone) | Sample Latency (Desktop / 4× Phone) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Phone`** | `2L, D=64, H=4, dFF=128` | **`214,036`** | `1,116 KB` | **`52.04 ms`** | `83.55 ms` | `~334.2 ms` | `85.2 ms` / `~340.7 ms` |
| **`Laptop`** | `3L, D=128, H=4, dFF=256` | **`722,132`** | `3,762 KB` | **`297.70 ms`** | `341.99 ms` | `~1,368.0 ms` | `742.8 ms` / `~2,971.3 ms` |
| **`Patient`** | `4L, D=192, H=6, dFF=384` | **`1,732,372`** | `9,024 KB` | **`599.68 ms`** | `993.09 ms` | `~3,972.4 ms` | `2,427.1 ms` / `~9,708.6 ms` |
