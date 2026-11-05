# The Autoencofabulator v0.3: Realized

**One model that learns any file in a single offline browser tab — made real with Masked Byte Diffusion, 2-D Patch Tokens, Dual-Path Audio (8-bit μ-law & AuraGraph v0.3.1 Tiles + Timing Helix), and Compiler-as-Validator Repair.**

> **v0.3.1 audit note (2026-11-04).** Everything below the line describes what v0.3 *is*, and the engine is real and
> verified (gradients, determinism, kernels). But the v0.3.0 results were inflated: repair fell back to copying
> training files, generation started from training files by default, and the audio comparison was unequal. Those
> paths are now off (repair is model-only; "Remix a training file" is an explicit, labelled option), and the
> honest numbers are in [research/AUDIT-1.md](research/AUDIT-1.md). In short: after a short train the model can't
> yet write parseable markup or code on its own, the count memory beats the transformer on Sherlock, and μ-law
> beats tiles when both get the same job.

---

## 1. What Changed in v0.3 ("Realized")

In v0.1–v0.2.3, The Autoencofabulator diffused raw byte values as continuous numbers in $[-1, +1]$ under Gaussian noise with a 4-layer chunk-independent MLP (`ScoreNet`). Even with exact analytical gradients (v0.2.1 / v0.2.3), that design had four fundamental walls:
1. **Bytes are categories, not quantities:** Gaussian noise between `'<'` (60) and `'>'` (62) passes through `'='` (61), turning text and markup into unparseable static.
2. **Independent chunks have no context:** Generating 256-byte chunks independently meant HTML tags opened in chunk 1 could never close in chunk 2, and BMPs were quilts of unrelated $8\times 8$ patches.
3. **Audio was capped by sum-of-sinusoids decoding:** `SpikeVox` could only ever produce band-limited synth wash.
4. **Nothing was measured against held-out data or baselines.**

**v0.3 fixes all four:**

| Pillar | v0.2.3 (Legacy) | v0.3 (*Realized*) |
| :--- | :--- | :--- |
| **Diffusion Process** | Continuous Gaussian DDPM over byte values $\in [-1, +1]$ | **Masked Byte Diffusion (MDLM / D3PM absorbing state)** over 256 byte categories (`0..255`) + `[MASK]` (`256`) |
| **Denoiser Backbone** | 4-layer MLP (`ScoreNet`) on independent 256-element chunks | **Bidirectional RoPE + Local-Conv + Dual-AdaLN Transformer (`FabulaTransformer`)** over 512-byte windows + Differentiably-Gated Bilateral Context Memory |
| **Long Files & Infilling** | Independent chunks (no boundary continuity) | **Sliding-window generation with 128-byte fixed context overlap** + arbitrary hole **Infilling** (`??` / `[MASK]` spans) |
| **2-D Images (BMP)** | Independent $8\times 8$ RGB float patches (`ElasticTok`) | **$4\times 4$ 2-D byte patch tokens** (`16` bytes/token) with 2-D row & column embeddings + top-row context continuity (`ElasticTok` kept as labelled legacy option) |
| **Audio Path A** | `SpikeVox` LIF spike rates + sum of sinusoids | **8-bit ITU-T G.711 μ-law** waveform bytes (`256` categories) with overlap context, exported as 16-bit PCM `.wav` (`SpikeVox` kept as labelled legacy option) |
| **Audio Path B** | None | **Inlined AuraGraph v0.3.1 tiles** (`160` log-frequency bands × time) + co-generated **Timing Helix** events decoded via **AuraLoom v0.5** (**Helix** & **Ghost** playback), exported as `.aura.png` and `.wav` |
| **Validation & Repair** | Magic bytes + Shannon entropy + ROSA bigram uniqueness | **Compiler-as-Validator in the tab** (`JSON.parse`, `DOMParser` for HTML/SVG/XML, sandboxed Worker `new Function` for JS, structural C + external `gcc -fsyntax-only` hook in Node) + **Repair-by-Remasking** around parser error byte offsets |
| **Evaluation** | Training MSE curve only | Held-out continuous-time **NELBO in bits/byte (bpb)** vs. Unigram, Bigram & Order-4 Kneser-Ney baselines on *Sherlock Holmes* (`80/10/10` split), property analyzers, and LCS memorisation checks (`research/REALIZED-1.md`) |

---

## 2. Mathematical Bound: Absorbing-State Masked Diffusion (MDLM NELBO)

Let $x_0 \in \{0, \dots, 255\}^L$ be a clean byte window of length $L$, and let $m = 256$ denote the `[MASK]` token. At continuous diffusion time $t \in (0, 1]$, the forward process $q(x_t \mid x_0, t)$ independently masks each non-fixed byte $x_0^{(i)}$ to $m$ with probability $\mu(t) = 1 - \alpha(t)$.

- **Linear schedule (`linear`):** $\alpha(t) = 1 - t$, $\mu(t) = t$, weight $w(t) = -\alpha'(t)/(1 - \alpha(t)) = 1/t$.
- **Cosine schedule (`cosine`):** $\alpha(t) = \cos(\pi t / 2)$, $\mu(t) = 1 - \cos(\pi t / 2)$, weight $w(t) = (\pi/2)\sin(\pi t / 2)/(1 - \cos(\pi t / 2))$.

The denoiser outputs a 256-way categorical distribution $p_\theta(x_0^{(i)} \mid x_t, t, c)$ at each masked position $i \in M_t = \{i : x_t^{(i)} = m\}$, conditioned via Adaptive LayerNorm (AdaLN) on format embedding $e_{\text{fmt}}$, normalized bag-of-words tag embedding $e_{\text{sem}}$, and sinusoidal time embedding $e_t$. The continuous-time negative evidence lower bound (NELBO) in **bits per byte (bpb)** is:

$$\mathcal{L}_{\text{NELBO}}(x_0;\theta) = \mathbb{E}_{t \sim \mathcal{U}(0, 1),\, x_t \sim q(\cdot \mid x_0, t)} \left[ \frac{w(t)}{L \ln 2} \sum_{i \in M_t} -\ln p_\theta\!\left(x_0^{(i)} \mid x_t, t, c\right) \right] \ge -\frac{1}{L}\log_2 p_\theta(x_0)$$

---

## 3. Model Presets & Parameter Counts

All three presets run 100% offline in a single browser tab (with optional inline `MicroWAT` / `WasmKernels` SIMD acceleration and automatic plain-JS fallback):

- **Phone (`phone`):** `2` layers, `dModel = 64`, `nHeads = 4`, `dFF = 128` — **`214,036` parameters** (`1,116 KB` `.fabula` file, `~25–52 ms/step` depending on machine).
- **Laptop (`laptop`):** `3` layers, `dModel = 128`, `nHeads = 4`, `dFF = 256` — **`722,132` parameters** (`3,762 KB` `.fabula` file, `~110–300 ms/step`).
- **Patient (`patient`):** `4` layers, `dModel = 192`, `nHeads = 6`, `dFF = 384` — **`1,732,372` parameters** (`9,024 KB` `.fabula` file, `~225–600 ms/step`).

Legacy v0.1/v0.2 `.fabula` weight files are detected by their version header and **rejected gracefully** with an explanatory toast banner without crashing the app.

---

## 4. Inlined Studio Code & Provenance

Every piece of inlined studio code in `index.html` carries an explicit provenance header comment:
- `/* from MicroWAT microwat.js v2.0.0, MicroWAT.assemble() */` — zero-dependency WAT-to-WASM assembler (scalar + 128-bit `v128` SIMD).
- `/* from Simulacra/OmniVocal kernels.js v0.2.0, WasmKernels */` — SIMD/WASM/JS matrix-vector & activation kernels.
- `/* from AuraGraph app.html v0.3.1 (TILE-FORMAT v0.4), mulberry32() & fnv1a32() */` — deterministic PRNG and FNV-1a seed hashing.
- `/* from Simulacra v0.2 / OmniVocal v0.1, Linear / LayerNorm / Adam / AdaLN, extended to multi-token & WasmKernels */` — autodiff layers.
- `/* from AuraGraph app.html v0.3.1 (TILE-FORMAT v0.4), fftOf(), PLAN, analyzeCore(), finalizeAnalyze(), analyzeSync(), buildHelixData(), LUT, drawHelixStripRGBA(), CRC_T, crc32(), makePngChunk(), pngWithMeta(), pngReadMeta(), meanRiseTimeMs(), evaluateMetrics() */` — real AuraGraph tile encoder, PNG `tEXt` chunk writer/reader, and Timing Helix extractor.
- `/* from AuraLoom index.html v0.5 (TILE-FORMAT v0.4), invertGhost(), invertHelix(), pixLevel() */` — real AuraLoom Ghost (Griffin-Lim) and Helix (phase-tethered ISTFT) audio decoders.
- `/* from Once Bytten v0.1, SpikeVox */` — legacy LIF spike-rate audio encoder (retained as a labelled legacy option).
- `/* from FPSS v0.2, shannonEntropy() & rosaRatio() */` — statistical validation metrics.

---

## 5. Running the App and Test Suite

### In the Browser
Open `index.html` directly from disk (`file://`) on desktop or mobile. Zero network requests, zero external scripts or fonts.
1. **Corpus Tab:** Load the built-in seeded multi-format test corpus (or *Sherlock Holmes* excerpt), or drop your own `.txt`, `.html`, `.json`, `.svg`, `.js`, `.c`, `.bmp`, `.wav`, or `.aura.png` files with custom tags.
2. **Train Tab:** Pick a preset (`Phone`, `Laptop`, `Patient`), noise schedule (`linear` or `cosine`), kernel backend (`Auto SIMD/WASM/JS`), and seed. Monitor the training NELBO (`bpb`), held-out validation NELBO (`bpb`), and Unigram / Bigram / Order-4 Kneser-Ney baselines live. Pause, resume, or export `.fabula` weights mid-run.
3. **Generate Tab:** Choose any format, audio path (`8-bit μ-law`, `AuraGraph Tile + Helix`, `AuraGraph Tile + Ghost`, or legacy `SpikeVox`), unmasking strategy (`Confidence-ordered` or `Random`), optional low-confidence re-masking, and optional **Infilling** template. Export `.wav`, `.aura.png` (opens directly in AuraLoom v0.5), `.bmp`, or text/code files.
4. **Validate Tab:** Inspect compiler/parser diagnostics (`JSON.parse`, `DOMParser`, Worker `new Function`, C syntax check, BMP/WAV structure), Shannon entropy, ROSA bigram ratio, LCS memorisation fraction, and trigger **Repair by Re-Masking**.

### Headless Node Verification & Benchmarks
```bash
# 1. Analytical vs. Finite-Difference Gradient Check (Float64, 53/53 checks at cos = 1.000000)
node tests/gradcheck.js

# 2. Bit-for-Bit Determinism, .fabula v0.2 Rejection / v0.3 Round-Trip, and JS/WASM/SIMD Equivalence
node tests/determinism.js

# 3. Full Tier 4 Empirical Evaluation Suite (outputs tests/metrics-results.json)
node tests/metrics.js
```
See [research/REALIZED-1.md](research/REALIZED-1.md) for Argon's v0.3.0 report and **[research/AUDIT-1.md](research/AUDIT-1.md) for the audit and corrected numbers**.

```bash
# 4. Audit: each experiment run the v0.3.0 way and the controlled way side by side
node tests/audit-v03.js
```
