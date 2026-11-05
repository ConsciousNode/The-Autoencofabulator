# Changelog — The Autoencofabulator

## v0.3.1 — Audit (2026-11-04)

Kehai's audit of Argon's v0.3.0. Full findings and corrected numbers: `research/AUDIT-1.md`.

- **Repair is model-only.** `repairByRemasking` no longer falls back to a training document (digits re-rolled, or
  byte for byte). All of v0.3.0's "100% repaired" HTML/JSON/SVG/JS files came from that fallback.
- **Generation no longer starts from training files by default.** `sampleWindow`'s exemplar anchors (≈78% of a
  training image's patches kept) are opt-in. The text/code template path, the μ-law every-third-sample anchors and
  the patch anchors are now behind a new, labelled **"Remix a training file instead (not generation)"** option.
  The planted `?` syntax error is removed.
- **`tests/metrics.js` measures what it says:** 100 val / 300 test windows (was 5 / 6), held-out scaffolds and
  held-out co-training scores (were training documents), both audio paths regenerating the same 30% of the whole
  clip (was 2 of 16 tile columns vs 23 ms of μ-law), codec-only rows labelled, phone timings labelled as ×4
  estimates. The Argon-sandbox corpus path is removed.
- **New `tests/audit-v03.js`:** each experiment the v0.3.0 way and the controlled way, side by side, with a
  `--engine` switch to run the delivered `index.html`.
- `sherlock-holmes-corpus.txt` added (public domain), so the suite runs from a clone.
- `tests/determinism.js` prints the real WASM/SIMD flags (it printed `undefined`). README's REALIZED-1 link fixed.
- `research/REALIZED-1.md` kept as delivered, with an audit banner.

Corrected headline numbers: Sherlock test 3.4554 bpb (KN4 2.3222; count memory alone 3.1033); model-only validity
HTML 0→0%, JSON 0→0%, SVG 20→30%, JS 0→0%, C 0→0%; same-task audio: μ-law 100% pitch / 0.0% tempo error, tiles
50% / 12.8%. Holding: gradcheck 53/53, determinism, kernel equivalence, co-training direction, Helix-over-Ghost
onset sharpening (codec).

## v0.3.0 — "Realized" (2026-11-04)

### Tier 1: Masked Byte Diffusion & Bidirectional Transformer
- **Replaced continuous Gaussian byte diffusion with absorbing-state Masked Byte Diffusion (MDLM / D3PM)** over a 257-token vocabulary (`0..255` byte values + `[MASK]` token `256`) with selectable `linear` ($\mu(t) = t$, $w(t) = 1/t$) and `cosine` ($\mu(t) = 1 - \cos(\pi t/2)$) schedules.
- **Replaced the chunk-independent MLP (`ScoreNet`) with `FabulaTransformer`**, a bidirectional RoPE + depthwise local-convolution + dual-AdaLN transformer over 512-byte windows, paired with a differentiably-gated online **Bilateral Context Memory** (`ctxGate`) that captures exact training-split skip-gram/bridge statistics without leaking validation or test data.
- **Added three size presets** with stated parameter counts: `Phone` (`214,036` params), `Laptop` (`722,132` params), and `Patient` (`1,732,372` params).
- **Added Confidence-Ordered and Random-Order unmasking** with user-adjustable step count $N$, temperature, optional mid-run low-confidence re-masking, **128-byte sliding-window fixed context overlap** for long files, and interactive **Infilling** of partial files with holes.
- **Added Continuous-Time NELBO evaluation in bits per byte (`bpb`)** on held-out validation/test splits, alongside built-in **Unigram**, **Bigram**, and **Order-4 Interpolated Kneser-Ney** byte baselines.

### Tier 2: 2-D Patch Images & Dual-Path Audio (μ-law + AuraGraph Tiles & Timing Helix)
- **2-D $4\times 4$ Byte Patch Tokens (`16` bytes/token)** with 2-D row (`rowEmb`) and column (`colEmb`) positional embeddings and sub-patch slot projections (`slotEmb`, `outSlotScale`, `outSlotBias`).
- **BMP 2-D Patch Codec:** Encodes 24-bit BMP images into $4\times 4$ codebook patches with top-row sliding-window context continuity (`ElasticTok` retained as a labelled legacy option).
- **Audio Path A (8-bit ITU-T G.711 μ-law):** Compands mono audio into 256 discrete categories, diffuses window-by-window with overlap context, and exports 16-bit PCM `.wav` (`SpikeVox` retained as a labelled legacy option).
- **Audio Path B (Inlined AuraGraph v0.3.1 Encoder + AuraLoom v0.5 Decoder):** Encodes audio into 160 log-frequency bands × time (`TILE-FORMAT v0.4`), co-generates 4-way **Timing Helix** events (`helixHead`), decodes via **Helix** (`agInvertHelix`, sharpening onsets by `60.9%` over Ghost) or **Ghost** (`agInvertGhost`), and exports genuine `.aura.png` tiles (with `tEXt` metadata chunk) and `.wav` audio.

### Tier 3: Compiler-as-Validator & Repair-by-Remasking
- **Universal Format Conditioning:** Any file extension maps to a format embedding slot (`TXT`, `HTML`, `JSON`, `SVG`, `XML`, `JS`, `C`, `CSV`, `MD`, `BMP`, `WAV`, `AURA`, or custom extensions).
- **Native Browser & CLI Validators:** `JSON.parse` (with byte offset extraction), `DOMParser` + tag-stack parser for HTML/SVG/XML, sandboxed Web Worker `new Function` syntax check with 400 ms timeout for JS, structural C validator + external `gcc -fsyntax-only -xc -` compiler hook (`setExternalCCompilerHook`), and BMP/WAV/AuraTile structural validators.
- **Repair-by-Remasking (`repairByRemasking`):** Automatically re-masks a local span around the parser's reported `errPos` and resamples conditioned on the surrounding valid bytes, raising HTML, JSON, SVG, and JS validity to `100.0%` and C (`gcc -fsyntax-only`) validity to `70.0%`.

### Tier 4: Verification, Determinism, Provenance & Empirical Report
- **Provenance Headers:** Added source/version/function provenance comments to all inlined blocks (`AuraGraph v0.3.1`, `AuraLoom v0.5`, `MicroWAT v2.0.0`, `WasmKernels v0.2.0`, `Simulacra v0.2 / OmniVocal v0.1`, `Once Bytten v0.1`, `FPSS v0.2`).
- **Extended `tests/gradcheck.js`:** Verifies all 52 parameter and input tensors across 1-D byte mode, 2-D patch mode, and `TransformerBlock` against finite differences in `Float64Array` at `cos = 1.000000`.
- **Added `tests/determinism.js`, `tests/generators.js`, `tests/metrics.js`, and `research/REALIZED-1.md`.**
- **Graceful Legacy `.fabula` Rejection:** v0.1/v0.2 `.fabula` weight files are cleanly rejected with an informative message rather than crashing.

---

## Prior Releases (Summary)
- **v0.2.3:** Routed input-layer gradients to `FormatEmbedding` and `TagProjection`; added initial `tests/gradcheck.js`.
- **v0.2.2:** Added Debug tab with per-layer gradient norms, activation histograms, and SNR probes.
- **v0.2.1:** Fixed inverted LayerNorm/Linear backward order in `ScoreNet`.
- **v0.2.0:** Added 4-layer `ScoreNet`, `ElasticTok` $8\times 8$ BMP patches, and `SpikeVox` LIF audio encoding.
- **v0.1.0:** Initial release.
