# The Autoencofabulator

[![Version](https://img.shields.io/badge/version-0.2.2-7c5cfc?style=flat-square)](#changelog)
[![MIT License](https://img.shields.io/badge/license-MIT-7c5cfc?style=flat-square)](LICENSE)
[![Xinu Compliant](https://img.shields.io/badge/xinu-compliant-3d2d8a?style=flat-square)](https://consciousnode.github.io)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-17152a?style=flat-square)](#)
[![Browser Native](https://img.shields.io/badge/runtime-browser_native-0d0b1e?style=flat-square)](#)
[![Single File](https://img.shields.io/badge/build-single_file-17152a?style=flat-square)](#)
[![ConsciousNode SoftWorks](https://img.shields.io/badge/ConsciousNode-SoftWorks-7c5cfc?style=flat-square)](https://github.com/ConsciousNode)

**[→ Open in Browser](https://consciousnode.github.io/The-Autoencofabulator/)**

A browser-native diffusion model that trains on your files and generates new ones. Never leaves the browser. No upload. No server. No account.

> *"Give me a WAV that sounds upbeat and jazzy."*
> Drop some WAV files. Train. Tag. Generate. Download.

---

## What It Is

The Autoencofabulator treats every file as what it fundamentally is: **a byte sequence with statistical structure**. Diffusion models learn to reverse a noise process — train on enough examples of a format and the model learns the statistical structure of that format implicitly, without being told what a WAV header is, or what HTML looks like, or how BMP pixel rows are laid out.

This is diffusion applied directly to raw bytes. No latent space. No encoder. No external dependencies. One HTML file, offline forever.

---

## Supported Formats — v0.2

| Format | Encoding | What gets diffused | Notes |
|--------|----------|--------------------|-------|
| **TXT** | Raw bytes | Full byte sequence | UTF-8, ASCII; model learns character distributions |
| **HTML** | Raw bytes | Full byte sequence | Single-file artifacts; model learns tag structure implicitly |
| **BMP** | **ElasticTok** | 8×8px spatial patches (192 RGB floats + position) | Preserves 2D topology; header templated |
| **WAV** | **SpikeVox** | LIF spike-rate features | Biologically-grounded audio encoding; header templated |

BMP and WAV headers are deterministic for given dimensions/sample rates — no point diffusing them. We template the header and diffuse what actually varies: the pixels and the samples.

---

## How It Works

### Training

1. Drop files into the corpus (TXT, HTML, BMP, or WAV — mix formats freely)
2. Files are validated, stripped to their payload, split into 256-element chunks
3. Each training step:
   - Random chunk sampled from corpus
   - Random timestep `t ∈ [1, 200]` sampled
   - Chunk noised via forward diffusion: `x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε`
   - **ScoreNet** predicts the noise `ε`
   - MSE loss computed; gradients flow back through ScoreNet
   - Adam optimizer updates weights
4. Loss curve updates every 8 steps; browser stays responsive via cooperative yielding

### Generation

1. Select format, enter semantic tags (`upbeat, jazz, minimal, calculator`...)
2. Specify output size (chars / pixels / duration)
3. Start from pure Gaussian noise, chunk by chunk
4. DDPM reverse sampling for `t = T → 1` per chunk
5. Denormalize; prepend templated header (BMP/WAV)
6. Validation layer runs automatically
7. Download

### Validation (from Once Bytten)

Every generated output is scanned:
- **Magic bytes** — header matches expected format signature
- **Shannon entropy** — within expected range for format
- **ROSA structural ratio** — bigram uniqueness within expected range
- **Format-specific checks** — BMP dimension consistency, WAV data size, HTML marker presence

Result: PASS / WARN / FAIL. FAIL triggers a resample suggestion.

---

## Architecture: ScoreNet

A ~664K parameter noise-prediction MLP. No external ML library.

```
Input: [noisy_chunk(256) | timestep_emb(64) | format_emb(16) | semantic_emb(32)] → 368-dim
  LayerNorm
  Linear(368 → 512)
  AdaLN conditioning: shift + scale from Linear(48 → 512) on [format_emb | semantic_emb]
  SiLU
  LayerNorm
  Linear(512 → 512)
  SiLU
  LayerNorm
  Linear(512 → 256) → predicted noise ε
```

**Diffusion:** DDPM (Ho et al. 2020). Linear schedule, β₁=1e-4 → β_T=0.02, T=200.
**Optimizer:** Adam (β₁=0.9, β₂=0.999, lr=1e-4).
**Conditioning:** format as learned dense embedding; semantic tags as mean-pooled bag-of-words over a 256-word vocabulary; timestep as sinusoidal embedding.

---

## Semantic Tags

Type comma or space-separated tags when generating. Recognized vocabulary covers:

- **Tone/mood:** `upbeat dark moody bright intense gentle calm aggressive playful mysterious`
- **Color (BMP):** `red green blue yellow purple orange teal warm cool vivid muted`
- **Audio (WAV):** `bass treble jazz classical electronic ambient percussion melody rhythm noise`
- **Text (TXT/HTML):** `formal casual technical poetic minimal calculator game dashboard tool`
- **Style:** `retro modern geometric organic dense sparse layered flat`
- **...and ~200 more**

Unknown tags are hashed into the embedding space — they still condition the model, just without a pre-learned direction.

---

## Weight Files (`.fabula`)

Export and import trained weights via the Model tab. Format:

```json
{
  "version": "0.1",
  "steps": 12400,
  "config": { "T": 200, "chunk": 256, "hid": 512, "vocab": 256 },
  "weights": { "...": "base64 Float32Arrays" }
}
```

Share `.fabula` files to distribute trained models. Load one to continue training or generate immediately.

---

## Integration Map

| Component | Source |
|-----------|--------|
| `Linear` (forward + backward) | Simulacra — ConsciousNode |
| `LayerNorm` (forward + backward) | Simulacra — ConsciousNode |
| `Adam` optimizer | Simulacra — ConsciousNode |
| AdaLN conditioning pattern | OmniVocal — ConsciousNode |
| `ElasticTok` spatial patch encoding | FPSS — ConsciousNode (Kehai Interim) |
| `SpikeVox` LIF audio encoding | FPSS — ConsciousNode (Kehai Interim) |
| `shannonEntropy` | Once Bytten — ConsciousNode |
| ROSA structural ratio | Once Bytten / Simulacra — ConsciousNode |
| Magic byte validation | Once Bytten — ConsciousNode |
| DDPM diffusion math | Ho et al. 2020 |

No new ML primitives. The ConsciousNode catalog assembling itself.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for full version history.

**v0.2.2** — Debug panel: live console, gradient health canvas, generation journal, format step distribution, optimizer state.  
**v0.2.1** — Critical backward pass fix (l2/ln2 order was inverted — primary cause of non-convergence). DDIM deterministic sampling toggle. Live LR control. Best practices guide.  
**v0.2** — ScoreNet v0.2 (4-layer, 768-dim, dual AdaLN, ~1.54M params). ElasticTok BMP encoding. SpikeVox WAV encoding. HTML iframe preview. WAV Web Audio playback. ⚠ breaks v0.1 `.fabula` compatibility.  
**v0.1.x** — OLA seam fix, per-file tagging, dynamic corpus, DDPM Algorithm 2 form, TXT truncation fix, corpus warnings.  
**v0.1** — Initial release.

---

## Roadmap

| Version | Target |
|---------|--------|
| **v0.1** | TXT + HTML + BMP + WAV · ScoreNet · DDPM · validation · `.fabula` I/O ✓ |
| **v0.1.x** | OLA seam fix · per-file tagging · dynamic corpus · bug fixes ✓ |
| **v0.2** | ElasticTok (BMP) · SpikeVox (WAV) · 4-layer ScoreNet · debug panel ✓ |
| **v0.3** | Compiler-as-validator — generate source → compile → result as validation signal |
| **v0.4** | Compiler feedback loop — error messages as fine-tuning signal |
| **v0.5** | PNG (uncompressed RGB diffusion, zlib on export) · multi-format co-training |
| **v0.6** | Transfer: Simulacra RWKV state as text conditioning for TXT/HTML |
| *research* | **EXE via compiled source** — diffuse source code, compile to binary. The binary was never the goal; the *program* was. |

### The EXE reframe

The original EXE target assumed binary diffusion — valid space too narrow, header synthesis too hard. The correct path: train on source code (TXT tier, already supported) → compiler validates output → compiler errors become fine-tuning signal → compile to binary as post-generation step. The EXE goal was always a code generation problem wearing a diffusion model's clothes.

### ElasticTok + SpikeVox

**ElasticTok (BMP):** 8×8px spatial patch chunking preserves 2D image topology. Each chunk encodes one patch: 192 RGB floats + normalized (row, col) position at slots [192, 193]. The model sees patch position twice — in the input chunk AND in the conditioning vector. Ported from FPSS (Kehai Interim).

**SpikeVox (WAV):** LIF (leaky integrate-and-fire) audio encoding converts PCM to spike-rate feature vectors. Parameters: threshold=3.5, leakRate=0.95, refractoryPeriod=2. Decode: log-spaced sinusoidal band synthesis. The cochlea does something structurally similar. Ported from FPSS (Kehai Interim).

---

## Philosophy

Xinu-compliant. The browser is bare metal. A file is a byte sequence with statistical structure. Diffusion learns statistical structure. Therefore: one model architecture, trained in your browser, on your files, generating new ones — across any format whose valid space is wide enough to sample from.

No install. No server. No upload. Works offline. Works on the phone in your pocket. Opens as a text file. Flash was a tenant. ConsciousNode builds on land it owns.

---

## Credits

**Kham** (Khamerron Ramsey Kizer) — concept, architecture, direction
**Ed Interim** — implementation
**Kehai Interim** — Simulacra ML stack (Linear, LayerNorm, Adam)
**Vael Interim** — OmniVocal AdaLN conditioning pattern

Validation layer adapted from **Once Bytten** (Ed Interim).
DDPM: Ho et al., "Denoising Diffusion Probabilistic Models," NeurIPS 2020.

---

## License

MIT — See [LICENSE](LICENSE)

*ConsciousNode SoftWorks · consciousnode.github.io*
