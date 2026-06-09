# Changelog — The Autoencofabulator

All notable changes documented here.

---

## [0.1.2] — 2026-06-08

### Added

- **Per-file semantic tagging (Approach A).** Each corpus file entry now has an editable tag input field below the filename. Tags update live — `updateFileTags()` recomputes `tagIds` on every keystroke. Training loop now samples `tagIds` from the corpus entry rather than always passing `[]`. Model learns tag→output associations proportional to corpus diversity and tag quality.

- **Auto-tag from filename (Approach B).** `autoTagFromFilename()` strips the file extension, splits on `_`, `-`, `.`, and space, lowercases, and matches tokens against the 256-word semantic vocabulary. Pre-populates the tag input on load. `dark_ambient_loop.wav` → `dark ambient`. User can edit, extend, or clear at will. Unknown tokens still get hashed into embedding space — graceful degradation.

- **Null filter in training loop.** `CORPUS` entries set to `null` on file removal are now filtered before building the chunk pool.

---

## [0.1.1] — 2026-06-08

### Fixed

- **WAV/BMP chunk seam artifacts via OLA (overlap-add).** Previously, generated chunks were chained independently — each starting from its own pure noise with zero knowledge of adjacent chunks. For WAV this produced audible clicks at every ~11.6ms boundary (~86Hz artifact). For BMP it produced faint discontinuities at pixel row boundaries.

  Now uses **triangular-windowed overlap-add** at 50% overlap (hop size = 128) for WAV and BMP formats. The triangular window satisfies perfect reconstruction: `w[i] + w[i + CHUNK/2] = 1` for all `i`, meaning the overlap-add sum normalizes back to 1 at every position. Seams become smooth crossfades.

  TXT/HTML retain simple chaining — byte boundaries are invisible for text formats, OLA would add overhead without benefit.

  Trade-off: WAV and BMP generation now runs ~2× more forward passes (correct output over speed — optimization deferred to v0.2).

---

## [0.1] — 2026-06-07

### Initial release

**ScoreNet**
- ~664K parameter noise-prediction MLP: Linear(368→512) → AdaLN → SiLU → Linear(512→512) → SiLU → Linear(512→256)
- Full forward and backward pass — no external autograd library
- AdaLN conditioning: format + semantic embeddings projected to per-layer shift/scale (pattern from OmniVocal)
- Sinusoidal timestep embedding (T_dim=64, no learned params)
- Learned format embedding (N_formats=4, dim=16)
- Learned semantic embedding (vocab=256, dim=32), mean-pooled bag-of-words

**Diffusion**
- DDPM (Ho et al. 2020), browser-native
- Linear noise schedule: β₁=1e-4 → β_T=0.02, T=200
- Forward: `x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε` (Box-Muller Gaussian)
- Reverse: DDPM posterior mean + stochastic term (σ_t · ε for t > 1)
- x₀ prediction clamped to [-1, 1] for stability

**Optimizer**
- Adam (β₁=0.9, β₂=0.999, ε=1e-8, lr=1e-4)
- Cooperative yielding via `yieldToMain()` every 8 steps — browser stays responsive

**Supported formats — v0.1 tier**
- TXT: full byte sequence, normalized to [-1, 1]
- HTML: full byte sequence, normalized to [-1, 1]
- BMP: 54-byte header templated from dimensions; RGB pixel payload diffused
- WAV: 44-byte RIFF header templated from sample rate + channels; Int16 PCM decoded to Float32 and diffused

**Format parsing**
- BMP: header validation (magic bytes `0x42 0x4D`), width/height/pixelOffset extraction, payload slicing
- WAV: RIFF header validation, `data` chunk location by scan, Int16→Float32 normalization
- Corpus integrity: files with invalid headers flagged and excluded before training

**Header generation**
- BMP: correct 54-byte BITMAPFILEHEADER + BITMAPINFOHEADER for given width × height, 24bpp, no compression
- WAV: correct 44-byte RIFF/WAVE/fmt/data structure for given sample rate, 1 channel, 16-bit PCM

**Validation layer (from Once Bytten)**
- Magic byte check (BMP, WAV)
- HTML marker check (`<html` / `<!doctype`)
- Shannon entropy within expected range per format
- ROSA structural ratio (bigram uniqueness) within expected range per format
- BMP: dimension/file-size consistency
- WAV: data chunk size vs declared sample count
- Verdict: PASS / WARN / FAIL

**Semantic vocabulary**
- 256 words across tone, color, audio, text style, HTML function, structure, and signal/noise descriptors
- Unknown tags hashed into embedding space (graceful degradation)

**Weight serialization**
- `.fabula` JSON format: all weights as base64-encoded Float32Arrays
- Export from Model tab; import by file drop or selector
- Steps counter preserved across sessions

**UI**
- Four tabs: TRAIN / GENERATE / VALIDATE / MODEL
- Drag-and-drop corpus file zone with format badge + integrity status per file
- Rolling 400-step loss curve canvas with shaded recent average
- Generation controls: format selector, semantic tag input, size controls (format-adaptive)
- Configurable reverse steps (20–200, default 100)
- BMP preview rendered to canvas (bottom-up BGR→RGB correct)
- WAV/TXT/HTML text preview
- Mobile-responsive header (stats wrap, logo tags hide at ≤420px)
- Download generated output as correctly-typed file
- Dark theme: void `#07060f`, accent `#7c5cfc`, signal `#2de0a5`

**Integration**
- Linear / LayerNorm / Adam: ported from Simulacra (ConsciousNode)
- AdaLN pattern: ported from OmniVocal VocoderUpsampleBlock (ConsciousNode)
- shannonEntropy / rosaRatio / magic byte DB: ported from Once Bytten (ConsciousNode)
- Xinu-compliant: zero external dependencies, single HTML file
