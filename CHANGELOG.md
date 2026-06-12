# Changelog — The Autoencofabulator

All notable changes documented here.

---

## [0.2.2] — 2026-06-10

### Added

- **Debug tab (⚙ Debug).** Full instrument panel for monitoring training health without opening browser devtools.

  - **Live Console:** Intercepts `console.log` / `warn` / `error` / `info` from the moment the page loads. Color-coded (dim/yellow/red/green), timestamped, auto-scroll toggle, clear button, max 500 entries. Every generation automatically logs `[GEN] format | entropy | ROSA | verdict | step`.

  - **Gradient Health canvas:** Per-weight RMS gradient norm plotted over training time. Green shaded band = healthy range (0.001–0.5). Red zone = exploding (>1.0). Line color reflects health status in real time. The single most useful diagnostic for whether learning is actually happening.

  - **Generation Journal:** Table of every generation, newest first (max 50). Columns: timestamp, training step, format, tags, Shannon entropy, ROSA ratio, verdict. Entropy and ROSA values colored when out of expected range. Lets you watch output quality trend over training time — when entropy drops below 7.0 on TXT, you'll see it here.

  - **Format Step Distribution:** Bar chart showing what percentage of training steps went to each format. Useful when corpus is mixed and you want to know if one format is dominating.

  - **Optimizer State:** Active learn rate, Adam step count, β params, momentum and velocity RMS with automatic health indicator.

---

## [0.2.1] — 2026-06-10

### Fixed

- **CRITICAL: ScoreNet backward pass — l2/ln2 order inverted.** In the four-layer ScoreNet, the gradient flow through the second layer pair (ln2→l2) was reversed. The correct backward order is `ln2.backward(l2.backward(siluGrad))` — LayerNorm on the outside, Linear on the inside, matching every other layer pair in the network. The code had them swapped: `l2.backward(ln2.backward(siluGrad))`. As a result, `ln2` was receiving the gradient at `x2` (l2's output) instead of `x2ln` (l2's input), corrupting weight gradients for both `l2` and `ln2` and poisoning the gradient signal flowing back to the first layers. This was the primary cause of near-zero learning despite training for 150k+ steps.

- **DDPM reverse step — Algorithm 2 form (v0.1.4 fix carried forward correctly).** Confirmed correct: `(1/√αₜ) * (xₜ - βₜ/√(1-ᾱₜ) * ε_pred)` — no x₀ clamping bias.

### Added

- **DDIM deterministic sampling toggle** in Generate tab. Checkbox disables the stochastic `σ_t · ε` noise term in `reverseStep`. Use when stochastic sampling produces incoherent output, especially on undertrained models where the noise term can overwhelm the signal.

- **Live learning rate control** in Model tab. Input updates `CFG.LR` and `optimizer.lr` immediately. No restart needed. Suggested values: `5e-4` if loss stuck after 50k steps, `1e-5` for fine-tuning after 500k+.

- **Best practices guide** in Model tab. Corpus recommendations, training milestones for TXT/HTML (30k / 200k / 500k / 1M+ steps), loss interpretation (watch the rolling average, not individual spikes), generation tips.

---

## [0.2] — 2026-06-08

### Breaking change

v0.2 `.fabula` weight files are **not compatible** with v0.1. ScoreNet architecture changed (wider, deeper, new conditioning dimension). The import dialog will warn and reject v0.1 files gracefully.

### Added

- **ScoreNet v0.2:** 4-layer architecture, 768-dim hidden (up from 512), dual AdaLN injection at layers 1 and 3. ~1.54M parameters (up from ~664K). Conditioning vector expanded to 50-dim: `[fmt(16), sem(32), patch_r(1), patch_c(1)]`. Second AdaLN injection lets the model reshape the representation at two depths with format+semantic+position signal.

- **ElasticTok (BMP):** Spatial 8×8px patch chunking, ported from FPSS (Kehai Interim). Each chunk encodes one patch: 192 RGB floats (8×8×3) at slots [0..191], normalized patch position `(r, c)` at slots [192, 193], zero-padding at [194..255]. Position flows through both the chunk input AND the conditioning vector — the model sees spatial location twice. BMP bottom-up BGR encoding handled correctly throughout. Loss backpropagation only touches the 192 active pixel dims, not padding noise. Generation assembles patches back into correct spatial positions.

- **SpikeVox (WAV):** LIF (leaky integrate-and-fire) audio encoding, ported from FPSS (Kehai Interim). Encode: PCM → band energy + ZCR + RMS feature extraction → LIF membrane/refractory step → EMA rate-coded spike features, normalized to [-1,1]. Decode: generated spike features → log-spaced sinusoidal band synthesis → PCM resampling. LIF parameters: threshold=3.5, leakRate=0.95, refractoryPeriod=2.

- **WAV playback:** Web Audio API `decodeAudioData` + `BufferSource` on generated WAV bytes. Play/stop controls in Generate tab output section.

- **HTML sandboxed iframe preview:** Generated HTML rendered in `<iframe sandbox="allow-scripts" srcdoc="...">` — scripts run in isolation, no same-origin access.

- **Encoding type badge** shown per corpus file entry (ElasticTok / SpikeVox).

- **Auto-switch to Validate tab** on generation completion.

- **Expanded semantic vocabulary:** 296 words (up from 266), adding spike/patch-relevant terms: spotted, striped, patterned, textured, pitched, tonal, transient, sustained, attack, decay, pulsing, fading, rising, falling, shifting, blending, flickering, stuttering, swelling, decaying, and more.

---

## [0.1.4] — 2026-06-08

### Fixed

- **DDPM reverse step — Algorithm 2 form.** Replaced x₀hat→posterior mean path (DDPM Eq. 7 with clamped x₀) with direct ε-prediction form from Algorithm 2: `(1/√αₜ) * (xₜ - βₜ/√(1-ᾱₜ) * ε_pred)`. Both are equivalent but Algorithm 2 avoids the bias introduced by clamping x₀ to [-1,1]. Scalars precomputed outside the sample loop.

- **TXT/HTML output truncated to exact requested length.** Previously generated `nChunks × CHUNK` bytes regardless of what the user specified.

- **Validation FAIL verdict** now suggests regenerating or training longer.

- **Minimum corpus size warning.** `startTraining()` warns at <10 total chunks (soft), blocks at 0 chunks (hard).

---

## [0.1.3] — 2026-06-08

### Fixed (from Grok/DeepSeek review)

- **Dynamic corpus resampling.** Training pool rebuilt each step — new files picked up immediately without stop/resume.
- **Null guard in `updateTrainUI` corpus count** — removed files no longer crash the counter.
- **`setStatus()` cleanup** — removed redundant id reassignment and spurious `.train-status` class (CSS uses `#id`, not `.class`).
- **Chunk progress text** — `Chunk X / N (Y%)` displayed live during generation alongside progress bar.
- **`CFG.VOCAB` derived from `VOCAB_WORDS.length`** — can never mismatch the actual vocabulary table.

---

## [0.1.2] — 2026-06-08

### Added

- **Per-file semantic tagging (Approach A).** Editable tag input per corpus file. Tags update live, training loop uses them.
- **Auto-tag from filename (Approach B).** `autoTagFromFilename()` matches filename tokens against vocabulary. Pre-populates on load.
- **Null filter in training loop** for removed files.

---

## [0.1.1] — 2026-06-08

### Fixed

- **WAV/BMP chunk seam artifacts via OLA.** Triangular-windowed overlap-add at 50% overlap eliminates click artifacts (WAV) and pixel row discontinuities (BMP). TXT/HTML unchanged.

---

## [0.1] — 2026-06-07

### Initial release

Browser-native DDPM diffusion model. Trains on user files (TXT, HTML, BMP, WAV). Generates new ones. ScoreNet ~664K params (Linear/LayerNorm/Adam from Simulacra, AdaLN from OmniVocal). Validation layer (magic bytes/Shannon/ROSA from Once Bytten). `.fabula` weight I/O. Mobile-responsive. Zero dependencies. Single HTML file. MIT License.
