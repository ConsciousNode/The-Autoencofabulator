/*
 * tests/generators.js — Seeded, copyright-clean synthetic test corpora + splits
 * Shared between The Autoencofabulator v0.3 (index.html) and the Node evaluation harness (tests/metrics.js).
 *
 * Includes:
 *   1. Deterministic PRNG (mulberry32)
 *   2. Templated HTML generator (nested lists, tables, forms with controlled nesting depth 2..5)
 *   3. Fixed-schema JSON generator ({"id", "kind", "active", "score", "tags"})
 *   4. Simple-scene BMP generator (1..3 filled circles or rectangles of palette colours on plain background, 32x32 & 64x64)
 *   5. Known-property WAV generator (single tones, two-note intervals, drum patterns at known tempos)
 *   6. SVG & C & JS synthetic generators for Tier 3 compiler-as-validator testing
 *   7. Sherlock Holmes 80/10/10 contiguous split helper
 */
(function (root) {
  'use strict';

  /* from AuraGraph app.html v0.3.1 (TILE-FORMAT v0.4), mulberry32() */
  function mulberry32(seed) {
    let a = (seed >>> 0) || 0xA02A0400;
    return function () {
      let t = (a += 0x6D2B79F5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function randInt(rng, lo, hi) {
    return lo + Math.floor(rng() * (hi - lo + 1));
  }

  function pick(rng, arr) {
    return arr[Math.floor(rng() * arr.length)];
  }

  // ── 1. Templated HTML Generator ──────────────────────────────────────────────
  // Generates valid HTML documents with nested lists, tables, and forms.
  // Controls and records exact maximum DOM nesting depth (depths 4..7).
  const HTML_WORDS = ['alpha', 'beta', 'gamma', 'delta', 'sigma', 'omega', 'pulse', 'node', 'flux', 'spark', 'loom', 'tile'];
  const HTML_TITLES = ['System Report', 'Sensor Matrix', 'Audio Telemetry', 'Corpus Index', 'Patch Catalog', 'Signal Log'];

  function generateHTMLSample(rng, opts = {}) {
    const targetDepth = opts.depth || randInt(rng, 4, 7); // exact total DOM nesting depth from <html>
    const variant = opts.variant || pick(rng, ['list', 'table', 'form', 'mixed']);
    const title = pick(rng, HTML_TITLES);

    // <html> (1) -> <body> (2) -> <section> (3) -> inner Chain of depth (targetDepth - 3)
    function makeDepthChain(remDepth) {
      if (remDepth <= 1) {
        const w1 = pick(rng, HTML_WORDS), w2 = pick(rng, HTML_WORDS);
        return `<span>${w1}-${randInt(rng, 10, 99)} ${w2}</span>`;
      }
      const mode = (variant === 'mixed') ? pick(rng, ['list', 'table', 'form', 'div']) : variant;
      if (mode === 'list' && remDepth >= 2) {
        const tag = rng() < 0.5 ? 'ul' : 'ol';
        const leaf = pick(rng, HTML_WORDS) + '-' + randInt(rng, 10, 99);
        const child = remDepth === 2 ? leaf : makeDepthChain(remDepth - 2);
        return `<${tag}><li>${child}</li><li>${leaf}</li></${tag}>`;
      } else if (mode === 'table' && remDepth >= 3) {
        const leaf = String(randInt(rng, 100, 999));
        const child = remDepth === 3 ? leaf : makeDepthChain(remDepth - 3);
        return `<table><tbody><tr><td>${pick(rng, HTML_WORDS)}</td><td>${child}</td></tr></tbody></table>`;
      } else if (mode === 'form' && remDepth >= 2) {
        const fname = pick(rng, HTML_WORDS);
        if (remDepth === 2) {
          return `<form action="#${fname}"><label>${fname}</label><input type="text" name="${fname}" value="${randInt(rng, 1, 99)}"></form>`;
        }
        return `<form action="#${fname}"><fieldset>${makeDepthChain(remDepth - 2)}</fieldset></form>`;
      } else {
        return `<div class="box">${makeDepthChain(remDepth - 1)}</div>`;
      }
    }

    const rem = Math.max(1, targetDepth - 3);
    const bodyInner = `<section><h1>${title}</h1>${makeDepthChain(rem)}</section>`;
    const html = `<!DOCTYPE html><html><head><meta charset="utf-8"><title>${title}</title></head><body>${bodyInner}</body></html>`;
    const bytes = new TextEncoder().encode(html);
    const measuredDepth = measureHTMLDepth(html);
    return {
      text: html,
      bytes,
      meta: { targetDepth: measuredDepth, measuredDepth, variant },
      tags: `html ${variant} minimal structured`
    };
  }

  function measureHTMLDepth(htmlStr) {
    const voidTags = new Set(['meta', 'input', 'br', 'hr', 'img', 'link', 'area', 'base', 'col', 'embed', 'param', 'source', 'track', 'wbr']);
    const tagRe = /<\/?([a-zA-Z][a-zA-Z0-9]*)\b[^>]*>/g;
    let depth = 0, maxDepth = 0, m;
    while ((m = tagRe.exec(htmlStr)) !== null) {
      const full = m[0], name = m[1].toLowerCase();
      if (full.startsWith('<!') || full.startsWith('<?')) continue;
      if (full.startsWith('</')) {
        depth = Math.max(0, depth - 1);
      } else if (!voidTags.has(name) && !full.endsWith('/>')) {
        depth++;
        if (depth > maxDepth) maxDepth = depth;
      }
    }
    return maxDepth;
  }

  // ── 2. Fixed-Schema JSON Generator ───────────────────────────────────────────
  // Schema: {"id": int(100..999), "kind": enum, "active": bool, "score": float, "tags": [str, str]}
  const JSON_KINDS = ['sensor', 'synth', 'patch', 'codec', 'filter'];
  const JSON_TAGS = ['fast', 'clean', 'warm', 'bright', 'deep', 'sharp', 'dense', 'calm'];

  function generateJSONSample(rng) {
    const id = randInt(rng, 100, 999);
    const kind = pick(rng, JSON_KINDS);
    const active = rng() >= 0.5;
    const score = +(rng() * 100).toFixed(2);
    const t1 = pick(rng, JSON_TAGS);
    let t2 = pick(rng, JSON_TAGS);
    if (t2 === t1) t2 = JSON_TAGS[(JSON_TAGS.indexOf(t1) + 1) % JSON_TAGS.length];
    const obj = { id, kind, active, score, tags: [t1, t2] };
    const text = JSON.stringify(obj, null, 2);
    const bytes = new TextEncoder().encode(text);
    return {
      text,
      bytes,
      obj,
      meta: { id, kind, active, score, tags: [t1, t2] },
      tags: `json ${kind} ${t1} ${t2} structured`
    };
  }

  function checkJSONSchema(strOrBytes) {
    const str = typeof strOrBytes === 'string' ? strOrBytes : new TextDecoder('utf-8', { fatal: false }).decode(strOrBytes);
    try {
      const o = JSON.parse(str);
      if (!o || typeof o !== 'object' || Array.isArray(o)) return { validParse: true, schemaOk: false, reason: 'not object' };
      const req = ['id', 'kind', 'active', 'score', 'tags'];
      for (const k of req) if (!(k in o)) return { validParse: true, schemaOk: false, reason: 'missing ' + k };
      if (typeof o.id !== 'number' || !Number.isInteger(o.id)) return { validParse: true, schemaOk: false, reason: 'id type' };
      if (typeof o.kind !== 'string' || !JSON_KINDS.includes(o.kind)) return { validParse: true, schemaOk: false, reason: 'kind enum' };
      if (typeof o.active !== 'boolean') return { validParse: true, schemaOk: false, reason: 'active type' };
      if (typeof o.score !== 'number' || !isFinite(o.score)) return { validParse: true, schemaOk: false, reason: 'score type' };
      if (!Array.isArray(o.tags) || o.tags.length < 1 || !o.tags.every(t => typeof t === 'string'))
        return { validParse: true, schemaOk: false, reason: 'tags array' };
      return { validParse: true, schemaOk: true, obj: o };
    } catch (e) {
      return { validParse: false, schemaOk: false, error: e.message };
    }
  }

  // ── 3. Simple-Scene BMP Generator ────────────────────────────────────────────
  // 1..3 non-overlapping filled circles or rectangles of distinct palette colours on a plain dark background.
  const BMP_BG = { name: 'navy', r: 16, g: 18, b: 32 };
  const BMP_PALETTE = [
    { name: 'red',    r: 230, g: 55,  b: 70  },
    { name: 'green',  r: 45,  g: 210, b: 120 },
    { name: 'blue',   r: 65,  g: 140, b: 245 },
    { name: 'yellow', r: 245, g: 205, b: 45  },
    { name: 'purple', r: 170, g: 85,  b: 240 },
    { name: 'orange', r: 245, g: 130, b: 40  }
  ];

  function buildBMPFile(width, height, rgbPixels) {
    const rowStride = Math.ceil((width * 3) / 4) * 4;
    const dataSize = rowStride * height;
    const fileSize = 54 + dataSize;
    const out = new Uint8Array(fileSize);
    const v = new DataView(out.buffer);
    out[0] = 0x42; out[1] = 0x4D; // 'BM'
    v.setUint32(2, fileSize, true);
    v.setUint32(10, 54, true);
    v.setUint32(14, 40, true);
    v.setInt32(18, width, true);
    v.setInt32(22, height, true);
    v.setUint16(26, 1, true);
    v.setUint16(28, 24, true);
    v.setUint32(34, dataSize, true);
    v.setInt32(38, 2835, true);
    v.setInt32(42, 2835, true);

    for (let y = 0; y < height; y++) {
      const bmpRow = height - 1 - y;
      const dstRow = 54 + bmpRow * rowStride;
      for (let x = 0; x < width; x++) {
        const src = (y * width + x) * 3;
        const dst = dstRow + x * 3;
        out[dst]     = rgbPixels[src + 2]; // B
        out[dst + 1] = rgbPixels[src + 1]; // G
        out[dst + 2] = rgbPixels[src];     // R
      }
    }
    return out;
  }

  function generateBMPSample(rng, opts = {}) {
    const size = opts.size || (rng() < 0.5 ? 32 : 64);
    const width = size, height = size;
    const nShapes = opts.nShapes || randInt(rng, 1, 3);
    const rgb = new Uint8Array(width * height * 3);
    for (let i = 0; i < width * height; i++) {
      rgb[i * 3]     = BMP_BG.r;
      rgb[i * 3 + 1] = BMP_BG.g;
      rgb[i * 3 + 2] = BMP_BG.b;
    }

    const shapes = [];
    const scale = size / 32;
    const slots = [
      { x0: Math.round(4 * scale),  y0: Math.round(4 * scale),  x1: Math.round(14 * scale), y1: Math.round(14 * scale) },
      { x0: Math.round(18 * scale), y0: Math.round(4 * scale),  x1: Math.round(28 * scale), y1: Math.round(15 * scale) },
      { x0: Math.round(8 * scale),  y0: Math.round(18 * scale), x1: Math.round(24 * scale), y1: Math.round(28 * scale) }
    ];

    for (let s = 0; s < nShapes; s++) {
      const slot = slots[s];
      const kind = rng() < 0.5 ? 'rect' : 'circle';
      const col = pick(rng, BMP_PALETTE);
      const w = randInt(rng, Math.max(4, Math.round(5 * scale)), Math.max(5, slot.x1 - slot.x0));
      const h = randInt(rng, Math.max(4, Math.round(5 * scale)), Math.max(5, slot.y1 - slot.y0));
      const x0 = slot.x0 + Math.floor((slot.x1 - slot.x0 - w) * rng());
      const y0 = slot.y0 + Math.floor((slot.y1 - slot.y0 - h) * rng());

      if (kind === 'rect') {
        for (let y = Math.max(1, y0); y < Math.min(height - 1, y0 + h); y++) {
          for (let x = Math.max(1, x0); x < Math.min(width - 1, x0 + w); x++) {
            const p = (y * width + x) * 3;
            rgb[p] = col.r; rgb[p + 1] = col.g; rgb[p + 2] = col.b;
          }
        }
      } else {
        const cx = x0 + w * 0.5, cy = y0 + h * 0.5, rad = Math.max(2.2, Math.min(w, h) * 0.5);
        for (let y = Math.max(1, Math.floor(cy - rad)); y <= Math.min(height - 2, Math.ceil(cy + rad)); y++) {
          for (let x = Math.max(1, Math.floor(cx - rad)); x <= Math.min(width - 2, Math.ceil(cx + rad)); x++) {
            if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= rad * rad) {
              const p = (y * width + x) * 3;
              rgb[p] = col.r; rgb[p + 1] = col.g; rgb[p + 2] = col.b;
            }
          }
        }
      }
      shapes.push({ kind, color: col.name, r: col.r, g: col.g, b: col.b, x0, y0, w, h });
    }

    const bytes = buildBMPFile(width, height, rgb);
    const colorNames = [...new Set(shapes.map(s => s.color))];
    return {
      bytes,
      rgb,
      width,
      height,
      meta: { width, height, nShapes, shapes, colors: colorNames },
      tags: `bmp geometric ${colorNames.join(' ')} ${shapes.map(s => s.kind).join(' ')}`
    };
  }

  function analyzeBMPScene(bmpBytes) {
    if (!bmpBytes || bmpBytes.length < 54 || bmpBytes[0] !== 0x42 || bmpBytes[1] !== 0x4D) {
      return { valid: false, nShapes: 0, colors: [], paletteMatchRatio: 0 };
    }
    const v = new DataView(bmpBytes.buffer, bmpBytes.byteOffset, bmpBytes.byteLength);
    const width = v.getInt32(18, true);
    const height = Math.abs(v.getInt32(22, true));
    const pixOff = v.getUint32(10, true);
    const rowStride = Math.ceil((width * 3) / 4) * 4;
    if (width <= 0 || height <= 0 || pixOff + rowStride * height > bmpBytes.length) {
      return { valid: false, nShapes: 0, colors: [], paletteMatchRatio: 0 };
    }

    const fgMask = new Uint8Array(width * height);
    const colorCounts = {};
    let fgPixels = 0, matchedPalettePixels = 0;

    for (let y = 0; y < height; y++) {
      const bmpRow = height - 1 - y;
      const rowBase = pixOff + bmpRow * rowStride;
      for (let x = 0; x < width; x++) {
        const B = bmpBytes[rowBase + x * 3];
        const G = bmpBytes[rowBase + x * 3 + 1];
        const R = bmpBytes[rowBase + x * 3 + 2];
        const dBg = Math.hypot(R - BMP_BG.r, G - BMP_BG.g, B - BMP_BG.b);
        if (dBg > 42) {
          fgMask[y * width + x] = 1;
          fgPixels++;
          let bestCol = null, bestDist = 1e9;
          for (const c of BMP_PALETTE) {
            const d = Math.hypot(R - c.r, G - c.g, B - c.b);
            if (d < bestDist) { bestDist = d; bestCol = c.name; }
          }
          if (bestDist < 65) {
            matchedPalettePixels++;
            colorCounts[bestCol] = (colorCounts[bestCol] || 0) + 1;
          }
        } else if (dBg < 35) {
          matchedPalettePixels++;
        }
      }
    }

    const visited = new Uint8Array(width * height);
    const minArea = Math.max(4, Math.round((width / 32) * (width / 32) * 5));
    let nShapes = 0;
    const queue = new Int32Array(width * height);

    for (let i = 0; i < width * height; i++) {
      if (!fgMask[i] || visited[i]) continue;
      let head = 0, tail = 0;
      queue[tail++] = i;
      visited[i] = 1;
      while (head < tail) {
        const cur = queue[head++];
        const cx = cur % width, cy = (cur / width) | 0;
        const neighbors = [
          cy > 0 ? cur - width : -1,
          cy + 1 < height ? cur + width : -1,
          cx > 0 ? cur - 1 : -1,
          cx + 1 < width ? cur + 1 : -1
        ];
        for (const nb of neighbors) {
          if (nb >= 0 && fgMask[nb] && !visited[nb]) {
            visited[nb] = 1;
            queue[tail++] = nb;
          }
        }
      }
      if (tail >= minArea) nShapes++;
    }

    const dominantColors = Object.entries(colorCounts)
      .filter(([, cnt]) => cnt >= minArea)
      .sort((a, b) => b[1] - a[1])
      .map(([name]) => name);

    return {
      valid: true,
      width,
      height,
      nShapes,
      colors: dominantColors,
      paletteMatchRatio: matchedPalettePixels / (width * height)
    };
  }

  // ── 4. Known-Property WAV Generator ──────────────────────────────────────────
  const WAV_PITCHES = [
    { name: 'C4',  freq: 261.63, midi: 60 },
    { name: 'E4',  freq: 329.63, midi: 64 },
    { name: 'G4',  freq: 392.00, midi: 67 },
    { name: 'A4',  freq: 440.00, midi: 69 },
    { name: 'C5',  freq: 523.25, midi: 72 },
    { name: 'E5',  freq: 659.25, midi: 76 }
  ];
  const WAV_TEMPOS = [96, 120, 144]; // BPM

  function pcmFloatToWavBytes(pcmFloat, sr = 8000) {
    const nSamp = pcmFloat.length;
    const dataSize = nSamp * 2;
    const out = new Uint8Array(44 + dataSize);
    const v = new DataView(out.buffer);
    const wStr = (off, s) => { for (let i = 0; i < s.length; i++) out[off + i] = s.charCodeAt(i); };
    wStr(0, 'RIFF'); v.setUint32(4, 36 + dataSize, true); wStr(8, 'WAVE'); wStr(12, 'fmt ');
    v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true);
    v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    wStr(36, 'data'); v.setUint32(40, dataSize, true);
    for (let i = 0; i < nSamp; i++) {
      const s = Math.max(-1, Math.min(1, pcmFloat[i]));
      v.setInt16(44 + i * 2, Math.round(s * 32767), true);
    }
    return out;
  }

  function generateWAVSample(rng, opts = {}) {
    const sr = opts.sr || 22050;
    const duration = opts.duration || 1.6;
    const nSamp = Math.round(sr * duration);
    const kind = opts.kind || pick(rng, ['tone', 'interval', 'drums']);
    const pcm = new Float32Array(nSamp);
    const p1 = opts.pitch1 || pick(rng, WAV_PITCHES);
    const p2 = opts.pitch2 || WAV_PITCHES[(WAV_PITCHES.indexOf(p1) + 2) % WAV_PITCHES.length];
    const bpm = opts.bpm || pick(rng, WAV_TEMPOS);
    const beatSec = 60 / bpm;

    if (kind === 'tone') {
      // Rhythmic plucked harmonic tone pulsed on every beat at known frequency p1.freq and tempo bpm
      for (let i = 0; i < nSamp; i++) {
        const t = i / sr;
        const posInBeat = t % beatSec;
        const env = Math.min(1, posInBeat / 0.003) * Math.exp(-posInBeat * 3.5);
        pcm[i] = 0.85 * env * (
          0.72 * Math.sin(2 * Math.PI * p1.freq * t) +
          0.20 * Math.sin(2 * Math.PI * p1.freq * 2 * t) +
          0.08 * Math.sin(2 * Math.PI * p1.freq * 3 * t)
        );
      }
    } else if (kind === 'interval') {
      const half = nSamp >> 1;
      for (let i = 0; i < nSamp; i++) {
        const t = i / sr;
        const f = i < half ? p1.freq : p2.freq;
        const posInBeat = t % beatSec;
        const env = Math.min(1, posInBeat / 0.003) * Math.exp(-posInBeat * 3.5);
        pcm[i] = 0.85 * env * (
          0.72 * Math.sin(2 * Math.PI * f * t) +
          0.20 * Math.sin(2 * Math.PI * f * 2 * t) +
          0.08 * Math.sin(2 * Math.PI * f * 3 * t)
        );
      }
    } else {
      // Pitched percussion / drum hits on every beat at known BPM and fundamental p1.freq
      for (let i = 0; i < nSamp; i++) {
        const t = i / sr;
        const stepIdx = Math.floor(t / beatSec);
        const dt = t - stepIdx * beatSec;
        const env = Math.min(1, dt / 0.002) * Math.exp(-dt * 12);
        const noise = Math.sin(i * 12.9898 + stepIdx * 78.233) * 43758.5453;
        const nz = (noise - Math.floor(noise)) * 2 - 1;
        pcm[i] = 0.85 * env * (
          0.75 * Math.sin(2 * Math.PI * p1.freq * t) +
          0.25 * nz * Math.exp(-dt * 25)
        );
      }
    }

    const bytes = pcmFloatToWavBytes(pcm, sr);
    return {
      bytes,
      pcm,
      sr,
      meta: {
        kind,
        sr,
        duration,
        freq: p1.freq,
        freq2: kind === 'interval' ? p2.freq : p1.freq,
        note: p1.name,
        midi: p1.midi,
        bpm
      },
      tags: `wav ${kind} ${p1.name.toLowerCase()} ${kind === 'drums' ? 'percussion rhythm' : 'melody harmonic'}`
    };
  }

  // First-peak normalized autocorrelation (YIN-style) pitch + onset-comb tempo + 10-90% rise time
  function analyzeWAVProperties(pcm, sr = 22050) {
    if (!pcm || pcm.length < 256) return { freq: 0, midi: 0, bpm: 0, riseMs: 0 };

    // 1. Dominant pitch via first-peak normalized autocorrelation (avoids subharmonic lag multiples)
    const win = Math.min(1024, pcm.length >> 1);
    const minLag = Math.max(4, Math.floor(sr / 900));
    const maxLag = Math.min(win - 4, Math.ceil(sr / 150));
    const offset = Math.min(Math.floor(sr * 0.015), Math.max(0, pcm.length - win - maxLag - 1));
    const corrs = new Float64Array(maxLag + 2);
    let e0 = 0;
    for (let i = 0; i < win; i++) e0 += pcm[offset + i] ** 2;

    let maxCorr = -1, bestLag = minLag;
    if (e0 > 1e-8) {
      for (let lag = minLag; lag <= maxLag + 1; lag++) {
        let c = 0, e1 = 0;
        for (let i = 0; i < win; i++) {
          const a = pcm[offset + i], b = pcm[offset + i + lag] || 0;
          c += a * b; e1 += b * b;
        }
        corrs[lag] = c / (Math.sqrt(e0 * Math.max(e1, 1e-12)) + 1e-12);
        if (lag <= maxLag && corrs[lag] > maxCorr) {
          maxCorr = corrs[lag];
          bestLag = lag;
        }
      }
      // Pick the first local peak that reaches >= 82% of the global maxCorr (eliminates 2*T0, 3*T0 subharmonics)
      const peakThresh = Math.max(0.25, maxCorr * 0.82);
      for (let lag = minLag + 1; lag <= maxLag; lag++) {
        if (corrs[lag] >= peakThresh && corrs[lag] >= corrs[lag - 1] && corrs[lag] >= corrs[lag + 1]) {
          // Parabolic interpolation around lag
          const y1 = corrs[lag - 1], y2 = corrs[lag], y3 = corrs[lag + 1];
          const denom = 2 * (2 * y2 - y1 - y3);
          const delta = Math.abs(denom) > 1e-9 ? (y1 - y3) / denom : 0;
          bestLag = lag + Math.max(-0.5, Math.min(0.5, delta));
          break;
        }
      }
    }
    const freq = bestLag > 0 ? sr / bestLag : 0;
    const midi = freq > 20 ? Math.round(69 + 12 * Math.log2(freq / 440)) : 0;

    // 2. Tempo via onset-envelope comb correlation
    const hop = Math.max(16, Math.round(sr * 0.005)); // 5 ms hop
    const nFrames = Math.floor(pcm.length / hop);
    const env = new Float64Array(nFrames);
    for (let f = 0; f < nFrames; f++) {
      let s = 0;
      for (let i = 0; i < hop; i++) s += pcm[f * hop + i] ** 2;
      env[f] = Math.sqrt(s / hop);
    }
    const odf = new Float64Array(nFrames);
    for (let f = 1; f < nFrames; f++) odf[f] = Math.max(0, env[f] - env[f - 1]);

    const envSr = sr / hop;
    let bestBpm = 120, bestBpmScore = -1;
    for (let candBpm = 84; candBpm <= 160; candBpm += 2) {
      const period = (60 / candBpm) * envSr;
      const lag1 = Math.round(period);
      const lag2 = Math.round(period * 2);
      let sc = 0, cnt = 0;
      for (let f = 0; f + lag1 < nFrames; f++) {
        sc += odf[f] * odf[f + lag1];
        if (f + lag2 < nFrames) sc += 0.5 * odf[f] * odf[f + lag2];
        cnt++;
      }
      if (cnt > 0 && sc / cnt > bestBpmScore) {
        bestBpmScore = sc / cnt;
        bestBpm = candBpm;
      }
    }

    // 3. Mean onset rise time (10% to 90% attack slope in ms, as in HELIX-1.md)
    const step = Math.max(8, Math.round(sr * 0.002)); // 2 ms step
    const winLen = step * 2;
    const M = Math.floor((pcm.length - winLen) / step);
    const rmsEnv = new Float64Array(Math.max(1, M));
    for (let m = 0; m < M; m++) {
      let s = 0;
      for (let i = 0; i < winLen; i++) s += pcm[m * step + i] ** 2;
      rmsEnv[m] = Math.sqrt(s / winLen);
    }
    const diff = new Float64Array(Math.max(1, M));
    for (let m = 1; m < M; m++) diff[m] = Math.max(0, rmsEnv[m] - rmsEnv[m - 1]);
    const sortedDiff = Float64Array.from(diff).sort();
    const thresh = sortedDiff[Math.floor(sortedDiff.length * 0.90)] || 1e-4;
    const rises = [];
    for (let m = 2; m < M - 6; m++) {
      if (diff[m] > thresh && diff[m] >= diff[m - 1] && diff[m] >= diff[m + 1]) {
        let pkIdx = m, pkVal = rmsEnv[m];
        for (let j = m; j < Math.min(M, m + 20); j++) {
          if (rmsEnv[j] > pkVal) { pkVal = rmsEnv[j]; pkIdx = j; }
          else if (rmsEnv[j] < pkVal * 0.85) break;
        }
        let vlIdx = m, vlVal = rmsEnv[m];
        for (let j = m; j >= Math.max(0, m - 15); j--) {
          if (rmsEnv[j] <= vlVal) { vlVal = rmsEnv[j]; vlIdx = j; }
          else if (rmsEnv[j] > vlVal * 1.2) break;
        }
        const amp = pkVal - vlVal;
        if (amp > thresh * 0.6 && pkIdx > vlIdx) {
          const v10 = vlVal + 0.1 * amp, v90 = vlVal + 0.9 * amp;
          let t10 = vlIdx, t90 = pkIdx;
          for (let j = vlIdx; j <= pkIdx; j++) {
            if (rmsEnv[j] >= v10) {
              const prev = rmsEnv[Math.max(0, j - 1)];
              t10 = j - 1 + (v10 - prev) / Math.max(1e-9, rmsEnv[j] - prev);
              break;
            }
          }
          for (let j = vlIdx; j <= pkIdx; j++) {
            if (rmsEnv[j] >= v90) {
              const prev = rmsEnv[Math.max(0, j - 1)];
              t90 = j - 1 + (v90 - prev) / Math.max(1e-9, rmsEnv[j] - prev);
              break;
            }
          }
          rises.push(Math.max(0.4, ((t90 - t10) * step / sr) * 1000));
          m = pkIdx + 2;
        }
      }
    }
    const riseMs = rises.length ? rises.reduce((a, b) => a + b, 0) / rises.length : 12.0;

    return { freq, midi, bpm: bestBpm, riseMs };
  }

  // ── 5. SVG, JS, and C Source Generators (Tier 3 Compiler-as-Validator) ──────
  function generateSVGSample(rng) {
    const col = pick(rng, BMP_PALETTE);
    const cx = randInt(rng, 20, 80), cy = randInt(rng, 20, 80), r = randInt(rng, 10, 30);
    const text = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect width="100" height="100" fill="#101220"/><circle cx="${cx}" cy="${cy}" r="${r}" fill="rgb(${col.r},${col.g},${col.b})"/></svg>`;
    return {
      text,
      bytes: new TextEncoder().encode(text),
      meta: { color: col.name, cx, cy, r },
      tags: `svg ${col.name} geometric art`
    };
  }

  function generateJSSample(rng) {
    const fnName = pick(rng, ['computeGain', 'stepFilter', 'mixChannel', 'scalePatch']);
    const factor = (0.2 + rng() * 1.6).toFixed(2);
    const bias = randInt(rng, -10, 10);
    const text = `function ${fnName}(x) {\n  const scale = ${factor};\n  const offset = ${bias};\n  return x * scale + offset;\n}\n`;
    return {
      text,
      bytes: new TextEncoder().encode(text),
      meta: { fnName, factor: +factor, bias },
      tags: `js code function minimal`
    };
  }

  function generateCSample(rng) {
    const fnName = pick(rng, ['clamp_byte', 'mix_step', 'hash_val', 'scale_int']);
    const mult = randInt(rng, 2, 15);
    const add = randInt(rng, 1, 63);
    const text = `int ${fnName}(int x) {\n  int y = x * ${mult} + ${add};\n  return y & 255;\n}\n`;
    return {
      text,
      bytes: new TextEncoder().encode(text),
      meta: { fnName, mult, add },
      tags: `c code minimal`
    };
  }

  // ── 6. Sherlock Holmes 80 / 10 / 10 Contiguous Split ─────────────────────────
  function splitSherlockCorpus(rawBytes) {
    const u8 = rawBytes instanceof Uint8Array ? rawBytes : new TextEncoder().encode(String(rawBytes));
    const N = u8.length;
    const nTrain = Math.floor(N * 0.80);
    const nVal = Math.floor(N * 0.10);
    return {
      totalBytes: N,
      train: u8.subarray(0, nTrain),
      val: u8.subarray(nTrain, nTrain + nVal),
      test: u8.subarray(nTrain + nVal)
    };
  }

  function buildSeededSuite(seed = 42, countPerFormat = 16) {
    const rng = mulberry32(seed);
    const html = [], json = [], bmp = [], wav = [], svg = [], js = [], c = [];
    for (let i = 0; i < countPerFormat; i++) {
      html.push(generateHTMLSample(rng, { depth: 4 + (i % 4) }));
      json.push(generateJSONSample(rng));
      bmp.push(generateBMPSample(rng, { size: i % 2 === 0 ? 32 : 64, nShapes: 1 + (i % 3) }));
      wav.push(generateWAVSample(rng, {
        kind: ['tone', 'interval', 'drums'][i % 3],
        pitch1: WAV_PITCHES[i % WAV_PITCHES.length],
        bpm: WAV_TEMPOS[i % WAV_TEMPOS.length]
      }));
      svg.push(generateSVGSample(rng));
      js.push(generateJSSample(rng));
      c.push(generateCSample(rng));
    }
    return { html, json, bmp, wav, svg, js, c };
  }

  const Generators = {
    mulberry32,
    HTML_WORDS,
    JSON_KINDS,
    JSON_TAGS,
    BMP_BG,
    BMP_PALETTE,
    WAV_PITCHES,
    WAV_TEMPOS,
    generateHTMLSample,
    measureHTMLDepth,
    generateJSONSample,
    checkJSONSchema,
    generateBMPSample,
    analyzeBMPScene,
    buildBMPFile,
    generateWAVSample,
    analyzeWAVProperties,
    pcmFloatToWavBytes,
    generateSVGSample,
    generateJSSample,
    generateCSample,
    splitSherlockCorpus,
    buildSeededSuite
  };

  if (typeof module === 'object' && module.exports) module.exports = Generators;
  else root.Generators = Generators;
})(typeof globalThis !== 'undefined' ? globalThis : this);