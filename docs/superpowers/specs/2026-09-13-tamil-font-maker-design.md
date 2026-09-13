# Tamil Font Maker (`tamil-font-maker`) — Design Spec

- **Date:** 2026-09-13
- **Status:** Draft for review
- **Location:** `workspace/default/tamil-font-maker` (own git repo, sibling to `storage-optimizer`, `wordsearch-llm`)

## 1. Goal

A ZCode plugin + Python pipeline that produces a usable Tamil TrueType font from reference imagery, **fully agentically**: the agent runs every stage, reads every JSON report, visually inspects every QA artifact, and iterates parameters until the font passes its gates. Human effort is limited to supplying source material and redrawing cells the agent flags.

Business framing: **any image → TTF**. Cinema posters, signboards, book pages, screenshots, handwriting scans are all just *sample images*; there are no image-type-specific code paths. Structured input (template sheets, drawn digitally or on paper) is the other supported source. A single font project may mix all sources freely.

## 2. Non-goals (v1)

- Multi-weight or variable fonts — one style per project.
- Full Indic OpenType shaping: conjuncts (க்ஷ…), reph/rakar forms, GPOS mark positioning — phase 2+.
- Compose mode (synthesizing uyirmei from base+matra components) — phase 2. V1 uses drawn/precomposed cells only; partial fonts degrade gracefully instead (§9).
- GUI font editor.
- Rights clearance for extracted lettering: extracted movie/poster lettering is typically someone's artwork; personal/experimental use is the user's call, commercial redistribution is a legal question outside the pipeline. Documented in the skill, not enforced in code.

## 3. Deliverables

1. `pipeline/` — Python package, one CLI with per-stage subcommands.
2. ZCode plugin — `plugin.json` + `skills/tamil-font-maker/SKILL.md` defining the agent orchestration protocol (§8).
3. Bootstrap validation suite (`test/`) proving the entire pipeline end-to-end with **zero human-drawn art** (§10).
4. This spec + an implementation plan.

## 4. Glyph set — single source of truth: `pipeline/mapping.py`

Every cell: glyph id, codepoint sequence, category, sheet placement, GSUB rule (if any). All later stages derive from this table; nothing else hardcodes Tamil.

### Core (260 cells)

| Category | Count | Codepoints | Mapped by |
|---|---|---|---|
| Uyir (vowels) | 12 | அ ஆ இ ஈ உ ஊ எ ஏ ஐ ஒ ஓ ஔ (U+0B85–0B8A, 0B8E–0B90, 0B92–0B94) | direct cmap |
| Ayutham | 1 | ஃ (U+0B83) | direct cmap |
| Consonants (bare) | 18 | க ங ச ஞ ட ண த ந ப ம ய ர ல வ ழ ள ற ன (U+0B95…) | direct cmap |
| Signs (matras + pulli) | 13 | ா ி ீ ு ூ ெ ே ை ொ ோ ௌ ௗ (12) + pulli ் (U+0BCD) | direct cmap |
| Pulli forms | 18 | க் ங் ச் … (consonant + ் drawn precomposed) | GSUB: (C, ்) → glyph |
| Uyirmei | 198 | consonant × 11 vowel-sign combinations (ஆ–ஔ), drawn precomposed; the அ column is the bare consonant itself, not a separate glyph | GSUB ligatures (§9) |

### Extras (36 cells)

| Category | Count | Codepoints | Notes |
|---|---|---|---|
| Tamil numerals | 10 | ௦–௯ (U+0BE6–0BEF) | direct cmap |
| Grantha (bare) | 4 | ஜ ஷ ஸ ஹ (U+0B9C, 0BB7, 0BB8, 0BB9) | pulli forms & grantha uyirmei are phase 2 |
| Latin digits | 10 | 0–9 (U+0030–0039) | direct cmap |
| Punctuation | 12 | `. , ; : ! ? - ' " ( ) /` | direct cmap |

Non-drawn: space (U+0020, advance-only), `.notdef`.

**Total: 296 drawn cells.** (The folk number "247" counts characters, not cells; proper decomposition adds bare consonants and pulli forms, while the inherent-அ column collapses into the consonants.)

## 5. Template specification

Generated at 300 dpi as PNG (digital mode) and print-ready PDF (paper mode), A4 portrait. 12 sheets:

| Sheets | Content | Grid |
|---|---|---|
| S1 | uyir + ஃ (13) | loose grid |
| S2 | consonants (18) | 5×4 |
| S3 | signs (13) | loose grid |
| S4 | pulli forms (18) | 5×4 |
| S5–S10 | uyirmei, 198 cells (3 consonant-groups × 2 vowel-groups: three 6×6 + three 6×5) | 6 cols × 6 or 5 rows |
| S11 | Tamil numerals + grantha (14) | loose grid |
| S12 | digits + punctuation (22) | 5×5 |

Cell geometry ≈ 400×520 px (≈34×44 mm at print size). Each cell carries:

- light border + codepoint/glyph label (under cell, outside drawing area),
- **faint baseline and headline guide lines** — the primary consistency lever: trace stage derives scale and vertical placement from them,
- for sign cells: an attachment tick marking where the sign visually joins a consonant,
- four **corner fiducial markers** per sheet (paper mode deskew).

**Completion mode:** given a coverage report (§9), generate sheets containing only missing cells — used to finish fonts that started from sample-image extraction.

## 6. Stage pipeline

Each stage is a CLI subcommand, idempotent, resumable, and emits a JSON report under `reports/`. Project state:

```
projects/<name>/
  config.toml      # family name, em size, presets, per-cell overrides
  samples/         # user-provided arbitrary images (any content)
  sheets/          # generated templates; user-filled sheets dropped back here
  glyphs/          # <glyph-id>.png + manifest.json (provenance per glyph)
  outlines/        # <glyph-id>.json — quadratic paths, metrics
  reports/         # stage JSON reports
  qa/              # rendered contact sheets, charts, comparisons
  out/<family>.ttf
```

### 6.1 `template`
Generates sheets (full or completion) per §5 into `sheets/`.

### 6.2 `ingest` — three modes, one output

All modes produce the same artifact: binarized, cropped `glyphs/<glyph-id>.png` (320×416 working raster inside the guides) + manifest entries with provenance (source file, mode, cell, preset, timestamp).

- **digital:** user fills template PNG in any image editor, drops it back in `sheets/`. Canvas is assumed unmodified → cell boxes are known constants; validate dimensions, crop, binarize (Otsu). No computer vision needed.
- **paper:** user prints, draws, photographs/scans. Detect 4 corner fiducials (numpy — dark-square detection), order corners, compute perspective warp (hand-rolled 3×3 homography + bilinear resample), segment grid by known geometry, adaptive-threshold, despeckle.
- **extract (agent-in-loop, general images):** input is any images in `samples/`. Division of labor:
  - **Agent (vision):** locates text regions, proposes per-glyph crops, **classifies each crop** (reads the Tamil letter), decides extraction viability (occluded by artwork, fused with background → flag unextractable), picks cleanup preset per glyph.
  - **Stage (deterministic):** crop → cleanup preset (`faithful`: preserve integral decorative detail in the filled outline; `clean`: aggressive simplification toward plain stroke) → binarize, despeckle, hole-fill, size-normalize → write glyph + provenance.
  - Target: reasonably isolated, separable lettering. Dense stylized paragraphs or letters welded into complex art are the honest edge — those cells get flagged, not forced.

Cell-level re-ingest is supported in all modes: redo one cell without touching the rest (needed by the redraw loop, §8).

### 6.3 `trace`
Binary glyph raster → vector outline via `potracer` (pure-Python potrace) → cubic-to-quadratic via `cu2qu` → placement in the 2048-unit em box: scale from headline/baseline distance, vertical position from baseline; sidebearings from outline bounds with a minimum floor. Sign cells additionally record attachment offsets. Output: `outlines/<glyph-id>.json`. Per-cell parameter overrides (threshold, smoothing, despeckle level) read from `config.toml`.

### 6.4 `build`
fontTools TTF assembly: glyf (quadratic outlines), cmap (direct-mapped glyphs; **matras get their own advance widths** — this is what makes partial fonts degrade legibly, §9), hmtx, vertical metrics derived from guide geometry, OS/2, head, hhea, name table from `config.toml`, and generated GSUB:

- Script/lang: `taml` / dflt.
- Feature: `liga` (registration validated empirically in bootstrap B1; if an engine path needs `ccmp`, lookups are additionally registered there — decision procedure, not a guess).
- Ligature rules, generated from mapping.py: per consonant — 8 simple matra rules (ா ி ீ ு ூ ெ ே ை) + 6 rules covering the split signs in composed and NFC-decomposed form: (C,ொ)+(C,ெ,ா), (C,ோ)+(C,ே,ா), (C,ௌ)+(C,ெ,ௗ) = 14 uyirmei rules, plus 1 × (C, ்) pulli rule = 15 × 18 = **270 rules total**. (Decomposed forms matter because text may arrive as க+ெ+ா instead of க+ொ; both must ligate to the same precomposed cell. C + independent-vowel sequences like க+ஔ are intentionally not ligated — the independent vowel never follows a consonant in canonical Tamil, and real fonts render such input as two glyphs.)
- **Rules are emitted only for glyphs that exist** in `outlines/`.
- WOFF2 if `brotli` is importable, skipped otherwise.

### 6.5 `verify`
Two gate classes; all results into a report + human/agent-viewable PNG artifacts in `qa/`.

Programmatic gates:
- every cell in the project's target set is non-empty in `glyphs/`;
- `hb-shape` identity tests: e.g. கி shapes to exactly one precomposed glyph id; க் shapes to its pulli-form glyph; NFC-decomposed கௌ shapes identically to precomposed input;
- no outline coordinates outside the em box;
- every test string renders with zero `.notdef` references for covered text.

Visual gates (rendered via uharfbuzz + Pillow at multiple sizes):
- full uyirmei chart (198 cells),
- pulli-form chart, numerals/digits/punctuation line,
- sample Tamil sentences (real prose),
- **side-by-side contact sheets**: rendered output vs source glyph crops, per glyph — the core comparison artifact the agent inspects.

Coverage report: filled/missing cells against the §4 target set, overall %, missing list. A build is **green** when: all programmatic gates pass for covered glyphs, visual gates have no open flags, and the coverage % is reported (coverage below 100 % is a warning, not a failure — §9).

## 7. Iteration and redraw protocol

When verify flags a cell (clipped, drifted, mistraced, colliding):

1. Agent tries per-cell parameter overrides (threshold, smoothing, despeckle) → re-ingest cell → re-trace → re-verify cell.
2. If parameters can't fix it (bad source art), the cell is flagged **needs-redraw** with the offending crop + a marked-up template cell appended to a redraw packet (a small sheet containing only flagged cells).
3. User redraws; cell-level re-ingest picks it up; loop repeats until green or user accepts remaining warnings.

## 8. The agent protocol (SKILL.md contract)

The skill instructs the agent to: run stage → read its JSON report → Read the QA PNGs (vision) → decide (override params / flag redraw / proceed) → repeat. The agent owns every judgment call: cleanup presets, classification of extracted glyphs, pass/fail on visual gates, convergence. Convergence: programmatic gates are mandatory; visual flags require either a fix or an explicit recorded waiver with reason. The skill also carries the usage workflow (init → template/extract → ingest → trace → build → verify → iterate) and the rights caveat from §2.

## 9. Partial coverage & degradation

Fonts from sample extraction typically cover few cells. Degradation ladder:

1. **Covered precomposed combos** → GSUB ligature → drawn glyph (best).
2. **Uncovered combos** → matra renders as its own spacing glyph after the consonant (legacy-style, legible, not pretty). Requires no extra work beyond matra advances (§6.4).
3. Missing cells render `.notdef` — visible as boxes, which is honest.

Every build report states coverage %. `template --completion` emits sheets for exactly the missing cells, enabling the standard hybrid workflow: extract from samples → complete by drawing in the same style.

## 10. Bootstrap validation (zero human art)

Fixture: Noto Sans Tamil (OFL; downloaded once into `test/fixtures/`, license file included). Its metrics also seed the guide-line geometry defaults (§5), then frozen.

- **B1 digital roundtrip:** rasterize known Tamil text into our template PNGs → digital ingest → trace → build → verify. Gates: per-glyph pixel tolerance vs source rendering; shaping identity tests; the `liga`-vs-`ccmp` registration decision is made here empirically.
- **B2 paper distortion:** apply synthetic perspective, rotation, noise, uneven lighting to B1 sheets → paper ingest → same gates (validates fiducials, homography, thresholding).
- **B3 extract:** render sample text onto synthetic backgrounds (gradients, noise, mild occlusion, non-white paper tones) → agent-driven extract → gates: classification accuracy = 100 % on the fixture set (ground truth known), glyph IoU/pixel fidelity within tolerance, unextractable cases correctly flagged.

All three green = the pipeline is proven fully agentically; real user art then rides validated rails. B1–B3 run as pytest integration tests (marked slow).

## 11. Dependencies & environment

- System Python 3.14; project venv. Already installed: fonttools 4.61.1, Pillow 12, numpy 2.3.
- pip: `uharfbuzz` (abi3 wheels), `potracer` (pure Python), `brotli` (optional).
- **No opencv** — deliberate: fiducial detection + perspective warp are ~100 lines of numpy, fully unit-tested, and sidestep cp314 wheel risk.
- Risks: cp314 wheels for uharfbuzz/brotli (both ship abi3; if uharfbuzz fails, shaping gates fall back to a system `hb-shape` CLI, else the gate is marked blocked rather than silently skipped). potracer speed on ~296 small rasters is negligible.

## 12. Testing strategy

pytest unit tests: mapping-table completeness (counts and codepoints vs §4 tables — the table is literally checked against the spec), homography math, fiducial detector on fixtures, cell extraction, tracer on synthetic shapes (circle → known area tolerance), GSUB rule generation (270 rules, including decomposed-form variants for ொ ோ ௌ), build smoke (TTF parses, tables present). Integration: B1–B3.

## 13. Project layout

```
tamil-font-maker/
  plugin.json
  skills/tamil-font-maker/SKILL.md
  pipeline/            # mapping.py, template.py, ingest.py, trace.py, build.py, verify.py, cli.py
  test/                # fixtures/, unit/, integration B1–B3
  projects/            # user data — gitignored
  docs/superpowers/specs/
  README.md
```

## 14. Success criteria

1. Bootstrap B1–B3 green, driven entirely by the agent, no human-drawn input.
2. A real template-set font (user-drawn) builds, passes all gates, installs on KDE, and types correct Tamil in HarfBuzz applications.
3. Arbitrary sample image(s) → coverage report + partial TTF that renders covered text correctly and degrades legibly; completion template generated for the gaps.

## 15. Open items (defaults chosen; override at review)

- Guide-line ratios (baseline/headline): seeded from Noto Sans Tamil in B1, then frozen.
- Plugin name `tamil-font-maker`; font family names come from project config at build time.
- Grantha pulli forms, ஶ (U+0BB6), grantha uyirmei: phase 2.
