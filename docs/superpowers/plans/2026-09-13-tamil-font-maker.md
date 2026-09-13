# Tamil Font Maker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `tamil-font-maker` ZCode plugin + Python pipeline that turns reference imagery (template sheets or arbitrary images) into a valid Tamil TTF, verified by the B1–B3 bootstrap suite with zero human-drawn art.

**Architecture:** Deterministic stage pipeline (template → ingest → trace → build → verify) behind one CLI (`python -m pipeline <stage>`), all state per-project; `mapping.py` is the single source of truth for the 314-cell glyph set and 306 GSUB rules. A shared rasterizer (uharfbuzz shaping + fontTools outline → XOR-filled polygons) powers fixtures, verify renders, and pixel-diff gates. The plugin skill makes the agent the orchestrator/QA driver.

**Tech Stack:** Python 3.14 (venv), fonttools 4.61 (+ feaLib, otlLib, cu2qu), Pillow, numpy, potracer, uharfbuzz, brotli (optional). No opencv — fiducials + homography in numpy.

**Spec:** `docs/superpowers/specs/2026-09-13-tamil-font-maker-design.md`

## Global Constraints

- Python 3.14 venv at `.venv/`; deps: `fonttools uharfbuzz potracer brotli` (+ pytest). No opencv, no fontforge, no system packages.
- UPM 2048; guides: baseline→0, headline→1400 units; cell 400×520 px; working raster 320×416.
- Glyph naming: `g_<hex>[_<hex>…]` e.g. `g_u0B95_u0BBF`; FEA-safe (letters/digits/underscore only). Not `u0B95` alone (add `g_` prefix to avoid reserved-name ambiguity).
- GSUB: script `taml`, feature `liga`, 306 rules generated from mapping (spec §6.4 as corrected: 16 uyirmei + 1 pulli per consonant × 18). Rules emitted only for glyphs present.
- Sheet fiducials: 60 px black squares, 20 px white ring, centers 100 px from each corner.
- Every stage: CLI subcommand, idempotent, JSON report to `projects/<name>/reports/`.
- Reports and tests must never contain credentials (none needed).
- Commit after every task, `feat:`/`test:`/`chore:` prefixes.

---

### Task 1: Venv + dependencies

**Files:** Create `.venv/` (gitignored), `.gitignore`.

- [ ] `python3 -m venv .venv && .venv/bin/pip install fonttools uharfbuzz potracer brotli pytest`
- [ ] Verify: `.venv/bin/python -c "import uharfbuzz, potracer, fontTools, brotli; print('ok')"` — if uharfbuzz/brotli lack cp314 wheels, retry `--pre`, else record fallback (system `hb-shape` CLI / skip woff2) in `.venv-setup-report.json` at repo root.
- [ ] `.gitignore`: `.venv/`, `projects/`, `test/fixtures/*.ttf`, `__pycache__/`, `*.pyc`
- [ ] Commit.

### Task 2: `pipeline/mapping.py` — glyph table + rules

**Files:** Create `pipeline/__init__.py`, `pipeline/mapping.py`. Test: `test/test_mapping.py`.

**Produces (exact interface):**
```python
UPM = 2048
@dataclass(frozen=True)
class Cell:
    gid: str            # "g_u0B85", "g_u0B95_u0BBF"
    cps: tuple[int, ...]  # cmap codepoint(s); ligature cells: sequence
    kind: str           # uyir|ayutham|consonant|sign|pulli_form|uyirmei|numeral|grantha|digit|punct
    sheet: str; row: int; col: int
    cmap_cp: int | None # direct cmap mapping (None for pulli_form/uyirmei)
CELLS: tuple[Cell, ...]            # exactly 314
def cell(gid) -> Cell
def ligature_rules() -> list[tuple[tuple[int,...], str]]  # 306 (seq -> gid), sorted longest-first
SAMPLE_TEXTS: list[str]            # test sentences for verify
```
Data: UYIR 12 (அ0B85 ஆ0B86 இ0B87 ஈ0B88 உ0B89 ஊ0B8A எ0B8E ஏ0B8F ஐ0B90 ஒ0B92 ஓ0B93 ஔ0B94); CONSONANTS 18 (க0B95 ங0B99 ச0B9A ஞ0B9E ட0B9F ண0BA3 த0BA4 ந0BA8 ப0BAA ம0BAE ய0BAF ர0BB0 ல0BB2 வ0BB5 ழ0BB4 ள0BB3 ற0BB1 ன0BA9); SIGNS: ா0BBE ி0BBF ீ0BC0 ு0BC1 ூ0BC2 ெ0BC6 ே0BC7 ை0BC8 ொ0BCA ோ0BCB ௌ0BCC ௗ0BD7 + pulli ்0BCD; VOWEL→SIGN map: ஆ→[ா] இ→[ி] ஈ→[ீ] உ→[ு] ஊ→[ூ] எ→[ெ] ஏ→[ே] ஐ→[ை] ஒ→[ொ | ெ,ா] ஓ→[ோ | ே,ா] ஔ→[ௌ | ஒ,ௗ] (அ→bare consonant, no cell). Rules per consonant: each vowel-sign combo (composed and decomposed) + (C, ்)→pulli_form. Sheets: S1 uyir+ஃ, S2 consonants, S3 signs, S4 pulli forms, S5–S10 uyirmei (3 consonant-groups × 2 vowel-groups, 6×6 grids), S11 numerals+grantha, S12 digits+punct.

- [ ] Write failing tests: `len(CELLS)==314`; per-kind counts (uyir 12, ayutham 1, consonant 18, sign 13, pulli_form 18, uyirmei 216, numeral 10, grantha 4, digit 10, punct 12); all cps in Tamil/ASCII ranges; `ligature_rules()` == 306, every target gid exists, every sequence non-empty, longest-first order; every codepoint of sample texts resolves via cmap or a full ligature prefix. Run → fail (module missing).
- [ ] Implement `mapping.py`. Run tests → pass. Commit.

### Task 3: `pipeline/gfx.py` — shared rasterizer + image utils

**Files:** Create `pipeline/gfx.py`. Test: `test/test_gfx.py`.

**Produces:**
```python
def shape_text(font_bytes: bytes, text: str) -> list[(gid: int, dx: int, dy: int)]
def rasterize_glyphs(ttfont: fontTools TTFont, shaped, px: int, canvas=(w,h), origin=(x,y)) -> Image  # XOR even-odd fill, quadratic flattening tol 0.5px
def render_text(font_path, text, px) -> Image   # shape+rasterize convenience
def otsu(gray: Image) -> int
def otsu_binarize(img: Image) -> Image          # ink=255 on black
def despeckle(binary: Image, min_area: int) -> Image     # connected-component (4-neigh, numpy BFS)
def fit_to_box(binary, box_wh, anchor_baseline_frac: float|None) -> Image  # scale+crop ink bbox into box, optional baseline alignment
```
- [ ] Tests: render_text on a tiny 2-glyph fixture font built inline via fontTools (a filled square + a square-with-square-hole) → pixel counts within tolerance, hole is white (even-odd works); otsu on bimodal histogram finds the valley; despeckle removes a 1-px dot, keeps a 10×10 blob. Run → pass. Commit.

### Task 4: fixture font

**Files:** `test/fixtures/` (font committed? spec says gitignore ttf; instead commit a *generator*): `test/make_fixture.py`.

- [ ] Try download Noto Sans Tamil TTF (raw.githubusercontent.com/google/fonts/main/ofl/notosanstamil/NotoSansTamil%5Bwdth,wght%5D.ttf) via curl; if network blocked, fall back to `test/make_fixture.py` building a synthetic Tamil fixture with fonttools (simple geometric strokes per codepoint — deterministic, OFL-free). Either way, run `test/make_fixture.py --out test/fixtures/ref.ttf --license test/fixtures/LICENSE.txt` and record provenance in `test/fixtures/PROVENANCE.md`. Commit generator + provenance (not the ttf).
- [ ] Smoke: render_text(ref.ttf, "தமிழ்", 96) has ink > 3% of canvas. Commit.

### Task 5: `pipeline/template.py`

**Files:** Create `pipeline/template.py`, `pipeline/cli.py` (subcommand dispatch). Test: `test/test_template.py`.

**Produces:** `generate(sheets_dir: Path, only: set[str] | None = None) -> TemplateReport` — writes `S1.png`…`S12.png` (2480×3508) + `S{n}.pdf` (Pillow multi-page-capable single-page saves), returns counts. Cell layout per spec §5; guides: headline y=100, baseline y=420 within cell (light gray 1px full cell width), label + codepoint under cell; fiducials per global constraints. Completion mode: `only` = set of gids → sheets re-flowed into minimal cell grids, filename prefix `C`.
- [ ] Failing test first: S5 exists, is 2480×3508, pixel at each fiducial center is black, pixel just outside ring is white; cell (row,col) of S1 contains label text region (dark pixels below cell box); guides present (gray rows); completion template with only={2 uyirmei gids} produces 1 sheet with 2 labeled cells. Run → fail → implement → pass. Commit.

### Task 6: `pipeline/ingest.py` — digital + paper + extract

**Files:** Create `pipeline/ingest.py`, `pipeline/homography.py`. Tests: `test/test_homography.py`, `test/test_ingest.py`.

**Produces:**
```python
# homography.py
def solve_homography(src4, dst4) -> np.ndarray  # 3x3, DLT, h33=1
def warp_perspective(img, H, out_wh) -> Image   # bilinear, numpy
def find_fiducials(img) -> list[(x,y)]          # 4 corners ordered TL,TR,BR,BL; raises IngestError if !=4
# ingest.py
def ingest_digital(project: Path, sheet_path: Path) -> IngestReport   # dims match template → crop cells → otsu → glyphs/
def ingest_paper(project: Path, sheet_path: Path) -> IngestReport     # fiducials → warp → same
def ingest_extract(project: Path, crops: dict[gid, Path], preset: str) -> IngestReport  # faithful|clean cleanup per spec §6.2
def ingest_cells(project: Path, cells: dict[gid, Path]) -> IngestReport # single-cell re-ingest, preserves manifest provenance history
```
Manifest: `glyphs/manifest.json` = {gid: {source, mode, preset, ts, history[]}}. Reports include per-cell warnings (empty, low ink <0.5%, overflow). Warnings never crash.
- [ ] Homography tests: identity; known projective map on synthetic points recovers to <0.5 px; warp of a rect grid lands labels in right cells. Fiducial test: template PNG → find_fiducials returns the 4 drawn centers. Run → pass. Commit.
- [ ] Ingest tests: fill 3 cells of a digital S1 programmatically → ingest_digital → 3 glyphs binarized with expected ink bboxes; same sheet printed-then-warped (apply random H + noise via warp) → ingest_paper recovers cells within IoU > 0.8 vs digital; re-ingest one cell updates only it + appends history. Run → pass. Commit.

### Task 7: `pipeline/trace.py`

**Files:** Create `pipeline/trace.py`. Test: `test/test_trace.py`.

**Produces:** `trace(project: Path) -> TraceReport` — for each glyph PNG: potracer → paths → cu2qu (cubic→quadratic, max err 1.0 em-px) → em placement (scale = 1400/(baseline−headline px) = 1400/320; y-flip; x centered by ink bbox with lsb ≥ 40) → `outlines/<gid>.json` = {contours: [[(x,y,on), …]], xmin, xmax, advance_hint}. Per-cell overrides from `config.toml [cells.<gid>]` (threshold, despeckle, smooth).
- [ ] Tests: circle raster → traced contour ≈ circle (area within 8%, quadratic-only points); square stays square (corner count); baseline alignment: glyph ink bottom sits at y≈0±10 units. Run → pass. Commit.

### Task 8: `pipeline/build.py`

**Files:** Create `pipeline/build.py`. Test: `test/test_build.py`.

**Produces:** `build(project: Path) -> BuildReport` — fontTools TTFont from outlines: glyf via TTGlyphPen replay; cmap format 4 (direct cells; **signs get positive advances** — degradation mode); hmtx (advance = xmax + 80, lsb = xmin; sign floor 60); head/hhea/OS2 vertical metrics (ascent 1500, descent −500, upm 2048); name from config (family, psname ASCII slug); GSUB: generate FEA text (`lookup` type 4 ligature, script taml, feature liga, longest-first, only existing gids) → `feaLib.compile`. Output `out/<family>.ttf` + `out/<family>.woff2` when brotli importable. Report: coverage %, emitted rule count.
- [ ] Tests: build from 6 synthetic outlines (2 uyir, 1 consonant, 1 pulli_form, 2 uyirmei) → TTF parses; cmap has exactly the direct cps; GSUB liga lookup has rules only for existing glyphs; shaping via uharfbuzz: "கி" → 1 glyph == gid of கி cell; decomposed "கொ" (if ொ cell exists as கொ… use a present combo) → same glyph; uncovered combo "வி" → 2 glyphs (degradation, no notdef). Run → pass. Commit.

### Task 9: `pipeline/verify.py`

**Files:** Create `pipeline/verify.py`. Test: `test/test_verify.py`.

**Produces:** `verify(project: Path) -> VerifyReport` — programmatic gates (per spec §6.5): non-empty cells; shaping identity set (கி→1 glyph g_u0B95_u0BBF; க்→g_u0B95_u0BCD pulli form; கௌ composed vs decomposed identical; SAMPLE_TEXTS zero-notdef for covered chars, em-bounds check). Visual gates: render uyirmei/pulli/numeral charts + sample sentences at 48/96/160 px via `gfx.render_text` → `qa/` contact sheets; side-by-side per-glyph output-vs-source crops → `qa/compare_S{n}.png`. Coverage report + green/warn/fail verdict.
- [ ] Tests: on a deliberately broken project (one empty glyph, one outline outside em) → verdict fail with both flags named; on good 6-glyph project → green, contact sheets exist and contain ink in expected halves. Run → pass. Commit.

### Task 10: Bootstrap B1 — digital roundtrip

**Files:** `test/test_b1.py` (marked slow).

- [ ] Fixture project: for all 314 cells render source text via gfx (uyirmei: shaped ligature string from ref.ttf) into template-cell geometry → digital sheets → ingest_digital → trace → build → verify. Gates: per-glyph IoU(actual render, source render) ≥ 0.85 (fill fonts, allow antialias margin); shaping identity gates green; 306/306 rules (all glyphs present). Run → pass. Commit.

### Task 11: Bootstrap B2 — paper distortion

**Files:** `test/test_b2.py` (slow).

- [ ] B1 sheets + random rotation ≤6°, perspective H (corner jitter ≤60 px), gaussian noise σ12, uneven illumination (radial gradient) → ingest_paper → same pipeline. Gates: cell recovery IoU ≥ 0.7 vs B1 glyphs; verify green. Run → pass. Commit.

### Task 12: Bootstrap B3 — extract

**Files:** `test/test_b3.py` (slow).

- [ ] Render 12 sample glyphs (mixed uyir/consonant/uyirmei) onto synthetic backgrounds (gradient, noise, poster-ish color blobs, one partial occlusion) → crop truth regions as `crops/<gid>.png` → ingest_extract (both presets) → trace/build/verify on partial set. Gates: classification injected as ground truth (stage is deterministic; agent-vision loop is exercised in the plugin workflow, not CI); IoU ≥ 0.75 faithful / ≥ 0.6 clean vs source cell renders; occluded glyph correctly flagged; partial coverage: verify warns, TTF builds, degradation shaping works. Run → pass. Commit.

### Task 13: Plugin packaging

**Files:** Create `plugin.json`, `skills/tamil-font-maker/SKILL.md`, `README.md`.

- [ ] Read an installed plugin's manifest for exact schema (e.g. `~/.zcode/cli/plugins/cache/zcode-plugins-official/browser-use/0.4.2/`) and mirror it: name, version, skills list. SKILL.md: frontmatter description (triggers: tamil font, font from image, எழுத்துரு, ttf from poster/handwriting), agent protocol per spec §8 (stage loop, report reading, QA image inspection, per-cell overrides, redraw packets, extract classification loop, coverage/completion workflow), CLI examples, rights caveat. README: quickstart.
- [ ] Commit.

### Task 14: Final verification

- [ ] `pytest -m "not slow"` green; `pytest` (all, incl. B1–B3) green.
- [ ] Full demo via CLI only: `demo` project → template → synthetic fill (B1 path) → ingest → trace → build → verify → green report; agent Reads one qa/ contact sheet and confirms visually; install `out/*.ttf` to `~/.local/share/fonts/` check optional (skip if perms block; report either way).
- [ ] Completion audit vs spec §14 criteria; final commit + tag `v0.1.0`.
