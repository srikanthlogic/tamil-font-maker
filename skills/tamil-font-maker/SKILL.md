---
name: tamil-font-maker
description: Use when the user wants to create a Tamil font from reference material — handwriting template sheets (digital or paper), or arbitrary images containing Tamil lettering (posters, signboards, scans, screenshots). Turns images into a working Tamil TTF via a staged pipeline with agent-driven QA. Triggers: tamil font, font from image, handwriting font, எழுத்துரு, TTF from poster, calligraphr.
---

# Tamil Font Maker — agent protocol

You are the orchestrator and QA driver of a deterministic pipeline. The pipeline
does pixel work; **you do every judgment call**: reading reports, inspecting QA
images visually (Read the PNGs), tuning per-cell parameters, classifying
extracted glyphs, and deciding pass/fail. Work in the plugin root
(`pipeline/` + this skill live together); all user data lives under
`projects/<name>/`.

## Setup (once per session)

```bash
cd <plugin-root>
test -x .venv/bin/python || { python3 -m venv --without-pip .venv && \
  pip3 --python .venv/bin/python install fonttools uharfbuzz potracer brotli cu2qu Pillow pytest; }
```

All commands below run with `.venv/bin/python -m pipeline.cli …` from the
plugin root. Every stage prints and stores a JSON report in
`projects/<name>/reports/`. **Always read the report after each stage.**

## Standard workflow (template path)

1. `init projects/<name> --family "<FontName>"`
2. `template projects/<name>` → generates 14 sheets (296 labeled cells:
   12 uyir + ஃ + 18 consonants + 13 signs + 18 pulli forms + 198 uyirmei +
   numerals/grantha/digits/punct) as PNG + PDF in `sheets/`.
   - Cells carry faint baseline/headline guide lines — tell the user to keep
     ink between them; consistency there is the main quality lever.
3. User fills cells **digitally** (any image editor, save PNG back into
   `sheets/`, dimensions must stay exactly 2480×3508) or **on paper**
   (print PDF, draw, photograph or scan; all 4 corner fiducial squares must
   be visible and the page roughly flat).
4. `ingest-digital projects/<name> sheets/S1.png` (per sheet) or
   `ingest-paper projects/<name> photo.png --sheet S1`.
5. `trace projects/<name>` → `build projects/<name>` → `verify projects/<name>`.

## Sample-image workflow (extract path)

For arbitrary images with Tamil lettering:

1. User drops images into `projects/<name>/samples/`.
2. **You** Read each image, locate the lettering, and crop per-glyph regions
   (save crops to `projects/<name>/crops/`). Judge each crop's viability:
   letters fused with artwork or heavily occluded → skip and tell the user.
3. **You classify** each crop (which Tamil letter/combo it is) — there is no
   ML model; your vision is the classifier. Cleanup presets: `faithful`
   (keeps integral decorative detail — default for display lettering) or
   `clean` (aggressive simplification).
4. `ingest-extract projects/<name> g_u0B95=crops/ka.png … --preset faithful`
5. Then trace/build/verify as above. Expect **partial coverage** — that is
   normal: uncovered combos degrade legibly (matra renders after the
   consonant), missing chars show honest boxes.
6. `template projects/<name> --only <missing gids>` generates **completion
   sheets** with just the gaps, so the user can finish the font by hand in
   the extracted style.

## The QA loop (your main job)

`verify` emits `reports/verify.json` plus PNG artifacts in `projects/<name>/qa/`
(combo chart, sample sentences, per-glyph source-vs-output comparisons).

- **Read the JSON.** Programmatic gates (`failed_gates`) are mandatory:
  non-empty rasters, em-bounds, shaping identities (கி → one precomposed
  glyph; க் → pulli form; composed == decomposed கௌ), notdefs matching
  coverage. A `fail` verdict means fix, not ship.
- **Read the QA images.** You are the visual judge: look for clipped glyphs,
  baseline drift, matra collisions, mistraces, ink blobs.
- **Iterate per cell.** Override cleanup/trace parameters in
  `projects/<name>/config.toml` (`[cells.g_u0BBF]`, keys `threshold`,
  `despeckle`, `smooth`), re-run the affected stages (stages are idempotent;
  single cells can be re-ingested). If parameters cannot fix a cell, add it
  to a redraw packet: `template projects/<name> --only <gid> …` produces a
  sheet with just those cells for the user to redraw.
- Converge when programmatic gates pass and every visual flag is either
  fixed or explicitly waived (say why, in the final summary).

## Commands

```
init <project> [--family NAME]
template <project> [--only GID...]      # full or completion sheets
ingest-digital <project> <sheet.png>
ingest-paper <project> <photo> --sheet S1
ingest-extract <project> gid=crop.png... [--preset faithful|clean]
trace <project> / build <project> / verify <project>
```

Exit codes: stage commands 0 on success; `verify` returns 2 on `fail` verdict.

## Fidelity expectations (tell the user)

- TTF glyphs are filled outlines: gradients/3D/photo textures do not survive;
  outline and inline decorations do.
- Bootstrap B1–B3 (in `test/`) prove this pipeline end-to-end on Noto Sans
  Tamil with zero human art: digital roundtrip, distorted paper photos, and
  poster-like extraction are all gated on shape fidelity (IoU) + shaping.
- Rights: extracted movie/poster lettering is typically owned artwork.
  Personal/experimental use is the user's call; distribution or commercial
  use of an extracted font is a legal question outside this tool.

## Output

The finished font is at `projects/<name>/out/<family>.ttf` (+ `.woff2` when
brotli is available). To install locally: copy to `~/.local/share/fonts/` and
run `fc-cache -f` (may require the user to run it).
