# tamil-font-maker

"Calligraphr for Tamil" — a ZCode plugin that turns reference imagery into a
working Tamil TrueType font, fully agentically. Business framing: **any image
→ TTF** (posters, signboards, pages, screenshots, handwriting scans), plus
drawn template sheets (digitally in any image editor, or printed on paper).

- Spec: `docs/superpowers/specs/2026-09-13-tamil-font-maker-design.md`
- Plan: `docs/superpowers/plans/2026-09-13-tamil-font-maker.md`
- Agent protocol: `skills/tamil-font-maker/SKILL.md`

## How it works

```
template → ingest (digital | paper | extract) → trace → build → verify
                ↑        redraw packet / completion sheets     │
                └────────────── agent QA loop ←────────────────┘
```

- `pipeline/mapping.py` is the single source of truth: **296 drawn cells**
  (12 uyir + ஃ + 18 consonants + 13 signs + 18 pulli forms + 198 uyirmei +
  10 Tamil numerals + 4 grantha + digits + punctuation) and **270 GSUB
  ligature rules** (precomposed uyirmei — sidesteps matra reordering).
- Partial fonts (from sample extraction) degrade legibly: uncovered combos
  render consonant + spacing matra; `template --only` emits completion
  sheets for exactly the missing cells.
- The agent is the QA driver: reads each stage's JSON report, visually
  inspects rendered charts and per-glyph comparisons, tunes per-cell
  parameters, flags cells for redraw.

## Bootstrap validation (zero human art)

`test/test_b1.py` … `test_b3.py` rasterize Noto Sans Tamil (OFL, in
`test/fixtures/`) into the pipeline's own formats and gate the full roundtrip:

- **B1** digital sheets → font: 296/296 traced, 270/270 GSUB rules, shaping
  identities green, per-glyph IoU (mean ≥ 0.82, none broken).
- **B2** the same sheets as synthesized phone photos (perspective ≤ ~70 px
  corner jitter, noise, uneven light) → fiducial deskew recovers
  everything.
- **B3** glyphs on poster-like backgrounds → agent-vision extract (ground
  truth injected), occluded glyph flagged, partial font builds and degrades
  legibly, completion template covers exactly the gaps.

```bash
.venv/bin/python -m pytest test/ -m "not slow"   # unit suite
.venv/bin/python -m pytest test/                 # + bootstrap B1-B3 (~4 min)
```

## Setup

```bash
python3 -m venv --without-pip .venv
pip3 --python .venv/bin/python install fonttools uharfbuzz potracer cu2qu Pillow brotli pytest
```

No opencv / fontforge / sudo required — fiducial detection and the
perspective warp are hand-rolled numpy.

## Rights note

Extracted movie/poster lettering is typically owned artwork. Personal and
experimental use is your call; distributing or commercializing a font
extracted from someone's lettering is a legal question this tool does not
solve.
