# Font notice

The two fonts in `docs/fonts/` are outputs of this pipeline's poster runs,
named for the movies whose posters seeded them:

- `VikramTitle.*` — from the *Vikram* (2022) poster run, Raaj Kamal Films
  International.
- `PonniyinSelvanTitle.*` — from the *Ponniyin Selvan: Part I* (2022) poster
  run, Madras Talkies / Lyca Productions.

**All 296 glyphs in both fonts are derived from Noto Sans Tamil** (SIL Open
Font License 1.1, see `test/fixtures/OFL.txt`); OFL terms apply. The poster
title lettering was extracted experimentally but rejected by the pipeline's
legibility gate (the lettering is fused with poster artwork), so no film
typography remains in these files. The film names are used only to describe
the pipeline runs, not as branding of the fonts.

`NotoTamil-fallback.woff2` is a subset of the same Noto Sans Tamil, served so
the page itself renders Tamil on systems without a Tamil font installed.
