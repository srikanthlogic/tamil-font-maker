# test/fixtures/NotoSansTamil.ttf

Variable font **Noto Sans Tamil** (default instance presents as Regular),
version 2.004, from the google/fonts repository
(`ofl/notosanstamil/NotoSansTamil[wdth,wght].ttf`).

- sha256: `aa3a9b321f4b0bb2c40203ffbde9af89713227866e0e13f76e5b9eeea727cf88`
- License: SIL Open Font License 1.1 — see `OFL.txt` in this directory.
  Redistribution is permitted; the license file ships alongside the font.
- Source: <https://github.com/google/fonts/tree/main/ofl/notosanstamil>
  (download: `https://raw.githubusercontent.com/google/fonts/main/ofl/notosanstamil/NotoSansTamil%5Bwdth,wght%5D.ttf`)

The font is committed (not generated) because the B1–B3 bootstrap IoU gates
are tuned to real Noto outlines — the earlier plan idea of a synthetic
geometric fixture would shape and measure differently. To replace it,
download the current file from the URL above, update the sha256 here, and
re-run the full suite: `.venv/bin/python -m pytest test/`.
