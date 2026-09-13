"""Tests for pipeline.backfill — the stage that completed the movie fonts.
Guards the two defects found on 2026-09-13: backfilled glyphs rendered at
the wrong em scale (300px instead of the guide-calibrated reference size,
making the fonts ~35% smaller than their base font), and glyphs seated by
ink-bottom instead of baseline, lifting descending letters off the baseline.
"""
import numpy as np
import pytest
from PIL import Image

from pipeline.backfill import REF_PX, backfill
from pipeline.ingest import BASELINE_IN_RASTER, WORK_H
from pipeline.mapping import CELLS, UPM

FIX_FONT = "test/fixtures/NotoSansTamil.ttf"
NOTO_UPEM = 1000


def _blob(img):
    Image.fromarray(img, "L").convert("L")


def _stub_project(tmp_path, missing_gids):
    """Fill every cell except missing_gids with a minimal blob so backfill
    only has to render the real ones."""
    glyphs = tmp_path / "glyphs"
    glyphs.mkdir(parents=True)
    blob = np.full((48, 48), 0, dtype=np.uint8)
    blob[10:38, 10:38] = 255
    for c in CELLS:
        if c.gid not in missing_gids:
            Image.fromarray(blob, "L").save(glyphs / f"{c.gid}.png")
    return tmp_path


def _ink_bbox(path):
    a = np.asarray(Image.open(path))
    ys, xs = np.nonzero(a > 127)
    return ys.min(), ys.max()


# The reference render size is guide-calibrated: a glyph of H units (base
# upem) must rasterize to H * REF_PX / base_upem pixels of ink.
def test_backfill_renders_at_em_true_scale(tmp_path):
    # எ = 566 units in the 1000-upem fixture -> 566 * REF_PX / 1000 px
    proj = _stub_project(tmp_path, {"g_u0B8E"})
    backfill(proj, FIX_FONT)
    top, bot = _ink_bbox(proj / "glyphs" / "g_u0B8E.png")
    expected = round(566 * REF_PX / NOTO_UPEM)
    assert (bot - top + 1) == pytest.approx(expected, abs=8)


def test_backfill_seats_descenders_below_baseline(tmp_path):
    # ர's ink descends below the baseline in the base font; the cell raster
    # must keep that tail below BASELINE_IN_RASTER, not sit ink-bottom on it.
    proj = _stub_project(tmp_path, {"g_u0BB0"})
    backfill(proj, FIX_FONT)
    top, bot = _ink_bbox(proj / "glyphs" / "g_u0BB0.png")
    assert bot > BASELINE_IN_RASTER, "descender was lifted onto the baseline"


def test_backfill_seats_flat_glyphs_on_baseline(tmp_path):
    # a non-descending letter seats by baseline: its bottom may overshoot the
    # line only by the glyph's own optical yMin (எ = -12 units ≈ 6px)
    proj = _stub_project(tmp_path, {"g_u0B8E"})
    backfill(proj, FIX_FONT)
    top, bot = _ink_bbox(proj / "glyphs" / "g_u0B8E.png")
    assert BASELINE_IN_RASTER <= bot <= BASELINE_IN_RASTER + 10


def test_backfill_tall_glyphs_fit_the_cell(tmp_path):
    # the tallest matra combos must survive inside the 480px raster
    proj = _stub_project(tmp_path, {"g_u0BB4_u0BCC"})
    report = backfill(proj, FIX_FONT)
    assert report["failed"] == []
    top, bot = _ink_bbox(proj / "glyphs" / "g_u0BB4_u0BCC.png")
    assert top >= 0 and bot < WORK_H


def test_backfill_does_not_shrink_wide_glyphs(tmp_path):
    # கா at reference scale is ~656px wide — wider than the 360px cell.
    # Shrinking it to fit the cell halves the glyph's em size (the 2nd
    # 2026-09-13 defect); the raster may simply be wider instead. Trace and
    # build are raster-size agnostic, so wide rasters are safe.
    proj = _stub_project(tmp_path, {"g_u0B95_u0BBE"})
    report = backfill(proj, FIX_FONT)
    assert report["failed"] == []
    a = np.asarray(Image.open(proj / "glyphs" / "g_u0B95_u0BBE.png"))
    ys, xs = np.nonzero(a > 127)
    expected_w = 656  # em-true ink width of கா at REF_PX (measured)
    assert (xs.max() - xs.min() + 1) == pytest.approx(expected_w, rel=0.06)


def test_ref_px_is_guide_calibrated():
    # 2048 upm * 320px guide band / 1400 em ascent -> ~468px per em
    assert REF_PX == pytest.approx(UPM * 320 / 1400, rel=0.01)
