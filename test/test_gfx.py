"""Tests for pipeline.gfx — shaping, rasterization, binarization utilities."""
import numpy as np
import pytest
from PIL import Image

from gfx_fixture import make_square_font
from pipeline.gfx import despeckle, otsu, otsu_binarize, render_text


@pytest.fixture(scope="module")
def square_font(tmp_path_factory):
    path = tmp_path_factory.mktemp("font") / "sq.ttf"
    make_square_font(path)
    return str(path)


def test_render_filled_square(square_font):
    img = render_text(square_font, "a", 100)
    assert img.mode == "L"
    arr = np.array(img)
    ink = arr > 127
    frac = ink.mean()
    # glyph is a 600x600 box in a 1000upm font with ~800x1000 canvas
    assert 0.15 < frac < 0.60
    ys, xs = np.where(ink)
    # bbox roughly square
    assert 0.7 < (ys.max() - ys.min()) / (xs.max() - xs.min()) < 1.4


def test_render_hole_even_odd(square_font):
    img = render_text(square_font, "b", 100)
    arr = np.array(img)
    ink = arr > 127
    ys, xs = np.where(ink)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    assert arr[cy, cx] < 127, "hole center must be background (even-odd fill)"


def test_shape_text_maps_both_chars(square_font):
    from pipeline.gfx import shape_text

    font_bytes = open(square_font, "rb").read()
    shaped = shape_text(font_bytes, "ab")
    assert [g[0] for g in shaped] == [1, 2]  # glyph ids of a, b
    # second glyph offset to the right
    assert shaped[1][1] > shaped[0][1]


def test_otsu_bimodal():
    arr = np.zeros((100, 100), dtype=np.uint8)
    arr[20:80, 20:80] = 240
    img = Image.fromarray(arr)
    t = otsu(img)
    assert 0 <= t <= 255
    bin_img = otsu_binarize(img)  # dark ink -> 255; here bright square is bg
    a = np.array(bin_img)
    assert a[50, 50] == 0        # bright region: background
    assert a[5, 5] == 255        # dark surround: ink


def test_despeckle():
    arr = np.zeros((100, 100), dtype=np.uint8)
    arr[40:60, 40:60] = 255      # real blob 20x20
    arr[5:7, 5:7] = 255          # speckle 2x2
    out = np.array(despeckle(Image.fromarray(arr), min_area=8))
    assert out[50, 50] == 255
    assert out[6, 6] == 0


def test_ink_extents_lone_two_part_mark_uses_composite_glyph():
    # HarfBuzz splits a lone கொ-sign (ொ) into ா+ெ (yMax ~554/1000), but the
    # cell raster draws the composite glyph itself (yMax ~843/1000). The
    # extents must describe the drawn glyph, or backfill clips its top.
    # (Real Noto values: கொ-sign composite yMax 843 -> 395px at 468px.)
    import uharfbuzz as hb  # noqa: F401 — proves nothing shaped is needed
    from pipeline.gfx import ink_extents

    FIX = "test/fixtures/NotoSansTamil.ttf"
    above, below = ink_extents(FIX, "ொ", 468)
    assert above > 350, above
    shaped = __import__("pipeline.gfx", fromlist=["shape_text"]).shape_text(
        open(FIX, "rb").read(), "ொ")
    assert len(shaped) >= 2  # the shaper does split it — extents must not


def test_render_ref_lone_mark_canvas_covers_wide_glyph():
    # a wide two-part sign (ொ ≈ 1.23em in the fixture) must be drawn fully
    # inside render_ref's canvas — ink touching/exceeding the edge silently
    # clips the gate's comparison (2026-09-13 waterfall regression)
    from pipeline.gfx import render_ref

    ref = render_ref("test/fixtures/NotoSansTamil.ttf", "ொ", 160)
    a = np.asarray(ref) > 127
    ys, xs = np.nonzero(a)
    assert xs.min() >= 1 and xs.max() <= ref.width - 2, (xs.min(), xs.max(), ref.size)
    assert ys.min() >= 1 and ys.max() <= ref.height - 2
    # and the full ink width survives: the composite's own bbox at 160px
    from fontTools.ttLib import TTFont

    tt = TTFont("test/fixtures/NotoSansTamil.ttf")
    g = tt["glyf"][tt.getBestCmap()[0x0BCA]]
    expected = round((g.xMax - g.xMin) * 160 / tt["head"].unitsPerEm)
    assert (xs.max() - xs.min() + 1) == pytest.approx(expected, rel=0.05)
