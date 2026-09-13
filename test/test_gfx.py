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
