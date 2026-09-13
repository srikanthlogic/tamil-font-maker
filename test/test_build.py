"""Tests for pipeline.build — TTF assembly with cmap + GSUB ligatures."""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from pipeline.build import build
from pipeline.ingest import BASELINE_IN_RASTER
from pipeline.trace import trace_glyph_raster

TARGET_GIDS = [
    "g_u0B85",            # uyir அ
    "g_u0B95",            # consonant க
    "g_u0BBE",            # sign ா
    "g_u0BBF",            # sign ி
    "g_u0BCD",            # pulli ்
    "g_u0B95_u0BCD",      # pulli form க்
    "g_u0B95_u0BBF",      # uyirmei கி
    "g_u0B95_u0BCC",      # uyirmei கௌ
]


def _blob(kind: str) -> np.ndarray:
    img = Image.new("L", (360, 480), 0)
    d = ImageDraw.Draw(img)
    if kind == "round":
        d.ellipse([80, BASELINE_IN_RASTER - 160, 240, BASELINE_IN_RASTER], fill=255)
    else:
        d.rectangle([80, BASELINE_IN_RASTER - 160, 240, BASELINE_IN_RASTER], fill=255)
    return np.array(img)


@pytest.fixture(scope="module")
def font_path(tmp_path_factory):
    proj = tmp_path_factory.mktemp("proj")
    outlines = proj / "outlines"
    outlines.mkdir()
    for i, gid in enumerate(TARGET_GIDS):
        r = trace_glyph_raster(_blob("round" if i % 2 else "square"))
        (outlines / f"{gid}.json").write_text(json.dumps(r))
    build(proj)
    return proj / "out" / "TamilMaker.ttf"


import json  # noqa: E402


def test_ttf_loads_and_tables(font_path):
    from fontTools.ttLib import TTFont
    f = TTFont(str(font_path))
    assert "GSUB" in f
    assert f["head"].unitsPerEm == 2048
    cmap = f.getBestCmap()
    assert cmap[0x0B95] == "g_u0B95"
    assert cmap[0x0BBF] == "g_u0BBF"
    assert cmap[0x20] == "g_u0020"
    assert 0x0BB5 not in cmap  # uncovered consonant stays out of cmap


def test_shaping_ligatures(font_path):
    from pipeline.gfx import shape_text
    fb = open(font_path, "rb").read()
    from fontTools.ttLib import TTFont
    order = TTFont(str(font_path)).getGlyphOrder()

    ki = shape_text(fb, "கி")           # covered uyirmei -> one precomposed glyph
    assert len(ki) == 1
    assert order[ki[0][0]] == "g_u0B95_u0BBF"

    kk = shape_text(fb, "க்")           # pulli form
    assert len(kk) == 1
    assert order[kk[0][0]] == "g_u0B95_u0BCD"

    ka = shape_text(fb, "கா")           # uncovered combo -> legacy degradation
    assert len(ka) == 2
    assert all(g != 0 for g, _, _ in ka)  # no .notdef: matra renders spacing

    miss = shape_text(fb, "வ")          # uncovered consonant -> notdef
    assert miss[0][0] == 0


def test_woff2_alongside(font_path):
    assert font_path.with_suffix(".woff2").exists()


def test_space_advance_is_word_proportionate(font_path):
    # 160 units (~0.08em) made words run together in running text; a word
    # space should sit near a quarter em like reference fonts
    from fontTools.ttLib import TTFont

    tt = TTFont(str(font_path))
    advance = tt["hmtx"]["g_u0020"][0]
    assert 0.20 * tt["head"].unitsPerEm <= advance <= 0.30 * tt["head"].unitsPerEm
