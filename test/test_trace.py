"""Tests for pipeline.trace — raster to quadratic outlines in em units."""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from pipeline.ingest import BASELINE_IN_RASTER
from pipeline.trace import EM_ASCENT, GUIDE_PX, trace, trace_glyph_raster


def _circle_raster() -> Image.Image:
    img = Image.new("L", (360, 480), 0)
    d = ImageDraw.Draw(img)
    cx, r = 180, 90
    cy = BASELINE_IN_RASTER - r           # circle sits on the baseline
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    return img


def _square_raster() -> Image.Image:
    img = Image.new("L", (360, 480), 0)
    d = ImageDraw.Draw(img)
    s = 160
    x0, y0 = 100, BASELINE_IN_RASTER - s
    d.rectangle([x0, y0, x0 + s, y0 + s], fill=255)
    return img


def test_circle_geometry():
    out = trace_glyph_raster(np.array(_circle_raster()))
    assert out["contours"], "no contours traced"
    # em bounds: circle top at baseline-2r, scale EM_ASCENT/GUIDE_PX
    scale = EM_ASCENT / GUIDE_PX
    r_units = 90 * scale
    assert abs(out["ymax"] - 2 * r_units) < 20      # top of circle
    assert abs(out["ymin"] - 0) < 20                # seated on baseline
    # shoelace area ≈ pi r^2 (outer contour)
    pts = out["contours"][0]["poly"]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    area = 0.5 * abs(sum(xs[i] * ys[(i + 1) % len(xs)] -
                         xs[(i + 1) % len(xs)] * ys[i] for i in range(len(xs))))
    # flattened quad polygon underestimates pi r^2 slightly
    assert 0.85 * np.pi * r_units ** 2 < area < 1.05 * np.pi * r_units ** 2


def test_square_is_quad():
    out = trace_glyph_raster(np.array(_square_raster()))
    contour = out["contours"][0]
    # a square traces to 4 corner segments; each emits vertex + endpoint
    pts = contour["poly"]
    assert 8 <= len(pts) <= 10
    scale = EM_ASCENT / GUIDE_PX
    assert abs(out["ymax"] - 160 * scale) < 15


def test_lsb_from_ink_position():
    img = _square_raster()
    shifted = Image.new("L", (360, 480), 0)
    shifted.paste(img, (60, 0))  # shift right by 60px
    out = trace_glyph_raster(np.array(shifted))
    scale = EM_ASCENT / GUIDE_PX
    assert abs(out["lsb"] - 160 * scale) < 25  # original x0=100 + 60 shift


def test_trace_project(project_with_glyph):
    report = trace(project_with_glyph)
    assert report["traced"] >= 1
    assert not report["failed"]


@pytest.fixture
def project_with_glyph(tmp_path):
    g = tmp_path / "glyphs"
    g.mkdir()
    Image.fromarray(np.array(_circle_raster()), "L").save(g / "g_u0B95.png")
    return tmp_path
