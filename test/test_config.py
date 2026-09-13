"""Tests for per-cell parameter overrides (config.toml [cells.<gid>] sections
— the SKILL.md QA-loop contract)."""
import json

import numpy as np
import pytest
from PIL import Image

from bootstrap import FIX_FONT
from pipeline import trace
from pipeline.config import ConfigError, cell_params
from pipeline.gfx import render_text
from pipeline.ingest import ingest_extract, manifest_of


def test_cell_params_validation():
    assert cell_params({}, "g_u0B95") == {}
    assert cell_params({"cells": {"g_u0B95": {"smooth": 1.2}}},
                       "g_u0B95") == {"smooth": 1.2}
    with pytest.raises(ConfigError):
        cell_params({"cells": {"g_u0B95": {"threshhold": 100}}}, "g_u0B95")
    with pytest.raises(ConfigError):
        cell_params({"cells": {"g_u0B95": {"smooth": "very"}}}, "g_u0B95")
    with pytest.raises(ConfigError):
        cell_params({"cells": {"g_u0B95": 4}}, "g_u0B95")


def _slab_crop(tmp_path):
    """க (near-black 0) on a dark-60 slab on a light-170 background. Otsu
    cuts between 60 and 170, so by default the slab comes in as ink; a fixed
    threshold of 30 keeps only the glyph."""
    from PIL import ImageDraw

    crop = render_text(FIX_FONT, "க", 300)
    bg = Image.new("L", (crop.width + 80, crop.height + 80), 170)
    ImageDraw.Draw(bg).rectangle([15, 15, bg.width - 16, bg.height - 16],
                                 fill=60)
    bg.paste(0, (40, 40), mask=crop)
    p = tmp_path / "raw.png"
    bg.save(p)
    return p


def _ink_frac(proj):
    return (np.array(Image.open(proj / "glyphs" / "g_u0B95.png"))
            > 127).mean()


def test_extract_threshold_override(tmp_path):
    plain, fixed = tmp_path / "plain", tmp_path / "fixed"
    plain.mkdir()
    fixed.mkdir()
    report_plain = ingest_extract(plain, {"g_u0B95": _slab_crop(plain)},
                                  preset="faithful")
    assert "param_overrides" not in report_plain

    (fixed / "config.toml").write_text(
        'family = "X"\n\n[cells.g_u0B95]\nthreshold = 30\n')
    report = ingest_extract(fixed, {"g_u0B95": _slab_crop(fixed)},
                            preset="faithful")
    # threshold 30 cuts the 60-gray slab out — the override changed the cut
    assert _ink_frac(fixed) < 0.5 * _ink_frac(plain)
    assert report["param_overrides"] == ["g_u0B95"]
    m = manifest_of(fixed)
    assert m["g_u0B95"]["params"] == {"threshold": 30}


def _glyph_raster_with_speck(path):
    raster = np.zeros((480, 360), dtype=np.uint8)
    raster[100:300, 100:260] = 255     # main slab
    raster[40:43, 40:43] = 255         # 9 px speck (kept at default turdsize 4)
    Image.fromarray(raster, "L").save(path)


def _contours(proj, gid):
    d = json.loads((proj / "outlines" / f"{gid}.json").read_text())
    return d["contours"]


def test_trace_despeckle_override(tmp_path):
    proj = tmp_path
    g = proj / "glyphs"
    g.mkdir()
    _glyph_raster_with_speck(g / "g_u0B95.png")

    trace.trace(proj)
    assert len(_contours(proj, "g_u0B95")) == 2   # slab + speck

    (proj / "config.toml").write_text("[cells.g_u0B95]\ndespeckle = 20\n")
    t = trace.trace(proj)
    assert t["overrides"] == {"g_u0B95": {"despeckle": 20}}
    assert len(_contours(proj, "g_u0B95")) == 1   # speck suppressed


def test_trace_smooth_override_reaches_potrace(tmp_path, monkeypatch):
    proj = tmp_path
    g = proj / "glyphs"
    g.mkdir()
    _glyph_raster_with_speck(g / "g_u0B95.png")
    (proj / "config.toml").write_text("[cells.g_u0B95]\nsmooth = 0.1\n")

    seen = {}
    orig = trace.trace_glyph_raster

    def spy(arr, **kw):
        seen.update(kw)
        return orig(arr, **kw)

    monkeypatch.setattr(trace, "trace_glyph_raster", spy)
    assert trace.trace(proj)["traced"] == 1
    assert seen["alphamax"] == 0.1 and seen["turdsize"] == 4


def test_trace_bad_override_fails_loudly(tmp_path):
    proj = tmp_path
    g = proj / "glyphs"
    g.mkdir()
    _glyph_raster_with_speck(g / "g_u0B95.png")
    (proj / "config.toml").write_text("[cells.g_u0B95]\nthreshhold = 5\n")
    # a typo'd key must raise, not dissolve into per-cell isolation
    with pytest.raises(ConfigError):
        trace.trace(proj)
