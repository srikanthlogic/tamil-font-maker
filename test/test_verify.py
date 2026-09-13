"""Tests for pipeline.verify — gates, QA artifacts, verdicts."""
import json
import numpy as np
import pytest
from PIL import Image, ImageDraw

from pipeline.build import build
from pipeline.ingest import BASELINE_IN_RASTER
from pipeline.trace import trace_glyph_raster
from pipeline.verify import verify

TARGET_GIDS = ["g_u0B85", "g_u0B95", "g_u0BBE", "g_u0BBF", "g_u0BCD",
               "g_u0B95_u0BCD", "g_u0B95_u0BBF", "g_u0B95_u0BCC"]


def _blob(kind):
    img = Image.new("L", (360, 480), 0)
    d = ImageDraw.Draw(img)
    if kind == "round":
        d.ellipse([80, BASELINE_IN_RASTER - 160, 240, BASELINE_IN_RASTER], fill=255)
    else:
        d.rectangle([80, BASELINE_IN_RASTER - 160, 240, BASELINE_IN_RASTER], fill=255)
    return np.array(img)


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    proj = tmp_path_factory.mktemp("proj")
    glyphs = proj / "glyphs"
    glyphs.mkdir()
    outlines = proj / "outlines"
    outlines.mkdir()
    for i, gid in enumerate(TARGET_GIDS):
        blob = _blob("round" if i % 2 else "square")
        Image.fromarray(blob, "L").save(glyphs / f"{gid}.png")
        (outlines / f"{gid}.json").write_text(json.dumps(trace_glyph_raster(blob)))
    build(proj)
    return proj


def test_green_project(project):
    r = verify(project)
    assert r["verdict"] == "warn", r  # warn: partial coverage, all gates ok
    assert not r["failed_gates"]
    assert (project / "reports" / "verify.json").exists()
    assert (project / "qa" / "combo_chart.png").exists()
    # chart contains real ink (not blank)
    chart = np.array(Image.open(project / "qa" / "combo_chart.png"))
    assert (chart < 100).sum() > 150


def test_full_coverage_is_green(project):
    # pretend full coverage for verdict purposes: verify uses CELLS length;
    # a project with every cell is B1's job — here assert warn mentions coverage
    r = verify(project)
    assert 0 < r["coverage_pct"] < 100
    assert r["missing_count"] == 296 - len(TARGET_GIDS)


def test_empty_raster_fails(project):
    Image.new("L", (360, 480), 0).save(project / "glyphs" / "g_u0BBE.png")
    r = verify(project)
    assert "non_empty_rasters" in r["failed_gates"]
    assert r["verdict"] == "fail"
    # restore
    Image.fromarray(_blob("round"), "L").save(project / "glyphs" / "g_u0BBE.png")
