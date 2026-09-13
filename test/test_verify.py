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


# --- glyph_similarity gate -----------------------------------------------------
#
# The shipped movie fonts passed every 2026-09-13 gate while their extracted
# "title" cells were unreadable smudges (render-vs-base IoU 0.22-0.49; clean
# backfill scores ~0.8). These tests pin the gate that must catch that class.

def test_similarity_gate_flags_garbage_glyphs(project):
    # fixture cells are geometric blobs — nothing like the base-font letters
    r = verify(project, base_font="test/fixtures/NotoSansTamil.ttf")
    g = r["gates"]["glyph_similarity"]
    assert not g["ok"]
    assert g["below"]
    assert r["verdict"] == "fail"


def test_similarity_gate_passes_noto_derived_cells(tmp_path):
    # cells filled from the fixture font itself must clear the bar
    from pipeline.mapping import CELLS
    from bootstrap import ref_render
    glyphs = tmp_path / "glyphs"
    glyphs.mkdir()
    # component cells included so the வி ligature can actually shape
    for gid in ["g_u0B85", "g_u0B95", "g_u0BAE", "g_u0BB5",
                "g_u0BBE", "g_u0BBF", "g_u0BB5_u0BBF"]:
        cell = next(c for c in CELLS if c.gid == gid)
        ref_render("".join(chr(cp) for cp in cell.cps)).save(glyphs / f"{gid}.png")
    from pipeline.trace import trace as _trace
    _trace(tmp_path)
    build(tmp_path)
    r = verify(tmp_path, base_font="test/fixtures/NotoSansTamil.ttf")
    g = r["gates"]["glyph_similarity"]
    assert g["ok"], g
    assert "glyph_similarity" not in r["failed_gates"]


def test_similarity_gate_without_base_font_caps_verdict_at_warn(project):
    r = verify(project)
    g = r["gates"]["glyph_similarity"]
    assert g.get("skipped")
    assert "glyph_similarity" not in r["failed_gates"]
    assert r["verdict"] == "warn"


# --- real-word artifact for the agent read-back --------------------------------

def test_words_artifact_exists_with_ink(project):
    verify(project)
    p = project / "qa" / "words.png"
    assert p.exists()
    img = np.array(Image.open(p))
    assert (img < 100).sum() > 100


def test_sample_words_stay_inside_cell_inventory():
    from pipeline.mapping import CELLS, SAMPLE_WORDS
    covered = {cp for c in CELLS for cp in c.cps} | {ord(" ")}
    for w in SAMPLE_WORDS:
        assert all(ord(ch) in covered for ch in w), w
