"""Tests for pipeline.ingest — digital, paper, extract, re-ingest, errors."""
import numpy as np
import pytest
from PIL import Image

from bootstrap import FIX_FONT, synthesize_photo
from pipeline.gfx import render_text
from pipeline.homography import IngestError
from pipeline.ingest import (ingest_digital, ingest_extract, ingest_paper,
                             manifest_of)
from pipeline.template import BASELINE_Y, cell_box, generate


def _fill_cell(sheet_img: Image.Image, row: int, col: int, char: str):
    """Paste a rendered Tamil glyph into a cell's inner drawing area."""
    x0, y0, x1, y1 = cell_box(row, col)
    glyph = render_text(FIX_FONT, char, 300)
    a = np.array(glyph) > 127
    ys, xs = np.nonzero(a)
    glyph = glyph.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    # seat the ink bottom on the baseline, centered
    px = x0 + (x1 - x0 - glyph.width) // 2
    py = y0 + BASELINE_Y - glyph.height
    sheet_img.paste(0, (px, py, px + glyph.width, py + glyph.height),
                    mask=Image.fromarray(np.array(glyph)))


@pytest.fixture(scope="module")
def filled_s1(tmp_path_factory):
    """Digital S1 sheet with three cells filled (அ ஆ ஃ), others empty."""
    d = tmp_path_factory.mktemp("proj_s1")
    sheets = d / "sheets"
    generate(sheets)
    img = Image.open(sheets / "S1.png").convert("L")
    _fill_cell(img, 0, 0, "அ")
    _fill_cell(img, 0, 1, "ஆ")
    _fill_cell(img, 2, 2, "ஃ")   # ஃ is the 13th cell: row 2, col 2
    img.save(sheets / "S1.png")
    return d, sheets / "S1.png"


def test_ingest_digital(filled_s1):
    d, sheet = filled_s1
    report = ingest_digital(d, sheet)
    gids = set(report["ingested_gids"])
    assert gids == {"g_u0B85", "g_u0B86", "g_u0B83"}
    for gid in gids:
        assert (d / "glyphs" / f"{gid}.png").exists()
    cells = _cells_of("S1")
    filled = {"g_u0B85", "g_u0B86", "g_u0B83"}
    assert report["empty"] == [c.gid for c in cells if c.gid not in filled]
    m = manifest_of(d)
    assert m["g_u0B85"]["mode"] == "digital"


def _cells_of(sheet):
    from pipeline.template import sheet_cells
    return sheet_cells()[sheet]


def test_ingest_paper_recovers_cells(filled_s1, tmp_path):
    proj, sheet = filled_s1
    out = tmp_path / "photo.png"
    synthesize_photo(sheet, out, seed=7)
    paper_glyphs = proj / "glyphs_paper"
    report = ingest_paper(proj, out, sheet="S1", glyphs_dir=paper_glyphs)
    assert set(report["ingested_gids"]) == {"g_u0B85", "g_u0B86", "g_u0B83"}
    # glyphs recovered from photo should closely match the digital ones
    for gid in ("g_u0B85", "g_u0B86", "g_u0B83"):
        a = np.array(Image.open(proj / "glyphs" / f"{gid}.png")) > 127
        b = np.array(Image.open(paper_glyphs / f"{gid}.png")) > 127
        inter = (a & b).sum()
        union = (a | b).sum()
        assert union and inter / union > 0.7, f"{gid} IoU too low"


def test_ingest_extract_cleanup_and_manifest(tmp_path):
    proj = tmp_path
    (proj / "crops").mkdir()
    crop = render_text(FIX_FONT, "க", 300)
    # put it on a noisy gray background to exercise binarization
    bg = Image.new("L", (crop.width + 60, crop.height + 60), 170)
    bg.paste(0, (30, 30), mask=crop)
    bg.save(proj / "crops" / "raw.png")
    report = ingest_extract(proj, {"g_u0B95": proj / "crops" / "raw.png"},
                            preset="faithful")
    assert report["ingested_gids"] == ["g_u0B95"]
    a = np.array(Image.open(proj / "glyphs" / "g_u0B95.png"))
    assert (a > 127).mean() > 0.02
    m = manifest_of(proj)
    assert m["g_u0B95"]["preset"] == "faithful"


def test_reingest_appends_history(tmp_path):
    proj = tmp_path
    (proj / "crops").mkdir()
    crop = render_text(FIX_FONT, "ம", 300)
    p1 = proj / "crops" / "m1.png"
    p2 = proj / "crops" / "m2.png"
    crop.save(p1)
    crop.save(p2)
    ingest_extract(proj, {"g_u0BAE": p1}, preset="clean")
    ingest_extract(proj, {"g_u0BAE": p2}, preset="clean")
    m = manifest_of(proj)
    # two ingests -> current entry + one historical entry
    assert len(m["g_u0BAE"]["history"]) == 1


def test_reingest_history_capped_at_five(tmp_path):
    proj = tmp_path
    (proj / "crops").mkdir()
    crop = render_text(FIX_FONT, "ம", 300)
    for i in range(7):
        p = proj / "crops" / f"m{i}.png"
        crop.save(p)
        ingest_extract(proj, {"g_u0BAE": p}, preset="clean")
    m = manifest_of(proj)
    # manifest keeps at most the current entry + 5 historical ones
    assert len(m["g_u0BAE"]["history"]) == 5


def test_ingest_digital_rejects_modified_sheet(tmp_path):
    bad = tmp_path / "bad.png"
    Image.new("L", (1000, 1400), 255).save(bad)
    with pytest.raises(IngestError):
        ingest_digital(tmp_path, bad)


def test_ingest_paper_requires_four_fiducials(tmp_path):
    blank = tmp_path / "blank.png"
    Image.new("L", (800, 1100), 255).save(blank)
    with pytest.raises(IngestError):
        ingest_paper(tmp_path, blank, sheet="S1")


def test_ingest_extract_unknown_preset(tmp_path):
    crop = render_text(FIX_FONT, "க", 300)
    p = tmp_path / "ka.png"
    crop.save(p)
    with pytest.raises(IngestError):
        ingest_extract(tmp_path, {"g_u0B95": p}, preset="nope")
