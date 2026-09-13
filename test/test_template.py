"""Tests for pipeline.template — sheet generation, guides, fiducials."""
import numpy as np
import pytest

from pipeline.mapping import cell
from pipeline.template import (BASELINE_Y, CELL_H, CELL_W, CELL_COLS,
                               HEADLINE_Y, SHEET_H, SHEET_W, cell_box,
                               generate, sheet_cells)


@pytest.fixture(scope="module")
def sheets_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("sheets")
    report = generate(d)
    return d, report


def test_generates_all_14_sheets(sheets_dir):
    d, report = sheets_dir
    for i in range(1, 15):
        assert (d / f"S{i}.png").exists(), f"S{i}.png missing"
        assert (d / f"S{i}.pdf").exists()
    assert report["cells_placed"] == 296


def test_sheet_geometry(sheets_dir):
    d, _ = sheets_dir
    img = np.array(__import__("PIL.Image", fromlist=["Image"]).open(d / "S5.png"))
    assert img.shape == (SHEET_H, SHEET_W)
    # fiducial centers black, ring white, outside white
    assert img[100, 100] < 50
    assert img[100, 55] > 200
    assert img[100, 30] == 255
    assert img[SHEET_H - 100, SHEET_W - 100] < 50


def test_cell_guides_and_labels(sheets_dir):
    d, _ = sheets_dir
    img = np.array(__import__("PIL.Image", fromlist=["Image"]).open(d / "S1.png"))
    x0, y0, x1, y1 = cell_box("S1", 0, 0)
    assert (x1 - x0, y1 - y0) == (CELL_W, CELL_H)
    # headline guide (gray) and baseline guide present
    assert img[y0 + HEADLINE_Y, x0 + 50: x0 + 350].mean() < 230
    assert img[y0 + BASELINE_Y, x0 + 50: x0 + 350].mean() < 230
    # cell interior above headline is white
    assert img[y0 + 20, x0 + 50: x0 + 350].mean() > 245
    # label strip below cell has dark pixels
    strip = img[y1 + 5: y1 + 40, x0: x1]
    assert (strip < 100).sum() > 30


def test_sheet_cells_matches_mapping(sheets_dir):
    sc = sheet_cells()
    assert set(sc) == {f"S{i}" for i in range(1, 15)}
    total = sum(len(v) for v in sc.values())
    assert total == 296
    # first uyirmei is கா (the ஆ column precedes இ)
    assert sc["S5"][0].gid == "g_u0B95_u0BBE"


def test_completion_sheets(tmp_path):
    only = {"g_u0B95_u0BBF", "g_u0B95"}
    report = generate(tmp_path, only=only)
    files = sorted(p.name for p in tmp_path.glob("C*.png"))
    assert files == ["C1.png"]
    assert report["cells_placed"] == 2
    img = np.array(__import__("PIL.Image", fromlist=["Image"]).open(tmp_path / "C1.png"))
    x0, y0, x1, y1 = cell_box("C1", 0, 0)
    strip = img[y1 + 5: y1 + 40, x0: x1]
    assert (strip < 100).sum() > 30
    # slots (0,0) and (0,1) hold the two cells; slot (1,0) must be empty
    bx0, by0, bx1, by1 = cell_box("C1", 1, 0)
    strip2 = img[by1 + 5: by1 + 40, bx0: bx1]
    assert (strip2 < 100).sum() == 0
