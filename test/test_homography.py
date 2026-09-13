"""Tests for pipeline.homography — DLT solve, warp, fiducial detection."""
import numpy as np
import pytest
from PIL import Image

from pipeline.homography import find_fiducials, solve_homography, warp_perspective
from pipeline.template import SHEET_H, SHEET_W, generate


@pytest.fixture(scope="module")
def sheet_s2_png(tmp_path_factory):
    d = tmp_path_factory.mktemp("sheets")
    generate(d)
    return d / "S2.png"


def test_solve_identity():
    src = [(0, 0), (10, 0), (10, 10), (0, 10)]
    H = solve_homography(src, src)
    for x, y in src:
        u, v, w = H @ [x, y, 1]
        assert abs(u / w - x) < 1e-6 and abs(v / w - y) < 1e-6


def test_solve_projective():
    src = [(0, 0), (100, 0), (100, 100), (0, 100)]
    dst = [(10, 15), (120, 25), (140, 130), (5, 105)]  # a quadrilateral
    H = solve_homography(src, dst)
    for (x, y), (u, v) in zip(src, dst):
        pu, pv, pw = H @ [x, y, 1]
        assert abs(pu / pw - u) < 1e-6
        assert abs(pv / pw - v) < 1e-6


def test_warp_moves_content():
    img = Image.new("L", (200, 200), 255)
    img.paste(0, (60, 60, 140, 140))  # black square
    # shift everything by (+20, +10)
    src = [(0, 0), (200, 0), (200, 200), (0, 200)]
    dst = [(20, 10), (220, 10), (220, 210), (20, 210)]
    H = solve_homography(src, dst)
    out = warp_perspective(img, H, (260, 260))
    a = np.array(out)
    assert a[100, 100] < 50   # was (80, 90) -> now (100, 100): ink
    assert a[50, 50] == 255   # background outside the moved square


def test_find_fiducials_positions_and_order(sheet_s2_png):
    img = Image.open(sheet_s2_png)
    fids = find_fiducials(img)
    expected = sorted([(100, 100), (SHEET_W - 100, 100),
                       (100, SHEET_H - 100), (SHEET_W - 100, SHEET_H - 100)])
    assert sorted(tuple(f) for f in fids) == expected
    # TL, TR, BR, BL ordering
    tl, tr, br, bl = fids
    assert tl[0] < tr[0] and tl[1] < bl[1]
    assert br[0] > bl[0] and br[1] > tr[1]
