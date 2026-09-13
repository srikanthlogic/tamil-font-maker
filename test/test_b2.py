"""Bootstrap B2 (spec §10): paper mode under synthetic photo distortion —
perspective, rotation, noise, uneven light — must recover all 296 cells."""
import numpy as np
import pytest
from PIL import Image

from pipeline import build as build_mod
from pipeline import ingest, template, trace, verify as verify_mod
from test_b1 import _fill_all_cells, _norm_iou, _ref_render
from pipeline.gfx import render_text
from pipeline.mapping import CELLS
from test_ingest import _synthesize_photo

pytestmark = pytest.mark.slow


def test_b2_paper_roundtrip(tmp_path):
    proj = tmp_path
    sheets = proj / "sheets"
    template.generate(sheets)
    _fill_all_cells(sheets, {})

    paper = proj / "glyphs_paper"
    for i in range(1, 15):
        photo = tmp_path / f"photo_S{i}.png"
        _synthesize_photo(sheets / f"S{i}.png", photo, seed=100 + i)
        r = ingest.ingest_paper(proj, photo, sheet=f"S{i}", glyphs_dir=paper)
        assert not r["empty"], f"S{i}: {r['empty'][:4]}"

    t = trace.trace(proj, glyphs_dir=paper)
    assert t["traced"] == 296 and not t["failed"], t["failed"][:5]

    b = build_mod.build(proj)
    assert b["gsub_rules"] == 270
    assert b["coverage_pct"] == 100.0

    v = verify_mod.verify(proj, glyphs_dir=paper)
    assert v["verdict"] == "green", v["failed_gates"]

    # shape check vs reference on a sample of cells (paper noise costs some
    # fidelity; 0.6 floor catches systematic deskew failures)
    font_path = proj / "out" / "TamilMaker.ttf"
    worst = []
    sample = CELLS[::24]  # ~13 cells across all sheets/kinds
    for c in sample:
        text = "".join(chr(cp) for cp in c.cps)
        got = render_text(str(font_path), text, 300)
        ref = _ref_render(text, 300)
        iou = _norm_iou(got, ref)
        if iou < 0.6:
            worst.append((c.gid, round(iou, 3)))
    assert not worst, f"paper-recovered cells below IoU 0.6: {worst}"
