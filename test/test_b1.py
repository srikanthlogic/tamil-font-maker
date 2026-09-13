"""Bootstrap B1 (spec §10): full 296-cell digital roundtrip with zero human
art. Noto Sans Tamil renders into our template cells; the pipeline must
rebuild a font whose rendering matches within pixel tolerance, with all
270 GSUB rules and green shaping gates."""
import pytest

from bootstrap import direct_render, fill_all_cells, norm_iou, ref_render
from pipeline import build, ingest, template, trace, verify as verify_mod
from pipeline.gfx import render_text
from pipeline.mapping import CELLS

pytestmark = pytest.mark.slow


def test_b1_full_roundtrip(tmp_path):
    proj = tmp_path
    sheets = proj / "sheets"
    template.generate(sheets)
    fill_all_cells(sheets, {})

    report = ingest.ingest_digital(proj, sheets / "S1.png")
    assert report["empty"] == [], "no cell should be empty in B1"
    for i in range(2, 15):
        ingest.ingest_digital(proj, sheets / f"S{i}.png")

    t = trace.trace(proj)
    assert t["traced"] == 296 and not t["failed"], t["failed"][:5]

    b = build.build(proj)
    assert b["gsub_rules"] == 270, "all ligature rules must be emitted"
    assert b["coverage_pct"] == 100.0

    v = verify_mod.verify(proj)
    assert v["verdict"] == "green", {"gates": v["failed_gates"],
                                     "missing": v["missing_count"]}
    assert v["qa_artifacts"]

    # per-glyph shape fidelity: rebuilt font render vs Noto source render.
    # Sign glyphs render outline-directly on both sides — shaping a lone
    # combining mark inserts a dotted circle (correct runtime behavior,
    # wrong comparison target).
    font_path = proj / "out" / "TamilMaker.ttf"
    worst = []
    ious = []
    for c in CELLS:
        text = "".join(chr(cp) for cp in c.cps)
        if c.kind == "sign":
            got = direct_render(str(font_path), c.cps[0], 300)
        else:
            got = render_text(str(font_path), text, 300)
        ref = ref_render(text, 300)
        iou = norm_iou(got, ref)
        ious.append(iou)
        if iou < 0.55:
            worst.append((c.gid, round(iou, 3)))
    # vectorization roundtrip loses some fidelity (binarize -> potrace ->
    # quadratics -> re-rasterize): the widest split-matra combos lose thin
    # connector strokes and bottom out ~0.58; truly broken cells (clipped,
    # shredded, misassigned) measure <0.35, so 0.55 separates them cleanly
    assert not worst, f"cells below IoU 0.55: {worst[:10]}"
    assert sum(ious) / len(ious) >= 0.82, f"mean IoU {sum(ious)/len(ious):.3f}"
