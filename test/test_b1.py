"""Bootstrap B1 (spec §10): full 296-cell digital roundtrip with zero human
art. Noto Sans Tamil renders into our template cells; the pipeline must
rebuild a font whose rendering matches within pixel tolerance, with all
270 GSUB rules and green shaping gates."""
import json
import os

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pipeline import build, ingest, template, trace, verify as verify_mod
from pipeline.gfx import render_text
from pipeline.mapping import CELLS, cell

FIX_FONT = "test/fixtures/NotoSansTamil.ttf"

pytestmark = pytest.mark.slow


def _ref_render(text: str, px: int = 300):
    """Render reference ink for a cell. Combining marks (signs) are rendered
    directly from the glyph outline — shaping a lone matra would make
    HarfBuzz insert a dotted circle."""
    from fontTools.ttLib import TTFont
    from pipeline.gfx import rasterize_glyphs

    if len(text) == 1 and 0x0BBE <= ord(text) <= 0x0BCD:
        tt = TTFont(FIX_FONT)
        gid_name = tt.getBestCmap()[ord(text)]
        gid = tt.getGlyphOrder().index(gid_name)
        upm = tt["head"].unitsPerEm
        asc, desc = tt["hhea"].ascent, tt["hhea"].descent
        canvas = (int((asc - desc) * px / upm) + 40,
                  int((asc - desc) * px / upm) + 40)
        img = rasterize_glyphs(tt, [(gid, 0, 0)], px, canvas,
                               (20, 20 + asc * px / upm))
        if not (np.array(img) > 127).any() and ord(text) == 0x0BCD:
            # Noto's standalone pulli glyph is empty by design; a hand-drawn
            # template cell would contain a simple dot
            img = Image.new("L", (200, 200), 0)
            ImageDraw.Draw(img).ellipse([76, 76, 124, 124], fill=255)
        return img
    return render_text(FIX_FONT, text, px)


def _fill_all_cells(sheets_dir, work):
    """Rasterize each cell's text from Noto into its template cell."""
    for c in CELLS:
        text = "".join(chr(cp) for cp in c.cps)
        ref = _ref_render(text)
        a = np.array(ref) > 127
        if not a.any():
            # Noto's standalone pulli (்) glyph is empty by design; a
            # hand-drawn template cell would contain a simple dot
            if c.cps[-1] == 0x0BCD:
                ref = Image.new("L", (200, 200), 0)
                ImageDraw.Draw(ref).ellipse([76, 76, 124, 124], fill=255)
            else:
                raise RuntimeError(f"fixture render empty for {c.gid}")
            a = np.array(ref) > 127
        ys, xs = np.nonzero(a)
        ref = ref.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        sheet_img = work.setdefault(c.sheet, Image.open(sheets_dir / f"{c.sheet}.png").convert("L"))
        from pipeline.template import BASELINE_Y, HEADLINE_Y, cell_box
        x0, y0, x1, y1 = cell_box(c.sheet, c.row, c.col)
        # seat ink bottom on the baseline (signs float around the headline)
        max_h = BASELINE_Y - HEADLINE_Y + 60
        max_w = x1 - x0 - 24
        if ref.height > max_h or ref.width > max_w:
            # re-render at fitted size instead of shrinking binary art —
            # preserves stroke weight for wide split-matra combos
            fit = min(max_h / ref.height, max_w / ref.width)
            bigger = _ref_render(text, max(48, int(300 * fit)))
            ba = np.array(bigger) > 127
            if ba.any():
                bys, bxs = np.nonzero(ba)
                ref = bigger.crop((bxs.min(), bys.min(), bxs.max() + 1, bys.max() + 1))
        bottom = BASELINE_Y if c.kind != "sign" else HEADLINE_Y + 80
        px = x0 + ((x1 - x0) - ref.width) // 2
        py = max(y0 + 40, y0 + bottom - ref.height)  # keep tall matras inside
        ink = Image.fromarray(np.where(np.array(ref) > 127, 255, 0).astype(np.uint8))
        sheet_img.paste(0, (px, py, px + ref.width, py + ref.height), mask=ink)
    for name, img in work.items():
        img.save(sheets_dir / f"{name}.png")


def _norm_iou(a_img, b_img):
    a = np.array(a_img) > 127
    b = np.array(b_img) > 127
    for m in (a, b):
        if not m.any():
            return 0.0
    def crop(m):
        ys, xs = np.nonzero(m)
        return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    ca, cb = crop(a), crop(b)
    cb = np.array(Image.fromarray(cb.astype(np.uint8) * 255).resize(
        (ca.shape[1], ca.shape[0]), Image.LANCZOS)) > 127
    return (ca & cb).sum() / (ca | cb).sum()


def _direct_render(font_path: str, cp: int, px: int = 150):
    """Render one glyph by outline (no shaping) — used for combining marks,
    where shaped rendering would insert a dotted circle."""
    from fontTools.ttLib import TTFont
    from pipeline.gfx import rasterize_glyphs

    tt = TTFont(font_path)
    gid_name = tt.getBestCmap()[cp]
    gid = tt.getGlyphOrder().index(gid_name)
    upm = tt["head"].unitsPerEm
    asc, desc = tt["hhea"].ascent, tt["hhea"].descent
    canvas = (int((asc - desc) * px / upm) + 40, int((asc - desc) * px / upm) + 40)
    return rasterize_glyphs(tt, [(gid, 0, 0)], px, canvas,
                            (20, 20 + asc * px / upm))


def test_b1_full_roundtrip(tmp_path):
    proj = tmp_path
    sheets = proj / "sheets"
    template.generate(sheets)
    _fill_all_cells(sheets, {})

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
            got = _direct_render(str(font_path), c.cps[0], 300)
        else:
            got = render_text(str(font_path), text, 300)
        ref = _ref_render(text, 300)
        iou = _norm_iou(got, ref)
        ious.append(iou)
        if iou < 0.55:
            worst.append((c.gid, round(iou, 3)))
    # vectorization roundtrip loses some fidelity (binarize -> potrace ->
    # quadratics -> re-rasterize): the widest split-matra combos lose thin
    # connector strokes and bottom out ~0.58; truly broken cells (clipped,
    # shredded, misassigned) measure <0.35, so 0.55 separates them cleanly
    assert not worst, f"cells below IoU 0.55: {worst[:10]}"
    assert sum(ious) / len(ious) >= 0.82, f"mean IoU {sum(ious)/len(ious):.3f}"
