"""Build stage (spec §6.4): outlines -> TTF with cmap, spacing matras,
generated GSUB ligature feature, guide-derived metrics. WOFF2 if brotli."""
from __future__ import annotations

import io
import json
from pathlib import Path

from fontTools.fontBuilder import FontBuilder
from fontTools.feaLib.builder import Builder
from fontTools.feaLib.parser import Parser
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

from .mapping import CELLS, UPM, ligature_rules

ASCENT, DESCENT = 1500, -500
SPACE_ADVANCE = 160
DEFAULT_FAMILY = "TamilMaker"


def _load_config(project: Path) -> dict:
    cfg = project / "config.toml"
    if cfg.exists():
        import tomllib
        return tomllib.loads(cfg.read_text())
    return {}


def _glyph_from_outline(data: dict):
    pen = TTGlyphPen(None)
    for contour in data["contours"]:
        pen.moveTo(tuple(contour["start"]))
        for seg in contour["segs"]:
            if "l" in seg:
                pen.lineTo(tuple(seg["l"]))
            else:
                cx, cy, ex, ey = seg["q"]
                pen.qCurveTo((cx, cy), (ex, ey))
        pen.closePath()
    return pen.glyph()


def _fea(rules: list[tuple[tuple[int, ...], str]]) -> str:
    lines = ["languagesystem taml dflt;", "lookup LIGA_LIG {"]
    for seq, target in rules:
        names = " ".join(f"g_u{cp:04X}" for cp in seq)
        lines.append(f"  sub {names} by {target};")
    lines.append("} LIGA_LIG;")
    # Empirically decided (spec §6.4): HarfBuzz's Tamil complex shaper does
    # not apply generic 'liga'; it does always apply 'ccmp'. Register both.
    lines.append("feature ccmp { lookup LIGA_LIG; } ccmp;")
    lines.append("feature liga { lookup LIGA_LIG; } liga;")
    return "\n".join(lines)


def build(project: Path) -> dict:
    project = Path(project)
    config = _load_config(project)
    family = config.get("family", DEFAULT_FAMILY)
    psname = config.get("psname", "".join(
        ch if ch.isalnum() else "" for ch in family) or DEFAULT_FAMILY)

    outlines_dir = project / "outlines"
    available = {p.stem for p in outlines_dir.glob("g_*.json")}
    order_cells = [c for c in CELLS if c.gid in available]

    glyph_order = [".notdef"] + [c.gid for c in order_cells] + ["g_u0020"]
    glyphs = {".notdef": TTGlyphPen(None).glyph(),
              "g_u0020": TTGlyphPen(None).glyph()}
    metrics = {".notdef": (400, 0), "g_u0020": (SPACE_ADVANCE, 0)}
    for c in order_cells:
        data = json.loads((outlines_dir / f"{c.gid}.json").read_text())
        glyphs[c.gid] = _glyph_from_outline(data)
        advance = int(round(data.get("advance_hint") or 400))
        metrics[c.gid] = (max(advance, 80), int(round(data.get("lsb", 0))))

    cmap: dict[int, str] = {0x0020: "g_u0020"}
    for c in order_cells:
        if c.cmap_cp is not None:
            cmap[c.cmap_cp] = c.gid

    fb = FontBuilder(UPM, isTTF=True)
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=ASCENT, descent=DESCENT)
    fb.setupNameTable({"familyName": family, "styleName": "Regular",
                       "fullName": f"{family} Regular",
                       "psName": f"{psname}-Regular"})
    fb.setupOS2(sTypoAscender=ASCENT, sTypoDescender=DESCENT,
                usWinAscent=2000, usWinDescent=800)  # win must cover full ink:
    # extracted title art can rise above the typo ascent, and some Windows
    # renderers clip at usWinAscent
    fb.setupPost()

    # GSUB: only rules whose precomposed glyph (and sequence members) exist
    emitted = []
    present = set(glyph_order)
    for seq, target in ligature_rules():
        if target in present and all(f"g_u{cp:04X}" in present for cp in seq):
            emitted.append((seq, target))
    fea = _fea(emitted)
    glyph_names = set(glyph_order)
    fea_ast = Parser(io.StringIO(fea), glyphNames=glyph_names).parse()
    Builder(fb.font, fea_ast).build()
    font = fb.font

    out = project / "out"
    out.mkdir(exist_ok=True)
    ttf_path = out / f"{family}.ttf"
    font.save(str(ttf_path))

    try:
        import brotli  # noqa: F401
        w = TTFont(str(ttf_path))
        w.flavor = "woff2"
        w.save(str(out / f"{family}.woff2"))
        woff2 = True
    except ImportError:
        woff2 = False

    covered = len(order_cells)   # all drawn glyphs (incl. ligature-only cells)
    return {"family": family, "ttf": str(ttf_path), "woff2": woff2,
            "glyphs": len(glyph_order) - 2, "coverage": covered,
            "coverage_pct": round(100 * covered / (len(CELLS)), 1),
            "gsub_rules": len(emitted)}
