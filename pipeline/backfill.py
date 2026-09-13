"""Backfill stage: complete a partial (extracted) font by rendering all
missing cells from a base font (e.g. Noto Sans Tamil). Title letters stay
extracted art; everything else becomes clean body text."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from .gfx import rasterize_glyphs, render_text
from .ingest import BASELINE_IN_RASTER, HEADLINE_IN_RASTER, WORK_H, WORK_W, _record
from .mapping import CELLS


def _ref_render(font_path: str, text: str, px: int = 300) -> Image.Image:
    """Shaped render; lone combining marks render outline-directly (a shaped
    single mark would get HarfBuzz's dotted circle)."""
    from fontTools.ttLib import TTFont

    if len(text) == 1 and 0x0BBE <= ord(text) <= 0x0BCD:
        tt = TTFont(font_path)
        glyph_set = tt.getGlyphSet()
        name = tt.getBestCmap()[ord(text)]
        gid = tt.getGlyphOrder().index(name)
        upm = tt["head"].unitsPerEm
        asc, desc = tt["hhea"].ascent, tt["hhea"].descent
        canvas = (int((asc - desc) * px / upm) + 40, int((asc - desc) * px / upm) + 40)
        return rasterize_glyphs(tt, [(gid, 0, 0)], px, canvas,
                                (20, 20 + asc * px / upm))
    return render_text(font_path, text, px)


def backfill(project: Path, font_path: str, px: int = 300,
             glyphs_dir: Path | None = None) -> dict:
    project = Path(project)
    glyphs = Path(glyphs_dir) if glyphs_dir else project / "glyphs"
    glyphs.mkdir(exist_ok=True)
    filled, backfilled, failed = 0, [], []
    for c in CELLS:
        out = glyphs / f"{c.gid}.png"
        if out.exists():
            filled += 1
            continue
        try:
            text = "".join(chr(cp) for cp in c.cps)
            ref = _ref_render(font_path, text, px)
            a = np.array(ref) > 127
            if not a.any() and c.cps[-1] == 0x0BCD:
                ref = Image.new("L", (200, 200), 0)
                from PIL import ImageDraw
                ImageDraw.Draw(ref).ellipse([76, 76, 124, 124], fill=255)
                a = np.array(ref) > 127
            if not a.any():
                failed.append({"gid": c.gid, "reason": "empty render"})
                continue
            ys, xs = np.nonzero(a)
            ink = ref.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
            # fit into the working raster: 360x480, baseline at 400
            max_h, max_w = int(WORK_H * 0.8), WORK_W - 8
            if ink.height > max_h or ink.width > max_w:
                s = min(max_h / ink.height, max_w / ink.width)
                ink = ink.resize((max(1, int(ink.width * s)),
                                  max(1, int(ink.height * s))), Image.LANCZOS)
            raster = Image.new("L", (WORK_W, WORK_H), 0)
            bottom = BASELINE_IN_RASTER if c.kind != "sign" else HEADLINE_IN_RASTER + 80
            px_x = (WORK_W - ink.width) // 2
            px_y = max(4, bottom - ink.height)
            raster.paste(255, (px_x, px_y, px_x + ink.width, px_y + ink.height),
                         mask=ink)
            raster.save(out)
            _record(glyphs, c.gid, {"source": Path(font_path).name,
                                    "mode": "backfill", "preset": None})
            backfilled.append(c.gid)
        except Exception as exc:  # noqa: BLE001 — per-cell isolation
            failed.append({"gid": c.gid, "reason": repr(exc)})
    return {"backfilled": len(backfilled), "already_present": filled,
            "failed": failed, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
