"""Backfill stage: complete a partial (extracted) font by rendering all
missing cells from a base font (e.g. Noto Sans Tamil). Title letters stay
extracted art; everything else becomes clean body text."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from .gfx import ink_extents, render_ref
from .ingest import BASELINE_IN_RASTER, HEADLINE_IN_RASTER, WORK_H, WORK_W, _record
from .mapping import CELLS, UPM
from .trace import EM_ASCENT, GUIDE_PX

# Em-true raster size: the trace stage maps the template's GUIDE_PX guide band
# to EM_ASCENT em units, so one em of the base font must occupy
# UPM * GUIDE_PX / EM_ASCENT raster pixels for backfilled glyphs to carry
# their true relative size (300px here shrunk fonts to ~2/3 scale).
REF_PX = round(UPM * GUIDE_PX / EM_ASCENT)


def backfill(project: Path, font_path: str, px: int = REF_PX,
             glyphs_dir: Path | None = None) -> dict:
    project = Path(project)
    glyphs = Path(glyphs_dir) if glyphs_dir else project / "glyphs"
    glyphs.mkdir(exist_ok=True)
    filled, backfilled, failed, synthetic_dot = 0, [], [], []
    for c in CELLS:
        out = glyphs / f"{c.gid}.png"
        if out.exists():
            filled += 1
            continue
        try:
            text = "".join(chr(cp) for cp in c.cps)
            ref = render_ref(font_path, text, px)
            a = np.array(ref) > 127
            if not a.any() and c.cps[-1] == 0x0BCD:
                # base font's standalone pulli glyph is empty by design;
                # substitute a programmatic dot and SAY SO in the report
                ref = Image.new("L", (200, 200), 0)
                from PIL import ImageDraw
                ImageDraw.Draw(ref).ellipse([76, 76, 124, 124], fill=255)
                a = np.array(ref) > 127
                synthetic_dot.append(c.gid)
            if not a.any():
                failed.append({"gid": c.gid, "reason": "empty render"})
                continue
            ys, xs = np.nonzero(a)
            ink = ref.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
            px_above, px_below = ink_extents(font_path, text, px)
            # shrink only if the glyph cannot fit the cell height at
            # reference scale; width may exceed the cell — the raster is
            # just an intermediate and trace/build are raster-size agnostic,
            # so wide syllables keep their true em width
            scale = min(1.0, (BASELINE_IN_RASTER - 4) / max(px_above, 1.0),
                        (WORK_H - 4.0) / max(px_above + px_below, 1.0))
            if scale < 1.0:
                ink = ink.resize((max(1, int(ink.width * scale)),
                                  max(1, int(ink.height * scale))), Image.LANCZOS)
            raster_w = max(WORK_W, ink.width + 8)
            raster = Image.new("L", (raster_w, WORK_H), 0)
            px_x = (raster_w - ink.width) // 2
            if c.cps[-1] == 0x0BCD and c.gid in synthetic_dot:
                px_y = HEADLINE_IN_RASTER + 80
            else:
                # seat by BASELINE: ink's yMax sits on the baseline, so
                # descending strokes stay below it
                px_y = BASELINE_IN_RASTER - round(px_above * scale)
                px_y = min(max(4, px_y), WORK_H - 4 - ink.height)
            raster.paste(255, (px_x, px_y, px_x + ink.width, px_y + ink.height),
                         mask=ink)
            raster.save(out)
            _record(glyphs, c.gid, {"source": Path(font_path).name,
                                    "mode": "backfill", "preset": None})
            backfilled.append(c.gid)
        except Exception as exc:  # noqa: BLE001 — per-cell isolation
            failed.append({"gid": c.gid, "reason": repr(exc)})
    return {"backfilled": len(backfilled), "already_present": filled,
            "failed": failed, "synthetic_dot": synthetic_dot,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
