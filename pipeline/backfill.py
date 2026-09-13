"""Backfill stage: complete a partial (extracted) font by rendering all
missing cells from a base font (e.g. Noto Sans Tamil). Title letters stay
extracted art; everything else becomes clean body text."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from .gfx import render_ref
from .ingest import BASELINE_IN_RASTER, HEADLINE_IN_RASTER, WORK_H, WORK_W, _record
from .mapping import CELLS


def backfill(project: Path, font_path: str, px: int = 300,
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
            "failed": failed, "synthetic_dot": synthetic_dot,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
