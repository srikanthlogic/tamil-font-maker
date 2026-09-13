"""Trace stage (spec §6.3): binary glyph raster -> quadratic outlines in em
units. Baseline maps to y=0; the headline guide maps to EM_ASCENT."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import potrace
from cu2qu import curve_to_quadratic
from PIL import Image

from .ingest import BASELINE_IN_RASTER
from .template import BASELINE_Y, HEADLINE_Y

GUIDE_PX = BASELINE_Y - HEADLINE_Y     # 320 px between guides
EM_ASCENT = 1400                       # units from baseline to headline (2048 upm)
SCALE = EM_ASCENT / GUIDE_PX           # em units per pixel
CU2QU_ERR = 4.0                        # ~1 px in em units
RSB_FLOOR = 80


def _pt(p):
    return (p.x * SCALE, (BASELINE_IN_RASTER - p.y) * SCALE)


def _flatten_quad(p0, c, p1, steps=8):
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * c[0] + t ** 2 * p1[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * c[1] + t ** 2 * p1[1])
            for t in (i / steps for i in range(1, steps + 1))]


def _shoelace(poly):
    s = 0.0
    for i in range(len(poly)):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % len(poly)]
        s += x0 * y1 - x1 * y0
    return 0.5 * s


def _fix_orientations(contours):
    """After the y-flip, potrace's windings invert; TrueType needs outer
    contours clockwise (negative shoelace) and holes counter-clockwise."""
    bboxes = []
    for c in contours:
        xs = [p[0] for p in c["poly"]]
        ys = [p[1] for p in c["poly"]]
        bboxes.append((min(xs), min(ys), max(xs), max(ys)))
    for i, c in enumerate(contours):
        depth = 0
        xi0, yi0, xi1, yi1 = bboxes[i]
        for j, (xj0, yj0, xj1, yj1) in enumerate(bboxes):
            if i == j:
                continue
            if xj0 < xi0 and yj0 < yi0 and xj1 > xi1 and yj1 > yi1:
                depth += 1
        area = _shoelace(c["poly"])
        outer_cw = depth % 2 == 0
        if outer_cw and area > 0 or not outer_cw and area < 0:
            _reverse(c)


def _reverse(c):
    """Reverse contour direction, keeping quads valid (a quadratic reversed
    is the same curve: (p0,c,p1) -> (p1,c,p0))."""
    start = tuple(c["start"])
    ends = [tuple(s["l"]) if "l" in s else (s["q"][2], s["q"][3])
            for s in c["segs"]]
    new_segs = []
    for idx in range(len(c["segs"]) - 1, -1, -1):
        s = c["segs"][idx]
        prior = start if idx == 0 else ends[idx - 1]
        if "l" in s:
            new_segs.append({"l": [prior[0], prior[1]]})
        else:
            new_segs.append({"q": [s["q"][0], s["q"][1], prior[0], prior[1]]})
    c["start"] = list(ends[-1])
    c["segs"] = new_segs
    c["poly"] = list(reversed(c["poly"]))


def trace_glyph_raster(arr: np.ndarray, turdsize: int = 10,
                       alphamax: float = 1.0) -> dict:
    """arr: L-mode raster (ink=255). Returns contours with quadratic segs.

    Note: potracer traces the *background* of the array it is given, so the
    bitmap is passed inverted.
    """
    bmp = potrace.Bitmap(~(arr > 127))
    paths = bmp.trace(turdsize=turdsize, alphamax=alphamax, opticurve=0,
                      opttolerance=0.2)
    contours = []
    xs, ys = [], []
    for curve in paths:
        start = _pt(curve.start_point)
        segs, poly = [], [start]
        cur = start
        for seg in curve.segments:
            if seg.is_corner:
                e = _pt(seg.end_point)
                segs.append({"l": [e[0], e[1]]})
                poly += [e]
                cur = e
            else:
                c1, c2 = _pt(seg.c1), _pt(seg.c2)
                e = _pt(seg.end_point)
                pts = curve_to_quadratic((cur, c1, c2, e), CU2QU_ERR)
                # cu2qu returns a flat [p0, ctrl, on, ctrl, on, ...] list
                for i in range(1, len(pts), 2):
                    c, p1 = pts[i], pts[i + 1]
                    segs.append({"q": [c[0], c[1], p1[0], p1[1]]})
                    poly += _flatten_quad(pts[i - 1], c, p1)
                cur = e
        contours.append({"start": [start[0], start[1]], "segs": segs,
                         "poly": poly})
        for x, y in poly:
            xs.append(x)
            ys.append(y)
    _fix_orientations(contours)
    xmin, xmax = min(xs), max(xs)
    return {"contours": contours,
            "lsb": round(xmin, 1),
            "xmin": round(xmin, 1), "xmax": round(xmax, 1),
            "ymin": round(min(ys), 1), "ymax": round(max(ys), 1),
            "advance_hint": round(xmax + RSB_FLOOR, 1)}


def trace(project: Path, glyphs_dir: Path | None = None) -> dict:
    project = Path(project)
    glyphs = Path(glyphs_dir) if glyphs_dir else project / "glyphs"
    outlines = project / "outlines"
    outlines.mkdir(exist_ok=True)
    traced, failed = [], []
    for png in sorted(glyphs.glob("g_*.png")):
        try:
            arr = np.asarray(Image.open(png).convert("L"))
            result = trace_glyph_raster(arr)
            if not result["contours"]:
                failed.append({"gid": png.stem, "reason": "no contours"})
                continue
            (outlines / f"{png.stem}.json").write_text(json.dumps(result))
            traced.append(png.stem)
        except Exception as exc:  # noqa: BLE001 — per-cell isolation by design
            failed.append({"gid": png.stem, "reason": repr(exc)})
    return {"traced": len(traced), "gids": traced, "failed": failed}
