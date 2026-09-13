"""Ingest stage (spec §6.2): digital, paper, extract — one glyph format.

Glyph artifact: inner-cell crop (WORK_INSET px inside the cell box),
360x480, mode L, ink=255. Baseline sits at BASELINE_IN_RASTER within it.
"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

from .gfx import despeckle, fit_to_box, otsu
from .homography import IngestError, find_fiducials, solve_homography, warp_perspective
from .template import (BASELINE_Y, CELL_H, CELL_W, FID_C, HEADLINE_Y,
                       SHEET_H, SHEET_W, sheet_cells)

WORK_INSET = 20
WORK_W = CELL_W - 2 * WORK_INSET          # 360
WORK_H = CELL_H - 2 * WORK_INSET          # 480
BASELINE_IN_RASTER = BASELINE_Y - WORK_INSET   # 400
HEADLINE_IN_RASTER = HEADLINE_Y - WORK_INSET   # 80

DIGITAL_THRESHOLD = 190   # template grays: guides (200) stay background,
                          # antialiased ink edges are kept (less thinning)
OTSU_MAX = 170            # clamp for paper: guides must stay background
EMPTY_INK_FRAC = 0.0015   # small punctuation (~300 px) must still register
LOW_INK_FRAC = 0.02


# --- project manifest ---------------------------------------------------------


def _glyphs_dir(project: Path, glyphs_dir: Path | None) -> Path:
    d = Path(glyphs_dir) if glyphs_dir else Path(project) / "glyphs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def manifest_of(project: Path) -> dict:
    p = Path(project) / "glyphs" / "manifest.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _write_manifest(glyphs_dir: Path, manifest: dict):
    (glyphs_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))


def _record(glyphs_dir: Path, gid: str, entry: dict):
    manifest = {}
    mp = glyphs_dir / "manifest.json"
    if mp.exists():
        manifest = json.loads(mp.read_text())
    prev = manifest.pop(gid, None)
    if prev:
        entry["history"] = ([prev] + prev.pop("history", []))[:5]
    entry["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest[gid] = entry
    _write_manifest(glyphs_dir, manifest)


# --- shared cell extraction ----------------------------------------------------


def _sheet_cells_for(sheet: str, sheets_dir: Path) -> list[str]:
    """Ordered gids for a sheet — completion sheets come from the manifest."""
    mpath = Path(sheets_dir) / "manifest.json"
    if mpath.exists():
        m = json.loads(mpath.read_text())
        if sheet in m:
            return m[sheet]
    sc = sheet_cells()
    if sheet not in sc:
        raise IngestError(f"unknown sheet {sheet!r}")
    return [c.gid for c in sc[sheet]]


def _extract_cells(sheet_img: Image.Image, gids: list[str], threshold: int):
    """Inner-crop each cell, binarize, drop empties. -> (artifacts, empty, low)"""
    from .template import GAP_H, GAP_X, MARGIN_X, MARGIN_Y

    artifacts, empty, low = {}, [], []
    for i, gid in enumerate(gids):
        row, col = divmod(i, 5)
        bx0 = MARGIN_X + col * (CELL_W + GAP_X)
        by0 = MARGIN_Y + row * (CELL_H + GAP_H)
        crop = sheet_img.crop((bx0 + WORK_INSET, by0 + WORK_INSET,
                               bx0 + CELL_W - WORK_INSET,
                               by0 + CELL_H - WORK_INSET))
        a = np.asarray(crop)
        ink = a < threshold            # dark strokes on light cell
        frac = ink.mean()
        if frac < EMPTY_INK_FRAC:
            empty.append(gid)
            continue
        if frac < LOW_INK_FRAC:
            low.append(gid)
        artifacts[gid] = Image.fromarray(
            np.where(ink, 255, 0).astype(np.uint8), "L")
    return artifacts, empty, low


def _threshold_for(img: Image.Image, fixed: int | None,
                   clamp: int | None = OTSU_MAX) -> int:
    if fixed is not None:
        return fixed
    t = otsu(img)
    return min(t, clamp) if clamp is not None else t


def _store(glyphs_dir: Path, artifacts: dict, empty: list, low: list,
           mode: str, source: str, preset: str | None) -> dict:
    for gid, img in artifacts.items():
        img.save(glyphs_dir / f"{gid}.png")
        _record(glyphs_dir, gid, {"source": source, "mode": mode,
                                  "preset": preset})
    return {"ingested_gids": sorted(artifacts), "empty": empty,
            "low_ink": low, "mode": mode}


# --- the three modes ------------------------------------------------------------


def ingest_digital(project: Path, sheet_path: Path, glyphs_dir: Path | None = None) -> dict:
    sheet_path = Path(sheet_path)
    img = Image.open(sheet_path).convert("L")
    if img.size != (SHEET_W, SHEET_H):
        raise IngestError(
            f"digital sheet must be unmodified template: {img.size} != "
            f"{(SHEET_W, SHEET_H)}")
    gids = _sheet_cells_for(sheet_path.stem, Path(project) / "sheets")
    artifacts, empty, low = _extract_cells(img, gids, DIGITAL_THRESHOLD)
    return _store(_glyphs_dir(project, glyphs_dir), artifacts, empty, low,
                  "digital", str(sheet_path.name), None)


def ingest_paper(project: Path, photo_path: Path, sheet: str,
                 glyphs_dir: Path | None = None) -> dict:
    photo = Image.open(Path(photo_path)).convert("L")
    corners = find_fiducials(photo)
    target = [(FID_C, FID_C), (SHEET_W - FID_C, FID_C),
              (SHEET_W - FID_C, SHEET_H - FID_C), (FID_C, SHEET_H - FID_C)]
    H = solve_homography(corners, target)
    warped = warp_perspective(photo, H, (SHEET_W, SHEET_H))
    gids = _sheet_cells_for(sheet, Path(project) / "sheets")
    artifacts, empty, low = _extract_cells(warped, gids, _threshold_for(warped, None))
    return _store(_glyphs_dir(project, glyphs_dir), artifacts, empty, low,
                  "paper", Path(photo_path).name, None)


_PRESETS = {
    "faithful": {"min_area": 30, "threshold": None},
    "clean": {"min_area": 80, "threshold": None},
}


def ingest_extract(project: Path, crops: dict[str, Path], preset: str,
                   glyphs_dir: Path | None = None) -> dict:
    if preset not in _PRESETS:
        raise IngestError(f"unknown preset {preset!r}")
    glyphs_dir = _glyphs_dir(project, glyphs_dir)
    cleaned: dict[str, Image.Image] = {}
    for gid, path in crops.items():
        img = Image.open(Path(path)).convert("L")
        # no OTSU clamp here: extract crops have no template guides, and the
        # clamp would admit mid-gray background as ink
        t = _threshold_for(img, None, clamp=None)
        a = np.asarray(img)
        # polarity auto-detect: posters carry light lettering on dark ground
        # as often as the reverse. Background always touches the crop borders
        # extensively; letter ink mostly does not — pick the polarity whose
        # ink has the LOWER border-contact ratio.
        def _border_contact(mask):
            b = np.zeros_like(mask)
            b[0, :] = b[-1, :] = b[:, 0] = b[:, -1] = True
            return (mask & b).sum() / max(b.sum(), 1)

        dark = a < t
        light = ~dark
        polarity = dark if _border_contact(dark) <= _border_contact(light) \
            else light
        binary = Image.fromarray(np.where(polarity, 255, 0).astype(np.uint8), "L")
        binary = despeckle(binary, min_area=_PRESETS[preset]["min_area"])
        cleaned[gid] = binary

    # No batch size normalization here: natural letter heights DIFFER
    # (ஃ is short, combos tall) and forcing a median distorts them. Size
    # consistency is the crop stage's job — cut equal-height boxes for
    # letters that share an optical height.
    artifacts, empty, low = {}, [], []
    for gid in crops:
        binary = cleaned[gid]
        m = np.asarray(binary) > 127
        if not m.any():
            empty.append(gid)
            continue
        ys, xs = np.nonzero(m)
        ink = binary.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        normalized = fit_to_box(ink, (WORK_W, WORK_H),
                                anchor_baseline_frac=BASELINE_IN_RASTER / WORK_H)
        frac = m.mean()
        if frac < LOW_INK_FRAC:
            low.append(gid)
        artifacts[gid] = normalized
    report = _store(glyphs_dir, artifacts, empty, low, "extract", "", preset)
    return report
