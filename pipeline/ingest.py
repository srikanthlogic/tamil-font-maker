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

from .config import cell_params, load as load_config
from .gfx import despeckle, otsu
from .homography import IngestError, find_fiducials, solve_homography, warp_perspective
from .template import (BASELINE_Y, CELL_COLS, CELL_H, CELL_W, FID_C,
                       HEADLINE_Y, SHEET_H, SHEET_W, cell_box, sheet_cells)

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


def _extract_cells(sheet_img: Image.Image, gids: list[str], threshold: int,
                   overrides: dict[str, dict] | None = None):
    """Inner-crop each cell, binarize, drop empties. -> (artifacts, empty, low)

    A per-cell `threshold` override ([cells.<gid>] in config.toml) wins over
    the mode default."""
    artifacts, empty, low = {}, [], []
    for i, gid in enumerate(gids):
        row, col = divmod(i, CELL_COLS)
        x0, y0, x1, y1 = cell_box(row, col)
        crop = sheet_img.crop((x0 + WORK_INSET, y0 + WORK_INSET,
                               x1 - WORK_INSET, y1 - WORK_INSET))
        a = np.asarray(crop)
        th = (overrides or {}).get(gid, {}).get("threshold", threshold)
        ink = a < th            # dark strokes on light cell
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
           mode: str, source: str, preset: str | None,
           params_by_gid: dict[str, dict] | None = None) -> dict:
    for gid, img in artifacts.items():
        entry = {"source": source, "mode": mode, "preset": preset}
        params = (params_by_gid or {}).get(gid)
        if params:
            entry["params"] = params
        img.save(glyphs_dir / f"{gid}.png")
        _record(glyphs_dir, gid, entry)
    report = {"ingested_gids": sorted(artifacts), "empty": empty,
              "low_ink": low, "mode": mode}
    if params_by_gid:
        report["param_overrides"] = sorted(params_by_gid)
    return report


# --- the three modes ------------------------------------------------------------


def ingest_digital(project: Path, sheet_path: Path, glyphs_dir: Path | None = None) -> dict:
    sheet_path = Path(sheet_path)
    img = Image.open(sheet_path).convert("L")
    if img.size != (SHEET_W, SHEET_H):
        raise IngestError(
            f"digital sheet must be unmodified template: {img.size} != "
            f"{(SHEET_W, SHEET_H)}")
    gids = _sheet_cells_for(sheet_path.stem, Path(project) / "sheets")
    config = load_config(project)
    params = {g: p for g in gids if (p := cell_params(config, g))}
    artifacts, empty, low = _extract_cells(img, gids, DIGITAL_THRESHOLD, params)
    return _store(_glyphs_dir(project, glyphs_dir), artifacts, empty, low,
                  "digital", str(sheet_path.name), None, params)


def ingest_paper(project: Path, photo_path: Path, sheet: str,
                 glyphs_dir: Path | None = None) -> dict:
    photo = Image.open(Path(photo_path)).convert("L")
    corners = find_fiducials(photo)
    target = [(FID_C, FID_C), (SHEET_W - FID_C, FID_C),
              (SHEET_W - FID_C, SHEET_H - FID_C), (FID_C, SHEET_H - FID_C)]
    H = solve_homography(corners, target)
    warped = warp_perspective(photo, H, (SHEET_W, SHEET_H))
    gids = _sheet_cells_for(sheet, Path(project) / "sheets")
    config = load_config(project)
    params = {g: p for g in gids if (p := cell_params(config, g))}
    artifacts, empty, low = _extract_cells(
        warped, gids, _threshold_for(warped, None), params)
    return _store(_glyphs_dir(project, glyphs_dir), artifacts, empty, low,
                  "paper", Path(photo_path).name, None, params)


_PRESETS = {
    "faithful": {"min_area": 30, "threshold": None},
    "clean": {"min_area": 80, "threshold": None},
}


def ingest_extract(project: Path, crops: dict[str, Path], preset: str,
                   glyphs_dir: Path | None = None,
                   scale_to_body: bool = True) -> dict:
    if preset not in _PRESETS:
        raise IngestError(f"unknown preset {preset!r}")
    glyphs_dir = _glyphs_dir(project, glyphs_dir)
    config = load_config(project)
    cleaned: dict[str, Image.Image] = {}
    params_by_gid: dict[str, dict] = {}
    for gid, path in crops.items():
        params = cell_params(config, gid)
        if params:
            params_by_gid[gid] = params
        img = Image.open(Path(path)).convert("L")
        # no OTSU clamp here: extract crops have no template guides, and the
        # clamp would admit mid-gray background as ink
        t = _threshold_for(img, params.get("threshold"), clamp=None)
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
        binary = despeckle(
            binary, min_area=params.get("despeckle",
                                        _PRESETS[preset]["min_area"]))
        cleaned[gid] = binary

    # No batch size normalization here: natural letter heights DIFFER
    # (ஃ is short, combos tall) and forcing a median distorts them. Size
    # consistency is the crop stage's job — cut equal-height boxes for
    # letters that share an optical height.
    #
    # One scale correction IS wanted: match the body. Extracted letters
    # otherwise land ~1.4x taller than the backfilled body text (their crops
    # get upscaled to fill the working raster), which reads as slop the
    # moment title letters are typed inside a sentence. Reference height =
    # median ink height of already-present uyir/consonant glyphs (i.e. run
    # `backfill` BEFORE `ingest-extract`).
    target_body_h = None
    if scale_to_body:
        from .mapping import cell as _cell
        body_heights = []
        for p in glyphs_dir.glob("g_*.png"):
            try:
                if _cell(p.stem).kind not in ("uyir", "consonant"):
                    continue
            except KeyError:
                continue
            m = np.asarray(Image.open(p)) > 127
            ys, xs = np.nonzero(m)
            if ys.size:
                body_heights.append(int(ys.max() - ys.min() + 1))
        if len(body_heights) >= 3:
            target_body_h = float(np.median(body_heights))

    artifacts, empty, low = {}, [], []
    for gid in crops:
        binary = cleaned[gid]
        m = np.asarray(binary) > 127
        if not m.any():
            empty.append(gid)
            continue
        ys, xs = np.nonzero(m)
        ink = binary.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        # body-match when a body reference exists (backfill-first workflow);
        # otherwise just ensure enough resolution for a clean trace
        target = target_body_h if target_body_h else max(ink.height, 200)
        if abs(ink.height - target) > 2:
            s = target / ink.height
            ink = ink.resize((max(1, int(ink.width * s)),
                              max(1, int(ink.height * s))), Image.LANCZOS)
        # place like backfill does: baseline-anchored paste, NO upscaling —
        # filling the raster height would break the body match
        max_h, max_w = int(WORK_H * 0.92), WORK_W - 8
        if ink.height > max_h or ink.width > max_w:
            s2 = min(max_h / ink.height, max_w / ink.width)
            ink = ink.resize((max(1, int(ink.width * s2)),
                              max(1, int(ink.height * s2))), Image.LANCZOS)
        raster = Image.new("L", (WORK_W, WORK_H), 0)
        from .mapping import cell as _cell
        bottom = BASELINE_IN_RASTER
        try:
            if _cell(gid).kind == "sign":
                bottom = HEADLINE_IN_RASTER + 80
        except KeyError:
            pass
        px = (WORK_W - ink.width) // 2
        py = max(4, bottom - ink.height)
        raster.paste(255, (px, py, px + ink.width, py + ink.height), mask=ink)
        frac = m.mean()
        if frac < LOW_INK_FRAC:
            low.append(gid)
        artifacts[gid] = raster
    report = _store(glyphs_dir, artifacts, empty, low, "extract", "", preset,
                    params_by_gid)
    return report
