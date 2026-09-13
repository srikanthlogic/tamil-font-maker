"""Template sheet generation (spec §5): 14 A4 300dpi portrait sheets,
5x5 grids, in-cell guide lines, corner fiducials, completion mode."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .mapping import CELLS, Cell

# geometry (px @300dpi) — shared with ingest via cell_box
SHEET_W, SHEET_H = 2480, 3508          # A4 portrait @300dpi
CELL_W, CELL_H = 400, 520
HEADLINE_Y, BASELINE_Y = 100, 420      # within cell; 320px guide distance
MARGIN_X, MARGIN_Y = 160, 170
GAP_X, GAP_H = 20, 60
CELL_COLS = 5                          # 5x5 portrait grid
FID_C = 100                            # fiducial center inset from corners
FID_BLACK, FID_RING = 60, 100          # black square, white ring (sides)

CELLS_PER_SHEET = CELL_COLS * 5

_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}


def _label_font(size: int = 28):
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = ImageFont.load_default(size=size)
    return _FONT_CACHE[size]


def sheet_cells() -> dict[str, list[Cell]]:
    """sheet name -> ordered cells (mapping order, 25 per sheet)."""
    out: dict[str, list[Cell]] = {}
    for c in CELLS:
        out.setdefault(c.sheet, []).append(c)
    return out


def cell_box(sheet: str, row: int, col: int) -> tuple[int, int, int, int]:
    x0 = MARGIN_X + col * (CELL_W + GAP_X)
    y0 = MARGIN_Y + row * (CELL_H + GAP_H)
    return x0, y0, x0 + CELL_W, y0 + CELL_H


def _draw_fiducials(draw: ImageDraw.ImageDraw):
    for cx, cy in ((FID_C, FID_C), (SHEET_W - FID_C, FID_C),
                   (FID_C, SHEET_H - FID_C), (SHEET_W - FID_C, SHEET_H - FID_C)):
        draw.rectangle([cx - FID_RING // 2, cy - FID_RING // 2,
                        cx + FID_RING // 2, cy + FID_RING // 2], fill=255)
        draw.rectangle([cx - FID_BLACK // 2, cy - FID_BLACK // 2,
                        cx + FID_BLACK // 2, cy + FID_BLACK // 2], fill=0)


def _draw_cell(draw: ImageDraw.ImageDraw, cell_item: Cell, sheet: str,
               row: int, col: int):
    x0, y0, x1, y1 = cell_box(sheet, row, col)
    draw.rectangle([x0, y0, x1, y1], outline=120, width=2)
    draw.line([x0 + 4, y0 + HEADLINE_Y, x1 - 4, y0 + HEADLINE_Y], fill=200, width=2)
    draw.line([x0 + 4, y0 + BASELINE_Y, x1 - 4, y0 + BASELINE_Y], fill=200, width=2)
    cps = "+".join(f"U+{cp:04X}" for cp in cell_item.cps)
    draw.text((x0 + 4, y1 + 6), cell_item.gid, fill=0, font=_label_font())
    draw.text((x0 + 4, y1 + 38), cps, fill=90, font=_label_font(22))


def _new_sheet() -> Image.Image:
    img = Image.new("L", (SHEET_W, SHEET_H), 255)
    _draw_fiducials(ImageDraw.Draw(img))
    return img


def _label_of(c: Cell) -> str:
    return c.gid


def generate(sheets_dir: Path, only: set[str] | None = None) -> dict:
    """Write template sheets (or completion sheets when `only` is given).

    Full mode: S1..S14.png/.pdf per mapping. Completion mode: C1..Cn.png
    containing just `only` cells, re-gridded 5x5. Returns a report dict.
    """
    sheets_dir = Path(sheets_dir)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    if only is None:
        groups: dict[str, list[Cell]] = sheet_cells()
    else:
        selected = [c for c in CELLS if c.gid in only]
        groups = {}
        for i in range(0, len(selected), CELLS_PER_SHEET):
            groups[f"C{1 + i // CELLS_PER_SHEET}"] = selected[i:i + CELLS_PER_SHEET]

    placed = 0
    sheets_manifest: dict[str, list[str]] = {}
    for sheet_name, cells in groups.items():
        img = _new_sheet()
        draw = ImageDraw.Draw(img)
        sheets_manifest[sheet_name] = [c.gid for c in cells]
        for idx, cell_item in enumerate(cells):
            if only is None:
                row, col = cell_item.row, cell_item.col
            else:
                row, col = divmod(idx, CELL_COLS)
            _draw_cell(draw, cell_item, sheet_name, row, col)
            placed += 1
        draw.text((SHEET_W // 2 - 60, SHEET_H - 60), sheet_name, fill=0,
                  font=_label_font(30))
        img.save(sheets_dir / f"{sheet_name}.png")
        img.save(sheets_dir / f"{sheet_name}.pdf", "PDF", resolution=300.0)
    (sheets_dir / "manifest.json").write_text(json.dumps(sheets_manifest))

    return {"sheets": sorted(groups), "cells_placed": placed}
