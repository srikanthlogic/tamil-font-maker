"""Verify stage (spec §6.5): programmatic gates, visual QA artifacts,
coverage report, verdict."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .gfx import render_text
from .mapping import CELLS, SAMPLE_TEXTS, ligature_rules

EM_MIN, EM_MAX = -600, 2600     # generous em-box bounds (guides: 0..1400)
CHART_PX = 96


def _font_path(project: Path) -> Path:
    out = project / "out"
    ttfs = sorted(out.glob("*.ttf"))
    if not ttfs:
        raise FileNotFoundError("no TTF in out/ — run build first")
    return ttfs[0]


def _programmatic_gates(font_path: Path, project: Path, glyphs_dir: Path,
                        filled: set[str]) -> dict:
    from fontTools.ttLib import TTFont
    from .gfx import shape_text

    font_bytes = open(font_path, "rb").read()
    tt = TTFont(str(font_path))
    order = tt.getGlyphOrder()
    cmap = tt.getBestCmap()
    gates = {}

    # non-empty glyph rasters
    empty = [c.gid for c in CELLS if (glyphs_dir / f"{c.gid}.png").exists()
             and not (np.asarray(Image.open(glyphs_dir / f"{c.gid}.png")) > 127).any()]
    gates["non_empty_rasters"] = {"ok": not empty, "empty": empty}

    # em-bounds
    outside = []
    for p in (project / "outlines").glob("g_*.json"):
        d = json.loads(p.read_text())
        if not (EM_MIN <= d["xmin"] and d["xmax"] <= EM_MAX
                and EM_MIN <= d["ymin"] and d["ymax"] <= EM_MAX):
            outside.append(p.stem)
    gates["em_bounds"] = {"ok": not outside, "outside": outside}

    # shaping identities (run only when the needed cells are covered)
    def gid_shaped_once(text, target):
        shaped = shape_text(font_bytes, text)
        return len(shaped) == 1 and order[shaped[0][0]] == target

    if filled >= {"g_u0B95", "g_u0BBF", "g_u0B95_u0BBF"}:
        gates["shape_ka_i"] = {"ok": gid_shaped_once("கி", "g_u0B95_u0BBF")}
    if filled >= {"g_u0B95", "g_u0BCD", "g_u0B95_u0BCD"}:
        gates["shape_ka_pulli"] = {"ok": gid_shaped_once("க்", "g_u0B95_u0BCD")}
    if filled >= {"g_u0B95", "g_u0BCC", "g_u0BC6", "g_u0BD7"}:
        a = [g for g, _, _ in shape_text(font_bytes, "கௌ")]
        # decomposed: க + ெ (U+0BC6) + ௗ (U+0BD7)
        b = [g for g, _, _ in shape_text(
            font_bytes, "க" + chr(0x0BC6) + chr(0x0BD7))]
        gates["shape_kau_composed_equals_decomposed"] = {"ok": a == b}

    # notdef scan: sample chars resolvable from covered cells must never hit
    # .notdef; uncovered chars legitimately will (degradation ladder §9)
    covered_cps = set(cmap.keys())
    for seq, target in ligature_rules():
        if target in filled and all(f"g_u{cp:04X}" in filled for cp in seq):
            covered_cps.update(seq)
    sample = "".join(SAMPLE_TEXTS)
    expected_notdef = sum(1 for ch in sample
                          if not ch.isspace() and ord(ch) not in covered_cps)
    notdefs = 0
    for text in SAMPLE_TEXTS:
        shaped = shape_text(font_bytes, text)
        notdefs += sum(1 for g, _, _ in shaped if g == 0)
    gates["notdefs_match_coverage"] = {"ok": notdefs == expected_notdef,
                                       "notdefs": notdefs,
                                       "expected": expected_notdef}
    return gates


def _visual_artifacts(font_path: Path, project: Path, glyphs_dir: Path) -> list[str]:
    qa = project / "qa"
    qa.mkdir(exist_ok=True)
    made = []

    covered = [c for c in CELLS if (glyphs_dir / f"{c.gid}.png").exists()]

    # uyirmei + pulli-form chart
    chart_cells = [c for c in covered if c.kind in ("uyirmei", "pulli_form")]
    if chart_cells:
        cols = 18
        rows = (len(chart_cells) + cols - 1) // cols
        cw, ch = CHART_PX + 16, CHART_PX + 40
        img = Image.new("L", (cols * cw, rows * ch), 255)
        d = ImageDraw.Draw(img)
        for i, c in enumerate(chart_cells):
            text = "".join(chr(cp) for cp in c.cps)
            try:
                g = render_text(str(font_path), text, CHART_PX // 2)
                a = np.array(g)
                inked = np.array(g) > 127
                if inked.any():
                    ys, xs = np.nonzero(inked)
                    g = g.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
                x = (i % cols) * cw + 8
                y = (i // cols) * ch + 8
                img.paste(0, (x, y, x + g.width, y + g.height),
                          mask=Image.fromarray(
                              np.where(np.array(g) > 127, 255, 0).astype(np.uint8)))
            except Exception:  # noqa: BLE001 — a broken cell must not kill the chart
                pass
            d.text(((i % cols) * cw + 4, (i // cols) * ch + CHART_PX + 10),
                   c.gid[2:12], stroke_width=0, fill=60)
        img.save(qa / "combo_chart.png")
        made.append("qa/combo_chart.png")

    # sample sentences at three sizes
    line_imgs = []
    for px in (28, 48, 80):
        for text in SAMPLE_TEXTS:
            try:
                line_imgs.append((px, render_text(str(font_path), text, px)))
            except Exception:  # noqa: BLE001
                pass
    if line_imgs:
        w = max(i.width for _, i in line_imgs) + 20
        h = sum(i.height + 10 for _, i in line_imgs) + 10
        img = Image.new("L", (w, h), 255)
        y = 10
        for _, li in line_imgs:
            img.paste(li, (10, y))
            y += li.height + 10
        img.save(qa / "samples.png")
        made.append("qa/samples.png")

    # side-by-side: source raster vs font render, per covered cell (first sheet)
    cmp_cells = covered[:25]
    if cmp_cells:
        cw, ch = 200, 260
        img = Image.new("L", (cw * len(cmp_cells), ch * 2 + 30), 255)
        for i, c in enumerate(cmp_cells):
            src = Image.open(glyphs_dir / f"{c.gid}.png").resize((cw - 10, ch - 10))
            img.paste(src, (i * cw + 5, 5))
            text = "".join(chr(cp) for cp in c.cps)
            try:
                ren = render_text(str(font_path), text, 60)
                inked = np.array(ren) > 127
                if inked.any():
                    ys, xs = np.nonzero(inked)
                    ren = ren.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
                ren.thumbnail((cw - 10, ch - 10))
                img.paste(ren, (i * cw + 5, ch + 15))
            except Exception:  # noqa: BLE001 — one bad cell must not kill the sheet
                pass
        img.save(qa / "compare_first25.png")
        made.append("qa/compare_first25.png")
    return made


def verify(project: Path, glyphs_dir: Path | None = None) -> dict:
    project = Path(project)
    glyphs_dir = Path(glyphs_dir) if glyphs_dir else project / "glyphs"
    font_path = _font_path(project)

    filled = {c.gid for c in CELLS if (glyphs_dir / f"{c.gid}.png").exists()}
    missing = [c.gid for c in CELLS if c.gid not in filled]

    gates = _programmatic_gates(font_path, project, glyphs_dir, filled)
    artifacts = _visual_artifacts(font_path, project, glyphs_dir)

    failed_gates = [k for k, v in gates.items() if not v["ok"]]
    verdict = "fail" if failed_gates else ("warn" if missing else "green")

    report = {
        "verdict": verdict,
        "font": str(font_path),
        "coverage_pct": round(100 * len(filled) / len(CELLS), 1),
        "missing_count": len(missing),
        "missing_sample": missing[:20],
        "gates": gates,
        "failed_gates": failed_gates,
        "qa_artifacts": artifacts,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    reports = project / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "verify.json").write_text(json.dumps(report, indent=1))
    return report
