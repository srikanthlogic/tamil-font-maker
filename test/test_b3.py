"""Bootstrap B3 (spec §10): extract path — glyphs on poster-like synthetic
backgrounds (gradients, noise, blobs), agent classification injected as
ground truth, one glyph occluded (flagged unextractable), partial font
builds with graceful degradation and a completion template."""
import numpy as np
import pytest
from PIL import Image

from pipeline import build as build_mod
from pipeline import ingest, template, trace, verify as verify_mod
from pipeline.gfx import render_text, shape_text
from pipeline.mapping import cell
from bootstrap import norm_iou, ref_render

pytestmark = pytest.mark.slow

# agent-selected extractable set (ground-truth classification injected);
# g_u0BB5 is rendered but OCCLUDED — the agent flags it unextractable.
# Bare sign cells are included: GSUB ligature rules need them as components.
SELECT = ["g_u0B85", "g_u0B95", "g_u0BAE", "g_u0BBF", "g_u0BBE", "g_u0BC1",
          "g_u0BCD", "g_u0B95_u0BBF", "g_u0B95_u0BCD", "g_u0BAE_u0BBE",
          "g_u0BAE_u0BC1", "g_u0BE7", "g_u0035", "g_u0021", "g_u0B83"]
OCCLUDED = "g_u0BB5"
OCCLUDED_TEXT = "வ"


def _poster_background(w, h, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    # posters keep title zones readable: bright ground, ink far darker
    a = np.zeros((h, w, 3), dtype=np.float32)
    a[..., 0] = 150 + 80 * xx / w
    a[..., 1] = 160 + 70 * yy / h
    a[..., 2] = 170
    for _ in range(6):  # poster color blobs
        cx, cy, r = rng.integers(0, w), rng.integers(0, h), rng.integers(60, 200)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
        color = rng.integers(110, 230, 3)
        a[mask] = 0.55 * a[mask] + 0.45 * color
    a += rng.normal(0, 10, a.shape)
    return np.clip(a, 0, 255)


def _poster_text_glyph(gid, seed):
    c = cell(gid)
    text = "".join(chr(cp) for cp in c.cps)
    ref = ref_render(text, 300)
    arr = np.array(ref) > 127
    ys, xs = np.nonzero(arr)
    ref = ref.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    pad = 60
    bg = _poster_background(ref.width + 2 * pad, ref.height + 2 * pad, seed)
    ink_val = 25
    mask = np.array(ref) > 127
    bg[pad:pad + ref.height, pad:pad + ref.width][mask] = ink_val
    if gid == OCCLUDED:
        oc = np.zeros(bg.shape[:2], dtype=bool)
        yy, xx = np.mgrid[0:bg.shape[0], 0:bg.shape[1]]
        cx, cy = pad + ref.width // 2, pad + ref.height // 2
        oc[(xx - cx) ** 2 * 0.3 + (yy - cy) ** 2 < 130 ** 2] = True
        bg[oc] = [200, 180, 40]  # artwork blob crossing the glyph
    return Image.fromarray(bg.astype(np.uint8)), ref


def test_b3_extract(tmp_path):
    proj = tmp_path
    crops_dir = proj / "crops"
    crops_dir.mkdir(parents=True)

    crops = {}
    refs = {}
    for i, gid in enumerate(SELECT + [OCCLUDED]):
        img, ref = _poster_text_glyph(gid, seed=500 + i)
        p = crops_dir / f"{gid}.png"
        img.save(p)
        refs[gid] = ref
        if gid != OCCLUDED:  # agent flags the occluded one unextractable
            crops[gid] = p

    report = ingest.ingest_extract(proj, crops, preset="faithful")
    assert sorted(report["ingested_gids"]) == sorted(SELECT)

    t = trace.trace(proj)
    assert t["traced"] == len(SELECT) and not t["failed"]

    b = build_mod.build(proj)
    assert 0 < b["coverage_pct"] < 100
    font_path = proj / "out" / "TamilMaker.ttf"

    v = verify_mod.verify(proj)
    assert v["verdict"] == "warn"
    assert not v["failed_gates"], v["failed_gates"]
    assert v["missing_count"] == 296 - len(SELECT)

    # the occluded glyph must be among the missing (agent flagged it)
    from pipeline.mapping import CELLS
    full_missing = {c.gid for c in CELLS
                    if not (proj / "glyphs" / f"{c.gid}.png").exists()}
    assert OCCLUDED in full_missing

    # shape fidelity on the extracted glyphs
    worst = []
    for gid in SELECT:
        c = cell(gid)
        text = "".join(chr(cp) for cp in c.cps)
        got = render_text(str(font_path), text, 300)
        iou = norm_iou(got, refs[gid])
        if iou < 0.6:
            worst.append((gid, round(iou, 3)))
    assert not worst, f"extracted glyphs below IoU 0.6: {worst}"

    # degradation: மி typed -> ம (covered) + ி (covered spacing matra), no
    # notdef — the மி combo itself was not extracted
    font_bytes = open(font_path, "rb").read()
    vi = shape_text(font_bytes, "மி")
    assert len(vi) == 2 and all(g != 0 for g, _, _ in vi)

    # completion template covers exactly the gaps
    comp = tmp_path / "completion"
    r = template.generate(comp, only=full_missing)
    assert r["cells_placed"] == 296 - len(SELECT)
