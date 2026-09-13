"""Shared machinery for the B1–B3 bootstrap roundtrips (and ingest tests):
reference renders from the fixture font, template filling, synthetic phone
photos, and the normalized-IoU shape metric."""
import numpy as np
from PIL import Image, ImageDraw

from pipeline.gfx import ink_extents, rasterize_glyphs, render_ref, render_text
from pipeline.mapping import CELLS
from pipeline.template import BASELINE_Y, HEADLINE_Y, cell_box

FIX_FONT = "test/fixtures/NotoSansTamil.ttf"


def ref_render(text: str, px: int = 300) -> Image.Image:
    """Fixture-font reference ink for a cell's text. Noto's standalone pulli
    (்) glyph is empty by design; a hand-drawn template cell would hold a
    simple dot, so substitute one."""
    img = render_ref(FIX_FONT, text, px)
    if len(text) == 1 and ord(text) == 0x0BCD \
            and not (np.array(img) > 127).any():
        img = Image.new("L", (200, 200), 0)
        ImageDraw.Draw(img).ellipse([76, 76, 124, 124], fill=255)
    return img


def fill_all_cells(sheets_dir, work):
    """Rasterize each cell's text from the fixture font into its template cell,
    at the guide-calibrated reference scale, seated by baseline."""
    from pipeline.backfill import REF_PX

    for c in CELLS:
        text = "".join(chr(cp) for cp in c.cps)
        ref = ref_render(text, REF_PX)
        a = np.array(ref) > 127
        if not a.any():
            raise RuntimeError(f"fixture render empty for {c.gid}")
        ys, xs = np.nonzero(a)
        ref = ref.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        above, below = ink_extents(FIX_FONT, text, REF_PX)
        sheet_img = work.setdefault(
            c.sheet, Image.open(sheets_dir / f"{c.sheet}.png").convert("L"))
        x0, y0, x1, y1 = cell_box(c.row, c.col)
        band = BASELINE_Y - HEADLINE_Y          # guide-to-guide px
        room_below = y1 - y0 - BASELINE_Y - 40  # cell room under the baseline
        max_w = x1 - x0 - 24                    # sheet cells are shared: ink
        scale = min(1.0, (band + 60) / max(above + below, 1.0),  # must stay
                    room_below / max(below, 1.0), max_w / max(ref.width, 1.0))
        if scale < 1.0:
            ink = ref.resize((max(1, int(ref.width * scale)),
                              max(1, int(ref.height * scale))), Image.LANCZOS)
            above *= scale
        else:
            ink = ref
        px = x0 + ((x1 - x0) - ink.width) // 2
        py = y0 + BASELINE_Y - round(above)     # baseline-seat the glyph
        ink = Image.fromarray(
            np.where(np.array(ink) > 127, 255, 0).astype(np.uint8))
        sheet_img.paste(0, (px, py, px + ink.width, py + ink.height), mask=ink)
    for name, img in work.items():
        img.save(sheets_dir / f"{name}.png")


def norm_iou(a_img, b_img) -> float:
    """Scale-free shape metric — canonical implementation lives in the
    pipeline (the verify similarity gate shares it); re-exported here for
    the B1-B3 roundtrips."""
    from pipeline.gfx import norm_iou as _n
    return _n(a_img, b_img)


def direct_render(font_path: str, cp: int, px: int = 150):
    """Render one glyph by outline (no shaping) — used for combining marks,
    where shaped rendering would insert a dotted circle."""
    from fontTools.ttLib import TTFont

    tt = TTFont(font_path)
    gid_name = tt.getBestCmap()[cp]
    gid = tt.getGlyphOrder().index(gid_name)
    upm = tt["head"].unitsPerEm
    asc, desc = tt["hhea"].ascent, tt["hhea"].descent
    canvas = (int((asc - desc) * px / upm) + 40,
              int((asc - desc) * px / upm) + 40)
    return rasterize_glyphs(tt, [(gid, 0, 0)], px, canvas,
                            (20, 20 + asc * px / upm))


def synthesize_photo(sheet_png, out_path, seed):
    """Simulate a phone photo: perspective + uneven light + noise."""
    from pipeline.homography import solve_homography, warp_perspective
    from pipeline.template import SHEET_H, SHEET_W

    rng = np.random.default_rng(seed)
    img = Image.open(sheet_png).convert("L")
    src = [(0, 0), (SHEET_W, 0), (SHEET_W, SHEET_H), (0, SHEET_H)]
    j = 70
    # all four corners pushed outward, kept safely inside the canvas
    dst = [(int(j * rng.uniform(0.5, 1.0)), int(j * rng.uniform(0.5, 1.0))),
           (SHEET_W + int(j * rng.uniform(0.5, 1.0)), int(j * rng.uniform(0.1, 0.8))),
           (SHEET_W + int(j * rng.uniform(0.5, 1.0)), SHEET_H + int(j * rng.uniform(0.5, 1.0))),
           (int(j * rng.uniform(0.1, 0.6)), SHEET_H + int(j * rng.uniform(0.3, 0.9)))]
    H = solve_homography(src, dst)
    warped = warp_perspective(img, H, (SHEET_W + 2 * j, SHEET_H + 2 * j))
    a = np.array(warped).astype(np.float32)
    # uneven illumination: radial gradient
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    light = 225 - 60 * ((xx - a.shape[1] * 0.3) ** 2 + (yy - a.shape[0] * 0.3) ** 2) \
        / (a.shape[0] ** 2 + a.shape[1] ** 2)
    a = np.clip(a * (light / 255.0), 0, 255)
    a = a + rng.normal(0, 6, a.shape)
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(out_path)
