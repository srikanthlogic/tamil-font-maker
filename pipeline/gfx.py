"""Shared rendering and image utilities (plan Task 3).

One rasterizer (uharfbuzz shaping + fontTools outlines XOR-filled as
polygons) powers fixtures, verify renders, and pixel-diff gates.
Binary convention everywhere: ink = 255, background = 0, mode "L".
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

# --- shaping -----------------------------------------------------------------


def shape_text(font_bytes: bytes, text: str, features: dict | None = None) -> list[tuple[int, int, int]]:
    """Shape `text` -> [(glyph_id, x_offset, y_offset), ...] in font units."""
    import uharfbuzz as hb

    face = hb.Face(font_bytes)
    font = hb.Font(face)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf, features or {"liga": True})
    out = []
    x = y = 0
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        out.append((info.codepoint, x + pos.x_offset, y + pos.y_offset))
        x += pos.x_advance
        y += pos.y_advance
    return out


# --- rasterizing --------------------------------------------------------------


def _quad(p0, c, p1, steps=14):
    return [(
        (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * c[0] + t ** 2 * p1[0],
        (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * c[1] + t ** 2 * p1[1],
    ) for t in (i / steps for i in range(1, steps + 1))]


def _flatten_contour(points, to_px):
    """points: list of (op, args) segments for one contour -> pixel polygon."""
    poly = []
    cur = None
    start = None
    for op, args in points:
        a = [to_px(p) for p in args]
        if op == "moveTo":
            cur = a[0]
            start = cur
            poly.append(cur)
        elif op == "lineTo":
            poly.append(a[0])
            cur = a[0]
        elif op == "qCurveTo":
            # TrueType: all args off-curve except the last (on-curve)
            offs, on = a[:-1], a[-1]
            for i in range(len(offs) - 1):
                mid = ((offs[i][0] + offs[i + 1][0]) / 2,
                       (offs[i][1] + offs[i + 1][1]) / 2)
                poly += _quad(cur, offs[i], mid)
                cur = mid
            poly += _quad(cur, offs[-1], on)
            cur = on
        elif op == "curveTo":
            c1, c2, p = a
            for i in range(1, 17):
                t = i / 16
                x = (1 - t) ** 3 * cur[0] + 3 * (1 - t) ** 2 * t * c1[0] \
                    + 3 * (1 - t) * t ** 2 * c2[0] + t ** 3 * p[0]
                y = (1 - t) ** 3 * cur[1] + 3 * (1 - t) ** 2 * t * c1[1] \
                    + 3 * (1 - t) * t ** 2 * c2[1] + t ** 3 * p[1]
                poly.append((x, y))
            cur = p
        elif op in ("closePath", "endPath"):
            if poly and start is not None:
                poly.append(start)
            break
    return poly


def rasterize_glyphs(ttfont, shaped, px: int, canvas: tuple[int, int],
                     origin: tuple[int, int]) -> Image.Image:
    """Rasterize shaped glyphs (see shape_text) onto an ink=255 canvas."""
    from fontTools.pens.recordingPen import DecomposingRecordingPen

    glyph_set = ttfont.getGlyphSet()
    order = ttfont.getGlyphOrder()
    scale = px / ttfont["head"].unitsPerEm
    ox, oy = origin

    def to_px(p, dx=0.0, dy=0.0):
        return (ox + (dx + p[0]) * scale, oy - (dy + p[1]) * scale)

    acc = np.zeros((canvas[1], canvas[0]), dtype=bool)
    for gid, dx, dy in shaped:
        pen = DecomposingRecordingPen(glyph_set)
        glyph_set[order[gid]].draw(pen)
        # split recording into contours
        contours, current = [], []
        for op, args in pen.value:
            current.append((op, args))
            if op in ("closePath", "endPath"):
                contours.append(current)
                current = []
        if current:
            contours.append(current)
        for contour in contours:
            poly = _flatten_contour(contour, lambda p: to_px(p, dx, dy))
            if len(poly) < 3:
                continue
            mask = np.zeros_like(acc)
            img = Image.fromarray(mask.astype(np.uint8) * 255)
            ImageDraw.Draw(img).polygon(poly, fill=255)
            acc ^= np.array(img, dtype=bool)
    return Image.fromarray(np.where(acc, 255, 0).astype(np.uint8), mode="L")


def render_text(font_path: str, text: str, px: int, pad: int = 8) -> Image.Image:
    """Render shaped text as ink=255 image, baseline-derived canvas sizing."""
    from fontTools.ttLib import TTFont

    font_bytes = open(font_path, "rb").read()
    tt = TTFont(font_path)
    shaped = shape_text(font_bytes, text)
    upm = tt["head"].unitsPerEm
    scale = px / upm
    ascent = tt["hhea"].ascent
    descent = tt["hhea"].descent  # negative
    order = tt.getGlyphOrder()
    hmtx = tt["hmtx"]
    if shaped:
        last_gid, last_x, _ = shaped[-1]
        width = last_x + hmtx[order[last_gid]][0]
    else:
        width = 0
    w = int(width * scale) + 2 * pad
    h = int((ascent - descent) * scale) + 2 * pad
    return rasterize_glyphs(tt, shaped, px, (max(w, 2), max(h, 2)),
                            (pad, pad + ascent * scale))


# --- binarization --------------------------------------------------------------


def otsu(gray: Image.Image) -> int:
    a = np.asarray(gray.convert("L")).ravel()
    hist = np.bincount(a, minlength=256).astype(np.float64)
    total = a.size
    sum_all = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0.0
    best_t, best_var = 127, -1.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return best_t


def otsu_binarize(img: Image.Image) -> Image.Image:
    """Dark ink on light background -> ink=255 binary."""
    a = np.asarray(img.convert("L"))
    t = otsu(img)
    return Image.fromarray(np.where(a <= t, 255, 0).astype(np.uint8), mode="L")


def despeckle(binary: Image.Image, min_area: int) -> Image.Image:
    """Remove connected ink components smaller than min_area (4-neighbour)."""
    a = np.asarray(binary) > 127
    out = np.zeros_like(a)
    h, w = a.shape
    visited = np.zeros_like(a)
    ys, xs = np.nonzero(a)
    from collections import deque
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if visited[sy, sx]:
            continue
        comp = []
        q = deque([(sy, sx)])
        visited[sy, sx] = True
        while q:
            y, x = q.popleft()
            comp.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and a[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    q.append((ny, nx))
        if len(comp) >= min_area:
            for y, x in comp:
                out[y, x] = True
    return Image.fromarray(np.where(out, 255, 0).astype(np.uint8), mode="L")


def fit_to_box(binary: Image.Image, box_wh: tuple[int, int],
               anchor_baseline_frac: float | None = None,
               margin: int = 8) -> Image.Image:
    """Scale a binary glyph crop into the box, optionally seating the ink
    bottom at anchor_baseline_frac of box height."""
    a = np.asarray(binary) > 127
    ys, xs = np.nonzero(a)
    if ys.size == 0:
        return Image.new("L", box_wh, 0)
    crop = binary.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    bw, bh = box_wh
    avail_w, avail_h = bw - 2 * margin, int(bh * 0.8)
    scale = min(avail_w / crop.width, avail_h / crop.height)
    nw, nh = max(1, int(crop.width * scale)), max(1, int(crop.height * scale))
    crop = crop.resize((nw, nh), Image.LANCZOS)
    out = Image.new("L", box_wh, 0)
    base_y = int(bh * anchor_baseline_frac) if anchor_baseline_frac else \
        (bh + nh) // 2
    out.paste(crop, ((bw - nw) // 2, max(0, base_y - nh)))
    return out
