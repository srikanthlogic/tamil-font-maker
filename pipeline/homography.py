"""Hand-rolled CV primitives (spec §11: no opencv): DLT homography,
bilinear perspective warp, fiducial-square detection."""
from __future__ import annotations

import numpy as np
from PIL import Image
from collections import deque


class IngestError(Exception):
    pass


def solve_homography(src4, dst4) -> np.ndarray:
    """3x3 homography mapping src4 -> dst4 (exactly 4 point pairs, DLT)."""
    A = []
    for (x, y), (u, v) in zip(src4, dst4):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, Vt = np.linalg.svd(np.array(A, dtype=np.float64))
    return Vt[-1].reshape(3, 3)


def warp_perspective(img: Image.Image, H: np.ndarray,
                     out_wh: tuple[int, int]) -> Image.Image:
    """Inverse-map warp with bilinear sampling; outside -> white (255)."""
    a = np.asarray(img.convert("L")).astype(np.float32)
    h, w = out_wh[1], out_wh[0]
    Hinv = np.linalg.inv(H)

    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    ones = np.ones_like(xs)
    pts = np.stack([xs, ys, ones], axis=0).reshape(3, -1)
    src = Hinv @ pts
    z = src[2]
    z[np.abs(z) < 1e-9] = 1e-9
    sx = (src[0] / z).reshape(h, w)
    sy = (src[1] / z).reshape(h, w)

    sh, sw = a.shape
    valid = (sx >= 0) & (sx < sw - 1) & (sy >= 0) & (sy < sh - 1)
    sx0 = np.clip(np.floor(sx).astype(np.int32), 0, sw - 2)
    sy0 = np.clip(np.floor(sy).astype(np.int32), 0, sh - 2)
    fx = np.clip(sx - sx0, 0, 1)
    fy = np.clip(sy - sy0, 0, 1)

    out = np.zeros((h, w), dtype=np.float32)
    for cy in (0, 1):
        for cx in (0, 1):
            val = a[sy0 + cy, sx0 + cx]
            weight = (fy if cy else 1 - fy) * (fx if cx else 1 - fx)
            out += np.where(valid, val * weight, 0)
    out = np.where(valid, out, 255.0)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="L")


def _components(binary: np.ndarray):
    """Yield (area, bbox, pixels_sample) for 4-connected ink components."""
    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    ys, xs = np.nonzero(binary)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if visited[sy, sx]:
            continue
        q = deque([(sy, sx)])
        visited[sy, sx] = True
        area = 0
        minx, maxx, miny, maxy = sx, sx, sy, sy
        while q:
            y, x = q.popleft()
            area += 1
            minx, maxx = min(minx, x), max(maxx, x)
            miny, maxy = min(miny, y), max(maxy, y)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] \
                        and not visited[ny, nx]:
                    visited[ny, nx] = True
                    q.append((ny, nx))
        yield area, (minx, miny, maxx, maxy), None


def find_fiducials(img: Image.Image) -> list[tuple[int, int]]:
    """Locate the 4 fiducial squares; returns centers ordered TL,TR,BR,BL."""
    a = np.asarray(img.convert("L"))
    binary = a < 128
    hits = []
    for area, (minx, miny, maxx, maxy), _ in _components(binary):
        bw, bh = maxx - minx + 1, maxy - miny + 1
        if not (45 <= bw <= 110 and 45 <= bh <= 110):
            continue
        if not (0.75 <= bw / bh <= 1.33):
            continue
        if area < 0.75 * bw * bh:      # filled square-ish only
            continue
        if area < 1500:                # at least a solid 45x45-ish blob
            continue
        hits.append(((minx + maxx) // 2, (miny + maxy) // 2))
    if len(hits) != 4:
        raise IngestError(
            f"expected 4 fiducial markers, found {len(hits)}")
    hits.sort(key=lambda p: (p[1], p[0]))
    tl, tr = sorted(hits[:2], key=lambda p: p[0])
    bl, br = sorted(hits[2:], key=lambda p: p[0])
    return [tl, tr, br, bl]
