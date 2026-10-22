"""Palette extraction from chart PNGs + pixel-level recoloring (NumPy only).

No labels needed: we find the flat 'data colors' in a chart, a model proposes a
replacement color for each, and `recolor_image` applies that mapping (handling
anti-aliased edges), so chart structure and data are preserved by construction.
"""
import numpy as np
from PIL import Image
from skimage.color import rgb2lab


def load_rgb(path):
    """Load PNG as float RGB in [0,1]; RGBA images (e.g. FigureQA) are composited over white."""
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(bg, im).convert("RGB"), dtype=np.float32) / 255.0


def _lab(x):
    return rgb2lab(x.reshape(-1, 1, 3)).reshape(-1, 3)


def interior_count(img, color, tol=0.03):
    """# of pixels within `tol` of `color` whose 4 neighbours are too: flat regions (bars, pie slices) have
    many; anti-aliasing blends along edges / thin lines have ~none."""
    m = np.linalg.norm(img - color, axis=-1) < tol
    inner = m.copy()
    inner[1:] &= m[:-1]; inner[:-1] &= m[1:]; inner[:, 1:] &= m[:, :-1]; inner[:, :-1] &= m[:, 1:]
    inner[0] = inner[-1] = False; inner[:, 0] = inner[:, -1] = False
    return int(inner.sum())


def find_background(img):
    q = (img.reshape(-1, 3) * 31).round().astype(np.int32)
    keys = q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    k = vals[counts.argmax()]
    mask = keys == k
    return img.reshape(-1, 3)[mask].mean(0)


def extract_palette(img, k_max=10, min_frac=0.0003, merge_de=12.0,
                    bg_tol=0.06, interior_min=8, ignore_neutral=True, max_pixels=200_000, seed=0):
    """Return dict(bg, colors(K,3), counts(K,)). Colors sorted by pixel count.
    min_frac is relative to ALL pixels (not just non-background), so small bars / thin lines survive.
    ignore_neutral drops every low-chroma color (text, axes, gridlines, black/gray series)."""
    rng = np.random.default_rng(seed)
    bg = find_background(img)
    px = img.reshape(-1, 3)
    if len(px) > max_pixels:
        px = px[rng.choice(len(px), max_pixels, replace=False)]
    n_total = len(px)
    px = px[np.linalg.norm(px - bg, axis=1) > bg_tol]          # drop background
    if ignore_neutral and len(px):                              # drop text/axes/gridlines
        lab = _lab(px)
        chroma = np.hypot(lab[:, 1], lab[:, 2])
        neutral = chroma < 8
        px = px[~neutral]
    if len(px) == 0:
        return dict(bg=bg, colors=np.zeros((0, 3), np.float32), counts=np.zeros(0))
    q = (px * 15).round().astype(np.int32)                      # 16 levels / channel
    keys = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    vals, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    order = np.argsort(-counts)
    cents = np.stack([px[inv == i].mean(0) for i in order])     # candidate centers
    cnts = counts[order]
    cl = _lab(cents)
    kept, assign = [], np.full(len(cents), -1)
    for i in range(len(cents)):                                 # greedy Lab merge
        if kept:
            d = np.linalg.norm(cl[kept] - cl[i], axis=1)
            j = int(d.argmin())
            if d[j] < merge_de:
                assign[i] = j
                continue
        assign[i] = len(kept)
        kept.append(i)
    K = len(kept)
    tot = np.zeros(K)
    sums = np.zeros((K, 3))
    for i, a in enumerate(assign):
        tot[a] += cnts[i]
        sums[a] += cents[i] * cnts[i]
    colors = sums / tot[:, None]
    keep = tot / n_total >= min_frac
    colors, tot = colors[keep], tot[keep]
    o = np.argsort(-tot)
    colors, tot = colors[o], tot[o]
    # drop anti-aliasing blends (colors without a flat interior):
    #  (1) on the segment between the background and a stronger color (edges of any shape / thin lines);
    #  (2) on the segment between two other series colors that are both >=5x more common
    #      (boundary pixels between adjacent pie slices / stacked regions).
    keep = np.ones(len(colors), bool)
    for i in range(len(colors)):
        is_blend = False
        for j in range(len(colors)):
            if j == i:
                continue
            d = colors[j] - bg
            dd = float(d @ d)
            if dd > 1e-6:
                t = float((colors[i] - bg) @ d) / dd
                if 0.02 < t < 0.95 and np.linalg.norm(colors[i] - (bg + t * d)) < 0.06:
                    is_blend = True
                    break
            for k in range(j + 1, len(colors)):
                if k == i or tot[j] < 5 * tot[i] or tot[k] < 5 * tot[i]:
                    continue
                d = colors[j] - colors[k]
                dd = float(d @ d)
                if dd < 1e-6:
                    continue
                al = float((colors[i] - colors[k]) @ d) / dd
                if 0.02 < al < 0.98 and np.linalg.norm(colors[i] - (colors[k] + al * d)) < 0.03:
                    is_blend = True
                    break
            if is_blend:
                break
        if is_blend and interior_count(img, colors[i]) < interior_min:
            keep[i] = False
    colors, tot = colors[keep][:k_max], tot[keep][:k_max]
    o = np.arange(len(colors))
    return dict(bg=bg.astype(np.float32), colors=colors[o].astype(np.float32), counts=tot[o])


def recolor_image(img, bg, old, new, res_tol=0.12, t_min=0.05, cos_min=0.97):
    """Apply old[k] -> new[k]. Pixels lying on the segment bg->old[k] (incl. anti-aliasing)
    are shifted by t * (new[k]-old[k]); everything else (text, axes) is untouched."""
    H, W, _ = img.shape
    px = img.reshape(-1, 3)
    out = px.copy()
    best_res = np.full(len(px), np.inf)
    best_k = np.full(len(px), -1)
    best_t = np.zeros(len(px))
    for k in range(len(old)):
        d = old[k] - bg
        dd = float(d @ d)
        if dd < 1e-6:
            continue
        t = np.clip(((px - bg) @ d) / dd, 0, 1)
        res = np.linalg.norm(px - (bg + t[:, None] * d), axis=1)
        v = px - bg
        cos = (v @ d) / (np.linalg.norm(v, axis=1) * np.linalg.norm(d) + 1e-8)
        ok = (res < res_tol) & (t > t_min) & (cos > cos_min) & (res < best_res)
        best_res[ok], best_k[ok], best_t[ok] = res[ok], k, t[ok]
    for k in range(len(old)):
        m = best_k == k
        if m.any():
            out[m] = px[m] + best_t[m, None] * (new[k] - old[k])
    return np.clip(out, 0, 1).reshape(H, W, 3)
