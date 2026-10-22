"""Fixed color-blind-friendly palette baseline (rule-based, non-learning)."""
import numpy as np


def _hex(lst):
    return np.array([[int(h[i:i + 2], 16) for i in (0, 2, 4)] for h in lst], np.float32) / 255.0


PALETTES = {
    # Okabe & Ito (2008)
    "okabe_ito": _hex(["E69F00", "56B4E9", "009E73", "F0E442", "0072B2", "D55E00", "CC79A7", "000000"]),
    # Petroff (2021), arXiv:2107.02270 -- please verify hex values against the paper
    "petroff6": _hex(["5790fc", "f89c20", "e42536", "964a8b", "9c9ca1", "7a21dd"]),
    "petroff8": _hex(["1845fb", "ff5e02", "c91f16", "c849a9", "adad7d", "86c8dd", "578dff", "656364"]),
    "petroff10": _hex(["3f90da", "ffa90e", "bd1f01", "94a4a2", "832db6", "a96b59", "e76300", "b9ac70", "717581", "92dadd"]),
}


def fixed_palette_map(n_colors, name="okabe_ito"):
    """Series k (sorted by pixel count, as returned by extract_palette) -> palette[k].
    Cycles if there are more series than palette entries."""
    p = PALETTES[name]
    return np.stack([p[i % len(p)] for i in range(n_colors)]) if n_colors else np.zeros((0, 3), np.float32)
