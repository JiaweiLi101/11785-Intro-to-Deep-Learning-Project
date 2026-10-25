"""Image-conditioned palette recoloring network (CNN on the chart + spatial attention bias).

Compared with model.PaletteRecolorNet (which only sees the extracted colors), this model also sees the chart:
  * a small CNN encodes a 128x128 thumbnail of the chart;
  * its feature map is (a) mean-pooled into one global context vector (chart type / density) and
    (b) pooled over each series' own pixels, giving every color token a description of its region
    (thick bar / thin line / pie slice / marker);
  * a spatial adjacency matrix (which series touch or run close together) is added to the attention
    logits with a learned per-head scale, so nearby series compare themselves with each other first.
The output is still a new color per series (color mapping), so chart structure and data are preserved.
"""
import os, pickle
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from scipy.ndimage import maximum_filter
from cvd import rgb_to_lab
from palette import load_rgb, recolor_image, low_chroma_mask

IMG_SIZE = 128      # thumbnail given to the CNN
MASK_SIZE = 64      # resolution of per-series masks / adjacency computation


# --------------------------------------------------------------------------------------
# Data: thumbnails, per-series masks and adjacency (NumPy; cached on disk)
# --------------------------------------------------------------------------------------
def series_label_map(img, bg, old, bg_tol=0.06, res_tol=0.12, t_min=0.05, cos_min=0.97):
    """Per-pixel label: 0 = background, k+1 = series k (incl. anti-aliased edges), -1 = other (text, axes)."""
    H, W, _ = img.shape
    px = img.reshape(-1, 3)
    lab = np.full(len(px), -1, np.int16)
    best = np.full(len(px), np.inf)
    v = px - bg
    vn = np.linalg.norm(v, axis=1)
    lowc = low_chroma_mask(old)
    for k in range(len(old)):
        t_lo = 0.9 if lowc[k] else t_min
        d = old[k] - bg
        dd = float(d @ d)
        if dd < 1e-6:
            continue
        t = np.clip((v @ d) / dd, 0, 1)
        res = np.linalg.norm(px - (bg + t[:, None] * d), axis=1)
        cos = (v @ d) / (vn * np.linalg.norm(d) + 1e-8)
        ok = (res < res_tol) & (t > t_lo) & (cos > cos_min) & (res < best)
        best[ok], lab[ok] = res[ok], k + 1
    lab[vn < bg_tol] = 0
    return lab.reshape(H, W)


def adjacency_from_masks(masks, radii=(4, 8, 12, 16, 24), thr=0.1, min_link=0.03):
    """masks (T,h,w) float fractions, index 0 = background. Returns (T,T) in [0,1]: how much series i's
    neighbourhood (dilated by r px) overlaps series j. Pie slices that touch, lines that cross or run side
    by side score high. The radius grows (4,8,...,24 px on the 64x64 grid) until every series has at least one
    neighbour, so widely spaced bars still get neighbours. Background row/column are zero."""
    m = (masks > thr).astype(np.float32)
    area = m.sum((1, 2))
    present = area[1:] > 0
    norm = np.sqrt(np.outer(area, area)) + 1.0
    for r in radii:
        dil = np.stack([maximum_filter(x, size=2 * r + 1) for x in m])
        ov = np.einsum("ihw,jhw->ij", dil, m)
        adj = (ov + ov.T) / norm
        np.fill_diagonal(adj, 0.0)
        adj[0, :] = 0.0
        adj[:, 0] = 0.0
        if len(present) == 0 or (adj[1:, 1:].max(1)[present] > min_link).all():
            break
    mx = adj.max()
    return (adj / mx if mx > 1e-6 else adj).astype(np.float32)


def build_entry(path, bg, old):
    """thumb (128,128,3) uint8, masks (K+1,64,64) uint8 (area fractions * 255), adj (K+1,K+1) float32."""
    img = load_rgb(path)
    thumb = np.asarray(Image.fromarray((img * 255).round().astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.BOX))
    lab = series_label_map(img, bg, old)
    T = len(old) + 1
    masks = np.zeros((T, MASK_SIZE, MASK_SIZE), np.float32)
    for c in range(T):
        ch = (lab == c).astype(np.float32)
        masks[c] = np.asarray(Image.fromarray(ch).resize((MASK_SIZE, MASK_SIZE), Image.BOX))
    return dict(thumb=thumb, masks=(masks * 255).round().astype(np.uint8), adj=adjacency_from_masks(masks))


def load_or_build_cache(rows, path=None, force=False, verbose=True):
    """Dict path -> entry for every palette row. Needs the original PNGs to exist at row['path']."""
    cache = {}
    if path and os.path.exists(path) and not force:
        with open(path, "rb") as fh:
            cache = pickle.load(fh)
    todo = [r for r in rows if r["path"] not in cache]
    for i, r in enumerate(todo):
        if not os.path.exists(r["path"]):
            raise FileNotFoundError(f"{r['path']} not found. The image model needs the chart PNGs: rebuild the palettes "
                                    "from your local images (REBUILD_PALETTES = True) so 'path' points to real files.")
        cache[r["path"]] = build_entry(r["path"], r["bg"], r["colors"])
        if verbose and i % 250 == 0:
            print(f"chart cache: {i}/{len(todo)}", flush=True)
    if todo and path:
        with open(path, "wb") as fh:
            pickle.dump(cache, fh)
    return cache


def entry_arrays(entry, k_max):
    """Pad one cache entry to fixed size: masks (k_max+1,64,64) float, adj (k_max+1,k_max+1) float."""
    T = k_max + 1
    masks = np.zeros((T, MASK_SIZE, MASK_SIZE), np.float32)
    adj = np.zeros((T, T), np.float32)
    n = min(entry["masks"].shape[0], T)
    masks[:n] = entry["masks"][:n].astype(np.float32) / 255.0
    adj[:n, :n] = entry["adj"][:n, :n]
    return masks, adj


def row_inputs(row, cache, k_max):
    """Image inputs (img, masks, adj), each with batch dim 1, for one palette row (used at evaluation)."""
    e = cache[row["path"]]
    masks, adj = entry_arrays(e, k_max)
    img = torch.tensor(e["thumb"].astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    return img, torch.tensor(masks)[None], torch.tensor(adj)[None]


def make_image_batch(prow, idxs, cache, k_max, synthetic_frac, rng, device):
    """Returns bg, colors, mask, (img, masks, adj). The last `synthetic_frac` of the batch are the same charts
    re-rendered with random series colors (label-free augmentation that keeps image and palette consistent)."""
    B = len(idxs)
    ns = int(B * synthetic_frac)
    bg = np.ones((B, 3), np.float32)
    col = np.full((B, k_max, 3), 0.5, np.float32)
    m = np.zeros((B, k_max), bool)
    imgs = np.zeros((B, IMG_SIZE, IMG_SIZE, 3), np.float32)
    masks = np.zeros((B, k_max + 1, MASK_SIZE, MASK_SIZE), np.float32)
    adj = np.zeros((B, k_max + 1, k_max + 1), np.float32)
    for i, ix in enumerate(idxs):
        r = prow[ix]
        e = cache[r["path"]]
        old = r["colors"][:k_max]
        K = len(old)
        thumb = e["thumb"].astype(np.float32) / 255.0
        if i >= B - ns:
            new = rng.random((K, 3)).astype(np.float32)
            thumb = recolor_image(thumb, r["bg"], old, new)
            old = new
        bg[i] = r["bg"]
        col[i, :K] = old
        m[i, :K] = True
        imgs[i] = thumb
        masks[i], adj[i] = entry_arrays(e, k_max)
    t = lambda x: torch.tensor(x, device=device)
    return t(bg), t(col), t(m), (t(imgs).permute(0, 3, 1, 2).contiguous(), t(masks), t(adj))


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
def _gn(c):
    return nn.GroupNorm(8, c)


class ChartEncoder(nn.Module):
    """Small CNN: (B,3,128,128) -> feature map (B,c,16,16) (stride 8)."""

    def __init__(self, c=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), _gn(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), _gn(64), nn.ReLU(),
            nn.Conv2d(64, c, 3, 2, 1), _gn(c), nn.ReLU(),
            nn.Conv2d(c, c, 3, 1, 1), _gn(c), nn.ReLU(),
        )

    def forward(self, img):
        return self.net(img)


def region_pool(feat, masks):
    """Average CNN features over each series' pixels. feat (B,C,h,w), masks (B,T,H,W) -> (B,T,C)."""
    m = F.adaptive_avg_pool2d(masks, feat.shape[-2:])                 # (B,T,h,w)
    num = torch.einsum("bthw,bchw->btc", m, feat)
    return num / (m.sum((2, 3)).unsqueeze(-1) + 1e-6)


class BiasedAttnBlock(nn.Module):
    """Post-norm transformer block whose attention logits get  + gamma_h * adjacency  (per head)."""

    def __init__(self, d=128, heads=4, ff=512):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.gamma = nn.Parameter(torch.full((heads,), 0.5))

    def forward(self, x, pad_bias, adj):
        """x (B,T,d); pad_bias (B,1,1,T) with -1e9 on padded keys; adj (B,T,T)."""
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)    # (3,B,H,T,Dh)
        bias = pad_bias + self.gamma.view(1, self.h, 1, 1) * adj[:, None]              # (B,H,T,T)
        a = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], attn_mask=bias)
        x = self.n1(x + self.proj(a.transpose(1, 2).reshape(B, T, D)))
        return self.n2(x + self.ff(x))


class ImagePaletteRecolorNet(nn.Module):
    """forward(bg (B,3), colors (B,K,3), mask (B,K) bool, cvd (B,) long,
               img (B,3,128,128), masks (B,K+1,64,64), adj (B,K+1,K+1)) -> new colors (B,K,3).
    Token = [RGB, Lab/100, is-background] -> Linear, + CVD embedding + global CNN context + region-pooled CNN
    features; 3 attention blocks with adjacency bias; Linear head; residual in logit space (starts near identity)."""

    def __init__(self, d=128, heads=4, layers=3, n_cvd=4, ff=512):
        super().__init__()
        self.in_proj = nn.Linear(7, d)
        self.cvd_emb = nn.Embedding(n_cvd, d)
        self.cnn = ChartEncoder(d)
        self.ctx_proj = nn.Linear(d, d)
        self.reg_proj = nn.Linear(d, d)
        self.blocks = nn.ModuleList([BiasedAttnBlock(d, heads, ff) for _ in range(layers)])
        self.out = nn.Linear(d, 3)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)

    def forward(self, bg, colors, mask, cvd, img, masks, adj):
        B, K, _ = colors.shape
        toks = torch.cat([bg[:, None], colors], 1)                                    # (B,K+1,3)
        lab = rgb_to_lab(toks) / 100.0
        flag = torch.zeros(B, K + 1, 1, device=toks.device)
        flag[:, 0] = 1.0
        feat = self.cnn((img - 0.5) / 0.5)                                            # (B,d,16,16)
        ctx = self.ctx_proj(feat.mean((2, 3)))                                        # (B,d)
        reg = self.reg_proj(region_pool(feat, masks))                                 # (B,K+1,d)
        x = self.in_proj(torch.cat([toks, lab, flag], -1)) + self.cvd_emb(cvd)[:, None] + ctx[:, None] + reg
        valid = torch.cat([torch.ones_like(mask[:, :1]), mask], 1)
        pad_bias = (~valid).float().mul(-1e9)[:, None, None, :]
        for blk in self.blocks:
            x = blk(x, pad_bias, adj)
        c = colors.clamp(1e-3, 1 - 1e-3)
        return torch.sigmoid(torch.log(c) - torch.log1p(-c) + self.out(x[:, 1:]))


def smoke_test(device="cpu", k_max=10):
    """Shape / NaN / gradient check on random tensors. Run once before training."""
    net = ImagePaletteRecolorNet().to(device)
    B = 4
    mask = torch.zeros(B, k_max, dtype=torch.bool)
    mask[:, :5] = True
    args = (torch.ones(B, 3), torch.rand(B, k_max, 3), mask, torch.randint(0, 4, (B,)),
            torch.rand(B, 3, IMG_SIZE, IMG_SIZE), torch.rand(B, k_max + 1, MASK_SIZE, MASK_SIZE),
            torch.rand(B, k_max + 1, k_max + 1))
    out = net(*[x.to(device) for x in args])
    assert out.shape == (B, k_max, 3) and torch.isfinite(out).all(), "bad output"
    out.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()), "bad gradients"
    print("smoke test OK | output", tuple(out.shape), "| parameters:", sum(p.numel() for p in net.parameters()))
