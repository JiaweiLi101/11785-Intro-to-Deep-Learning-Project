"""Train the CVD-conditioned recoloring net with the label-free 3-term loss.
Splits 70/15/15 by image. Mixes real extracted palettes with random synthetic ones."""
import argparse, pickle
import numpy as np, torch
from model import PaletteRecolorNet
from losses import total_loss
from cvd import CVD_TYPES

ap = argparse.ArgumentParser()
ap.add_argument("--palettes", default="palettes.pkl")
ap.add_argument("--out", default="ckpt.pt")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--lam", type=float, nargs=3, default=[1.0, 0.5, 0.5], help="acc struct visual (set 0 to ablate)")
ap.add_argument("--synthetic_frac", type=float, default=0.3)
ap.add_argument("--no_cvd_conditioning", action="store_true", help="ablation: ignore CVD type")
ap.add_argument("--k_max", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"

rows = pickle.load(open(a.palettes, "rb"))
perm = rng.permutation(len(rows)); n = len(rows)
split = dict(train=perm[:int(.7 * n)], val=perm[int(.7 * n):int(.85 * n)], test=perm[int(.85 * n):])
pickle.dump({k: [rows[i]["path"] for i in v] for k, v in split.items()}, open(a.out + ".split.pkl", "wb"))


def pack(items):
    B, K = len(items), a.k_max
    bg = np.ones((B, 3), np.float32); col = np.full((B, K, 3), 0.5, np.float32); m = np.zeros((B, K), bool)
    for i, (b, c) in enumerate(items):
        k = min(len(c), K); bg[i] = b; col[i, :k] = c[:k]; m[i, :k] = True
    return [torch.tensor(x, device=dev) for x in (bg, col, m)]


def synth(nb):
    out = []
    for _ in range(nb):
        k = int(rng.integers(2, a.k_max + 1))
        out.append((np.ones(3, np.float32), rng.random((k, 3)).astype(np.float32)))
    return out


def batch(idx):
    items = [(rows[i]["bg"], rows[i]["colors"]) for i in idx]
    ns = int(len(items) * a.synthetic_frac)
    items = items[:len(items) - ns] + synth(ns)
    return pack(items)


net = PaletteRecolorNet().to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr)
tr = split["train"]
for ep in range(a.epochs):
    net.train(); rng.shuffle(tr); logs = []
    for s in range(0, len(tr), a.bs):
        bg, col, m = batch(tr[s:s + a.bs])
        cvd = torch.randint(0, len(CVD_TYPES), (len(bg),), device=dev)
        cond = torch.zeros_like(cvd) if a.no_cvd_conditioning else cvd
        new = net(bg, col, m, cond)
        loss, d = total_loss(bg, col, new, m, cvd, lam=a.lam)   # loss always uses the TRUE cvd
        opt.zero_grad(); loss.backward(); opt.step(); logs.append(d)
    net.eval()
    with torch.no_grad():
        bg, col, m = batch(split["val"])
        cvd = torch.randint(0, len(CVD_TYPES), (len(bg),), device=dev)
        cond = torch.zeros_like(cvd) if a.no_cvd_conditioning else cvd
        vl, vd = total_loss(bg, col, net(bg, col, m, cond), m, cvd, lam=a.lam)
    print(f"ep {ep:3d} train " + " ".join(f"{k}={np.mean([l[k] for l in logs]):.4f}" for k in logs[0])
          + f" | val total={vl.item():.4f} " + " ".join(f"{k}={v:.4f}" for k, v in vd.items()), flush=True)
    torch.save(dict(model=net.state_dict(), args=vars(a)), a.out)
