"""Make a before/after figure for one chart: [original | original under CVD | ours | ours under CVD].
Usage: python render_demo.py --img chart.png --ckpt ckpt.pt --cvd deuteranopia --out demo.png
"""
import argparse
import numpy as np, torch
from PIL import Image
from cvd import CVD_TYPES, simulate_cvd
from model import PaletteRecolorNet
from palette import load_rgb, extract_palette, recolor_image
from baselines import fixed_palette_map

ap = argparse.ArgumentParser()
ap.add_argument("--img", required=True)
ap.add_argument("--ckpt", default="ckpt.pt")
ap.add_argument("--cvd", default="deuteranopia", choices=CVD_TYPES)
ap.add_argument("--out", default="demo.png")
ap.add_argument("--k_max", type=int, default=10)
ap.add_argument("--fixed", default="", help="optional fixed palette name for a 5th panel, e.g. okabe_ito")
a = ap.parse_args()

img = load_rgb(a.img)
p = extract_palette(img, k_max=a.k_max)
old, bg, K = p["colors"], p["bg"], len(p["colors"])
ck = torch.load(a.ckpt, map_location="cpu")
net = PaletteRecolorNet(); net.load_state_dict(ck["model"]); net.eval()
col = torch.full((1, a.k_max, 3), 0.5); col[0, :K] = torch.tensor(old)
m = torch.zeros(1, a.k_max, dtype=torch.bool); m[0, :K] = True
cidx = 0 if ck["args"].get("no_cvd_conditioning") else CVD_TYPES.index(a.cvd)
with torch.no_grad():
    new = net(torch.tensor(bg)[None], col, m, torch.tensor([cidx]))[0, :K].numpy()


def sim(x):
    H, W, _ = x.shape
    return simulate_cvd(torch.tensor(x).reshape(1, -1, 3).float(), CVD_TYPES.index(a.cvd))[0].reshape(H, W, 3).numpy()


rec = recolor_image(img, bg, old, new)
panels = [img, sim(img), rec, sim(rec)]
if a.fixed:
    fx = recolor_image(img, bg, old, fixed_palette_map(K, a.fixed))
    panels += [fx, sim(fx)]
Image.fromarray((np.concatenate(panels, 1) * 255).astype(np.uint8)).save(a.out)
print("saved", a.out, "| panels: original, original-under-CVD, ours, ours-under-CVD" + (", fixed, fixed-under-CVD" if a.fixed else ""))
