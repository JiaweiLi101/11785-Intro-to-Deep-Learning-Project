"""Evaluate Original vs Fixed-palette vs Ours on the held-out test split.
Reports, per CVD type: mean S_color (min pairwise dE under simulated CVD) and mean dS,
using CIE76 (what training optimizes) AND CIEDE2000 (not optimized -> less circular),
plus fidelity = mean normal-vision CIE76 shift from the original colors.
Usage: python evaluate.py --palettes palettes.pkl --ckpt ckpt.pt --out results.csv
"""
import argparse, csv, pickle, itertools
import numpy as np, torch
from skimage.color import rgb2lab, deltaE_ciede2000, deltaE_cie76
from cvd import CVD_TYPES, simulate_cvd
from model import PaletteRecolorNet
from baselines import fixed_palette_map

ap = argparse.ArgumentParser()
ap.add_argument("--palettes", default="palettes.pkl")
ap.add_argument("--ckpt", default="ckpt.pt")
ap.add_argument("--out", default="results.csv")
ap.add_argument("--k_max", type=int, default=10)
ap.add_argument("--fixed", nargs="+", default=["okabe_ito", "petroff10"])
a = ap.parse_args()


def s_color(colors, cvd, metric):
    if len(colors) < 2:
        return np.nan
    sim = simulate_cvd(torch.tensor(colors)[None].float(), CVD_TYPES.index(cvd))[0].numpy()
    lab = rgb2lab(sim[None])[0]
    fn = deltaE_ciede2000 if metric == "ciede2000" else deltaE_cie76
    return min(float(fn(lab[i], lab[j])) for i, j in itertools.combinations(range(len(lab)), 2))


def fidelity(old, new):
    return float(np.mean(deltaE_cie76(rgb2lab(old[None])[0], rgb2lab(new[None])[0])))


ck = torch.load(a.ckpt, map_location="cpu")
net = PaletteRecolorNet(); net.load_state_dict(ck["model"]); net.eval()
no_cond = ck["args"].get("no_cvd_conditioning", False)
rows = {r["path"]: r for r in pickle.load(open(a.palettes, "rb"))}
test = [rows[p] for p in pickle.load(open(a.ckpt + ".split.pkl", "rb"))["test"]]

res = {}   # (cvd, method) -> lists
for cvd in CVD_TYPES:
    for r in test:
        old, bg = r["colors"][:a.k_max], r["bg"]
        K = len(old)
        col = torch.full((1, a.k_max, 3), 0.5); col[0, :K] = torch.tensor(old)
        m = torch.zeros(1, a.k_max, dtype=torch.bool); m[0, :K] = True
        with torch.no_grad():
            ours = net(torch.tensor(bg)[None], col, m, torch.tensor([0 if no_cond else CVD_TYPES.index(cvd)]))[0, :K].numpy()
        methods = {"original": old, "ours": ours}
        for nm in a.fixed:
            methods[f"fixed_{nm}"] = fixed_palette_map(K, nm)
        for meth, new in methods.items():
            d = res.setdefault((cvd, meth), dict(s76=[], s00=[], fid=[], base76=[], base00=[]))
            d["s76"].append(s_color(new, cvd, "cie76")); d["s00"].append(s_color(new, cvd, "ciede2000"))
            d["base76"].append(s_color(old, cvd, "cie76")); d["base00"].append(s_color(old, cvd, "ciede2000"))
            d["fid"].append(fidelity(old, new))

with open(a.out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["cvd", "method", "S_cie76", "dS_cie76", "S_ciede2000", "dS_ciede2000", "fidelity_dE76", "n"])
    print(f"{'cvd':14s}{'method':22s}{'S76':>8s}{'dS76':>8s}{'S00':>8s}{'dS00':>8s}{'fid':>8s}")
    for (cvd, meth), d in res.items():
        s76, s00 = np.nanmean(d["s76"]), np.nanmean(d["s00"])
        ds76, ds00 = np.nanmean(np.array(d["s76"]) - np.array(d["base76"])), np.nanmean(np.array(d["s00"]) - np.array(d["base00"]))
        fid = np.mean(d["fid"])
        w.writerow([cvd, meth, f"{s76:.2f}", f"{ds76:.2f}", f"{s00:.2f}", f"{ds00:.2f}", f"{fid:.2f}", len(d["s76"])])
        print(f"{cvd:14s}{meth:22s}{s76:8.2f}{ds76:8.2f}{s00:8.2f}{ds00:8.2f}{fid:8.2f}")
print("wrote", a.out)
