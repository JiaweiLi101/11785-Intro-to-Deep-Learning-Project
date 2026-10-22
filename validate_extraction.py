"""Score palette extraction against FigureQA ground truth (annotations.json). Pie charts are skipped
(their annotations have color names, not hex). Usage:
  python validate_extraction.py --root /path/to/sample_train1 [--min_frac 0.001]"""
import argparse, json, collections
import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000
from palette import load_rgb, extract_palette

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--min_frac", type=float, default=0.0003)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
ann = json.load(open(f"{a.root}/annotations.json"))
ann = ann[:a.limit] if a.limit else ann
hex2rgb = lambda h: np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255
st = collections.defaultdict(lambda: collections.defaultdict(list))
for x in ann:
    if x["type"] == "pie":
        continue
    cs = []
    for m in x["models"]:
        cs += m["colors"] if "colors" in m else [m["color"]]
    g = np.stack([hex2rgb(c) for c in cs])
    chroma = np.hypot(*rgb2lab(g[None])[0][:, 1:].T)
    g = g[chroma >= 8]                                   # neutral series are intentionally not extracted
    if len(g) == 0:
        continue
    e = extract_palette(load_rgb(f"{a.root}/png/{x['image_index']}.png"), min_frac=a.min_frac)["colors"]
    if len(e) == 0:
        st[x["type"]]["recall"].append(0.0); st[x["type"]]["extra"].append(0); st[x["type"]]["exact"].append(0.0); continue
    lg, le = rgb2lab(g[None])[0], rgb2lab(e[None])[0]
    D = np.array([[deltaE_ciede2000(p, q) for q in le] for p in lg])
    found = D.min(1) < 6
    used = set(D.argmin(1)[found].tolist())
    st[x["type"]]["recall"].append(found.mean())
    st[x["type"]]["extra"].append(len(e) - len(used))
    st[x["type"]]["exact"].append(float(found.all() and len(e) == len(g)))
print(f"min_frac={a.min_frac}")
print(f"{'type':18s}{'n':>5s}{'recall':>9s}{'extras/img':>12s}{'exact set':>11s}")
for t, d in st.items():
    print(f"{t:18s}{len(d['recall']):5d}{np.mean(d['recall']):9.3f}{np.mean(d['extra']):12.2f}{np.mean(d['exact']):11.2f}")
