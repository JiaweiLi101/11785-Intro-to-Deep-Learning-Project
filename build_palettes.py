"""Extract palettes from every PNG under a directory -> palettes.pkl (no labels needed).
Usage: python build_palettes.py --images /path/to/FigureQA/train1 /path/to/ChartQA --out palettes.pkl
"""
import argparse, glob, os, pickle
from palette import load_rgb, extract_palette

ap = argparse.ArgumentParser()
ap.add_argument("--images", nargs="+", required=True)
ap.add_argument("--out", default="palettes.pkl")
ap.add_argument("--limit", type=int, default=0, help="max images per directory (0 = all)")
ap.add_argument("--k_max", type=int, default=10)
a = ap.parse_args()

rows = []
for root in a.images:
    files = sorted(glob.glob(os.path.join(root, "**", "*.png"), recursive=True))
    files = files[:a.limit] if a.limit else files
    for i, f in enumerate(files):
        try:
            p = extract_palette(load_rgb(f), k_max=a.k_max)
        except Exception as e:
            print("skip", f, e); continue
        if len(p["colors"]) >= 2:                  # need >=2 series for a distinguishability problem
            rows.append(dict(path=f, **p))
        if i % 500 == 0:
            print(root, i, "/", len(files), flush=True)
pickle.dump(rows, open(a.out, "wb"))
print("saved", len(rows), "palettes ->", a.out)
