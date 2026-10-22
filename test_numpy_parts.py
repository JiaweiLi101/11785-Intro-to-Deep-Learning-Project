"""Sanity test for the NumPy-only parts (palette extraction + recoloring) on a synthetic bar chart."""
import numpy as np
from PIL import Image, ImageDraw
from palette import extract_palette, recolor_image
from baselines import fixed_palette_map

im = Image.new("RGB", (400, 300), "white")
d = ImageDraw.Draw(im)
cols = [(214, 39, 40), (44, 160, 44), (31, 119, 180), (255, 127, 14)]
for i, c in enumerate(cols):
    d.rectangle([40 + i * 85, 60 + i * 30, 100 + i * 85, 260], fill=c)
d.line([30, 260, 390, 260], fill=(0, 0, 0), width=2)
d.line([30, 20, 30, 260], fill=(0, 0, 0), width=2)
d.line([35, 100, 385, 100], fill=(210, 210, 210), width=1)
img = np.asarray(im.filter(__import__("PIL.ImageFilter", fromlist=["x"]).GaussianBlur(0.7)), np.float32) / 255
p = extract_palette(img)
print("bg", p["bg"].round(2), "| K =", len(p["colors"]))
print((p["colors"] * 255).round().astype(int))
assert len(p["colors"]) == 4, "expected 4 series colors"
new = fixed_palette_map(4, "okabe_ito")
out = recolor_image(img, p["bg"], p["colors"], new)
Image.fromarray((np.concatenate([img, out], 1) * 255).astype(np.uint8)).save("/tmp/test_recolor.png")
# axes/text pixels must be unchanged
assert np.allclose(out[258:263, 105:120], img[258:263, 105:120], atol=0.05), "axis line was modified"
print("OK: palette extraction + recolor sane; saved /tmp/test_recolor.png")
