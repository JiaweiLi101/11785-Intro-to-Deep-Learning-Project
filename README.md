# CVD-accessible chart recoloring (color-mapping version)

Label-free pipeline: PNG chart -> extract series colors -> CVD-conditioned net proposes new colors -> apply mapping to the chart.
Chart structure and data are preserved by construction (only the color of series pixels changes).

## Run order
```bash
pip install -r requirements.txt
# 1. extract palettes from your PNGs (no labels needed)
python build_palettes.py --images /data/FigureQA /data/ChartQA --out palettes.pkl
# 2. train (loss-only; 70/15/15 split by image is written to ckpt.pt.split.pkl)
python train.py --palettes palettes.pkl --out ckpt.pt
# 3. evaluate: Original vs Fixed palette vs Ours, per CVD type
python evaluate.py --palettes palettes.pkl --ckpt ckpt.pt --out results.csv
# 4. qualitative figure: [orig | orig under CVD | ours | ours under CVD | fixed | fixed under CVD]
python render_demo.py --img some_chart.png --ckpt ckpt.pt --cvd deuteranopia --fixed okabe_ito
```

## Proposal experiments -> commands
| Experiment | Command |
|---|---|
| Original vs fixed palette | rows `original` / `fixed_*` in `results.csv` |
| Fixed vs generative | rows `fixed_*` vs `ours` |
| CVD conditioning | train with and without `--no_cvd_conditioning`, compare per-CVD rows |
| Remove accessibility loss | `--lam 0 0.5 0.5 --out ckpt_noacc.pt` |
| Remove structure loss | `--lam 1 0 0.5 --out ckpt_nostruct.pt` |

## Files
- `cvd.py` differentiable Machado (2009) CVD simulation, sRGB->Lab, pairwise dE (CIE76)
- `palette.py` palette extraction + anti-aliasing-aware recoloring (NumPy)
- `baselines.py` Okabe-Ito and Petroff palettes (verify Petroff hex values against the paper)
- `model.py` set-transformer, CVD embedding, near-identity init
- `losses.py` L_acc (hinge + soft-min of S_color), L_structure, L_visual
- `train.py`, `evaluate.py`, `render_demo.py`, `build_palettes.py`

## Known limitations (state these in the report)
- Only color is changed; marker shape / line style (extension 3) is not implemented yet.
- S_color is color-only; under monochromacy with many series, a ~30 dE margin is unreachable, so the hinge just maximizes spread.
- Neutral dark/light colors (black, gray, white) are treated as text/axes/background and are not recolored, so black or gray data series are skipped.
- Heatmaps/continuous colormaps and pie charts with touching slices are poorly handled by flat-color extraction.
- All CVD scores are simulation proxies, not proof of readability for people with CVD.
- Evaluation reports CIEDE2000 alongside CIE76; training only optimizes CIE76, which makes the CIEDE2000 numbers a less circular check.
- Downstream chart-QA accuracy (interpretability metric) is not included yet.

## Status
`palette.py` and `baselines.py` were tested on a synthetic chart. The PyTorch files (`cvd.py`, `model.py`, `losses.py`, `train.py`, `evaluate.py`, `render_demo.py`) were only syntax-checked: torch was not available where this was written, so run a short smoke test first (e.g. `--epochs 2` on ~200 images).
