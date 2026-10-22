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
- ALL low-chroma colors (black, gray, white, and near-gray like slate gray) are treated as text/axes/background and are not recolored: gray text is as large as gray bars in FigureQA, so pixel count cannot separate them. ~10-15% of sample charts contain a gray/black series that stays unchanged.
- Heatmaps/continuous colormaps and pie charts with touching slices are poorly handled by flat-color extraction.
- All CVD scores are simulation proxies, not proof of readability for people with CVD.
- Evaluation reports CIEDE2000 alongside CIE76; training only optimizes CIE76, which makes the CIEDE2000 numbers a less circular check.
- Downstream chart-QA accuracy (interpretability metric) is not included yet.

## Validation on the FigureQA sample (1000 PNGs, `annotations.json` gives true series colors)
`python validate_extraction.py --root /path/to/sample_train1` (neutral/gray series excluded from scoring; CIEDE2000 < 6 counts as a match)

| type | recall | extra colors / img | exact color set |
|---|---|---|---|
| vbar_categorical | 0.979 | 0.01 | 0.84 |
| hbar_categorical | 0.955 | 0.01 | 0.75 |
| line | 0.983 | 0.10 | 0.82 |
| dot_line | 0.997 | 0.02 | 0.94 |
| pie (no hex GT; slice count only) | - | - | 0.86 exact K, 0.995 within +-1 |

Result: 983 of 1000 images give a usable palette (>= 2 chromatic series); K ranges 2-10. `palettes_sample.pkl` is included.
Recoloring was checked visually on all 5 chart types (legends, anti-aliased lines, pies, dots; axes/text untouched).
FigureQA PNGs are RGBA (mostly opaque); `load_rgb` composites onto white.

## Status
- Tested on real data: `palette.py` (extraction, recoloring), `baselines.py`.
- NOT run: `cvd.py`, `model.py`, `losses.py`, `train.py`, `evaluate.py`, `render_demo.py` (syntax-checked only; torch was unavailable). Smoke test first: `python train.py --palettes palettes_sample.pkl --epochs 2`, then `evaluate.py`, then `render_demo.py`.
- Fixed-palette baseline cycles when a chart has more series than palette colors (Okabe-Ito has 8, 10 FigureQA charts-in-sample use up to 10 series). Report this, or use `petroff10` for the main comparison.
