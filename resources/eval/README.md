# Evaluation outputs

Raw `evaluate.py` output — the measurements behind every performance figure
claimed in this project. Each file holds two matrices (contour-wise and
pixel-wise), one row per decision threshold, with columns
`TP, FP, FN, precision, recall`.

They are kept under version control because they cannot be regenerated
without the trained models, which live outside the repository (see
`docs/` for the archive). At ~3 kB each, they cost nothing and let a
reader recompute any F1 from the raw counts rather than trusting a number.

## How to read them

Liege measurements (thresholds, surface bias, per-model calibration):

```bash
cd python
python misc/threshold.py ../resources/eval/<file>.txt -raw
```

`-raw` disables the "precision := 0 when no target is missed" correction.
It is the right choice here because these runs used `-negatives`, where
the artefact it patches is already visible in the false positives.

The Californian file is different — it reproduces the report's protocol
and is read with `misc/compare_report.py`, without `-raw`.

## Protocol

Unless stated otherwise: fold 0 of `via_liege_city.json` (133 tiles never
seen during training, 71 of them empty, ~140 annotated panels),
`-min 256`, **no morphological opening**, negatives scored. This is the
chain `walonmap.py` actually runs.

## Files

| file | model | grid | best threshold | recall | F1 | surface bias |
|---|---|---|---|---|---|---|
| `multiunet_liege_030_liege_fin.txt` | production | fine | **5e-5** | 0.819 | **0.8401** | 0.944 |
| `multiunet_bdwl_040_liege_fin.txt` | California→BDAPPV→Liege | fine | 7e-5 | **0.907** | **0.8439** | 0.838 |
| `multiunet_hue_030_liege_fin.txt` | hue jitter 0.1 | fine | 3e-5 | 0.804 | 0.8538 | 0.816 |
| `multiunet_liege_030_liege_ouverture_fin.txt` | production **+ opening** | fine | 2e-4 | 0.752 | 0.8306 | 0.909 |
| `multiunet_liege_030_liege.txt` | production | per-decade | 1e-3 | 0.752 | 0.8340 | 0.879 |
| `multiunet_bdwl_040_liege.txt` | California→BDAPPV→Liege | per-decade | 1e-3 | 0.763 | 0.8314 | 0.862 |
| `multiunet_hue_030_liege.txt` | hue jitter 0.1 | per-decade | 1e-5 | 0.856 | 0.8322 | 0.858 |
| `multiunet_liege_030_liege_ouverture.txt` | production **+ opening** | per-decade | 1e-4 | 0.759 | 0.8221 | 0.932 |
| `multiunet_bdw_030_liege.txt` | California→BDAPPV, no Liege stage | per-decade | — | — | degenerate | 76.7 |
| `multiunet_0_020.txt` | cross-validation (K=5, fold 0) | per-decade | — | — | — | — |

`multiunet_bdw_030` never saw the WalOnMap sensor: precision never exceeds
0.53 and the surface bias of 76 means the network paints blobs. It is not
a usable model, only the starting point of `bdwl`.

## Warning: the per-decade grid is not conclusive

The original threshold grid tested one value per decade
(`1e-9 … 1e-1, 0.5, 1-1e-1 …`). It is too coarse to compare models: it
steps over their true optima, and between 1e-5 and 1e-4 the production
model's false positives jump from 14 to 61.

Measured on a fine grid on 2026-08-12, two conclusions drawn from the
coarse one were overturned:

- the production model's own optimum moved from 1e-3 (F1 0.8340) to
  **5e-5 (F1 0.8401)**, worth +3.6 points of recall;
- BDAPPV was dismissed as bringing nothing (0.8314 vs 0.8340). On the
  fine grid `bdwl` **wins**: 0.8439, and above all 0.907 recall against
  0.819.

**Any model comparison made before 2026-08-12 should be treated as
non-conclusive.** The per-decade files are kept as a record of what was
measured, not as evidence.

The opening result held up on both grids: it costs more recall than it
saves false positives, which is why `OPENING=0` is the default in
`calibrate.sbatch`.

## Regenerating

```bash
cd python
G="1e-6 2e-6 5e-6 1e-5 1.5e-5 2e-5 3e-5 5e-5 7e-5 1e-4 2e-4 5e-4 1e-3"
NET=$HOME/adopptrs/products/models/<model>.pth K=5 FOLD=0 THRESHOLDS="$G" \
    sbatch calibrate.sbatch
```
