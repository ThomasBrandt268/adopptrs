# Pipeline

From aerial imagery to a capacity estimate. The first section is readable
without knowing the code; the rest descends into it.

## Overview

```
GeoJSON polygon
      │
      ▼   wms.py            tile geometry, WMS requests
  512 px tile @ 0.132 m/px
      │
      ▼   walonmap.py       U-Net forward pass
  probability map (one value per pixel)
      │
      ▼                     threshold  (5e-5)
  binary mask
      │
      ▼                     connected components
  contours
      │
      ▼                     -min 256 px  (~4.5 m²)
  retained detections
      │
      ▼   → VIA JSON        polygons, pixel coordinates
      │
      ▼   summarize.py      projection + bounding rectangle
  CSV: latitude, longitude, area, azimuth   (EPSG:4326)
      │
      ▼   wallonia_grid     installed capacity (kWp)
```

Two properties of this chain explain most of the design decisions
elsewhere in this documentation. The network outputs **a probability per
pixel**, not a verdict — hence a threshold, and hence the fact that it
must be recalibrated per model. And **`area` is a projected surface**:
seen from directly above, a tilted panel appears shorter than it is.

## Stage by stage

### Tiles — `wms.py`

Serves 512 px tiles at 0.132 m/px, indexed by `(row, col)`. A given
`(row, col)` always covers the same ground rectangle regardless of the
imagery year, so only the URL changes with `-vintage`.

:::{note}
The Wallonia geoportal decommissioned its WMTS service on 2026-08-03.
`wms.py` replaced it, freezing the constants of `TileMatrix "15"` read
from an archived snapshot of the original WMTS capabilities. `wmts.py` is
kept but deprecated. Grid alignment was verified against the 661
annotations from 2020.
:::

### Detection — `walonmap.py`

The production entry point. Takes a GeoJSON `Polygon` geometry, walks the
tiles covering it, runs the network, and writes a VIA file of detected
polygons.

```bash
cd python
python walonmap.py -p ../resources/walonmap/liege.geojson \
    -n ../products/models/multiunet_liege_030.pth -multitask \
    -d ../products/liege_city -o ../products/json/liege_city.json
```

It pauses 0.2 s between requests: the geoportal is a public service, and
firing thousands of requests without pause invites throttling mid-run.
This pause is what makes downloading, not the GPU, the bottleneck at
regional scale — see [Scaling to Wallonia](scaling.md).

### Summary — `summarize.py`

Converts the VIA into the CSV consumed downstream, computing each
installation's azimuth.

```bash
python summarize.py -i ../products/json/liege_city.json \
    -o ../products/csv/liege_city.csv
```

**The azimuth is the direction of the roof slope**, which is the direction
of the **short** side of the minimum bounding rectangle: panel rows follow
the ridge, and the slope descends perpendicular to it. Two irreducible
limits remain from a single nadir view — a rectangle cannot tell a slope
from its opposite (resolved by assuming the southern half, right under our
latitudes but wrong for rare north-facing arrays), and a flat roof has no
meaningful azimuth.

## Scripts

| script | role |
|---|---|
| `wms.py` | WMS client and tile geometry |
| `walonmap.py` | production: polygon → detections |
| `summarize.py` | VIA → CSV, azimuth computation |
| `dataset.py` | `VIADataset`, augmentations; also converts the Californian polygons |
| `models.py` | UNet / SegNet, multi-task variants |
| `criterions.py` | `DiceLoss`, `MultiTaskLoss` |
| `train.py` | training |
| `evaluate.py` | measurement over a threshold grid |
| `misc/threshold.py` | reads an `evaluate.py` output, places the threshold |
| `misc/compare_report.py` | compares a run against the original report |
| `misc/download.py` | downloads the tiles listed in a VIA |
| `misc/bdappv.py` | BDAPPV → VIA, with resampling to 0.132 m/px |
| `misc/empty_via.py` | contour → VIA of empty tiles, for negatives |
| `misc/collect_fp.py` | extracts false positives as thumbnails |
| `tests/check_*.py` | visual checks: alignment, predictions, azimuth, vintages |
| `tests/smoke.py` | GPU wheel compatibility, before reserving a node |

## SLURM launchers

The `.sbatch` files live beside the Python they call — they all `cd` into
`python/` and invoke a script there, so separating them by extension would
only put a launcher further from what it launches.

| launcher | runs | writes |
|---|---|---|
| `train.sbatch` | `train.py`, in resumable chunks | `products/models/<NAME>_<epoch>.pth` |
| `evaluate.sbatch` | `evaluate.py` + `compare_report.py` | `products/eval/` |
| `calibrate.sbatch` | `evaluate.py` + `threshold.py` | `products/eval/<net>_liege*.txt` |

**Nothing is hard-coded.** Every parameter is an environment variable with
a default (`${VAR:-default}`), so anything can be overridden from the
command line:

| launcher | overridable |
|---|---|
| `train.sbatch` | `MODEL EPOCHS CHUNK SCALE BATCH SPECIAL HUE NEGATIFS K FOLD DATA VIA` |
| `calibrate.sbatch` | `MODEL SCALE MIN OPENING K FOLD THRESHOLDS EPOCH NAME NET DATA VIA` |
| `evaluate.sbatch` | `MODEL K FOLD EPOCH MIN NAME NET DATA VIA` |

```bash
NET=$REPO/products/models/multiunet_liege_030.pth K=5 FOLD=0 \
    THRESHOLDS="1e-5 2e-5 5e-5 1e-4" sbatch calibrate.sbatch
```

:::{warning}
`train.sbatch` and `evaluate.sbatch` default to the **Californian** data,
`calibrate.sbatch` to **Liège**. Overriding `DATA` and `VIA` is therefore
mandatory when fine-tuning on Liège, and forgetting it costs a job.
:::

## Traps

**The two chains diverge on opening.** `evaluate.py` applies a 5×5
morphological opening; `walonmap.py` does not. Calibrating with one and
running the other makes the threshold meaningless — hence `OPENING=0` as
the default in `calibrate.sbatch`.

**`-min` differs by tool**: 64 in `evaluate.py`, 256 in `walonmap.py`.
Use 256 whenever the measurement is meant to describe production.

**`area` is a projected surface**, not the panel's real area. Downstream
conversion to kWp accounts for it.

**The threshold is per-model.** The default is now 5e-5, the measured
optimum of `multiunet_liege_030`; it is wrong for any other network, and
`walonmap.py` prints it at startup as a reminder.
