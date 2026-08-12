# Training, calibrating, measuring

Three steps that must be done in order and never separated: a model
without a recalibrated threshold is unusable, and a measurement on a
contaminated fold is worthless.

:::{note}
**`$REPO`** stands for the repository root, and the commands assume a
SLURM scheduler. Paths and partition names for a particular machine belong
in `LOCAL.md` — see [Setup](setup.md).
:::

## The one rule

:::{important}
**Fold 0 must never enter training.** Those 133 tiles are the only
yardstick against which every figure in this documentation is compared. A
model trained on them cannot be compared to anything.
:::

The fold is computed from the sorted, then shuffled, keys of the VIA file
(`random.seed(0)`, then `i % k == fold`). **Adding tiles to
`via_liege_city.json` therefore reshuffles the whole split** and destroys
the yardstick. Extra data goes into a separate VIA file, passed through
`-negatives`, which is merged *after* the split and only into training.

## The training stages

The production model is built in two stages, a third is optional:

| stage | data | mode | epochs | duration |
|---|---|---|---|---|
| Californian | Californian imagery | `SCALE=2` | 1–20 | **~10.5 h** on one GPU |
| BDAPPV *(optional)* | BDAPPV, resampled | — | 21–30 | ~17 min |
| Liège | WalOnMap tiles | `SPECIAL=1` | last 10 | ~10 min |

Durations measured on a single RTX A5000. The Californian imagery should
sit on a fast filesystem: 601 files are re-read in full at every epoch.

The Californian stage is the expensive one and rarely needs redoing —
`multiunet_020.pth` is the shared starting point of every variant. The
Liège stage is minutes, which is what makes experimentation practical.

`-special` (`SPECIAL=1`) feeds **whole 512 px images** instead of crops
centred on panels. That is how the 257 empty training tiles reach the
network at all, and why `SCALE` has no effect at this stage.

## Reproducing the production model

```bash
# 1. Californian stage (long)
cd $REPO/python
SCALE=2 K=0 EPOCHS=20 sbatch train.sbatch

# 2. seed the Liège stage from it -- a different NAME will not resume
#    from a checkpoint it cannot see
cp $REPO/products/models/multiunet_020.pth \
   $REPO/products/models/multiunet_liege_020.pth

# 3. Liège stage, fold 0 held out
NAME=multiunet_liege SPECIAL=1 K=5 FOLD=0 EPOCHS=30 \
DATA=$REPO/products/liege \
VIA=$REPO/resources/walonmap/via_liege_city.json \
sbatch train.sbatch
```

`train.sbatch` runs in resumable chunks of `CHUNK=5` epochs, so an
interrupted job restarts from its last checkpoint rather than from
scratch. Re-running the same command resumes.

For the three-stage variant (`multiunet_bdwl_040`), insert a BDAPPV stage
between 1 and 3, using `wallonie_0132` as `DATA` — the exact conversion
command is in the docstring of `misc/bdappv.py`.

## Calibrating — never optional

A newly trained network has **its own** threshold. Skipping this step and
reusing a previous value is the single most expensive mistake in this
pipeline.

```bash
G="1e-6 2e-6 5e-6 1e-5 1.5e-5 2e-5 3e-5 5e-5 7e-5 1e-4 2e-4 5e-4 1e-3"
NET=$REPO/products/models/<model>.pth K=5 FOLD=0 \
    THRESHOLDS="$G" sbatch calibrate.sbatch
```

:::{warning}
**Always pass a fine grid.** The historical default tests one value per
decade and is too coarse to compare models: between 1e-5 and 1e-4 the
production model's false positives jump from 14 to 61. That grid produced
two wrong conclusions, both overturned on 2026-08-12. See
[Limitations](limitations.md).
:::

`calibrate.sbatch` measures on the Liège tiles rather than the Californian
fold, with `-negatives` so that false alarms on empty tiles count — in
production the empty tile is the majority case. It writes to
`products/eval/<net>_liege_fin.txt` and prints the curve.

:::{note}
`misc/threshold.py` reports the degenerate threshold 0 as the F1 optimum
on some outputs. Read the table, not the announced line.
:::

## Measuring

The calibration output *is* the measurement: precision, recall, F1 and
surface bias at each threshold. Compare against
[Results](results.md) — the production model's F1 of **0.8401 at 5e-5**
is the number to beat, and it is only comparable if fold 0 stayed intact.

To re-read any past measurement:

```bash
python misc/threshold.py ../resources/eval/<file>.txt -raw
```

Visual confirmation, once the numbers look right:

```bash
python tests/check_predictions.py -n ../products/models/<model>.pth \
    -threshold <t> -no-opening -k 5 -f 0 -l 133 -d ../products/check
```

Its totals must match the calibration table. If they diverge, the visual
script is wrong, not the calibration. Green marks annotations, red
retained predictions, orange those rejected by `-min`.

## Adding negatives

The network already sees negatives — `alt=1` yields one randomly placed
crop per positive crop, and `-special` feeds whole empty tiles. What is
missing is **coverage** of what fools it: logistics depots, large car
parks, steel-deck warehouse roofs.

```bash
# 1. list the tiles covering an industrial zone (GeoJSON Polygon geometry)
python misc/empty_via.py -p ../resources/walonmap/zoning.geojson \
    -o ../resources/walonmap/via_negatifs.json

# 2. download them INTO products/liege -- VIADataset knows a single
#    directory and silently ignores files it does not find there
python misc/download.py -i ../resources/walonmap/via_negatifs.json \
    -d ../products/liege

# 3. check by eye, at full resolution, that none carries a panel

# 4. retrain the Liège stage only
NAME=multiunet_neg SPECIAL=1 K=5 FOLD=0 EPOCHS=30 \
DATA=$REPO/products/liege \
VIA=$REPO/resources/walonmap/via_liege_city.json \
NEGATIFS=$REPO/resources/walonmap/via_negatifs.json \
sbatch train.sbatch
```

`train.py` prints `Negatifs : N tiles added`. **If it reports 0, the tiles
are in the wrong directory** — the run would otherwise be identical to the
previous one, without any error.

Then recalibrate. Always.
