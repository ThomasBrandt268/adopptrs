# Results

What has been measured, and under which protocol. A figure without its
protocol is worthless here — that principle is what caught two wrong
conclusions on 2026-08-12.

## Protocol

Unless stated otherwise, every number on this page comes from the same
measurement:

- **fold 0 of `via_liege_city.json`** — 133 WalOnMap tiles never seen
  during training, of which 71 contain no panel at all, for ~140
  hand-annotated installations;
- **`-min 256`** — detected shapes below 256 px (~4.5 m² on the ground)
  are discarded as noise;
- **no morphological opening** — this is the chain `walonmap.py` actually
  runs (see [Limitations](limitations.md));
- **negatives scored** — empty tiles count, so false alarms on
  panel-free roofs are not silently ignored. In production the empty tile
  is the majority case.

The raw outputs are versioned under `resources/eval/`, so any figure below
can be recomputed from the counts rather than taken on trust.

## The metrics, and what they answer

**Precision** = TP / (TP + FP) — *when the model announces a panel, is it
right?* At 0.863: out of 100 announced installations, 86 exist and 14 are
roof artefacts (skylights, smoke vents, bright membranes).

**Recall** = TP / (TP + FN) — *of the panels that exist, how many are
found?* At 0.819: 82 found out of 100, 18 missed.

**F1** — the harmonic mean, a single score for ranking two settings.
Convenient, but **blind to purpose**: it weighs a miss and a false alarm
equally, which they are not for a capacity census.

**Surface bias** = predicted area / annotated area, over correctly matched
panels only. 1.00 means correctly sized; **0.944 means contours drawn
5.6 % too small**. It counts neither false positives nor misses, so it is
a bias of *shape*, not of volume — never read it alone. It matters because
the `area` column feeds the capacity estimate directly.

## Production model

**`multiunet_liege_030`**, threshold **5e-5**:

| TP / FP / FN | precision | recall | F1 | surface bias |
|---|---|---|---|---|
| 113 / 18 / 25 | 0.863 | 0.819 | 0.8401 | 0.944 |

Training chain: 20 epochs on Californian rooftops (`SCALE=2`), then 10
epochs fine-tuned on 528 Liège tiles in `-special` mode, with fold 0 held
out so it stays measurable.

:::{important}
The threshold is **not** transferable between models. This one's optimum
is 5e-5; the hue-jitter variant peaks at 3e-5, the BDAPPV chain at 7e-5.
Recalibrate with `calibrate.sbatch` after every training run. The former
default of `0.5` cost about 28 points of recall.
:::

## The choice of model is open

Three models have been measured on the same fold with the same fine grid.
**No single one dominates**, and the ranking flips depending on which
column you care about:

| model | threshold | precision | recall | F1 | surface bias |
|---|---|---|---|---|---|
| `multiunet_liege_030` | 5e-5 | 0.863 | 0.819 | 0.8401 | **0.944** |
| `multiunet_bdwl_040` | 7e-5 | 0.789 | **0.907** | 0.8439 | 0.838 |
| `multiunet_hue_030` | 3e-5 | **0.910** | 0.804 | **0.8538** | 0.816 |

`multiunet_hue_030` has the best F1 and the best precision, yet it is the
**worst choice** for this project: it shrinks contours the most, and the
deliverable is a surface converted to kWp, not a count. This is the
clearest illustration available that a better detection score does not
make a better tool.

`multiunet_bdwl_040` adds a BDAPPV stage (California → BDAPPV Wallonia →
Liège) and buys **9 points of recall**. At equal recall it dominates
outright: around 0.90 recall it holds 0.789 precision where the production
model collapses to 0.672.

**For a census, missing half as many buildings is a strong argument.** For
a capacity estimate the two are equivalent, as the next section shows.
The decision has not been made and should be explicit.

## From surface to capacity

Only the panels that are found contribute, and they are measured slightly
too small. Combining both effects approximates the captured share of real
surface:

| model | panels found | measured | real surface captured |
|---|---|---|---|
| `multiunet_liege_030` @ 5e-5 | 82 % | 5.6 % too small | **≈ 77 %** |
| `multiunet_bdwl_040` @ 5e-5 | 91 % | 14 % too small | **≈ 78 %** |

The two land in the same place by opposite routes: one finds fewer
installations but measures them well, the other finds many more and
shrinks them.

:::{warning}
`recall × surface bias` is an approximation. It assumes missed panels
average the same size as found ones, and it does not subtract the surface
of false positives. The order of magnitude is reliable; the third decimal
is not.
:::

Whichever model is used, a regional figure that ignores this correction
**under-reports installed capacity by roughly a quarter**. This is what
the manifest accompanying each published CSV is meant to carry.

## Methodological warning

The original threshold grid tested one value per decade. It is too coarse
to compare models, and it produced two wrong conclusions that were only
caught on 2026-08-12:

- the production model's own optimum moved from 1e-3 to **5e-5**, worth
  +3.6 points of recall and +5.7 % of captured surface, without retraining
  anything;
- BDAPPV was written off as bringing nothing (F1 0.8314 vs 0.8340). On a
  fine grid it **wins**, and by a wide margin on recall.

**Any model comparison predating 2026-08-12 should be treated as
non-conclusive.** See `resources/eval/README.md` for the full record.
