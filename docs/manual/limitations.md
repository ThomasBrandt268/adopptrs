# Limitations and dead ends

Where the detector fails, what the metrics cannot see, and which avenues
were explored and closed. This page exists so that a successor does not
spend a week rediscovering any of it.

## How the detector fails

Two families of error, identified visually with
`tests/check_predictions.py` and confirmed on 2026-08-12.

### False positives: regular geometry the model has never seen

**Bright ribbed structures** — steel-deck warehouse roofs, carports,
veranda roofs. Fine parallel ribs *in one direction only*.

**Aligned dark rectangles** — shipping containers in a depot, cars in a
parking lot. From above, a regular grid of dark rectangles has exactly the
signature of a panel array.

The model is not being unreasonable: neither Liège city centre nor
Californian suburbs contain a logistics zone, so it has never had the
chance to learn that these are not panels. **This is a coverage problem,
not a mechanism problem** (see the note on negatives below).

A practical discriminator, at 13 cm/px:

| | photovoltaic | something else |
|---|---|---|
| tone | dark, darker than the roof | bright, glossy |
| texture | grid, joints in **both** directions | fine ribs in one direction |
| opacity | opaque | you can see what is underneath |
| edge | clean rectangle aligned to the roof plane | follows an annex's shape |

### False negatives: low contrast and ground mounts

Dark panels on dark roofing, and **large ground-mounted arrays**. The
latter are the more serious case, for a reason that has nothing to do with
the model — see below.

## What the metrics cannot see

:::{danger}
A panel that is **neither annotated nor detected** appears in *no* counter.
It is not a false negative — that requires an annotation to have been
missed. It is simply absent.
:::

Recall is therefore computed **over the panels the annotator saw**. Visual
review on 2026-08-12 surfaced what appear to be ground-mounted arrays,
neither annotated nor detected, in a tile where the model had instead
flagged a neighbouring warehouse roof. If the annotation set
systematically misses that category, true recall across the full estate is
**lower than measured**, and the capacity shortfall larger than the
quarter estimated in [Results](results.md).

This is unquantified. It matters more at regional scale than in a city
centre, and it is the most consequential open question in the project.

### Annotation gaps

The same review found a tile where the model's "false positives" were
**real panels the annotation had missed**. Two consequences:

- measured precision is a **floor**, not the true value;
- worse, at training time those panels are taught as background — the
  network is actively trained to ignore real installations.

Correcting one such tile would move F1 from 0.840 to roughly 0.853.
Nothing indicates it is the only one. Usefully, **the model points at the
gaps in its own ground truth**: reviewing its false positives finds them.

### Ambiguity the imagery cannot resolve

Some cases cannot be settled from aerial imagery at all. Photovoltaic
carport canopies exist and are increasingly common; from above they are
hard to separate from a tinted polycarbonate canopy — both dark,
rectangular, modular. Opacity is the discriminator, and it is not always
readable.

Where a careful human hesitates, the model has no chance: it sees the same
pixels, without the context or the ability to check on the ground.

## Explored and closed

### Morphological opening — rejected, robustly

`evaluate.py` applies a 5×5 opening that `walonmap.py` does not. Measured
on a fine grid: F1 **0.8306 with** opening against **0.8401 without**, and
recall falls from 0.819 to 0.752. It removes a few false positives but
costs more true detections. Hence `OPENING=0` as the default in
`calibrate.sbatch` — calibrating one chain and running another would make
the threshold meaningless.

:::{note}
A corollary worth knowing: the CSV published in 2020 was produced without
opening, while the report quoted figures measured with it.
:::

### `-min 256` — closed, but the reasoning was wrong

The hypothesis that it excludes small residential installations is doubly
false: it rejects almost nothing, and what passes just above it is wrong
about half the time. If anything it is too permissive.

### BDAPPV alone — unusable, but not the whole story

`multiunet_bdw_030` (California → BDAPPV, no Liège stage) never exceeds
0.53 precision, with a surface bias of 76 — it paints blobs. It has never
seen the WalOnMap sensor.

**But adding the Liège stage changes everything**: `multiunet_bdwl_040` is
the best-recall model available (0.907). BDAPPV was wrongly written off on
2026-08-11 on a coarse threshold grid. See [Results](results.md).

### Hue jitter — works, but wrong for this deliverable

`HUE=0.1` colour augmentation was tried to reduce dependence on roof
colour. It succeeds on its own terms — best F1 (0.8538) and best precision
(0.910) of any model — but produces the **worst surface bias** (0.816).
Since the deliverable is a surface converted to kWp, it is rejected. Worth
retesting if negatives are ever added, since its weakness and theirs are
complementary.

### The "negatives lock" — it does not exist

It was believed that `VIADataset` structurally prevented learning from
absence, because `clusterize()` yields no crop for an image without
polygons. **Checked in the code on 2026-08-12: two mechanisms already
bypass it.**

- `alt=1`, which `train.py` always passes, yields **one randomly placed
  crop for every positive crop** (`dataset.py:159-166`);
- the `-special` mode used for the Liège stage sets `size=None`, which
  yields **whole images**, empty tiles included.

The lock only applies to an image entirely without polygons in normal
mode — effectively never, on the Californian set. **Do not rewrite
`VIADataset`.** What is missing is coverage of industrial zones, and
`misc/empty_via.py` plus `train.py -negatives` are already in place for
that.

### Manual triage of false positives — abandoned

`misc/collect_fp.py` extracts each false positive as a thumbnail for
sorting. At 13 cm/px the crops proved too blurry for reliable judgement
(and were saved as JPEG on top of already-JPEG tiles). The precedent that
motivated the approach — 251 empty tiles taking a model from 0.11 to 0.91
precision — was obtained **without fine triage**, so targeted selection is
a second-order refinement. The script remains useful for spotting
annotation gaps.

## Methodological lesson

Two conclusions were overturned in a single day, both for the same reason:
a threshold grid with one value per decade is too coarse to separate
models whose F1 differ by 0.003. Between 1e-5 and 1e-4 the production
model's false positives jump from 14 to 61 — an optimum can hide anywhere
in that gap.

**Treat any model comparison predating 2026-08-12 as non-conclusive**, and
always calibrate on the fine grid documented in
`resources/eval/README.md`.
