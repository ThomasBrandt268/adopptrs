# Interface with `wallonia_grid`

ADOPPTRS is not a standalone project: its detections are consumed by
`wallonia_grid` (branch `PVs_integration`, `sourcecode/core/pv/`), which
turns them into installed capacity.

:::{note}
This page states **what ADOPPTRS emits**, not what its current consumer
happens to expect. The consumer is under active development and its
internals will move; anything described here can be relied on as long as
`summarize.py` and `walonmap.py` say so, and those live in this
repository, versioned alongside this page.
:::

## The data contract

The published CSV — `docs/resources/csv/liege_province.csv`, copied into
`wallonia_grid` as `data/opensource/pv/pv_detections_liege_province.csv` —
carries exactly four columns, in this order:

| column | unit | note |
|---|---|---|
| `latitude` | degrees, **EPSG:4326** | |
| `longitude` | degrees, **EPSG:4326** | |
| `area` | m² | **projected** surface, not real panel area |
| `azimuth` | degrees | direction of the roof slope |

:::{important}
**Publish these four columns, in this order, with no missing values.**
`summarize.py -diagnostics` adds an `elongation` column; it is a working
aid and must stay out of published files. A consumer is entitled to
validate the header strictly, and at least one does — so an extra column,
a reordering or a stray blank is a breaking change, not a cosmetic one.
:::

### Settings the numbers depend on

The coordinates and areas above only mean something under the imagery grid
and detection settings that produced them. A consumer cannot read these
from the file, so it will hardcode its own copy — and if this side moves,
the copy goes wrong silently.

| setting | value | set in |
|---|---|---|
| tile grid | `TileMatrix "15"` / `default028mm`, EPSG:31370 | `wms.py` |
| pixel span | 472.4711830375448 × 0.28 mm = **0.1323 m/px** | `wms.py` |
| tile span | 512 px = **67.73 m** | `wms.py` |
| minimum contour | **256 px** (`-min`) | `walonmap.py` |

Changing any of them is a republication event: state the new values in the
manifest below, and treat the previous files as a different dataset rather
than an older one.

## The azimuth column

Stated precisely, because it is easy to misread and a sign error in it is
invisible downstream — the installations still produce, just wrongly, and
a commune total stays plausible.

- a **compass bearing from north**: 0 = north, 90 = east, 180 = south;
- folded to the slope facing south, so values fall in **`[90, 270)`** and
  a due-south roof reads **180**;
- known modulo 180° from a single nadir view, so the rare north-facing
  installation is reported as its southern opposite;
- meaningless on a near-square installation, where the short side is not
  a real direction. `-diagnostics` exposes the `elongation` that says how
  much to trust each row (median 2.46, above 1.2 for 91 % of them).

:::{warning}
**PVGIS counts azimuth from south, not north.** Under that convention due
south is 0, so `azimuth_pvgis = azimuth − 180`. Passing this column
straight to PVGIS turns every installation around.
:::

Files published before 2026-08 carry a different column entirely, and one
that was rightly judged unusable: `summarize.py` then kept only the
`angle` from `cv2.minAreaRect` and discarded the `(w, h)` pair, leaving
the orientation known modulo 90° and crushed into a ~90° band around
south, with 12.9 % of rows at exactly 180°. Reading the angle from the
rectangle's corners instead is what fixed it.

## Why a file boundary rather than a package

Decided on 2026-08-11, and worth restating because packaging looks
tempting: the dependencies are asymmetric, the two projects release at
different cadences, and this side carries a 120 MB model that
`wallonia_grid` has no use for. A shared file with a documented schema
costs less than a coupled dependency.

## Naming convention *(agreed, not implemented)*

```
pv_detections_<zone>_<vintage>.csv
```

with a JSON manifest alongside it, carrying:

- the model that produced it, and its `sha256`
- the decision threshold used
- the imagery vintage
- the **measured performance**: precision, recall, F1, surface bias, and
  the protocol they were measured under

## What the manifest unlocks

It is not bookkeeping. Two consumer-side problems depend on it.

**Recall correction.** The detector finds ~82 % of panels and draws them
~6 % too small, so a raw sum of `area` under-reports installed capacity by
roughly a quarter. Without the measured figures travelling next to the
data, `wallonia_grid` cannot correct for this — and the correction changes
with every model and threshold.

**Which azimuth column a file carries.** The two versions described above
are not comparable, and nothing inside the CSV tells them apart. Only the
manifest can say which one a consumer is holding — until it exists, a
consumer is right to distrust the column and default to a flat assumption,
which is what happens today.

## Rules for publishing a new dataset

:::{warning}
**Never substitute a new dataset for the 2018 one.** Add them side by
side.
:::

Two reasons. The 2018 CSV is the *published and independently evaluated*
result of the original 2020 work — that status is what makes it a credible
cross-check for `wallonia_grid`. A CSV produced by a locally retrained
model does not inherit it. And keeping two vintages measured with the same
detector allows growth between them to be estimated, which a single
snapshot cannot.

Note also that comparing 2018 detections against a present-day capacity
register mixes detection gaps with years of real growth. Publishing the
imagery vintage in the file name and the manifest is what lets a consumer
avoid that confusion.

## Status

| item | state |
|---|---|
| column contract | implemented, stable |
| azimuth from bounding-rectangle corners | **fixed** in `summarize.py` |
| vintage selection (`-vintage`) | implemented |
| naming convention | agreed, **not implemented** |
| JSON manifest | agreed, **not implemented** |
| threshold `5e-5` reflected in published CSVs | **no** — published files predate it |

The last line matters: the CSVs currently published were produced with the
former threshold. Regenerating them is a prerequisite to any claim based
on the figures in [Results](results.md).
