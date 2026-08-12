# Interface with `wallonia_grid`

ADOPPTRS is not a standalone project: it feeds the photovoltaic module of
`wallonia_grid` (branch `PVs_integration`, `sourcecode/core/pv/`). This
page describes the contract between them — part of which is **agreed but
not yet implemented**, and marked as such.

## The data contract

`docs/resources/csv/liege_province.csv` is copied into
`data/opensource/pv/pv_detections_liege_province.csv` and read by
`detections.py`.

| column | unit | note |
|---|---|---|
| `latitude` | degrees, **EPSG:4326** | |
| `longitude` | degrees, **EPSG:4326** | |
| `area` | m² | **projected** surface, not real panel area |
| `azimuth` | degrees | direction of the roof slope |

:::{important}
`wallonia_grid` validates these columns strictly and **rejects any extra
column**. Adding a field to the CSV breaks the consumer. Diagnostic
columns exist behind `summarize.py -diagnostics` and must stay out of
published files.
:::

`wallonia_grid` also derives its `ADOPPTRS_PIXEL_SPAN_M` and
`ADOPPTRS_TILE_SPAN_M` constants from the `TileMatrix "15"` grid. Changing
the zoom level on this side would silently invalidate them.

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

**Per-installation azimuth.** `wallonia_grid` currently applies a
`DEFAULT_AZIMUTH_DEG` (due south) to every installation, because the
azimuth column was judged unusable: `summarize.py` used to discard the
`(w, h)` pair from `cv2.minAreaRect`, leaving the orientation known only
modulo 90°, with 12.9 % of rows sitting at exactly 180°. **This has been
fixed on the ADOPPTRS side** — keeping `(w, h)` resolves the ambiguity —
but the consumer has no way to know which files carry the corrected column
and which carry the old one. The manifest is that signal.

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

Note also that comparing 2018 detections against a 2026 capacity register
mixes detection gaps with eight years of real growth — a confusion
`detections.py` already flags.

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
