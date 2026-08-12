# ADOPPTRS — status and handover

Detection of photovoltaic installations from aerial imagery, applied to
Wallonia. This documentation answers two questions that the code alone
cannot: **what has been established**, and **what remains to be done** to
turn a working detector into a regional census.

It is written for two readers. The *Status* pages require no familiarity
with the code. The *Handover* pages assume you intend to run it.

:::{note}
This is a fork of the original ADOPPTRS by François Rozet, extended for
Belgian imagery (WalOnMap) and regional-scale application.
:::

## In one paragraph

A U-Net segments photovoltaic panels on 512 px orthophoto tiles at
0.132 m/px. Trained on Californian rooftops, then fine-tuned on manually
annotated tiles from Liège, it is measured on 133 tiles it has never seen.
At its calibrated threshold it finds **82 % of annotated panels with 86 %
precision**, and draws their contours about **6 % too small**. Detections
are exported as a CSV of coordinates, areas and azimuths, which downstream
converts into installed capacity.

## What is settled

- The model reproduces, then exceeds, the reference evaluation of the
  original report.
- The decision threshold is **not** transferable between models: it must
  be recalibrated for each one, on a fine grid.
- Post-processing levers are exhausted. Remaining gains come from data.

## What is not

- **Spatial filtering is not implemented**, and it is what makes a
  region-wide run practical — see [Scaling to Wallonia](scaling.md).
- The choice of production model is **open**: one variant finds 91 % of
  installations instead of 82 %, at the cost of precision.
- Ground-mounted installations may be systematically missing from the
  annotation set, and therefore from every metric.

```{toctree}
:caption: Status
:maxdepth: 2

results
limitations
scaling
```

```{toctree}
:caption: Handover
:maxdepth: 2

setup
pipeline
training
interface
api
```
