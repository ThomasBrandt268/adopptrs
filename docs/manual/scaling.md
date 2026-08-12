# Scaling to Wallonia

What a region-wide photovoltaic census would actually require, in measured
numbers rather than estimates. This page exists because the answer is
counter-intuitive: the bottleneck is not the GPU, and the single missing
piece divides the workload by more than a hundred.

## Where the project stands

The detection chain is validated and measured, but it has only ever been
run on **Liège** — 661 annotated tiles for training and measurement, plus
full-city and full-province inference passes. Nothing in the code is
specific to Liège; `walonmap.py` takes any GeoJSON polygon. The obstacle
to covering Wallonia is throughput, not capability.

## The unit of work

A tile is 512 px at 0.132 m/px, so **67.6 m on a side, ~4 570 m² on the
ground**. Wallonia covers roughly 16 800 km², which gives about
**3.7 million tiles** at this zoom level — consistent with the figure
measured directly from the tile grid.

## The bottleneck is the geoportal, not the GPU

This is the finding that reorders everything else.

`walonmap.py` deliberately waits 0.2 s between requests, because WalOnMap
is a public service and firing thousands of requests without pause invites
throttling mid-run. Measured end-to-end, `misc/download.py` fetched the
661 Liège tiles in **3 min 38 s**, i.e. **0.33 s per tile** — the 0.2 s
pause plus ~0.119 s of median WMS latency.

A forward pass of the network on a 512 px tile costs on the order of
10 ms on a GPU. **Downloading therefore accounts for roughly 97 % of the
wall-clock time.** Adding GPUs would not shorten a full-region run.

| scope | tiles | download time |
|---|---|---|
| all of Wallonia | 3 700 000 | **~14 days** |
| tiles containing buildings | 429 000 | **~39 hours** |
| large non-residential roofs | 29 700 | **~2.7 hours** |

## The missing piece: spatial filtering

Going from 3.7 M to 29.7 k tiles is a **factor of 125**, and it turns an
impractical two-week run into an afternoon. This filtering is **not
implemented** — it is the single highest-value piece of work remaining.

The inputs exist. The `wallonia_grid` project holds
`buildings_with_consumption.parquet` with per-building coordinates
(columns `x`, `y`, EPSG:3812, to be reprojected to 31370). Turning those
into a tile list means projecting each building to its `(row, col)` and
deduplicating.

Two design constraints were settled and should be respected:

- **Do not import `buildings.py` into this repository.** The two projects
  keep a file-level boundary on purpose — asymmetric dependencies,
  different release cadences, and a 120 MB model on this side.
- Export a plain `row,col` CSV, and add a `--tiles <file.csv>` option to
  `walonmap.py` so it consumes the list instead of walking a polygon.

## Choosing the scope

The three tiers above are not equivalent, and the choice is a policy
decision rather than a technical one:

- **429 k tiles (all buildings)** is the honest scope for a census
  claiming completeness. It costs ~39 hours of downloading, which is
  feasible over a weekend but warrants contacting the geoportal
  operators first — 429 000 requests against a public service is not a
  polite thing to do unannounced.
- **29.7 k tiles (large non-residential roofs)** targets the installations
  that dominate installed capacity, at a fraction of the cost. It
  structurally misses residential rooftops, which are the majority by
  count.

## Also unresolved at this scale

**Vintage.** Inference must state which imagery year it ran on;
`walonmap.py` accepts `-vintage`. A summer vintage was chosen on measured
grounds, not assumed.

**Recall correction.** The model finds ~82 % of panels (or ~91 % with
`multiunet_bdwl_040`) and draws them ~6 % too small. A region-wide figure
that ignores this under-reports installed capacity by roughly a quarter.
The manifest accompanying each CSV is meant to carry the measured
performance so the downstream consumer can correct for it.

**Ground-mounted installations.** Visual review on 2026-08-12 surfaced
what appear to be ground-mounted arrays that are neither annotated nor
detected. Such panels appear in *no* metric — they are not false
negatives, since that requires an annotation. If the annotation set
systematically misses this category, true recall across the full estate
is lower than measured. This is unquantified and matters more at regional
scale than in a city.
