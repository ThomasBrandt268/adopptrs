# Local environment — template

Copy this file to `LOCAL.md` (gitignored) and fill it in. It holds
everything that depends on *your* machine, *your* account and *your*
cluster — none of which belongs in `docs/manual/`, which is meant to be
readable by someone working elsewhere.

Keep it up to date: it is the fastest way back after a break, and the
first thing to hand over along with the archives.

---

## Compute

- **Cluster / machine**: <name, institution, scheduler>
- **Access**: <ssh alias, VPN or network requirement>
- **Documentation**: <URL, and whether it is current>

### Partitions and GPUs

| partition | GPU | compute capability | verified wheel |
|---|---|---|---|
| | | | |

> The `torch` wheel must cover every GPU architecture you intend to use.
> A missing one only surfaces at the first computation. Run
> `python tests/smoke.py --no-data` on each partition before committing to
> a long job.

## Paths

| what | where |
|---|---|
| repository (`$REPO`) | |
| Californian imagery | *(a scratch/fast filesystem, never a slow home)* |
| WalOnMap tiles | |
| BDAPPV, resampled | |
| models | |
| scheduler logs | |

Note which filesystems are **persistent** and which are purged.

## Local archives

Artefacts that exist nowhere else — trained models above all.

| archive | contents | location |
|---|---|---|
| | | |

Record the `sha256` of each, or keep the `MANIFEST` files that ship with
them.

## Quirks worth remembering

Anything that cost you an hour once: broken transfer tools, throughput
that has to be throttled, scheduler limits, queues to avoid.
