# Setup

:::{note}
This page describes what the project needs, not where any particular
person keeps it. Machine-specific details — cluster name, partitions,
paths, archive locations — belong in a local `LOCAL.md`; see
`LOCAL.example.md` at the repository root for the template.

Throughout, **`$REPO`** stands for the repository root.
:::

## Cloning the repository is not enough

The repository holds the code, the 661 Liège annotations and the
evaluation records. Everything else — imagery, converted datasets, trained
models — lives outside it, and each piece is obtained differently.

| artefact | in repo | how to obtain |
|---|---|---|
| code, job launchers | ✅ | — |
| `resources/walonmap/via_liege_city.json` (661 annotations) | ✅ | hand-made, irreplaceable |
| `resources/eval/*.txt` (all measurements) | ✅ | — |
| Californian imagery, 45 GB / 601 files | ❌ | figshare, see README |
| `products/json/california.json` | ❌ | `python dataset.py` (README) |
| WalOnMap tiles, 661 files | ❌ | `misc/download.py`, a few minutes |
| BDAPPV, resampled to 0.132 m/px | ❌ | Zenodo + `misc/bdappv.py` |
| **trained models, 6 files / 360 MB** | ❌ | **no public source** |

:::{danger}
The trained models are the one artefact that cannot be obtained from
anywhere. The Californian stage alone represents more than ten hours of
GPU time, and every other model derives from it. They currently exist
only as private archives.

Publishing them — Zenodo gives a DOI and outlives any individual account
— would remove the project's single point of failure. Until then, whoever
hands the project over must hand over the archives with it.
:::

## Environment

```bash
conda env create -f environment.yml
conda activate adopptrs
```

`environment.yml` is **CPU-only**, deliberately: scientific packages come
from pip rather than conda, and the GPU wheel is a separate matter. This
is enough to re-read evaluation outputs, convert datasets and inspect
results — everything except training and inference at scale.

## GPU wheels

:::{warning}
PyTorch no longer compiles every GPU architecture into every wheel, and a
missing one **only surfaces at the first computation**, as
`no kernel image is available for execution on the device` — potentially
hours into a job.
:::

Pick the CUDA build that covers every GPU architecture you intend to use,
then install it *before* the rest:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/<build>
pip install -r requirements.txt
```

The order matters: `--index-url` **replaces** PyPI rather than adding to
it, and the PyTorch index carries neither `opencv-python` nor `pyproj`.
The two commands must stay separate.

Verify before reserving a GPU for real:

```bash
cd python && python tests/smoke.py --no-data
```

It compares the GPU's compute capability against `torch.cuda.get_arch_list()`
and forces an actual computation, so an incompatibility surfaces here
rather than mid-job. It also accounts for CUDA's binary compatibility
across minor revisions, which a naive comparison would miss.

Record in `LOCAL.md` which build you verified, and on which hardware.

## Compute requirements

Training is the only demanding step, and only its first stage:

| stage | cost |
|---|---|
| Californian stage, 20 epochs | ~10.5 h on one modern GPU *(measured on an RTX A5000, 24 GB)* |
| fine-tuning stage, 10 epochs | ~10 minutes |
| calibration | ~20 minutes |

The launchers assume **SLURM** (`sbatch`, `srun`, partitions). Adapting
them to another scheduler means changing the `#SBATCH` headers; everything
below them is plain shell.

A fast scratch filesystem matters more than it looks: the Californian
imagery is 601 files re-read in full at every epoch, which a slow or
network-mounted home directory will not sustain.

## Getting the data

**Californian imagery** — figshare links and the conversion to VIA format
are in the README. Roughly 45 GB.

**WalOnMap tiles** — everything needed is already in the repository:

```bash
cd $REPO/python
python misc/download.py -i ../resources/walonmap/via_liege_city.json \
    -d ../products/liege
```

`-vintage` selects the imagery year; the default matches the year the
annotations were made.

**BDAPPV** — [Zenodo 7358126](https://zenodo.org/record/7358126),
CC-BY-4.0, though the Google imagery remains subject to Google's terms.
The exact conversion command is in the docstring of `misc/bdappv.py`.

**Trained models** — from wherever they were archived. Retrieving them
from a cluster is an ordinary copy:

```bash
scp -O <host>:<path>/models.zip .
```
