# Automatic Detection Of Photovoltaic Panels Through Remote Sensing

Nowadays, photovoltaic panels are playing an increasingly important role in the global production of electrical energy. Unfortunately, since anyone owning a roof could potentially install PV panels, it is quite hard to assess their geographical deployement and, as a consequence, their impact on the electrical grids.

Therefore, this project, named *Automatic Detection Of Photovoltaic Panels Through Remote Sensing* or **ADOPPTRS**, aims to detect photovoltaic panels in high-resolution satellite images.

More specifically, the goal is to detect, as accurately as possible, photovoltaic panels in the [WalOnMap][walonmap] orthorectified images in the [Province of Liège](resources/walonmap/liege_province.geojson).

For further explanations and technicalities, please see the project [report](latex/main.pdf).

> All the photovoltaic installations that have been detected, can be visualized at [francois-rozet.github.io/adopptrs](https://francois-rozet.github.io/adopptrs/).

**This is a fork.** ADOPPTRS is the 2020 work of [François Rozet][upstream]; the networks, the training pipeline, the annotations and the published detections are his, under the MIT licence kept in [`LICENSE`](LICENSE). The fork ports the decommissioned WMTS service to WMS, modernizes the stack, retrains the network on Liège and documents the result. Its own documentation is [`docs/manual/`](docs/manual/) — a *Status* part readable without background, and a *Handover* part for taking the code over.

Two things follow. The map linked above shows the detections **published in 2020, on 2018 imagery**; no CSV has been regenerated since. And figures measured by the fork are not comparable to the report's — different training data, different threshold, different imagery vintage.

## Implementation

The [PyTorch](https://pytorch.org/) library has been used to implement and train several neural networks [models](python/models.py) one of which is the well known [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).

> For a short description of the arguments of the scripts (`train.py`, `evaluate.py`, etc.), use `--help`.

### Dependencies

If you wish to run the scripts or the [Jupyter](https://jupyter.org/) notebook(s), you will need to install several `Python` packages including `jupyter`, `torch`, `torchvision`, `opencv`, `matplotlib` and their dependencies.

To do so safely, one should create a new environment :

```bash
python3 -m venv ~/adopptrs-venv
source ~/adopptrs-venv/bin/activate
pip install -r requirements.txt
```

or with the `conda` package manager

```bash
conda env create -f environment.yml
conda activate adopptrs
```

Then check the installation, before reserving a GPU for good :

```bash
cd python
python tests/smoke.py         # add --net for a single real WMS request
```

> On a GPU, the `torch` wheel must be chosen for the target architecture: a missing one only shows up at the first computation, as `no kernel image is available for execution on the device`. Install `torch` from the PyTorch index **first**, then the rest from PyPI — `--index-url` replaces PyPI rather than adding to it. See [Setup](docs/manual/setup.md), and keep machine-specific paths and partitions in a `LOCAL.md` (gitignored, cf. [`LOCAL.example.md`](LOCAL.example.md)).

### Networks

The neural networks that have been implemented (cf. [`models.py`](python/models.py)) are [*U-Net*](https://arxiv.org/abs/1505.04597), [*SegNet*](https://arxiv.org/abs/1511.00561) and [*Multi-Task*](https://arxiv.org/abs/1709.05932) versions of them.

The legacy networks are trained with a *Dice loss* while the multi-task ones are trained with a *Multi-Task loss* (cf. [`criterions.py`](python/criterions.py)).

### Augmentation

During training, the dataset is *augmented*, meaning that each image undergoes a different random transformation at each epoch. The transformation is a combination of *rotations* (90°, 180° or 270°), *flips* (horizontal or vertical), *brightness* alteration, *contrast* alteration, *saturation* alteration, *blurring*, *smoothing*, *sharpening*, etc.

This improves greatly the *robustness* of the networks.

> A *hue* alteration is also available through `-hue`, disabled by default. It has no effect under `-special`, which applies no colour augmentation.

### Reproductibility

The production model is built in two stages: a long Californian one, then a short fine-tuning on hand-annotated Liège tiles. The first stage rarely needs redoing — it is the shared starting point of every variant — while the second takes minutes.

```bash
cd python
python train.py -m unet -multitask -n multiunet -e 20 -scale 2 -batch 5 \
    -i ../products/json/california.json -p ../resources/california/ \
    -d ../products/models -s ../products/csv/multiunet.csv -k 0
```

> `train.py` only writes a `.pth` at the last epoch of the requested range: a run interrupted at epoch 19 of 20 leaves nothing. Call it in shorter ranges with `-r` to make it resumable.

The `661` [hand-annotated](resources/walonmap/via_liege_city.json) tiles are then downloaded and the model fine tuned for `10` more epochs, fold `0` held out.

```bash
python misc/download.py -i ../resources/walonmap/via_liege_city.json -d ../products/liege
cp ../products/models/multiunet_020.pth ../products/models/multiunet_liege_020.pth
python train.py -m unet -multitask -special -n multiunet_liege -e 10 -r 21 -scale 2 -batch 2 \
    -i ../resources/walonmap/via_liege_city.json -p ../products/liege \
    -d ../products/models -s ../products/csv/multiunet_liege.csv -k 5 -f 0
```

> Note the use of the flag `-special` that removes images cropping and data augmentation. The copy seeds the second stage from the first: `train.py` resumes from the checkpoint matching its own `-n`, and will not find one under a different name.

Every network has **its own** decision threshold, and reusing a previous value is the most expensive mistake in this pipeline. It is calibrated on the Liège tiles, with `-negatives` so that false alarms on empty tiles count, and without the morphological opening — that is the chain `walonmap.py` actually runs.

```bash
python evaluate.py -m unet -multitask -n ../products/models/multiunet_liege_030.pth \
    -i ../resources/walonmap/via_liege_city.json -p ../products/liege \
    -k 5 -f 0 -min 256 -negatives -no-opening \
    -thresholds "1e-6 2e-6 5e-6 1e-5 1.5e-5 2e-5 3e-5 5e-5 7e-5 1e-4 2e-4 5e-4 1e-3" \
    | tee ../products/eval/multiunet_liege_030_liege_fin.txt
python misc/threshold.py ../products/eval/multiunet_liege_030_liege_fin.txt -raw
```

> Always pass an explicit grid. The historical default tests one value per decade and is too coarse to compare two networks — between `1e-5` and `1e-4` the production model's false positives jump from `14` to `61`.

The calibrated model is then applied to every image in the [Province of Liège](resources/walonmap/liege_province.geojson), and the resulting file *"summarized"* into a CSV.

```bash
python walonmap.py -m unet -multitask -n ../products/models/multiunet_liege_030.pth \
    -p ../resources/walonmap/liege_province.geojson -threshold 5e-5 \
    -o ../products/json/liege_province_via.json
python summarize.py -i ../products/json/liege_province_via.json -o liege_province.csv
```

> Unlike the original chain, these commands do **not** reproduce the published [`liege_province.csv`](docs/resources/csv/liege_province.csv): that file dates from 2020, no trained model is republished here (`products/` is gitignored), and the intermediate VIA file that produced it was not kept. They regenerate an equivalent from scratch.

The measured performance of each stage, the fold discipline these commands assume, and the SLURM wrappers used to run them are in [Training](docs/manual/training.md) and [Results](docs/manual/results.md).

## Training data

For training our models, we used the [Distributed Solar PV Array Location and Extent Data Set for Remote Sensing Object Identification][duke-dataset] provided by [Duke University Energy Initiative](https://energy.duke.edu/).

This dataset contains the geospatial coordinates and border vertices for over `19 000` solar panels across `601` high resolution images from four cities in California.

```bash
wget "https://ndownloader.figshare.com/articles/3385780/versions/3" -O polygons.zip
wget "https://ndownloader.figshare.com/articles/3385828/versions/1" -O Fresno.zip
wget "https://ndownloader.figshare.com/articles/3385789/versions/1" -O Modesto.zip
wget "https://ndownloader.figshare.com/articles/3385807/versions/1" -O Oxnard.zip
wget "https://ndownloader.figshare.com/articles/3385804/versions/1" -O Stockton.zip
mkdir -p resources/california/
unzip polygons.zip -d resources/california/
unzip Fresno.zip -d resources/california/
unzip Modesto.zip -d resources/california/
unzip Oxnard.zip -d resources/california/
unzip Stockton.zip -d resources/california/
rm *.zip resources/california/*.xml # optionally
```

Afterwards, the file `SolarArrayPolygons.json` has to be converted to the [VGG Image Annotator][via] format.

```bash
python3 python/dataset.py --output products/json/california.json --path resources/california/
```

> The Californian images are `45` GB and are re-read in full at every epoch; they should sit on a fast filesystem rather than a home directory.

## Documentation

[`docs/manual/`](docs/manual/) documents what this fork measured and what it did not. It reads in either order:

- **Status** — [what it detects and how well](docs/manual/results.md), [what it misses](docs/manual/limitations.md), and [what covering all of Wallonia would take](docs/manual/scaling.md).
- **Handover** — [installing](docs/manual/setup.md), [the pipeline end to end](docs/manual/pipeline.md), [training and calibrating](docs/manual/training.md), [the CSV contract](docs/manual/interface.md) and the [module reference](docs/manual/api.md).

[walonmap]: https://geoportail.wallonie.be/walonmap
[duke-dataset]: https://energy.duke.edu/content/distributed-solar-pv-array-location-and-extent-data-set-remote-sensing-object-identification
[via]: http://www.robots.ox.ac.uk/~vgg/software/via/
[upstream]: https://github.com/francois-rozet/adopptrs
