# Automatic Detection Of Photovoltaic Panels Through Remote Sensing

Nowadays, photovoltaic panels are playing an increasingly important role in the global production of electrical energy. Unfortunately, since anyone owning a roof could potentially install PV panels, it is quite hard to assess their geographical deployement and, as a consequence, their impact on the electrical grids.

Therefore, this project, named *Automatic Detection Of Photovoltaic Panels Through Remote Sensing* or **ADOPPTRS**, aims to detect photovoltaic panels in high-resolution satellite images.

More specifically, the goal is to detect, as accurately as possible, photovoltaic panels in the [WalOnMap][walonmap] orthorectified images in the [Province of Liège](resources/walonmap/liege_province.geojson).

For further explanations and technicalities, please see the project [report](latex/main.pdf).

> All the photovoltaic installations that have been detected, can be visualized at [francois-rozet.github.io/adopptrs](https://francois-rozet.github.io/adopptrs/).

## Implementation

The [PyTorch](https://pytorch.org/) library has been used to implement and train several neural networks [models](python/models.py) one of which is the well known [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).

> For a short description of the arguments of the scripts (`train.py`, `evaluate.py`, etc.), use `--help`.

### Dependencies

If you wish to run the scripts or the [Jupyter](https://jupyter.org/) notebook(s), you will need to install several `Python` packages including `jupyter`, `torch`, `torchvision`, `opencv`, `matplotlib` and their dependencies.

To do so safely, one should create a new environment :

```bash
virtualenv ~/adopptrs -p python3
source ~/adopptrs/bin/activate
pip3 install -r requirements.txt -y
```

or with the `conda` package manager

```bash
conda env create -f environment.yml
conda activate adopptrs
```

### Networks

The neural networks that have been implemented (cf. [`models.py`](python/models.py)) are [*U-Net*](https://arxiv.org/abs/1505.04597), [*SegNet*](https://arxiv.org/abs/1511.00561) and [*Multi-Task*](https://arxiv.org/abs/1709.05932) versions of them.

The legacy networks are trained with a *Dice loss* while the multi-task ones are trained with a *Multi-Task loss* (cf. [`criterions.py`](python/criterions.py)).

### Augmentation

During training, the dataset is *augmented*, meaning that each image undergoes a different random transformation at each epoch. The transformation is a combination of *rotations* (90°, 180° or 270°), *flips* (horizontal or vertical), *brightness* alteration, *contrast* alteration, *saturation* alteration, *blurring*, *smoothing*, *sharpening*, etc.

This improves greatly the *robustness* of the networks.

### Reproductibility

In order to produce the networks and plots that are presented in the [notebooks](notebooks/), the scripts [`train.py`](python/train.py) and [`evaluate.py`](python/evaluate.py) were used. For instance, to train *Multi-Task U-Net* on `5` folds (except fold `0`) for `20` epochs and then evaluate it on fold `0` :

```bash
python train.py -m unet -multitask -n multiunet_0 -e 20 -s multiunet.csv -k 5 -f 0
python evaluate.py -m unet -multitask -n ../products/models/multiunet_0_020.pth -k 5 -f 0
```

> The output of `evaluate.py` is not very user friendly, it should be improved in a future version.

Concerning the model used for [fine tuning](notebooks/tuning.ipynb), the images were twice upscaled and the whole Californian training set was used.

```bash
python train.py -m unet -multitask -n multiunet_x2 -e 20 -scale 2 -s multiunet_x2.csv -k 0
```

Then it was fine tuned for `10` more epochs on `661` [hand-annotated](resources/walonmap/via_liege_city.json) images.

```bash
python misc/download.py -d ../products/liege/ -i ../resources/walonmap/via_liege_city.json
python train.py -m unet -multitask -n multiunet_x2 -e 10 -r 21 -scale 2 -batch 2 -special -p ../products/liege/ -i ../resources/walonmap/via_liege_city.json -s multiunet_x2.csv -k 0
```

> Note the use of the flag `-special` that removes images cropping and data augmentation.

Afterwards, the fine-tuned model was applied to every images in the [Province of Liège](resources/walonmap/liege_province.geojson).

```bash
python walonmap.py -m unet -multitask -n ../products/models/multiunet_x2_030.pth -p ../resources/walonmap/liege_province.geojson -o ../products/json/liege_province_via.json
```

Finally, the resulting `liege_province_via.json` file was *"summarized"* using

```bash
python summarize.py -i ../products/json/liege_province_via.json -o liege_province.csv
```

which produced the [`liege_province.csv`](docs/resources/csv/liege_province.csv) file.

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

## État du fork (2026-08-03)

**Corrigé** — Le géoportail wallon a décommissionné son service WMTS (`supportedExtensions = WMSServer` seul sur les millésimes 2018-2024, `GetCapabilities` WMTS → HTTP 400). Portage vers WMS dans [`python/wms.py`](python/wms.py), qui fige les constantes du `TileMatrix` `"15"` (CRS, échelle, taille de tuile) relevées sur un [instantané archivé](https://web.archive.org/web/20211024033138/https://geoservices.wallonie.be/arcgis/rest/services/IMAGERIE/ORTHO_2018/MapServer/WMTS/1.0.0/WMTSCapabilities.xml) des capabilities WMTS d'origine (`python/wmts.py` conservé, marqué déprécié). Corrigés également : les incompatibilités Python 3.11+ (`TabError` d'indentation mixte dans `models.py`, `random.sample` sur une vue de dictionnaire dans `dataset.py`) et un `ModuleNotFoundError` dans `misc/download.py` (import cassé quand le script est lancé depuis `python/`, comme documenté ci-dessus). L'alignement du nouveau quadrillage a été vérifié visuellement contre les 661 annotations de 2020 ([`python/tests/check_alignment.py`](python/tests/check_alignment.py)), puis confirmé à l'échelle des 661 tuiles (aucune tuile blanche détectée sur le téléchargement complet).

**État actuel** — Les 661 tuiles annotées de [`via_liege_city.json`](resources/walonmap/via_liege_city.json) se téléchargent en ~3 min 38 s via `misc/download.py`. Aucun modèle entraîné n'est disponible dans ce fork (`products/` est gitignoré, aucun `.pth` republié en amont) : l'entraînement nécessite un GPU, indisponible pour l'instant.

**Bug connu, non corrigé** — `SegNet`/`MultiTaskSegNet` (profondeur par défaut) : la formule `128 * (2 ** i)` dans `SegNet.__init__` fait doubler les canaux du bottleneck à 256 au lieu de 128, ce qui provoque `RuntimeError: Shape of indices should match shape of input` dans `max_unpool2d` (indices sauvegardés pour 128 canaux, appliqués à un tenseur de 256 — reproduit sur les deux classes, `MultiTaskSegNet` héritant du même `forward`). Non corrigé car ce n'est pas le modèle retenu par le rapport ([`latex/main.pdf`](latex/main.pdf)) — *Multi-Task U-Net* est utilisé pour WalOnMap.

**Où reprendre** — Ordre du pipeline : entraînement californien → fine-tuning Liège → inférence WalOnMap → agrégation (§Reproductibilité ci-dessus). Mesures déjà faites : latence WMS ~119 ms médiane (40 requêtes, sans dégradation) ; la Wallonie représente environ 3,7 M de tuiles au niveau de zoom utilisé, ce qui rend un balayage exhaustif coûteux — des pistes de filtrage spatial préalable (couches d'occupation du sol type `HABITAT/TISSU_URBANISE`) sont explorées dans [`python/exploration/`](python/exploration/), non intégrées au pipeline ADOPPTRS.

[walonmap]: https://geoportail.wallonie.be/walonmap
[duke-dataset]: https://energy.duke.edu/content/distributed-solar-pv-array-location-and-extent-data-set-remote-sensing-object-identification
[via]: http://www.robots.ox.ac.uk/~vgg/software/via/
