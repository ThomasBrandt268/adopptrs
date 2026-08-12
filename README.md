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
python3 -m venv ~/adopptrs-venv
source ~/adopptrs-venv/bin/activate
pip install -r requirements.txt
```

or with the `conda` package manager

```bash
conda env create -f environment.yml
conda activate adopptrs
```

Then check the installation :

```bash
cd python
python tests/smoke.py         # add --net for a real WMS request
```

#### Cluster GPU

Le choix de la roue `torch` est contraint par le GPU : PyTorch ne compile plus toutes les architectures dans toutes ses roues, et une architecture absente ne se manifeste qu'au premier calcul, par un `no kernel image is available for execution on the device`.

| Build | Architectures compilées | GPU couverts |
| --- | --- | --- |
| `cu126` | 5.0 ; 6.0 ; 7.0 ; 7.5 ; 8.0 ; 8.6 ; 9.0 | Pascal, **Volta**, **Turing**, Ampere, Hopper |
| `cu128` / `cu129` | 7.5 ; 8.0 ; 8.6 ; 9.0 ; 10.0 ; 12.0 | Turing et plus récent |
| `cu13x` | Turing et plus récent | Turing et plus récent |

Choisir la build qui couvre **toutes** les architectures GPU visées, puis l'installer avant le reste :

```bash
# 1. torch et torchvision depuis l'index PyTorch, qui seul distribue les roues CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/<build>

# 2. le reste depuis PyPI ; torch est déjà satisfait, pip ne le réinstalle pas
pip install -r requirements.txt

# 3. vérification avant de réserver un GPU pour de bon
cd python && python tests/smoke.py --no-data
```

> L'ordre importe : `--index-url` **remplace** PyPI au lieu de s'y ajouter, et l'index PyTorch ne distribue ni `opencv-python`, ni `pyproj`, ni `matplotlib`. Les deux commandes doivent donc rester séparées.
>
> À l'étape 3, `tests/smoke.py` compare la *compute capability* du GPU à `torch.cuda.get_arch_list()`, puis force un calcul réel pour lever l'erreur au bon endroit plutôt qu'au milieu d'un job.
>
> Attention à la compatibilité binaire de CUDA : une roue peut ne pas contenir le noyau exact d'une carte tout en la faisant fonctionner, via une révision mineure inférieure de la même génération. Une comparaison naïve entre `get_arch_list()` et la *compute capability* conclurait à tort à une incompatibilité ; `tests/smoke.py` en tient compte.

Le relevé des partitions d'un cluster donné, la build vérifiée sur chacune et les chemins de données n'ont rien à faire ici : ils dépendent de la machine. Les consigner dans un `LOCAL.md` (gitignoré, cf. [`LOCAL.example.md`](LOCAL.example.md)).

La précision mixte (`torch.amp`) n'est pas implémentée dans [`train.py`](python/train.py), donc la présence de *Tensor Cores* ne change rien pour l'instant.

##### Soumettre un entraînement

[`python/train.sbatch`](python/train.sbatch) encapsule l'appel à `train.py` pour SLURM. Dans ce qui suit, `$REPO` désigne la racine du dépôt.

```bash
mkdir -p $REPO/products/logs   # une seule fois : sans ce dossier, SLURM refuse le job
cd $REPO/python

EPOCHS=1 sbatch train.sbatch        # essai court, pour mesurer le coût d'une époque
K=5 FOLD=0 sbatch train.sbatch      # validation croisée, évaluable par evaluate.py
SCALE=2 sbatch --time=20:00:00 train.sbatch   # modèle de production (cf. §Reproductibilité)
```

`train.py` n'écrit un `.pth` qu'à la dernière époque de la plage demandée : un job interrompu à l'époque 19 sur 20 ne laisse rien. Le script l'appelle donc par tranches de `CHUNK` époques (5 par défaut) et repart automatiquement du dernier checkpoint trouvé — relancer la même commande après une interruption reprend où l'on s'était arrêté. Le surcoût d'une tranche est d'une vingtaine de secondes, négligeable devant le coût d'une époque.

Suivi : `squeue --user $USER`, `tail -f $REPO/products/logs/<jobid>.log`, `sacct -j <jobid>` une fois terminé.

##### Évaluer un modèle

[`python/evaluate.sbatch`](python/evaluate.sbatch) évalue le fold laissé de côté par `K=5`, puis compare les chiffres obtenus à ceux du rapport.

```bash
cd $REPO/python

sbatch evaluate.sbatch                   # multiunet_0_020.pth sur le fold 0
FOLD=1 sbatch evaluate.sbatch            # un autre fold
EPOCH=10 sbatch evaluate.sbatch          # un checkpoint intermédiaire
```

Un passage sans rétropropagation sur un cinquième du jeu : **2 min 45 s** mesurées sur un RTX A5000, l'essentiel étant du CPU (décodage TIFF, morphologie OpenCV sur 21 seuils). Une partition GPU moins récente ne coûte donc presque rien de plus.

La sortie brute d'`evaluate.py` — deux matrices 21 × 5 — est enregistrée dans `products/eval/`, puis relue par [`misc/compare_report.py`](python/misc/compare_report.py), qui la met en regard des chiffres publiés :

```bash
python misc/compare_report.py ../products/eval/multiunet_0_020.txt -m multiunet -f 0
```

Les tableaux de référence sont ceux de [`misc/hardcoded_plots.py`](python/misc/hardcoded_plots.py), qui sont précisément les sorties d'`evaluate.py` ayant produit les figures du rapport. Le script les relit avec `ast` plutôt que par un `import` — `hardcoded_plots.py` trace ses figures dès l'import et réclame matplotlib avec LaTeX.

Deux retouches manuelles signalées par le N.B. d'`evaluate.py` sont réappliquées aux deux jeux de chiffres, faute de quoi la comparaison porterait sur des points aberrants : aux seuils très bas, la sortie contour-wise est une tache unique qui recouvre tout et « attrape » toutes les cibles (précision forcée à 0) ; aux seuils très hauts, sortie et cible sont vides et le rappel `0/0` vaut 1 par défaut (forcé à 0). Le script recalcule donc précision et rappel à partir des comptes bruts, identiquement des deux côtés.

Il sort en `1` si l'écart d'AP dépasse la tolérance (`-tol`, 0,05 par défaut) : le job apparaît alors `FAILED` dans `sacct`, ce qui est le signal recherché. Repère utile : sur les cinq folds du rapport, l'AP de détection de *Multi-Task U-Net* s'étale de 0,829 (fold 0, le plus sévère) à 0,919.

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

## État du fork (2026-08-10)

**Portage WMTS → WMS (2026-08-03)** — Le géoportail wallon a décommissionné son service WMTS (`supportedExtensions = WMSServer` seul sur les millésimes 2018-2024, `GetCapabilities` WMTS → HTTP 400). Portage vers WMS dans [`python/wms.py`](python/wms.py), qui fige les constantes du `TileMatrix` `"15"` (CRS, échelle, taille de tuile) relevées sur un [instantané archivé](https://web.archive.org/web/20211024033138/https://geoservices.wallonie.be/arcgis/rest/services/IMAGERIE/ORTHO_2018/MapServer/WMTS/1.0.0/WMTSCapabilities.xml) des capabilities WMTS d'origine (`python/wmts.py` conservé, marqué déprécié). Corrigés également : les incompatibilités Python 3.11+ (`TabError` d'indentation mixte dans `models.py`, `random.sample` sur une vue de dictionnaire dans `dataset.py`) et un `ModuleNotFoundError` dans `misc/download.py` (import cassé quand le script est lancé depuis `python/`, comme documenté ci-dessus). L'alignement du nouveau quadrillage a été vérifié visuellement contre les 661 annotations de 2020 ([`python/tests/check_alignment.py`](python/tests/check_alignment.py)), puis confirmé à l'échelle des 661 tuiles (aucune tuile blanche détectée sur le téléchargement complet).

**Environnement modernisé (2026-08-10)** — L'environnement d'origine (Python 3.8, `torch` 1.4, 2020) était en fin de vie et surtout plafonné à CUDA 10.1, donc inutilisable sur les GPU d'un cluster. Il a été remplacé par Python 3.12 / `torch` 2.x, et [`requirements.txt`](requirements.txt) comme [`environment.yml`](environment.yml) ont été réécrits en conséquence. Le portage n'a demandé aucune adaptation du code applicatif : le pipeline a été validé à l'identique sur les deux stacks via [`python/tests/smoke.py`](python/tests/smoke.py). Les points de rupture silencieux ont été vérifiés un à un — en particulier la convention d'angle de `cv2.minAreaRect`, qui a changé en OpenCV 4.5 et pilotait le calcul d'azimut de [`summarize.py`](python/summarize.py) : angle et azimut sont restés identiques entre OpenCV 4.2 et 5.0. Ce calcul a depuis été refait (cf. *Azimut résolu* ci-dessous) et ne dépend plus de cette convention du tout.

**`SegNet`/`MultiTaskSegNet` corrigé (2026-08-10)** — Le dernier bloc descendant double les canaux (256) sans `maxpool` correspondant, alors que `MaxUnpool2d` exige que l'entrée ait exactement autant de canaux que les indices sauvegardés (128) : d'où le `RuntimeError: Shape of indices should match shape of input`. Le correctif applique la convolution **avant** le désempilement plutôt qu'après, dans `SegNet.forward` — la convolution ramène alors les canaux au bon compte. Validé sur les profondeurs 1 à 3, en tailles paires et impaires (chemin `ceil_mode`), et sur `MultiTaskSegNet` en `train` comme en `eval`. Ce n'est pas le modèle retenu par le rapport ([`latex/main.pdf`](latex/main.pdf)) — *Multi-Task U-Net* reste utilisé pour WalOnMap — mais les quatre architectures sont désormais fonctionnelles.

**État actuel** — Les 661 tuiles annotées de [`via_liege_city.json`](resources/walonmap/via_liege_city.json) se téléchargent en ~3 min 38 s via `misc/download.py`. Aucun modèle entraîné n'est republié dans ce fork (`products/` est gitignoré), mais l'entraînement est désormais opérationnel sur GPU (cf. §Cluster GPU).

Mesures de référence pour `MultiTaskUNet` sur le jeu californien complet, à l'échelle 1 :

| | |
| --- | --- |
| Échantillons par époque | 21 632 |
| Époque sur RTX A5000 | 11 min 14 s |
| Époque sur GTX 1080 Ti | 18 min 24 s (1,6×) |
| Époque sur GTX 1080 Ti, `K=5` | 14 min 12 s (4/5 du jeu) |
| Pic mémoire vive | 6,6 Go |
| Taille d'un checkpoint | 120 Mo |

L'écart entre les deux cartes est modeste parce que `train.py` calcule en FP32 — les *Tensor Cores* de l'A5000 restent inutilisés — et surtout parce que le `DataLoader` est instancié sans `num_workers` : le décodage des TIFF occupe un seul cœur pendant que le GPU attend. Quand la partition la plus rapide est saturée, prendre une carte plus ancienne mais libre est donc souvent plus rapide que d'attendre son tour.

Soit environ **3 h 45 pour 20 époques** sur A5000. À l'échelle 2 (celle du modèle de production), chaque crop passe de 256 à 512 pixels : compter environ 4× ce coût, d'où le `--time=20:00:00` recommandé.

À titre de comparaison, la même passe avant + arrière prend ~20 s par tuile 512×512 sur un poste de travail sans GPU — le fine-tuning Liège y demanderait une quarantaine d'heures, et l'entraînement californien reste hors de portée.

**Où reprendre** — Ordre du pipeline : entraînement californien → fine-tuning Liège → inférence WalOnMap → agrégation (§Reproductibilité ci-dessus). La chaîne complète (`train_epoch` → `.pth` → inférence → `summarize`) a été exécutée bout-en-bout sur la stack modernisée, et l'entraînement tourne sur GPU.

Étape en cours : reproduire l'évaluation du rapport (`K=5 FOLD=0`, puis `evaluate.py` sur le fold 0). Comparer ces chiffres à ceux de [`latex/main.pdf`](latex/main.pdf) est le seul contrôle qui vérifie que le passage à `torch` 2.x / NumPy 2.x / OpenCV 5 n'a rien altéré numériquement — `tests/smoke.py` prouve que le code s'exécute, pas qu'il apprend aussi bien. Vient ensuite le modèle de production (`SCALE=2`, `-k 0`), puis le fine-tuning sur Liège.

**Reproduction confirmée (2026-08-10)** — Entraînement sur GTX 1080 Ti (20 époques en 4 h 44, perte finale 0,347), évaluation sur RTX A5000 (2 min 45 s). Fold 0, *Multi-Task U-Net* :

| | rapport | reproduction |
| --- | --- | --- |
| AP détection | 0,829 | **0,842** |
| AP segmentation | 0,896 | **0,903** |
| F1 détection maximal | 0,828 | **0,836** |
| F1 segmentation maximal | 0,894 | **0,901** |

Les quatre métriques tombent légèrement **au-dessus** du rapport, très en deçà de l'écart entre folds (l'AP de détection va de 0,829 à 0,919 selon le fold). `torch` 2.x, NumPy 2.x et OpenCV 5 n'ont donc rien altéré de mesurable : le réseau apprend aussi bien qu'en 2020.

**Le seuil de décision, en revanche, a bougé.** À qualité égale, ce réseau-ci est bien plus polarisé : au seuil 0,5 il affiche une précision de 0,964 pour un rappel de 0,725, là où le rapport lisait 0,754 / 0,839. Toute la courbe est décalée le long de l'axe des seuils, et le F1 de détection est maximal vers `1e-3` (0,836) et non plus à 0,5 (0,827). C'est attendu — la graine n'est fixée que pour le découpage en folds, pas pour l'augmentation ni pour cuDNN, si bien que deux entraînements ne convergent pas vers le même minimum — mais ça ne se voit que si l'on regarde la courbe entière plutôt que les seules AP.

Conséquence pratique : le `-threshold` de [`walonmap.py`](python/walonmap.py) (0,5 par défaut) n'est **pas** transposable d'un entraînement à l'autre. Il doit être recalibré sur le modèle qui sert effectivement à l'inférence, en arbitrant entre le nombre d'installations trouvées (rappel de détection) et le biais de surface (MRE pixel-wise, minimal vers 0,5), puisque `area` alimente l'estimation de capacité en aval.

**Azimut résolu (2026-08-10)** — [`summarize.py`](python/summarize.py) ne gardait de `cv2.minAreaRect` que son `angle`, en jetant le couple `(w, h)`. Cet angle ne dit pas lequel des deux côtés il décrit : l'orientation n'était donc connue que modulo 90°, et la formule de repli tassait toute la colonne dans une bande de 90° autour du sud — mesuré sur les 1043 polygones annotés de Liège : `[135,9° ; 225,0°]`.

Le calcul part désormais des quatre coins du rectangle englobant. La direction visée est celle du **petit côté** : les rangées de panneaux suivent le faîtage, la pente descend perpendiculairement, et la projection verticale raccourcit encore ce côté. Partir des coins plutôt que de l'angle rend au passage le calcul indépendant de la convention d'OpenCV *et* du sens de l'axe des ordonnées.

| | avant | après |
| --- | --- | --- |
| étendue de la colonne | 135,9° – 225,0° | 90,0° – 269,2° |
| écart-type | ~26° au maximum théorique | 43,9° |
| exactement 180,00° | 5,1 % | 3,1 % |

La distribution obtenue est physiquement crédible : 88 % des installations entre sud-est et sud-ouest, pic au sud — alors que la formule autorise désormais les 180° complets. Validé à trois niveaux : [`tests/check_azimuth.py`](python/tests/check_azimuth.py) (24 orientations construites, cas limites, invariance à l'ordre des coins), la distribution ci-dessus, et une superposition des flèches sur les tuiles annotées.

Deux limites subsistent, irréductibles depuis une seule vue nadir : un rectangle ne distingue pas une pente de son opposée (on retient celle qui regarde le sud, ce qui se trompe sur les rares installations orientées au nord), et sur une installation quasi carrée le petit côté n'a pas de sens — 9,1 % des cas, que l'option `-diagnostics` signale via une colonne `elongation`.

**Précision géométrique** — En traquant l'azimut, un second défaut est apparu : `summarize.py` passait à OpenCV des coordonnées Lambert 72 **absolues** en millimètres, jusqu'à 2,4 × 10⁸. OpenCV calcule en `float32`, dont le pas à cette magnitude vaut 16 mm : toute la géométrie était quantifiée à 16 mm avant d'être mesurée. Le polygone est maintenant recentré sur son centroïde au préalable, ce qui fait tomber le pas sous le micromètre. L'effet sur les colonnes publiées était modeste — 0,15° d'erreur médiane sur l'azimut, 0,14 % sur `area` — mais il touchait aussi `area`, qui alimente l'estimation de capacité.

**À coordonner avec `wallonia_grid`** — Le module `sourcecode/core/pv/detections.py` valide les colonnes à l'identique et documente longuement l'ancien comportement : `EXPECTED_AZIMUTH_RANGE_DEG = (120, 226)` et `AZIMUTH_ARTIFACT_VALUES_DEG` deviendront caducs le jour où un CSV régénéré y sera versé. Le format reste inchangé (quatre colonnes ; `elongation` n'apparaît que sur demande explicite), mais l'avertissement de plage se déclencherait sur presque toutes les lignes. Rien ne presse : les CSV publiés datent de 2020 et le fichier VIA intermédiaire qui les a produits n'a pas été conservé — les régénérer suppose de rejouer l'inférence, donc d'attendre le modèle de production.

Les données ne sont pas dans le dépôt : `resources/california/` (45 Go, 601 images) et `products/liege/` sont gitignorés. Sur le cluster, les images californiennes doivent vivre sur `/scratch` — 601 fichiers relus à chaque époque, c'est exactement l'accès aléatoire que `/home` ne doit pas subir.

Mesures déjà faites : latence WMS ~119 ms médiane (40 requêtes, sans dégradation) ; la Wallonie représente environ 3,7 M de tuiles au niveau de zoom utilisé, ce qui rend un balayage exhaustif coûteux — des pistes de filtrage spatial préalable (couches d'occupation du sol type `HABITAT/TISSU_URBANISE`) sont explorées dans [`python/exploration/`](python/exploration/), non intégrées au pipeline ADOPPTRS.

[walonmap]: https://geoportail.wallonie.be/walonmap
[duke-dataset]: https://energy.duke.edu/content/distributed-solar-pv-array-location-and-extent-data-set-remote-sensing-object-identification
[via]: http://www.robots.ox.ac.uk/~vgg/software/via/
