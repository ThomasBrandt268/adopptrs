# Carnets d'origine (2020)

Les six carnets de ce dossier sont ceux de François Rozet, datés du
2020-05-19. Ils ont servi à explorer les architectures (`segnet`, `unet`,
leurs variantes multi-tâches), à régler les hyperparamètres (`tuning`) et à
regarder le jeu californien (`california`). Le travail qu'ils contiennent a
depuis été repris par `python/train.py` et `python/evaluate.py`, qui sont
scriptés, reproductibles et lançables sur cluster.

Ils sont conservés comme **trace de méthode**, pas comme outil.

## Ils ne s'exécutent pas en l'état

Non pas à cause de la modernisation : l'API qu'ils appellent est intacte, et
`via.load`, `VIADataset`, `ToTensor`, `MultiTaskUNet`, `to_pil` et
`plot_alongside` existent toujours avec des signatures compatibles.
`torchsummary`, qu'ils utilisent, est encore dans `requirements.txt`.

Ce qui manque, ce sont leurs **entrées** :

| attendu | état |
|---|---|
| `products/models/multiunet_<fold>_020.pth` | absent — les poids de 2020 n'ont jamais été publiés |
| `products/csv/multiunet.csv` | absent — les courbes d'entraînement de l'époque |
| `products/json/california.json` | présent |
| `resources/california/` | présent (606 fichiers) |

`products/` est gitignoré, donc ces fichiers n'ont jamais fait partie du
dépôt : même en 2020, un clone frais exigeait de réentraîner d'abord. Les
modèles archivés par le fork ne les remplacent pas — ils suivent un autre
protocole (K=0, sans fold mis de côté) et portent d'autres noms.

Les faire tourner suppose donc de rejouer un entraînement K=5 puis de
renommer les sorties selon la convention de 2020. C'est possible, ce n'est
pas gratuit, et `evaluate.py` répond aujourd'hui aux mêmes questions.

## Ce qui est vivant

`notebooks/results.ipynb`, un cran au-dessus : lecture des mesures
d'évaluation, courbes, comparaison à rappel égal. Celui-là tourne sur
l'environnement courant.
