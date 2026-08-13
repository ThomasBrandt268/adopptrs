#!/usr/bin/env python

"""
Extrait les faux positifs du reseau en vignettes, pour tri manuel.

Pourquoi ce script
------------------
check_predictions.py dessine des tuiles entieres et repond a « le seuil
est-il bon ? ». Celui-ci decoupe autour de chaque detection non appariee
et repond a « de quoi le reseau se trompe-t-il ? ».

La revue du 2026-08-12 a montre que les faux positifs comptes par
evaluate.py sont en realite de deux natures, que le chiffre confond :

  - de vraies fausses alarmes -- conteneurs d'un depot, voitures d'un
    parking : des rectangles plats et alignes, qui ont vu du ciel la
    signature d'un champ de panneaux. Ce sont les negatifs a fournir au
    reseau, qui n'en a jamais vu : clusterize() ne produit aucune vignette
    pour une image sans polygone ;
  - des trous d'annotation -- de vrais panneaux que via_liege_city.json a
    oublies. Ceux-la ne sont pas des erreurs du reseau : ils rabaissent la
    precision mesuree, et surtout ils enseignent a l'entrainement qu'un
    panneau est du fond.

Les deux se corrigent, mais pas de la meme main, et seul l'oeil les
separe. D'ou ces vignettes.

Ouverture morphologique
-----------------------
Pas d'ouverture par defaut, contrairement a check_predictions.py : on
collecte la chaine que walonmap.py execute vraiment. L'ecart entre les
deux chaines est explique dans evaluate.py.

Usage
-----
    conda activate adopptrs
    cd python
    python misc/collect_fp.py \
        -n ../products/models/multiunet_liege_030.pth -threshold 5e-5

    # puis, le tri : deplacer chaque vignette dans l'un des deux dossiers
    cd ../products/fp
    mkdir -p manquant faux
    #   manquant/ = vrai panneau, oubli de l'annotation
    #   faux/     = vraie fausse alarme (conteneur, parking, verriere...)

Les vignettes sont numerotees par surface decroissante : 000_* est le faux
positif qui coute le plus de metres carres, donc celui qui merite le
premier coup d'oeil.

Code couleur
------------
    rouge   la detection non appariee (le faux positif lui-meme)
    vert    les annotations du VIA presentes dans la vignette
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import torch

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import via as VIA

from dataset import to_mask, to_contours, to_tensor
from models import UNet, SegNet, MultiTaskUNet, MultiTaskSegNet
from evaluate import bounding, intersection, surface


ANNOTATION = (0, 220, 0)
FALSE_POSITIVE = (255, 0, 0)


def predict(model, image, device, threshold, opening):
    """Rend le masque binaire, dans l'etat ou les contours en sortent."""
    with torch.no_grad():
        inpt = to_tensor(image).unsqueeze(0).to(device)
        outpt = model(inpt).cpu()[0]

    mask = (outpt[0].numpy() > threshold).astype(np.uint8) * 255

    if opening:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8))

    return mask


def unmatched(target_ctns, output_ctns, inter, minimum):
    """Les contours predits qu'aucune annotation ne recouvre.

    Meme appariement que evaluate.py : une cible et une prediction se
    correspondent des qu'elles partagent un pixel. Le filtre de surface
    reprend le -min de walonmap.py, pour ne pas faire trier des contours
    que la production ne publierait pas.
    """
    targets = [bounding(c) for c in target_ctns]

    keep = []

    for contour in output_ctns:
        if surface(contour) <= minimum:
            continue

        box = bounding(contour)
        matched = False

        for target in targets:
            common = intersection(target, box)

            if common is None:
                continue

            if inter[common[0]:(common[2] + 1), common[1]:(common[3] + 1)].sum() > 0:
                matched = True
                break

        if not matched:
            keep.append(contour)

    return keep


def window(box, shape, margin, minimum):
    """Fenetre carree autour d'une boite, recadree dans l'image.

    bounding() rend (ligne, colonne) malgre ses noms x/y ; on reste donc
    en (ligne, colonne) jusqu'a la conversion pour PIL, faite par
    l'appelant. Une fenetre carree evite d'etirer les vignettes, qu'on
    regarde les unes apres les autres.
    """
    height, width = shape

    side = max(box[2] - box[0], box[3] - box[1]) + 2 * margin
    side = max(side, minimum)
    side = min(side, height, width)

    row = (box[0] + box[2]) // 2 - side // 2
    col = (box[1] + box[3]) // 2 - side // 2

    row = max(0, min(row, height - side))
    col = max(0, min(col, width - side))

    return row, col, side


def main():
    parser = argparse.ArgumentParser(description='Extrait les faux positifs en vignettes')
    parser.add_argument('-n', '--network', required=True, help='fichier du reseau')
    parser.add_argument('-m', '--model', default='unet', choices=['unet', 'segnet'], help='schema du reseau')
    parser.add_argument('-multitask', default=True, action='store_true', help='reseau multi-taches')
    parser.add_argument('-i', '--input', default=None, help='fichier VIA')
    parser.add_argument('-p', '--path', default=None, help='dossier des tuiles')
    parser.add_argument('-d', '--destination', default=None, help='dossier de sortie')
    parser.add_argument('-threshold', type=float, default=5e-5, help='seuil de decision')
    parser.add_argument('-min', type=int, default=256, help='surface minimale, comme walonmap.py')
    parser.add_argument('-margin', type=int, default=48, help='contexte autour du contour, en pixels')
    parser.add_argument('-size', type=int, default=192, help='cote minimal des vignettes')
    # -k 0 : toutes les tuiles. On cherche ici des negatifs et des trous
    # d'annotation, pas une mesure -- il n'y a aucune raison de se limiter
    # au fold mis de cote.
    parser.add_argument('-k', type=int, default=0, help='nombre de folds (0 = toutes les tuiles)')
    parser.add_argument('-f', '--fold', type=int, default=0, help='fold retenu si -k > 0')
    parser.add_argument('-opening', default=False, action='store_true',
                        help="applique l'ouverture 5x5 (par defaut non, comme walonmap.py)")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))

    via_path = args.input or os.path.join(root, 'resources', 'walonmap', 'via_liege_city.json')
    tiles_path = args.path or os.path.join(root, 'products', 'liege')
    destination = args.destination or os.path.join(root, 'products', 'fp')

    for path in (via_path, tiles_path, args.network):
        if not os.path.exists(path):
            sys.exit('Introuvable : ' + path)

    os.makedirs(destination, exist_ok=True)

    # Fold : meme decoupage que evaluate.py, pour que -k 5 -f 0 designe
    # exactement les tuiles de la mesure.
    import random

    via = VIA.load(via_path)
    keys = sorted(list(via.keys()))

    random.seed(0)
    random.shuffle(keys)

    # Le fold du protocole (K=5), note pour chaque tuile meme quand on les
    # prend toutes. Un faux positif sur le fold 0 ne se lit pas comme un
    # autre : ces 133 tuiles-la sont les seules que le reseau n'a jamais
    # vues. Ailleurs, un panneau oublie par l'annotation lui a ete enseigne
    # comme du fond -- il a appris a ne pas le voir, donc les trous
    # d'annotation y restent invisibles.
    fold_of = {key: i % 5 for i, key in enumerate(keys)}

    if args.k > 0:
        keys = [key for i, key in enumerate(keys) if (i % args.k) == args.fold]

    print('{} tuiles | seuil {:g} | -min {} | ouverture {}'.format(
        len(keys), args.threshold, args.min, 'oui' if args.opening else 'non'
    ))
    print()

    # Reseau
    if args.model == 'unet':
        model = MultiTaskUNet(3, 1, R=5) if args.multitask else UNet(3, 1)
    else:
        model = MultiTaskSegNet(3, 1, R=5) if args.multitask else SegNet(3, 1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_state_dict(torch.load(args.network, map_location=device))
    model.eval()

    # Passage
    found = []
    tiles = 0

    for key in keys:
        filename = os.path.join(tiles_path, key)

        if not os.path.exists(filename):
            continue

        tiles += 1

        image = Image.open(filename).convert('RGB')
        polygons = via[key]

        target = np.array(to_mask((image.height, image.width), polygons))
        mask = predict(model, image, device, args.threshold, args.opening)

        inter = ((target > 0) & (mask > 0)).astype(np.uint8)

        contours = unmatched(to_contours(target), to_contours(mask), inter, args.min)

        for contour in contours:
            found.append((surface(contour), key, contour, image, polygons))

    # Les plus gros d'abord : ce sont eux qui pesent en metres carres, donc
    # ceux dont la nature change le plus le resultat.
    found.sort(key=lambda f: -f[0])

    index = os.path.join(destination, 'index.csv')

    with open(index, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rang', 'vignette', 'tuile', 'fold', 'ligne', 'colonne', 'surface', 'annotes_tuile'])

        for rank, (area, key, contour, image, polygons) in enumerate(found):
            box = bounding(contour)
            row, col, side = window(box, (image.height, image.width), args.margin, args.size)

            # PIL veut (gauche, haut, droite, bas), soit (colonne, ligne).
            crop = image.crop((col, row, col + side, row + side))
            draw = ImageDraw.Draw(crop)

            # Les points d'un contour OpenCV sont deja en (x, y) PIL.
            points = [(p[0] - col, p[1] - row) for p in contour[:, 0, :]]

            if len(points) >= 2:
                draw.line(points + [points[0]], fill=FALSE_POSITIVE, width=2)

            # Les annotations voisines : un panneau annote juste a cote
            # signale une detection qui deborde, pas une hallucination.
            for polygon in polygons:
                shifted = [(x - col, y - row) for x, y in polygon]

                if len(shifted) >= 2:
                    draw.line(shifted + [shifted[0]], fill=ANNOTATION, width=2)

            # Le fold est dans le nom : c'est lui qui dit si la vignette est
            # un temoin fiable (f0) ou une tuile deja vue a l'entrainement.
            name = '{:03d}_f{}_{}_{}.jpg'.format(rank, fold_of[key], key.split('.')[0], int(area))
            crop.save(os.path.join(destination, name))

            writer.writerow([rank, name, key, fold_of[key], row, col, int(area), len(polygons)])

    print('{} faux positifs sur {} tuiles'.format(len(found), tiles))

    if found:
        areas = np.array([f[0] for f in found])
        print('surface : mediane {:.0f} px, maximum {:.0f} px'.format(
            np.median(areas), areas.max()
        ))

        witnesses = sum(1 for f in found if fold_of[f[1]] == 0)
        print('dont {} sur le fold 0, les seules tuiles jamais vues'.format(witnesses))

    print()
    print('Vignettes dans {}'.format(destination))
    print('Index      : {}'.format(index))
    print('  rouge = la detection | vert = annotations du VIA')
    print()
    print('Tri : creer manquant/ et faux/ dans ce dossier, y deplacer chaque vignette.')
    print('  manquant/ = vrai panneau oublie par l\'annotation')
    print('  faux/     = vraie fausse alarme (conteneur, parking, verriere...)')

    return 0


if __name__ == '__main__':
    sys.exit(main())
