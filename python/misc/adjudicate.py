#!/usr/bin/env python

"""
Arbitre les detections sans annotation, sur un millesime posterieur a 2018.

Le probleme
-----------
Les annotations datent de 2018. Sur une image de 2022, un panneau pose en
2020 n'est annote nulle part : quand le modele le trouve, l'evaluation le
compte en faux positif. Les 59 "faux positifs" de ORTHO_2022_ETE sont
donc un melange de vraies erreurs et de vraies installations recentes, et
aucun calcul ne les separe -- seul un oeil humain le peut.

Ce script decoupe une vignette centree sur chacune, les numerote sur des
planches, et attend en retour la liste de celles qui ne sont pas des
panneaux. Il en tire ensuite la precision reelle et le nombre
d'installations posees depuis 2018.

Ce qu'il mesure, ce qu'il ne mesure pas
---------------------------------------
Il mesure la **precision** : parmi ce que le modele annonce, quelle part
est juste. Il ne mesure pas le **rappel** sur les installations recentes
-- pour savoir combien de panneaux de 2022 ont ete rates il faudrait les
avoir annotes, ce qui est precisement le travail qu'on cherche a eviter.
Le compte de croissance rendu ici est donc une borne basse, et son
extrapolation suppose que le rappel sur les installations recentes
ressemble a celui mesure sur les anciennes.

Usage
-----
    conda activate adopptrs

    # 1. extraire les vignettes
    cd python
    python misc/adjudicate.py -n ../products/models/multiunet_liege_030.pth \\
        --vintage ORTHO_2022_ETE

    # 2. regarder les planches, relever les numeros qui ne sont PAS des panneaux

    # 3. rendre le verdict
    python misc/adjudicate.py --score --false "3 7 12 41"

Le fichier verdicts.csv ecrit a l'etape 1 garde la trace de chaque
vignette (tuile, position, surface) ; l'etape 3 y consigne les verdicts.
"""

import argparse
import csv
import os
import random
import sys

import numpy as np
import torch

from PIL import Image, ImageDraw

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_PYTHON)

sys.path.insert(0, _PYTHON)
sys.path.insert(0, os.path.join(_PYTHON, 'tests'))

import via as VIA

from check_predictions import confusion, parse_name, predict
from dataset import to_mask, to_contours
from evaluate import surface
from models import MultiTaskUNet, UNet


CELL = 256          # cote d'une vignette sur la planche
COLUMNS = 6
ROWS = 5
GAP = 4
BACKGROUND = (24, 24, 24)
LABEL = (255, 255, 255)
DETECTION = (255, 0, 0)


def vignette(image, contour, size=CELL):
    """Decoupe autour d'une detection, avec du contexte, et l'entoure.

    Le cadrage suit la taille de l'objet : une rampe de vingt metres et un
    module isole n'ont pas besoin du meme recul. Sans le contour rouge, on
    ne saurait pas lequel des objets de la vignette est juge.
    """
    points = contour[:, 0, :]

    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)

    centre_x, centre_y = (x_min + x_max) / 2, (y_min + y_max) / 2
    side = max(x_max - x_min, y_max - y_min) * 2.5
    side = int(min(max(side, 96), 384))

    left = int(round(centre_x - side / 2))
    upper = int(round(centre_y - side / 2))

    # Recadrage dans la tuile plutot que remplissage : mieux vaut decentrer
    # l'objet que lui coller une bande noire qui gene le jugement.
    left = max(0, min(left, image.width - side))
    upper = max(0, min(upper, image.height - side))

    crop = image.crop((left, upper, left + side, upper + side)).resize(
        (size, size), Image.LANCZOS
    )

    scale = size / side
    draw = ImageDraw.Draw(crop)
    drawn = [
        ((int(x) - left) * scale, (int(y) - upper) * scale)
        for x, y in points
    ]

    if len(drawn) >= 2:
        draw.line(drawn + [drawn[0]], fill=DETECTION, width=2)

    return crop


def sheet(cells, numbers, columns=COLUMNS):
    """Assemble une planche de vignettes numerotees.

    La hauteur suit le nombre de vignettes : la derniere planche est
    rarement pleine, et une bande noire de plusieurs milliers de pixels
    donnerait l'impression qu'il manque des cas a juger.
    """
    band = 20
    rows = max(1, -(-len(cells) // columns))
    columns = min(columns, max(1, len(cells)))

    width = columns * CELL + (columns - 1) * GAP
    height = rows * (CELL + band) + (rows - 1) * GAP

    canvas = Image.new('RGB', (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    for i, (cell, number) in enumerate(zip(cells, numbers)):
        col, row = i % columns, i // columns

        x = col * (CELL + GAP)
        y = row * (CELL + band + GAP)

        draw.text((x + 4, y + 5), '#{}'.format(number), fill=LABEL)
        canvas.paste(cell, (x, y + band))

    return canvas


def extract(args):
    """Passe le modele, isole les detections sans annotation, les decoupe."""
    tiles = os.path.join(args.tiles, args.vintage)
    via_path = args.input or os.path.join(_ROOT, 'resources', 'walonmap', 'via_liege_city.json')

    for path in (tiles, via_path, args.network):
        if not os.path.exists(path):
            sys.exit('Introuvable : ' + path)

    os.makedirs(args.destination, exist_ok=True)

    via = VIA.load(via_path)
    keys = sorted(list(via.keys()))

    random.seed(0)
    random.shuffle(keys)

    if args.k > 0:
        keys = [key for i, key in enumerate(keys) if (i % args.k) == args.fold]

    model = MultiTaskUNet(3, 1, R=5) if args.multitask else UNet(3, 1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_state_dict(torch.load(args.network, map_location=device))
    model.eval()

    rows = []
    cells = []
    matched_total = 0
    kept_total = 0

    for key in keys:
        filename = os.path.join(tiles, key)

        if not os.path.exists(filename):
            continue

        image = Image.open(filename).convert('RGB')
        polygons = via[key]

        target = np.array(to_mask((image.height, image.width), polygons))
        mask = predict(model, image, device, args.threshold, args.opening)

        inter = ((target > 0) & (mask > 0)).astype(np.uint8)
        output_ctns = to_contours(mask)

        _, _, _, outputs = confusion(to_contours(target), output_ctns, inter, args.min)

        for contour, (_, area, matched) in zip(output_ctns, outputs):
            if area <= args.min:
                continue

            kept_total += 1

            if matched:
                matched_total += 1
                continue

            row, col = parse_name(key)
            points = contour[:, 0, :]

            rows.append({
                'id': len(rows) + 1,
                'tuile': key,
                'row': row,
                'col': col,
                'x': int(points[:, 0].mean()),
                'y': int(points[:, 1].mean()),
                'surface': int(area),
                'verdict': '',
            })

            cells.append(vignette(image, contour))

    # Vignettes individuelles : pour les cas ou la planche ne suffit pas.
    for cell, entry in zip(cells, rows):
        cell.save(os.path.join(
            args.destination, '{:03d}_{}_{}.jpg'.format(entry['id'], entry['row'], entry['col'])
        ), quality=92)

    per_sheet = COLUMNS * ROWS

    for start in range(0, len(cells), per_sheet):
        chunk = cells[start:start + per_sheet]
        numbers = [entry['id'] for entry in rows[start:start + per_sheet]]

        page = start // per_sheet + 1
        sheet(chunk, numbers).save(
            os.path.join(args.destination, 'planche_{}.jpg'.format(page)), quality=92
        )

    with open(os.path.join(args.destination, 'verdicts.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['id'])
        writer.writeheader()
        writer.writerows(rows)

    # Les comptes servent au calcul final ; les garder ici evite de
    # refaire tourner le modele a l'etape de notation.
    with open(os.path.join(args.destination, 'comptes.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['millesime', 'detections', 'appariees', 'sans_annotation'])
        writer.writerow([args.vintage, kept_total, matched_total, len(rows)])

    print('millesime          : {}'.format(args.vintage))
    print('detections totales : {}'.format(kept_total))
    print('  dont appariees   : {}'.format(matched_total))
    print('  sans annotation  : {}   <- a arbitrer'.format(len(rows)))
    print()
    print('Planches : {}'.format(os.path.join(args.destination, 'planche_*.jpg')))
    print()
    print('Parcours les planches et releve les numeros qui ne sont PAS des panneaux,')
    print('puis :')
    print('    python misc/adjudicate.py --score --false "3 7 12"')

    return 0


def score(args):
    """Calcule precision et croissance a partir des verdicts."""
    path = os.path.join(args.destination, 'verdicts.csv')
    counts = os.path.join(args.destination, 'comptes.csv')

    for f in (path, counts):
        if not os.path.exists(f):
            sys.exit('Introuvable : {} (lancer l\'extraction d\'abord)'.format(f))

    with open(counts, 'r') as f:
        row = list(csv.DictReader(f))[0]

    vintage = row['millesime']
    detections = int(row['detections'])
    matched = int(row['appariees'])

    with open(path, 'r') as f:
        entries = list(csv.DictReader(f))

    if args.false is not None:
        wrong = {int(n) for n in args.false.replace(',', ' ').split()}
    else:
        wrong = {int(e['id']) for e in entries if e['verdict'].strip().lower() in ('x', 'n', 'non')}

    unknown = wrong - {int(e['id']) for e in entries}

    if unknown:
        sys.exit('Numeros inconnus : {}'.format(sorted(unknown)))

    for entry in entries:
        entry['verdict'] = 'x' if int(entry['id']) in wrong else 'p'

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(entries[0].keys()))
        writer.writeheader()
        writer.writerows(entries)

    false_alarms = len(wrong)
    new_panels = len(entries) - false_alarms

    precision = (detections - false_alarms) / detections if detections else float('nan')

    print('=' * 60)
    print('Arbitrage -- {}'.format(vintage))
    print('=' * 60)
    print()
    print('detections totales          : {}'.format(detections))
    print('  appariees a une annotation: {}   (justes par construction)'.format(matched))
    print('  panneaux non annotes en 2018: {}'.format(new_panels))
    print('  fausses alarmes           : {}'.format(false_alarms))
    print()
    print('PRECISION reelle            : {:.3f}'.format(precision))
    print('  (contre {:.3f} si l\'on comptait toute detection sans annotation'.format(
        matched / detections if detections else float('nan')
    ))
    print('   comme une erreur -- la mesure d\'avant arbitrage)')
    print()
    print('Croissance sur ces tuiles')
    print('-------------------------')
    print('  installations trouvees, posees apres 2018 : {}'.format(new_panels))

    if args.recall > 0:
        print('  estimation corrigee du rappel ({:.2f})     : {:.0f}'.format(
            args.recall, new_panels / args.recall
        ))
        print()
        print('  La correction suppose que le rappel sur les installations recentes')
        print('  vaut celui mesure sur les anciennes. C\'est une hypothese, pas une')
        print('  mesure : la verifier demanderait d\'annoter un millesime recent.')

    print()
    print('Verdicts consignes dans {}'.format(path))

    return 0


def main():
    parser = argparse.ArgumentParser(description='Arbitre les detections sans annotation')
    parser.add_argument('-n', '--network', default=None, help='fichier du reseau')
    parser.add_argument('-multitask', default=True, action='store_true', help='reseau multi-taches')
    parser.add_argument('-i', '--input', default=None, help='fichier VIA des annotations 2018')
    parser.add_argument('--tiles', default=os.path.join(_ROOT, 'products', 'vintages'),
                        help='dossier contenant un sous-dossier par millesime')
    parser.add_argument('--vintage', default='ORTHO_2022_ETE', help='millesime a arbitrer')
    parser.add_argument('-d', '--destination', default=os.path.join(_ROOT, 'products', 'arbitrage'),
                        help='dossier des vignettes et des verdicts')
    parser.add_argument('-k', type=int, default=5, help='nombre de folds')
    parser.add_argument('-f', '--fold', type=int, default=0, help='fold mis de cote')
    parser.add_argument('-threshold', type=float, default=1e-4, help='seuil de decision')
    parser.add_argument('-min', type=int, default=256, help='surface minimale, comme walonmap.py')
    parser.add_argument('-opening', dest='opening', default=False, action='store_true',
                        help="applique l'ouverture 5x5 d'evaluate.py")
    parser.add_argument('--score', default=False, action='store_true', help='calcule le resultat')
    parser.add_argument('--false', default=None,
                        help='numeros qui ne sont PAS des panneaux, entre guillemets')
    parser.add_argument('--recall', type=float, default=0.72,
                        help='rappel mesure, pour corriger le compte de croissance ; 0 pour omettre')
    args = parser.parse_args()

    if args.score:
        return score(args)

    if not args.network:
        sys.exit('-n/--network est requis pour l\'extraction')

    return extract(args)


if __name__ == '__main__':
    sys.exit(main())
