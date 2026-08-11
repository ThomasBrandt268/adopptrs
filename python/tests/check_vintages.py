"""
Compare les predictions d'un meme modele sur plusieurs millesimes.

Le meme reseau, le meme seuil, les memes tuiles : seule l'annee de la
prise de vue change. C'est la condition pour qu'un ecart entre deux
millesimes soit attribuable a l'image ou au terrain, et non au detecteur.

Ce que la juxtaposition rend visible
------------------------------------
    - le decalage d'orthorectification : un meme batiment ne tombe pas au
      meme endroit d'un millesime a l'autre, ce qui condamne d'avance tout
      appariement point a point entre deux passes ;
    - la vegetation et les ombres, qui masquent des toitures selon la
      saison et l'heure de vol ;
    - la resolution native, plus grossiere sur les millesimes anciens :
      le serveur reechantillonne vers nos 0,132 m/px et l'image est molle.

Lire les chiffres
-----------------
Les annotations sont celles de 2018, les seules qui existent. Elles ne
sont donc une verite terrain exacte que pour ce millesime-la. Sur les
autres, elles restent exploitables grace a une asymetrie : un panneau
n'est presque jamais depose.

    millesime posterieur a 2018   un panneau annote et non retrouve est
                                  un echec du modele, pas une disparition
                                  -> FN = borne basse de l'erreur

    millesime anterieur a 2018    une detection sur un toit non annote en
                                  2018 est presque surement fausse, car
                                  l'installation aurait ete la en 2018
                                  -> FP = borne haute de l'erreur

    Les TP d'un millesime ancien, eux, sont des installations qui
    existaient deja : c'est la mesure de croissance, pas une erreur.

Usage
-----
    conda activate adopptrs
    cd python
    python tests/check_vintages.py \\
        -n ../products/models/multiunet_liege_030.pth \\
        --tiles ../products/vintages \\
        --vintages ORTHO_2009_2010 ORTHO_2017 ORTHO_2018 ORTHO_2024 \\
        -threshold 1e-4 -k 5 -f 0

Chaque millesime est attendu dans son propre sous-dossier de --tiles, tel
que misc/download.py -vintage ... -d ../products/vintages/<millesime> les
depose. Une tuile absente d'un millesime est dessinee en gris plutot
qu'ignoree : sa disparition est une information.

Produit une planche par tuile, millesimes cote a cote, triees par ecart
decroissant du nombre de detections -- les tuiles ou quelque chose a
change viennent en premier.
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import via as VIA

from check_predictions import ANNOTATION, REJECTED, RETAINED, confusion, parse_name, predict
from dataset import to_mask, to_contours
from evaluate import surface
from models import MultiTaskUNet, UNet


BAND = 26
GAP = 6
BACKGROUND = (24, 24, 24)
LABEL = (255, 255, 255)
MISSING = (70, 70, 70)


def panel(image, polygons, contours, minimum, title):
    """Une vignette : la tuile, ses contours, et un bandeau titre."""
    width, height = image.size
    canvas = Image.new('RGB', (width, height + BAND), BACKGROUND)
    canvas.paste(image, (0, BAND))

    draw = ImageDraw.Draw(canvas)
    draw.text((4, 7), title, fill=LABEL)

    for polygon in polygons:
        if len(polygon) >= 2:
            points = [(x, y + BAND) for x, y in polygon]
            draw.line(points + [points[0]], fill=ANNOTATION, width=3)

    for contour in contours:
        points = [(int(p[0]), int(p[1]) + BAND) for p in contour[:, 0, :]]

        if len(points) < 2:
            continue

        colour = RETAINED if surface(contour) > minimum else REJECTED
        draw.line(points + [points[0]], fill=colour, width=2)

    return canvas


def missing(size, title):
    """Vignette de remplacement quand la tuile manque a ce millesime."""
    canvas = Image.new('RGB', (size[0], size[1] + BAND), BACKGROUND)
    canvas.paste(Image.new('RGB', size, MISSING), (0, BAND))

    draw = ImageDraw.Draw(canvas)
    draw.text((4, 7), title, fill=LABEL)
    draw.text((8, BAND + 8), 'tuile absente', fill=LABEL)

    return canvas


def main():
    parser = argparse.ArgumentParser(description='Compare les millesimes sur les memes tuiles')
    parser.add_argument('-n', '--network', required=True, help='fichier du reseau')
    parser.add_argument('-m', '--model', default='unet', choices=['unet'], help='schema du reseau')
    parser.add_argument('-multitask', default=True, action='store_true', help='reseau multi-taches')
    parser.add_argument('-i', '--input', default=None, help='fichier VIA des annotations 2018')
    parser.add_argument('--tiles', default=None, help='dossier contenant un sous-dossier par millesime')
    parser.add_argument('--vintages', nargs='+', required=True, help='millesimes, dans l\'ordre chronologique')
    parser.add_argument('--reference', default='ORTHO_2018', help='millesime dont proviennent les annotations')
    parser.add_argument('-d', '--destination', default=None, help='dossier de sortie')
    parser.add_argument('-k', type=int, default=5, help='nombre de folds')
    parser.add_argument('-f', '--fold', type=int, default=0, help='fold mis de cote')
    parser.add_argument('-threshold', type=float, default=1e-4, help='seuil de decision')
    parser.add_argument('-min', type=int, default=256, help='surface minimale, comme walonmap.py')
    parser.add_argument('-l', '--limit', type=int, default=25, help='nombre de planches dessinees')
    # Defaut sans ouverture : c'est la chaine que walonmap.py execute, et
    # celle sur laquelle le seuil 1e-4 a ete calibre.
    parser.add_argument('-opening', dest='opening', default=False, action='store_true',
                        help="applique l'ouverture 5x5 d'evaluate.py")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))

    via_path = args.input or os.path.join(root, 'resources', 'walonmap', 'via_liege_city.json')
    tiles = args.tiles or os.path.join(root, 'products', 'vintages')
    destination = args.destination or os.path.join(root, 'products', 'check_vintages')

    for path in (via_path, tiles, args.network):
        if not os.path.exists(path):
            sys.exit('Introuvable : ' + path)

    os.makedirs(destination, exist_ok=True)

    # Fold : identique a evaluate.py, train.py et misc/download.py.
    via = VIA.load(via_path)
    keys = sorted(list(via.keys()))

    random.seed(0)
    random.shuffle(keys)

    if args.k > 0:
        keys = [key for i, key in enumerate(keys) if (i % args.k) == args.fold]

    print('{} tuiles | {} millesimes | seuil {:g} | ouverture {}'.format(
        len(keys), len(args.vintages), args.threshold, 'oui' if args.opening else 'non'
    ))
    print('annotations : {} (millesime de reference {})'.format(
        os.path.basename(via_path), args.reference
    ))
    print()

    model = MultiTaskUNet(3, 1, R=5) if args.multitask else UNet(3, 1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_state_dict(torch.load(args.network, map_location=device))
    model.eval()

    totals = {vintage: [0, 0, 0, 0] for vintage in args.vintages}  # TP, FP, FN, detections
    absent = {vintage: 0 for vintage in args.vintages}
    sheets = []

    for key in keys:
        panels = []
        counts = []

        for vintage in args.vintages:
            filename = os.path.join(tiles, vintage, key)

            if not os.path.exists(filename):
                absent[vintage] += 1
                panels.append(('missing', vintage))
                continue

            image = Image.open(filename).convert('RGB')
            polygons = via[key]

            target = np.array(to_mask((image.height, image.width), polygons))
            mask = predict(model, image, device, args.threshold, args.opening)

            inter = ((target > 0) & (mask > 0)).astype(np.uint8)
            output_ctns = to_contours(mask)

            tp, fp, fn, _ = confusion(to_contours(target), output_ctns, inter, args.min)
            kept = [c for c in output_ctns if surface(c) > args.min]

            for i, value in enumerate((tp, fp, fn, len(kept))):
                totals[vintage][i] += value

            counts.append(len(kept))
            panels.append((image, polygons, output_ctns, vintage, tp, fp, fn))

        # Les tuiles ou le compte bouge d'un millesime a l'autre sont
        # celles qui portent une information : croissance reelle, ou
        # instabilite du detecteur face a l'image.
        spread = (max(counts) - min(counts)) if counts else 0

        sheets.append((spread, key, panels))

    sheets.sort(key=lambda s: -s[0])

    print('{:<24} {:>7}   {}'.format('tuile', 'ecart', 'detections par millesime'))

    for spread, key, panels in sheets[:args.limit]:
        row, col = parse_name(key)
        drawn = []

        for item in panels:
            if item[0] == 'missing':
                drawn.append(missing((512, 512), item[1] + '  --'))
                continue

            image, polygons, contours, vintage, tp, fp, fn = item
            title = '{}   TP {} FP {} FN {}'.format(vintage.replace('ORTHO_', ''), tp, fp, fn)
            drawn.append(panel(image, polygons, contours, args.min, title))

        width = sum(p.width for p in drawn) + GAP * (len(drawn) - 1)
        height = max(p.height for p in drawn)

        sheet = Image.new('RGB', (width, height), BACKGROUND)

        offset = 0
        for p in drawn:
            sheet.paste(p, (offset, 0))
            offset += p.width + GAP

        out = '{:02d}_{}_{}.jpg'.format(spread, row, col)
        sheet.save(os.path.join(destination, out), quality=90)

        print('{:<24} {:>7}   {}'.format(key, spread, out))

    print()
    print('{:<20} {:>6} {:>6} {:>6} {:>12}'.format(
        'millesime', 'TP', 'FP', 'FN', 'detections'
    ))

    for vintage in args.vintages:
        tp, fp, fn, kept = totals[vintage]
        note = '' if not absent[vintage] else '  ({} tuile(s) absente(s))'.format(absent[vintage])

        print('{:<20} {:>6} {:>6} {:>6} {:>12}{}'.format(vintage, tp, fp, fn, kept, note))

    print()
    print('Rappel de lecture : les annotations datent de {}.'.format(args.reference))
    print('  millesime posterieur  -> les FN sont des echecs du modele (les panneaux ne disparaissent pas)')
    print('  millesime anterieur   -> les FP sont surtout de vraies fausses alarmes')
    print('  ... et les TP y mesurent le parc deja installe, pas une erreur.')
    print()
    print('Planches dans {}'.format(destination))
    print('  vert = annotation 2018 | rouge = detection retenue | orange = rejetee par -min')


if __name__ == '__main__':
    main()
