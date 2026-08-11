"""
Controle visuel des predictions du reseau sur les tuiles mises de cote.

check_alignment.py superpose les *annotations* aux tuiles et repond a la
question « le quadrillage est-il bon ? ». Celui-ci superpose les
*predictions* et repond a « le seuil est-il bon ? ». Il ne sert a rien
tant qu'un seuil n'a pas ete calibre -- il montre a quoi ressemblent, sur
l'image, les faux positifs et les oublis que calibrate.sbatch a comptes.

Le fold est choisi comme dans evaluate.py (memes cles triees, meme
random.seed(0)), donc les tuiles dessinees ici sont exactement celles qui
ont produit les chiffres. Les totaux imprimes a la fin doivent d'ailleurs
retomber sur ceux de la table -- s'ils divergent, c'est ce script qui a
tort.

Code couleur
------------
    vert    annotation
    rouge   prediction retenue (surface > -min)
    orange  prediction rejetee par -min : ce que le filtre de taille coute

Usage
-----
    conda activate adopptrs
    cd python
    python tests/check_predictions.py \
        -n ../products/models/multiunet_liege_030.pth \
        -threshold 1e-4 -k 5 -f 0

    # la chaine telle que walonmap.py l'execute vraiment
    python tests/check_predictions.py -n ... -threshold 1e-4 -no-opening

Puis, depuis Windows :
    scp -O -r alan:adopptrs/products/check ./check

Les tuiles sont numerotees par ordre decroissant de defauts : 00_* est la
plus fautive, les 'ok_*' sont les reussites, gardees pour comparaison.
"""

import argparse
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
RETAINED = (255, 0, 0)
REJECTED = (255, 150, 0)


def parse_name(imagename):
    """Meme logique que summarize.py::parse()."""
    return tuple(map(int, imagename.split('.')[0].split('_')[-2:]))


def predict(model, image, device, threshold, opening):
    """Rend le masque binaire du reseau, dans l'etat ou les contours en sortent.

    L'ouverture morphologique est optionnelle parce que les deux chaines du
    depot divergent ici : evaluate.py l'applique, walonmap.py non. Le seuil
    a ete calibre avec, la production tournerait sans -- pouvoir basculer
    d'un mode a l'autre est le seul moyen de voir ce que cet ecart change.
    """
    with torch.no_grad():
        inpt = to_tensor(image).unsqueeze(0).to(device)
        outpt = model(inpt).cpu()[0]

    mask = (outpt[0].numpy() > threshold).astype(np.uint8) * 255

    if opening:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8))

    return mask


def confusion(target_ctns, output_ctns, inter, minimum):
    """Apparie cibles et predictions, exactement comme evaluate.py.

    Une cible est trouvee des qu'une prediction la recouvre d'au moins un
    pixel. Les faux positifs et les oublis ne sont comptes qu'au-dela de
    -min : en dessous, walonmap.py ne les publierait pas.
    """
    targets = [[bounding(c), surface(c), False] for c in target_ctns]
    outputs = [[bounding(c), surface(c), False] for c in output_ctns]

    for t in targets:
        for o in outputs:
            box = intersection(t[0], o[0])

            if box is None:
                continue

            if inter[box[0]:(box[2] + 1), box[1]:(box[3] + 1)].sum() > 0:
                t[2] = True
                o[2] = True

    tp = sum(1 for _, _, matched in targets if matched)
    fn = sum(1 for _, area, matched in targets if not matched and area > minimum)
    fp = sum(1 for _, area, matched in outputs if not matched and area > minimum)

    return tp, fp, fn, outputs


def main():
    parser = argparse.ArgumentParser(description='Controle visuel des predictions')
    parser.add_argument('-n', '--network', required=True, help='fichier du reseau')
    parser.add_argument('-m', '--model', default='unet', choices=['unet', 'segnet'], help='schema du reseau')
    parser.add_argument('-multitask', default=True, action='store_true', help='reseau multi-taches')
    parser.add_argument('-i', '--input', default=None, help='fichier VIA')
    parser.add_argument('-p', '--path', default=None, help='dossier des tuiles')
    parser.add_argument('-d', '--destination', default=None, help='dossier de sortie')
    parser.add_argument('-k', type=int, default=5, help='nombre de folds')
    parser.add_argument('-f', '--fold', type=int, default=0, help='fold mis de cote')
    parser.add_argument('-threshold', type=float, default=1e-4, help='seuil de decision')
    parser.add_argument('-min', type=int, default=256, help='surface minimale, comme walonmap.py')
    parser.add_argument('-l', '--limit', type=int, default=20, help='nombre de tuiles dessinees')
    parser.add_argument('-no-opening', dest='opening', default=True, action='store_false',
                        help="n'applique pas l'ouverture 5x5, comme walonmap.py")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))

    via_path = args.input or os.path.join(root, 'resources', 'walonmap', 'via_liege_city.json')
    tiles_path = args.path or os.path.join(root, 'products', 'liege')
    destination = args.destination or os.path.join(root, 'products', 'check')

    for path in (via_path, tiles_path, args.network):
        if not os.path.exists(path):
            sys.exit('Introuvable : ' + path)

    os.makedirs(destination, exist_ok=True)

    # Fold : le meme decoupage que evaluate.py, sans quoi les tuiles
    # dessinees ne seraient pas celles qui ont produit les chiffres.
    import random

    via = VIA.load(via_path)
    keys = sorted(list(via.keys()))

    random.seed(0)
    random.shuffle(keys)

    if args.k > 0:
        keys = [key for i, key in enumerate(keys) if (i % args.k) == args.fold]

    print('{} tuiles dans le fold {} sur {}'.format(len(keys), args.fold, args.k))
    print('seuil {:g} | -min {} | ouverture {}'.format(
        args.threshold, args.min, 'oui' if args.opening else 'non'
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
    results = []
    totals = [0, 0, 0]

    for key in keys:
        filename = os.path.join(tiles_path, key)

        if not os.path.exists(filename):
            continue

        image = Image.open(filename).convert('RGB')
        polygons = via[key]

        target = np.array(to_mask((image.height, image.width), polygons))
        mask = predict(model, image, device, args.threshold, args.opening)

        target_ctns = to_contours(target)
        output_ctns = to_contours(mask)

        inter = ((target > 0) & (mask > 0)).astype(np.uint8)

        tp, fp, fn, outputs = confusion(target_ctns, output_ctns, inter, args.min)

        for i in range(3):
            totals[i] += (tp, fp, fn)[i]

        results.append((tp, fp, fn, key, image, polygons, output_ctns))

    # Les tuiles fautives d'abord : ce sont les seules a regarder longtemps.
    results.sort(key=lambda r: (-(r[1] + r[2]), -r[0]))

    print('{:<24} {:>4} {:>4} {:>4}   {}'.format('tuile', 'TP', 'FP', 'FN', 'fichier'))

    for rank, (tp, fp, fn, key, image, polygons, output_ctns) in enumerate(results[:args.limit]):
        row, col = parse_name(key)

        draw = ImageDraw.Draw(image)

        for polygon in polygons:
            if len(polygon) >= 2:
                draw.line(list(polygon) + [polygon[0]], fill=ANNOTATION, width=3)

        for contour in output_ctns:
            points = [tuple(p) for p in contour[:, 0, :]]

            if len(points) < 2:
                continue

            colour = RETAINED if surface(contour) > args.min else REJECTED
            draw.line(points + [points[0]], fill=colour, width=2)

        tag = 'ok' if (fp + fn) == 0 else 'fp{}fn{}'.format(fp, fn)
        out = '{:02d}_{}_{}_{}.jpg'.format(rank, tag, row, col)

        image.save(os.path.join(destination, out))

        print('{:<24} {:>4} {:>4} {:>4}   {}'.format(key, tp, fp, fn, out))

    print()
    print('Totaux sur le fold : TP {} | FP {} | FN {}'.format(*totals))
    print('Ils doivent retomber sur la ligne du seuil dans calibrate.sbatch.')
    print()
    print('Images dans {}'.format(destination))
    print('  vert = annotation | rouge = prediction retenue | orange = rejetee par -min')


if __name__ == '__main__':
    main()
