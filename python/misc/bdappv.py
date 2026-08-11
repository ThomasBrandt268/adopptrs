#!/usr/bin/env python

"""
Convertit BDAPPV au format attendu par train.py.

BDAPPV livre des images et des masques raster ; notre chaine veut des
polygones au format VIA et des images a notre resolution. Ce script fait
les deux, campagne par campagne.

Resolution
----------
Le reseau doit voir les panneaux a la taille qu'ils auront en production,
soit 0,132 m/px -- le pas de WalOnMap. Les deux campagnes n'y sont pas :

    google  0,1 m/px  ->  reduction (x0,756) : on jette du detail reel,
                          c'est honnete
    ign     0,2 m/px  ->  agrandissement (x1,512) : on fabrique du flou
                          que WalOnMap n'a pas

Techniquement google est donc le meilleur choix, et il offre 13 303
masques contre 7 685. Juridiquement c'est l'inverse : les images IGN sont
sous Licence Ouverte 2.0, celles de Google restent soumises aux conditions
de Google. Le script ne tranche pas, `--campaign` le fait.

Ce qui n'est PAS converti
-------------------------
Les images sans masque. Elles ne sont pas des toitures sans panneaux :
ce sont des installations connues du registre dont personne n'a trace le
contour. Les traiter en negatifs apprendrait au reseau a ignorer de vrais
panneaux -- le pire defaut possible pour notre usage.

Attention, limite qui reste entiere : chaque image de BDAPPV est centree
sur UNE installation connue, et une seconde installation presente dans le
cadre peut ne pas etre annotee. Le controle se fait a l'oeil sur quelques
dizaines de vignettes avant de lancer un entrainement (--check).

Usage
-----
    conda activate adopptrs
    cd python

    # controle visuel sur 12 images, avant tout le reste
    python misc/bdappv.py --source /scratch/users/$USER/bdappv/bdappv \\
        --campaign google --limit 12 --check

    # conversion complete
    python misc/bdappv.py --source /scratch/users/$USER/bdappv/bdappv \\
        --campaign google \\
        -d /scratch/users/$USER/bdappv/google_0132 \\
        -o ../products/json/bdappv_google.json
"""

import argparse
import os
import sys

import cv2
import numpy as np

from PIL import Image, ImageDraw

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON = os.path.dirname(_HERE)

sys.path.insert(0, _PYTHON)

import via as VIA

from wms import WMS


GSD = {'google': 0.1, 'ign': 0.2}
MIN_PIXELS = 12  # sous ce compte, le contour est du bruit d'annotation


def polygons(mask, scale_x, scale_y, minimum=MIN_PIXELS):
    """Contours exterieurs du masque, remis a l'echelle cible.

    Les coordonnees sont mises a l'echelle apres extraction plutot que le
    masque avant : redimensionner un masque binaire cree des bords
    intermediaires qu'il faut reseuiller, et le seuil choisi deplace le
    contour d'un pixel ou deux. Sur un panneau de vingt pixels de cote,
    ce n'est pas anodin.
    """
    contours, _ = cv2.findContours(
        (mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    out = []

    for c in contours:
        if cv2.contourArea(c) < minimum:
            continue

        points = [(float(x) * scale_x, float(y) * scale_y) for x, y in c[:, 0, :]]

        if len(points) >= 3:
            out.append(points)

    return out


def main():
    parser = argparse.ArgumentParser(description='Convertit BDAPPV au format VIA')
    parser.add_argument('--source', required=True, help='racine bdappv/ (contenant google/ et ign/)')
    parser.add_argument('--campaign', default='google', choices=sorted(GSD), help='campagne a convertir')
    parser.add_argument('-d', '--destination', default=None, help='dossier des images reechantillonnees')
    parser.add_argument('-o', '--output', default=None, help='fichier VIA produit')
    parser.add_argument('--gsd', type=float, default=None, help='resolution cible en m/px (defaut : celle de WalOnMap)')
    parser.add_argument('--limit', type=int, default=0, help='ne traiter que les N premieres (0 = tout)')
    parser.add_argument('--check', default=False, action='store_true',
                        help='ecrit des vignettes de controle au lieu du jeu complet')
    args = parser.parse_args()

    root = os.path.join(args.source, args.campaign)
    img_dir, mask_dir = os.path.join(root, 'img'), os.path.join(root, 'mask')

    for d in (img_dir, mask_dir):
        if not os.path.isdir(d):
            sys.exit('Introuvable : ' + d)

    target = args.gsd or WMS().pixel_span
    scale = GSD[args.campaign] / target

    names = sorted(os.listdir(mask_dir))

    if args.limit:
        names = names[:args.limit]

    print('campagne %s : %d masques' % (args.campaign, len(names)))
    print('resolution %.3f -> %.6f m/px, facteur %.4f' % (GSD[args.campaign], target, scale))

    destination = args.destination or os.path.join(args.source, args.campaign + '_converti')
    os.makedirs(destination, exist_ok=True)

    via = {}
    sans_image = sans_contour = 0

    for i, name in enumerate(names, 1):
        image_path = os.path.join(img_dir, name)

        if not os.path.exists(image_path):
            sans_image += 1
            continue

        image = Image.open(image_path).convert('RGB')
        size = (int(round(image.width * scale)), int(round(image.height * scale)))

        # Le facteur applique aux polygones doit etre celui que l'image a
        # reellement subi, arrondi compris, et non le facteur nominal :
        # 302/400 n'est pas 0,755903. L'ecart est infime mais systematique,
        # et un decalage systematique entre masque et image est exactement
        # ce qu'un reseau apprend le mieux.
        mask = np.array(Image.open(os.path.join(mask_dir, name)).convert('L'))
        polys = polygons(mask, size[0] / image.width, size[1] / image.height)

        if not polys:
            sans_contour += 1
            continue

        # LANCZOS a la reduction comme a l'agrandissement : c'est le seul
        # des filtres de PIL qui ne tremble pas sur les lignes fines, et
        # une rangee de modules n'est rien d'autre qu'une ligne fine.
        image = image.resize(size, Image.LANCZOS)

        if args.check:
            draw = ImageDraw.Draw(image)
            for p in polys:
                draw.line([(x, y) for x, y in p] + [p[0]], fill=(255, 0, 0), width=2)
            image.save(os.path.join(destination, 'check_' + name))
        else:
            image.save(os.path.join(destination, name))
            via[name] = [[(round(x, 1), round(y, 1)) for x, y in p] for p in polys]

        if i % 500 == 0:
            print('  %d/%d' % (i, len(names)))

    print()
    print('images ecrites   : %d' % (len(names) - sans_image - sans_contour))
    print('masque sans image: %d' % sans_image)
    print('masque vide      : %d' % sans_contour)
    print('taille cible     : %s px' % (str(size) if names else '-'))

    if args.check:
        print()
        print('Vignettes de controle dans %s' % destination)
        print('Regarde-les : un panneau visible SANS contour rouge signifie une')
        print('annotation incomplete, et c\'est ce qui degraderait le rappel.')
        return 0

    output = args.output or os.path.join(_PYTHON, '..', 'products', 'json',
                                         'bdappv_%s.json' % args.campaign)
    VIA.dump(via, output, path=destination)

    print()
    print('VIA ecrit : %s (%d images, %d polygones)' % (
        output, len(via), sum(map(len, via.values()))))

    return 0


if __name__ == '__main__':
    sys.exit(main())
