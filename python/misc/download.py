#!/usr/bin/env python

"""
Download annotated WalOnMap tiles

Un millesime au choix (-vintage) et, au besoin, un seul fold (-k / -f) :
comparer plusieurs millesimes sur les memes tuiles demande de rapatrier
la meme liste plusieurs fois, et la limiter au fold mis de cote divise la
facture par cinq.

    python misc/download.py -vintage ORTHO_2017 -k 5 -f 0 \\
        -d ../products/vintages/ORTHO_2017
"""

###########
# Imports #
###########

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import via as VIA

from PIL import Image
from wms import WMS
from summarize import parse


#############
# Constants #
#############

_DELAY = 0.2  # seconds between requests (matches the tested WMS benchmark spacing)
_SUSPECT_SIZE = 15_000  # bytes ; below this, check whether the tile is a uniform blank


########
# Main #
########

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description='Download images from WalOnMap')
    parser.add_argument('-d', '--destination', default='../products/liege/', help='destination of the tiles')
    parser.add_argument('-i', '--input', default='../resources/walonmap/via_liege_city.json', help='input VIA file')
    parser.add_argument('-vintage', default=None, help='service du millesime (ex. ORTHO_2024) ; defaut ORTHO_2018')
    parser.add_argument('-k', type=int, default=0, help='nombre de folds ; 0 pour tout telecharger')
    parser.add_argument('-f', '--fold', type=int, default=0, help='fold a telecharger')
    args = parser.parse_args()

    # Destination
    os.makedirs(args.destination, exist_ok=True)

    wm = WMS(vintage=args.vintage)

    # Download
    via = VIA.load(args.input)

    # Meme decoupage que evaluate.py et train.py : sans la meme graine sur
    # les memes cles triees, le fold telecharge ne serait pas celui qui a
    # ete mis de cote a l'entrainement.
    names = sorted(list(via))

    random.seed(0)
    random.shuffle(names)

    if args.k > 0:
        names = [name for i, name in enumerate(names) if (i % args.k) == args.fold]

    total = len(names)

    print('millesime {} | {} tuiles'.format(wm.vintage or wm.url, total))
    print()

    blank = []

    for i, imagename in enumerate(names, 1):
        row, col = parse(imagename)

        tile = wm.get_tile(row, col)
        size = len(tile.getvalue())

        status = '{:>6.1f} Ko'.format(size / 1000)

        if size < _SUSPECT_SIZE and Image.open(tile).convert('L').getextrema() == (255, 255):
            # Confirmed blank tile (out of WMS coverage) : skip, don't pollute the training set
            blank.append((imagename, row, col, size))
            status += '  BLANCHE (extrema=255,255) -> ignoree'
        else:
            # Write the bytes received from the server as-is (no JPEG re-encode)
            with open(os.path.join(args.destination, imagename), 'wb') as f:
                f.write(tile.getvalue())

            if size < _SUSPECT_SIZE:
                status += '  (petite mais non uniforme, conservee)'

        print('[{}/{}] {:<24} {}'.format(i, total, imagename, status))

        time.sleep(_DELAY)

    # Summary
    print()
    print('{}/{} tuiles telechargees'.format(total - len(blank), total))

    if blank:
        print('{} tuile(s) blanche(s) ignoree(s) (hors couverture WMS) :'.format(len(blank)))
        for imagename, row, col, size in blank:
            print('  {}  (row={}, col={}, {:.1f} Ko)'.format(imagename, row, col, size / 1000))
    else:
        print('Aucune tuile blanche detectee.')
