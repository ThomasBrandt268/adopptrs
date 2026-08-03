"""
Validation visuelle du portage WMTS -> WMS.

Telecharge quelques tuiles annotees via la nouvelle classe WMS et
superpose les polygones de via_liege_city.json. Si l'alignement est
correct, les contours doivent epouser les panneaux photovoltaiques.

C'est le seul test qui prouve reellement que le nouveau quadrillage
reproduit l'ancien : la geometrie peut etre coherente en interne tout
en etant decalee par rapport aux annotations d'origine.

Usage
-----
    conda activate adopptrs
    cd python
    python tests/check_alignment.py
    python tests/check_alignment.py --path ../products/liege/

Avec --path, les tuiles deja telechargees localement (misc/download.py)
sont utilisees telles quelles (pas de reencodage JPEG) ; repli sur le
WMS pour toute tuile absente du dossier local.

Produit check_<row>_<col>.jpg (dans le repertoire courant) pour chaque
tuile testee.
"""

import argparse
import json
import os
import random
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wms import WMS


_HERE = os.path.dirname(os.path.abspath(__file__))
VIA_PATH = os.path.join(_HERE, '..', '..', 'resources', 'walonmap', 'via_liege_city.json')
N_SAMPLES = 6
OUTLINE = (255, 0, 0)
WIDTH = 3


def load_via(path):
    with open(path, 'r') as f:
        data = json.load(f)

    if '_via_img_metadata' in data:
        data = data['_via_img_metadata']

    out = {}
    for key, entry in data.items():
        name = entry.get('filename', key)
        polys = []
        for r in entry.get('regions', []):
            shape = r.get('shape_attributes', {})
            xs = shape.get('all_points_x')
            ys = shape.get('all_points_y')
            if xs and ys:
                polys.append(list(zip(xs, ys)))
        out[name] = polys
    return out


def parse_name(imagename):
    """Meme logique que summarize.py::parse()."""
    return tuple(map(int, imagename.split('.')[0].split('_')[-2:]))


def main():
    parser = argparse.ArgumentParser(description='Valide l\'alignement du portage WMTS -> WMS')
    parser.add_argument('--path', default=None, help='dossier de tuiles locales (ex. ../products/liege/) ; repli sur le WMS si absent')
    args = parser.parse_args()

    if not os.path.exists(VIA_PATH):
        sys.exit('Introuvable : ' + VIA_PATH)

    via = load_via(VIA_PATH)
    print(len(via), 'images annotees')

    # On privilegie les tuiles bien fournies : un decalage y saute aux yeux
    annotated = [(k, v) for k, v in via.items() if len(v) >= 2]
    print(len(annotated), 'tuiles avec >= 2 polygones')

    if not annotated:
        annotated = [(k, v) for k, v in via.items() if v]

    random.seed(0)
    sample = random.sample(annotated, min(N_SAMPLES, len(annotated)))

    wm = WMS()

    for name, polys in sample:
        row, col = parse_name(name)

        local_file = os.path.join(args.path, name) if args.path else None

        try:
            if local_file is not None and os.path.exists(local_file):
                img = Image.open(local_file).convert('RGB')
                source = 'local'
            else:
                img = Image.open(wm.get_tile(row, col)).convert('RGB')
                source = 'WMS'
        except Exception as e:
            print('%-24s ECHEC : %s' % (name, e))
            continue

        draw = ImageDraw.Draw(img)
        for poly in polys:
            if len(poly) >= 2:
                draw.line(list(poly) + [poly[0]], fill=OUTLINE, width=WIDTH)

        out = 'check_%d_%d.jpg' % (row, col)
        img.save(out)
        print('%-24s %d polygone(s) [%s] -> %s' % (name, len(polys), source, out))

    print()
    print('Ouvre les fichiers check_*.jpg :')
    print('  - contours SUR les panneaux      -> alignement correct')
    print('  - contours decales d\'un offset   -> origine ou tile_span a revoir')
    print('  - contours sur du vide           -> mauvaise tuile (row/col inverses ?)')


if __name__ == '__main__':
    main()
