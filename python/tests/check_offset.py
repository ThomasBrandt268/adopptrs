"""
Mesure le decalage entre millesimes, tuile par tuile.

Pourquoi avant tout le reste
----------------------------
Trois causes deplacent un panneau d'un millesime a l'autre :

    1. le calage du millesime -- une erreur de georeferencement globale,
       identique sur toute l'image, donc mesurable et corrigeable ;
    2. le deplacement du toit par le relief -- un panneau a dix metres de
       haut est projete radialement depuis le nadir, d'une distance qui
       varie d'un batiment a l'autre ;
    3. le changement reel, seul signal recherche.

L'appariement par batiment prevu en phase 3 absorbe la cause 2 : le toit
se deplace d'un bloc. Il ne peut rien contre la cause 1 des que celle-ci
depasse l'ecartement des batiments -- a Liege une dizaine de metres --
car la detection ne tombe plus sur le voisin mais deux maisons plus loin,
ce qui produit une disparition et une apparition, toutes deux credibles.

Ce script isole la cause 1. Il repond a une question binaire : le
decalage est-il **constant** (georeferencement, corrigeable d'un vecteur)
ou **erratique** d'une tuile a l'autre (relief, orthorectification, et
alors le millesime n'est pas exploitable tel quel) ?

Methode
-------
Correlation de phase entre chaque tuile et son homologue du millesime de
reference. La correlation de phase n'utilise que la phase du spectre :
elle est donc insensible aux differences d'exposition, de saison et de
couleur, qui sont precisement ce qui separe deux millesimes. Une fenetre
de Hanning supprime les artefacts de bord.

Elle suppose en revanche une translation pure. Deux limites en decoulent :
une tuile qui a beaucoup change (chantier, deboisement) donne une reponse
basse, et un decalage superieur a un quart de tuile (128 px, 17 m) devient
peu fiable faute de recouvrement. La colonne 'reponse' sert a ecarter ces
cas, pas a les corriger.

Usage
-----
    conda activate adopptrs
    cd python
    python tests/check_offset.py \\
        --tiles ../products/vintages \\
        --vintages ORTHO_2009_2010 ORTHO_2017 ORTHO_2024 \\
        --reference ORTHO_2018 -k 5 -f 0

Auto-test de la convention de signe :
    python tests/check_offset.py --self-test --tiles ../products/liege
"""

import argparse
import os
import random
import sys

import cv2
import numpy as np

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import via as VIA

from wms import WMS


def grayscale(path):
    """Tuile en niveaux de gris flottants, prete pour la correlation."""
    return np.asarray(Image.open(path).convert('L'), dtype=np.float32)


def offset(reference, other, window):
    """Deplacement de `other` par rapport a `reference`, en pixels.

    Convention verifiee par --self-test : si `other` est `reference`
    translatee de (+dx, +dy), la fonction rend (+dx, +dy).
    """
    (dx, dy), response = cv2.phaseCorrelate(reference, other, window)

    return dx, dy, response


def robust(values):
    """Mediane et ecart absolu median -- resistants aux tuiles aberrantes.

    La moyenne et l'ecart-type ne conviennent pas ici : quelques tuiles
    ou la correlation echoue suffiraient a deplacer la mediane annoncee.
    """
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return float('nan'), float('nan')

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))

    return median, mad


def self_test(tiles):
    """Verifie la convention de signe sur un decalage connu."""
    names = sorted(n for n in os.listdir(tiles) if n.endswith('.jpg'))

    if not names:
        sys.exit('Aucune tuile dans ' + tiles)

    image = grayscale(os.path.join(tiles, names[0]))
    window = cv2.createHanningWindow(image.shape[::-1], cv2.CV_32F)

    print('Auto-test sur {} ({}x{})'.format(names[0], *image.shape[::-1]))
    print()
    print('{:>12} {:>12}   {:>12} {:>12}   {:>8}'.format(
        'dx attendu', 'dy attendu', 'dx mesure', 'dy mesure', 'reponse'
    ))

    ok = True

    for shift_x, shift_y in [(0, 0), (30, 12), (-25, 8), (7, -19)]:
        # np.roll decale le contenu de +shift : la tuile decalee est bien
        # la reference translatee de (+shift_x, +shift_y).
        shifted = np.roll(np.roll(image, shift_y, axis=0), shift_x, axis=1)

        dx, dy, response = offset(image, shifted, window)

        print('{:>12} {:>12}   {:>12.2f} {:>12.2f}   {:>8.3f}'.format(
            shift_x, shift_y, dx, dy, response
        ))

        if abs(dx - shift_x) > 1 or abs(dy - shift_y) > 1:
            ok = False

    print()

    if ok:
        print('Convention confirmee : la fonction rend le deplacement de la')
        print('seconde image par rapport a la premiere, en pixels image.')
        return 0

    print('ECHEC : la convention de signe ne tient pas, ne pas se fier aux mesures.')
    return 1


def main():
    parser = argparse.ArgumentParser(description='Mesure le decalage entre millesimes')
    parser.add_argument('--tiles', default=None, help='dossier contenant un sous-dossier par millesime')
    parser.add_argument('--vintages', nargs='+', default=None, help='millesimes a mesurer')
    parser.add_argument('--reference', default='ORTHO_2018', help='millesime de reference')
    parser.add_argument('-i', '--input', default=None, help='fichier VIA donnant la liste des tuiles')
    parser.add_argument('-k', type=int, default=5, help='nombre de folds')
    parser.add_argument('-f', '--fold', type=int, default=0, help='fold mis de cote')
    # La reponse de phaseCorrelate est le piquet du pic de correlation. Sur
    # de l'imagerie aerienne naturelle elle reste basse (quelques
    # centiemes) meme quand le recalage est excellent : elle ne fait donc
    # pas un filtre par tuile, et s'en servir comme tel ecarte des mesures
    # justes tout en masquant les cas ou tout echoue. Elle n'est ici qu'un
    # indicateur de qualite affiche. Le verdict vient de la dispersion.
    parser.add_argument('--min-response', type=float, default=0.0,
                        help='ecarte les tuiles sous cette reponse (0 = tout garder, defaut)')
    parser.add_argument('--detail', default=False, action='store_true', help='table tuile par tuile')
    parser.add_argument('--self-test', default=False, action='store_true', help='verifie la convention de signe')
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))

    if args.self_test:
        return self_test(args.tiles or os.path.join(root, 'products', 'liege'))

    if not args.vintages:
        sys.exit('--vintages est requis (ou --self-test)')

    tiles = args.tiles or os.path.join(root, 'products', 'vintages')
    via_path = args.input or os.path.join(root, 'resources', 'walonmap', 'via_liege_city.json')

    for path in (tiles, via_path):
        if not os.path.exists(path):
            sys.exit('Introuvable : ' + path)

    # Meme fold que partout ailleurs.
    via = VIA.load(via_path)
    keys = sorted(list(via.keys()))

    random.seed(0)
    random.shuffle(keys)

    if args.k > 0:
        keys = [key for i, key in enumerate(keys) if (i % args.k) == args.fold]

    span = WMS().pixel_span  # m/px, identique a tous les millesimes

    print('{} tuiles | reference {} | {:.6f} m/px'.format(len(keys), args.reference, span))
    print()

    summary = []

    for vintage in args.vintages:
        rows = []
        window = None

        for key in keys:
            a = os.path.join(tiles, args.reference, key)
            b = os.path.join(tiles, vintage, key)

            if not (os.path.exists(a) and os.path.exists(b)):
                continue

            reference, other = grayscale(a), grayscale(b)

            if reference.shape != other.shape:
                continue

            if window is None:
                window = cv2.createHanningWindow(reference.shape[::-1], cv2.CV_32F)

            dx, dy, response = offset(reference, other, window)
            rows.append((key, dx, dy, response))

        if args.detail:
            print('--- {} ---'.format(vintage))
            print('{:<24} {:>8} {:>8} {:>8} {:>9}'.format('tuile', 'dx px', 'dy px', 'norme m', 'reponse'))

            for key, dx, dy, response in sorted(rows, key=lambda r: -np.hypot(r[1], r[2])):
                print('{:<24} {:>8.1f} {:>8.1f} {:>8.2f} {:>9.3f}'.format(
                    key, dx, dy, np.hypot(dx, dy) * span, response
                ))
            print()

        kept = [r for r in rows if r[3] >= args.min_response]

        dx_median, dx_mad = robust([r[1] for r in kept])
        dy_median, dy_mad = robust([r[2] for r in kept])

        # La dispersion du vecteur autour de sa mediane : c'est elle qui
        # tranche entre un decalage constant et un decalage erratique.
        residual, _ = robust([
            np.hypot(r[1] - dx_median, r[2] - dy_median) for r in kept
        ])

        response, _ = robust([r[3] for r in kept])

        summary.append((
            vintage, len(rows), len(kept),
            dx_median, dy_median, dx_mad, dy_mad, residual, response
        ))

    print('{:<20} {:>6} {:>16} {:>16} {:>11} {:>9}'.format(
        'millesime', 'retenu', 'decalage median', 'dispersion MAD', 'residu', 'reponse'
    ))
    print('{:<20} {:>6} {:>16} {:>16} {:>11} {:>9}'.format(
        '', '', 'dx, dy (m)', 'dx, dy (m)', 'median (m)', 'mediane'
    ))

    for vintage, total, kept, dx, dy, dx_mad, dy_mad, residual, response in summary:
        print('{:<20} {:>6} {:>7.2f} {:>8.2f} {:>7.2f} {:>8.2f} {:>11.2f} {:>9.3f}'.format(
            vintage, kept,
            dx * span, dy * span,
            dx_mad * span, dy_mad * span,
            residual * span, response
        ))

    print()
    print('Lecture')
    print('-------')
    print('  Le residu median est la colonne decisive : c\'est ce qui resterait de')
    print('  travers apres avoir applique le decalage median a toutes les tuiles.')
    print()
    print('  residu bien sous 10 m   -> decalage systematique, corrigeable d\'un vecteur ;')
    print('                             le reste releve du relief, que l\'appariement par')
    print('                             batiment absorbe')
    print('  residu de l\'ordre de    -> erratique : aucun vecteur unique ne recale le')
    print('  la dizaine de metres       millesime, il n\'est pas utilisable tel quel')
    print()
    print('Repere : a Liege les batiments sont espaces d\'une dizaine de metres. Un decalage')
    print('median au-dela reporte les detections sur le batiment voisin, ce que l\'appariement')
    print('par batiment ne rattrape pas.')

    return 0


if __name__ == '__main__':
    sys.exit(main())
