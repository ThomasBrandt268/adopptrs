#!/usr/bin/env python

"""
Compare une sortie d'evaluate.py aux chiffres publies dans le rapport.

Le rapport original (Francois Rozet, 2020) a ete produit avec torch 1.x,
NumPy 1.x et OpenCV 3. Rejouer le meme protocole (K=5, un fold mis de cote,
20 epoques) sur la stack modernisee et retrouver les memes courbes est le
seul controle qui prouve que la migration n'a rien altere numeriquement.

Les tableaux de reference sont ceux de `misc/hardcoded_plots.py`, qui sont
justement les sorties brutes d'evaluate.py de l'epoque. Ils sont lus par
`ast` plutot qu'importes : hardcoded_plots.py dessine ses figures a
l'import et exige matplotlib avec LaTeX.

Usage
-----
    cd python
    python misc/compare_report.py ../products/eval/multiunet_0_020.txt
    python misc/compare_report.py sortie.txt -m multiunet -f 0

Deux retouches manuelles du rapport sont reappliquees ici aux deux jeux de
chiffres, sans quoi la comparaison porterait sur des points aberrants (cf.
le N.B. d'evaluate.py) :

  - contour-wise, precision := 0 quand aucun contour n'est manque (aux
    seuils tres bas la sortie est une seule tache qui couvre tout, et
    "matcher" toutes les cibles avec elle n'a pas de sens) ;
  - pixel-wise, rappel := 0 quand la sortie et la cible sont vides (aux
    seuils tres hauts), la division 0/0 valant 1 par defaut.
"""

import argparse
import ast
import numpy as np
import os
import sys


# Ordre des modeles dans hardcoded_plots.py
MODELS = ['unet', 'multiunet', 'segnet', 'multisegnet']
LABELS = ['U-Net', 'Multi-Task U-Net', 'SegNet', 'Multi-Task SegNet']

HARDCODED = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hardcoded_plots.py')


def thresholds():
    '''Reconstruit les etiquettes des seuils d'evaluate.py, dans le meme ordre.

    Les seuils hauts sont notes 1-1e-k plutot que developpes : au-dela de
    1-1e-7, l'affichage decimal les confond tous avec 1.
    '''
    labels = ['0']
    labels.extend('1e-{}'.format(-x) for x in range(-9, 0))
    labels.append('0.5')
    labels.extend('1-1e-{}'.format(-x) for x in range(-1, -10, -1))
    labels.append('1')

    return labels


def parse_output(path):
    '''Extrait les deux matrices d'une sortie d'evaluate.py.'''
    with open(path, 'r') as f:
        text = f.read()

    arrays = {}

    for key, label in [('contour', 'Contour wise'), ('pixel', 'Pixel wise')]:
        # rfind : evaluate.py -o ajoute a la suite, on veut la derniere passe
        i = text.rfind(label)

        if i < 0:
            raise ValueError("'{}' introuvable dans {}".format(label, path))

        start = text.index('[[', i)
        end = text.index(']]', start) + 2

        arrays[key] = np.array(ast.literal_eval(text[start:end]), dtype=float)

    return arrays


def parse_reference(model, fold, path=HARDCODED):
    '''Extrait les matrices du rapport, sans executer le module.'''
    with open(path, 'r') as f:
        tree = ast.parse(f.read())

    i = MODELS.index(model)
    wanted = {'detection': 'contour', 'segmentation': 'pixel'}
    arrays = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue

        target = node.targets[0]

        # On cherche les affectations de la forme detection[i][j] = [[...]]
        if not (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Subscript)):
            continue

        outer = target.value

        if not isinstance(outer.value, ast.Name) or outer.value.id not in wanted:
            continue

        try:
            if ast.literal_eval(outer.slice) != i or ast.literal_eval(target.slice) != fold:
                continue

            arrays[wanted[outer.value.id]] = np.array(ast.literal_eval(node.value), dtype=float)
        except ValueError:
            continue

    for key in wanted.values():
        if key not in arrays:
            raise ValueError('reference {} absente pour {} fold {}'.format(key, model, fold))

    return arrays


def rates(arr, contour_wise):
    '''(Re)calcule precision et rappel a partir des comptes bruts.

    Les colonnes 3 et 4 du rapport ont ete retouchees a la main ; les
    recalculer des colonnes 0-2 puis reappliquer les memes retouches met
    les deux jeux de chiffres sur le meme pied.
    '''
    tp, fp, fn = arr[:, 0], arr[:, 1], arr[:, 2]

    # 0/0 -> 1, comme safe_divide dans evaluate.py
    precision = np.divide(tp, tp + fp, out=np.ones_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.ones_like(tp), where=(tp + fn) != 0)

    # Sortie et cible vides : le rappel n'est pas 1, il n'existe pas
    recall[(tp + fn) == 0] = 0

    if contour_wise:
        # Une seule tache couvrant tout attrape toutes les cibles
        precision[(fn == 0) & (tp > 0)] = 0

    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros_like(tp), where=(precision + recall) != 0
    )

    return precision, recall, f1


def average_precision(precision, recall):
    '''AP au sens du rapport : integrale du rappel le long de la precision.'''
    trapezoid = getattr(np, 'trapezoid', None) or np.trapz

    return trapezoid(recall, precision)


def table(title, new, ref, contour_wise):
    '''Affiche la comparaison seuil par seuil.'''
    p, r, f = rates(new, contour_wise)
    pr, rr, fr = rates(ref, contour_wise)

    print()
    print(title)
    print('-' * len(title))
    print('{:>9}  {:>16}  {:>16}  {:>16}'.format('seuil', 'precision', 'rappel', 'F1'))
    print('{:>9}  {:>7} {:>8}  {:>7} {:>8}  {:>7} {:>8}'.format(
        '', 'rapport', 'nouveau', 'rapport', 'nouveau', 'rapport', 'ecart'
    ))

    for i, t in enumerate(thresholds()):
        print('{:>9}  {:7.4f} {:8.4f}  {:7.4f} {:8.4f}  {:7.4f} {:+8.4f}'.format(
            t, pr[i], p[i], rr[i], r[i], fr[i], f[i] - fr[i]
        ))

    half = 10  # index du seuil 0.5

    print()
    print('comptes au seuil 0.5 : TP {:.0f} -> {:.0f} | FP {:.0f} -> {:.0f} | FN {:.0f} -> {:.0f}'.format(
        ref[half, 0], new[half, 0], ref[half, 1], new[half, 1], ref[half, 2], new[half, 2]
    ))

    return {
        'AP': (average_precision(pr, rr), average_precision(p, r)),
        'F1 au seuil 0.5': (fr[half], f[half]),
        'F1 maximal': (fr.max(), f.max()),
    }


def spread(model, key, contour_wise):
    '''Etendue de l'AP du rapport sur ses cinq folds, pour situer l'ecart.'''
    aps = []

    for fold in range(5):
        arr = parse_reference(model, fold)[key]
        p, r, _ = rates(arr, contour_wise)
        aps.append(average_precision(p, r))

    return min(aps), max(aps)


def main():
    parser = argparse.ArgumentParser(description="Compare une sortie d'evaluate.py au rapport")
    parser.add_argument('output', help="fichier de sortie d'evaluate.py")
    parser.add_argument('-m', '--model', default='multiunet', choices=MODELS, help='modele evalue')
    parser.add_argument('-f', '--fold', type=int, default=0, help='fold mis de cote')
    parser.add_argument('-tol', type=float, default=0.05, help="ecart d'AP juge acceptable")
    args = parser.parse_args()

    new = parse_output(args.output)
    ref = parse_reference(args.model, args.fold)

    print('=' * 72)
    print('{} -- fold {} -- {}'.format(
        LABELS[MODELS.index(args.model)], args.fold, os.path.basename(args.output)
    ))
    print('=' * 72)

    summary = {}

    summary['Detection (contour-wise)'] = table(
        'Detection (contour-wise)', new['contour'], ref['contour'], True
    )
    summary['Segmentation (pixel-wise)'] = table(
        'Segmentation (pixel-wise)', new['pixel'], ref['pixel'], False
    )

    print()
    print('Resume')
    print('------')
    print('{:<32}{:>10}{:>10}{:>10}'.format('', 'rapport', 'nouveau', 'ecart'))

    verdict = True

    for scope, metrics in summary.items():
        for name, (r, n) in metrics.items():
            print('{:<32}{:10.4f}{:10.4f}{:+10.4f}'.format(
                '{} / {}'.format(scope.split()[0], name), r, n, n - r
            ))

            if name == 'AP' and abs(n - r) > args.tol:
                verdict = False

    lo, hi = spread(args.model, 'contour', True)
    print()
    print('Pour situer : sur les cinq folds du rapport, AP detection va de {:.4f} a {:.4f}.'.format(lo, hi))

    if verdict:
        print("Ecart d'AP sous {:g} : la migration n'a rien altere de mesurable.".format(args.tol))
    else:
        print("Ecart d'AP au-dela de {:g} : a investiguer avant d'aller plus loin.".format(args.tol))

    return 0 if verdict else 1


if __name__ == '__main__':
    sys.exit(main())
