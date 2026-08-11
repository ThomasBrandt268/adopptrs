#!/usr/bin/env python

"""
Choisit le -threshold de walonmap.py a partir d'une sortie d'evaluate.py.

Pourquoi ce script
------------------
compare_report.py repond a une autre question : « ce reseau reproduit-il le
rapport ? ». Il lui faut donc une reference publiee, et il n'en existe pas
pour le modele de production (K=0, pas de fold mis de cote) ni pour les
tuiles de Liege. Ici il n'y a rien a comparer : on lit la courbe complete du
reseau que l'on s'apprete a lancer sur WalOnMap et on y place le seuil.

Deux criteres, qui ne tombent pas au meme endroit :

  - le F1 de detection, qui arbitre entre installations trouvees et fausses
    alarmes -- c'est lui qui decide combien de lignes sortent du CSV ;
  - le biais de surface, rapport entre la surface predite et la surface
    annotee. `area` alimente l'estimation de capacite en aval, donc un seuil
    qui gonfle les contours gonfle les kWc de la meme main.

Le premier tire vers le bas, le second vers le haut. Le script affiche les
deux et laisse l'arbitrage visible plutot que de rendre un seul nombre.

Usage
-----
    cd python
    python misc/threshold.py ../products/eval/multiunet_020_liege.txt
    python misc/threshold.py sortie.txt -raw
"""

import argparse
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_report import parse_output, rates, thresholds


def values():
    '''Les seuils d'evaluate.py en clair, dans le meme ordre que thresholds().

    thresholds() rend des etiquettes lisibles ('1-1e-1') ; walonmap.py veut
    un flottant.
    '''
    v = [0.0]
    v.extend(10.0 ** x for x in range(-9, 0))
    v.append(0.5)
    v.extend(1 - 10.0 ** x for x in range(-1, -10, -1))
    v.append(1.0)

    return v


def surface_bias(pixel):
    '''Surface predite / surface annotee, sur les contours apparies.

    evaluate.py ne compte en faux positifs pixel-wise que le debord des
    contours apparies : un contour entierement faux n'entre pas dans ce
    rapport. C'est donc un biais de forme, pas un biais de volume -- il dit
    si les panneaux trouves sont mesures trop grands, pas si l'on en trouve
    trop.
    '''
    tp, fp, fn = pixel[:, 0], pixel[:, 1], pixel[:, 2]

    return np.divide(
        tp + fp, tp + fn,
        out=np.full_like(tp, np.nan), where=(tp + fn) != 0
    )


def main():
    parser = argparse.ArgumentParser(description="Place le seuil de walonmap.py")
    parser.add_argument('output', help="fichier de sortie d'evaluate.py")
    parser.add_argument(
        '-raw', default=False, action='store_true',
        help="n'applique pas la retouche de precision des seuils bas du rapport"
    )
    args = parser.parse_args()

    arrays = parse_output(args.output)
    contour, pixel = arrays['contour'], arrays['pixel']

    # La retouche « precision := 0 quand aucune cible n'est manquee » corrige
    # un artefact des seuils bas : une tache unique recouvrant la decoupe
    # attrape toutes les cibles. Elle a un sens quand les decoupes vides sont
    # ecartees. Avec -negatives, cet artefact se voit deja dans les faux
    # positifs, et la retouche peut alors effacer un seuil legitime.
    p, r, f1 = rates(contour, contour_wise=not args.raw)
    bias = surface_bias(pixel)

    labels = thresholds()

    print('=' * 78)
    print('Seuil de decision -- {}'.format(os.path.basename(args.output)))
    print('=' * 78)
    print()
    print('{:>9}  {:>6} {:>6} {:>6}  {:>9} {:>9} {:>9}  {:>7}'.format(
        'seuil', 'TP', 'FP', 'FN', 'precision', 'rappel', 'F1', 'surface'
    ))

    for i, t in enumerate(labels):
        print('{:>9}  {:6.0f} {:6.0f} {:6.0f}  {:9.4f} {:9.4f} {:9.4f}  {:7.3f}'.format(
            t, contour[i, 0], contour[i, 1], contour[i, 2], p[i], r[i], f1[i], bias[i]
        ))

    half = labels.index('0.5')
    best = int(np.argmax(f1))

    # Le seuil le moins biaise en surface, parmi ceux qui detectent encore
    # quelque chose : a rappel nul le biais n'a plus d'objet.
    usable = np.where(r > 0.1, np.abs(bias - 1), np.inf)
    fair = int(np.nanargmin(usable)) if np.isfinite(usable).any() else half

    print()
    print('Lecture')
    print('-------')
    print('{:<34}{:>10}{:>10}{:>10}{:>10}'.format('', 'seuil', 'rappel', 'F1', 'surface'))
    for name, i in [
        ('defaut actuel de walonmap.py', half),
        ('F1 de detection maximal', best),
        ('surface la moins biaisee', fair),
    ]:
        print('{:<34}{:>10}{:10.4f}{:10.4f}{:10.3f}'.format(name, labels[i], r[i], f1[i], bias[i]))

    print()

    if best == half:
        print('Le defaut 0.5 est deja le meilleur compromis de detection : rien a changer.')
    else:
        print('Passer de 0.5 a {} rend {:+.1f} points de rappel et {:+.1f} points de F1,'.format(
            labels[best], 100 * (r[best] - r[half]), 100 * (f1[best] - f1[half])
        ))
        print('et deplace le biais de surface de {:.3f} a {:.3f}.'.format(bias[half], bias[best]))

    # Les deux bornes ne sont pas des seuils exploitables : 0 retient tout,
    # 1 ne retient rien. Si le F1 y culmine, c'est que la courbe est plate ou
    # que le jeu est trop petit -- on retombe sur le defaut plutot que de
    # proposer une absurdite.
    retained = half if labels[best] in ('0', '1') else best

    print()
    # .10g et pas .g : 1-1e-7 s'affiche sinon « 1 », soit le seuil qui ne
    # retient rien -- l'inverse de ce qu'on vient de conclure.
    print("walonmap.py -threshold {:.10g}   (soit {}, a confirmer par un coup d'oeil aux tuiles)".format(
        values()[retained], labels[retained]
    ))

    return 0


if __name__ == '__main__':
    sys.exit(main())
