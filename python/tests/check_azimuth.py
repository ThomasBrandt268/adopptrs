"""
Verification du calcul d'azimut de summarize.py.

Construit des rectangles dont l'orientation est connue par construction,
les fait passer par la meme chaine que la production (`cv2.minAreaRect`
puis `cv2.boxPoints`) et compare l'azimut obtenu a l'azimut attendu.

Le test qui compte est celui de l'ambiguite modulo 90 degres : un
rectangle et le meme tourne d'un quart de tour ont le meme `angle` au
sens d'OpenCV, mais des azimuts qui different de 90 degres. C'est
exactement ce que l'ancienne version confondait.

Usage
-----
    conda activate adopptrs
    cd python
    python tests/check_azimuth.py
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from summarize import orientation


TOLERANCE = 1e-6


def rectangle(bearing, length, width, center=(0.0, 0.0)):
    """Rectangle dont le grand cote pointe vers `bearing` (0 = nord).

    Renvoie ses quatre coins en (est, nord), dans le sens direct.
    """
    rad = np.radians(bearing)

    along = np.array([np.sin(rad), np.cos(rad)])    # le faitage
    across = np.array([np.sin(rad + np.pi / 2), np.cos(rad + np.pi / 2)])

    center = np.array(center, dtype=float)

    return np.array([
        center - along * length / 2 - across * width / 2,
        center + along * length / 2 - across * width / 2,
        center + along * length / 2 + across * width / 2,
        center - along * length / 2 + across * width / 2,
    ])


def southern(bearing):
    """Le representant de `bearing` modulo 180 qui regarde le sud."""
    bearing %= 360
    return bearing if 90 <= bearing < 270 else (bearing + 180) % 360


def check(condition, label):
    print('  {:<58} {}'.format(label, 'ok' if condition else 'ECHEC'))
    return bool(condition)


def main():
    ok = True

    print('Rectangles construits, azimut lu directement')

    # Un faitage oriente vers `bearing` fait regarder les panneaux 90
    # degres plus loin ; seul le representant sud est retenu.
    for bearing in range(0, 360, 15):
        box = rectangle(bearing, length=12.0, width=4.0)
        azimuth, elongation = orientation(box)

        expected = southern(bearing + 90)

        ok &= check(
            abs(azimuth - expected) < TOLERANCE and abs(elongation - 3.0) < TOLERANCE,
            'faitage {:3d} deg -> azimut {:6.1f} (attendu {:6.1f})'.format(
                bearing, azimuth, expected
            )
        )

    print()
    print('Chaine complete : minAreaRect puis boxPoints, comme en production')

    # Comme summarize.py : recentrage, puis entiers en millimetres. Le
    # centre choisi est une coordonnee Lambert 72 realiste -- sans le
    # recentrage, le pas de float32 y vaut 16 mm et fait deriver l'azimut
    # de quelques dixiemes de degre.
    for bearing in (0, 22.5, 45, 67.5, 90, 123.4):
        corners = rectangle(bearing, length=9.0, width=3.5, center=(150000.0, 130000.0))
        corners = (corners - corners.mean(axis=0)) / 1e-3

        box = cv2.boxPoints(cv2.minAreaRect(corners.astype(np.int32)))

        azimuth, elongation = orientation(box)
        expected = southern(bearing + 90)

        # 0,01 deg : le passage par des entiers en millimetres laisse
        # environ 5e-3 deg de residu, deux ordres de grandeur sous
        # l'arrondi a deux decimales du CSV.
        ok &= check(
            abs(azimuth - expected) < 0.01 and abs(elongation - 9.0 / 3.5) < 1e-3,
            'faitage {:5.1f} deg -> azimut {:6.2f} (attendu {:6.2f})'.format(
                bearing, azimuth, expected
            )
        )

    print()
    print("Ambiguite a 90 degres : c'est ce que l'ancienne version confondait")

    est_ouest = orientation(rectangle(90, 12.0, 4.0))[0]   # faitage est-ouest
    nord_sud = orientation(rectangle(0, 12.0, 4.0))[0]     # faitage nord-sud

    ok &= check(abs(est_ouest - 180.0) < TOLERANCE, 'faitage est-ouest  -> plein sud (180)')
    ok &= check(abs(nord_sud - 90.0) < TOLERANCE, 'faitage nord-sud   -> plein est (90)')
    ok &= check(abs(est_ouest - nord_sud - 90.0) < TOLERANCE, 'les deux different de 90 degres')

    print()
    print('Cas limites')

    # Un carre n'a pas de petit cote : l'azimut ne veut rien dire, et
    # c'est l'elongation qui doit le signaler.
    ok &= check(abs(orientation(rectangle(30, 5.0, 5.0))[1] - 1.0) < 1e-9, 'carre -> elongation 1')

    # L'ordre des coins ne doit rien changer.
    box = rectangle(37.0, 10.0, 2.5)
    azimuths = [orientation(np.roll(box, k, axis=0))[0] for k in range(4)]
    ok &= check(max(azimuths) - min(azimuths) < TOLERANCE, 'azimut insensible au coin de depart')

    # Le sens de parcours non plus.
    ok &= check(
        abs(orientation(box[::-1])[0] - azimuths[0]) < TOLERANCE,
        'azimut insensible au sens de parcours'
    )

    # Une installation qui regarderait le nord est ramenee au sud.
    ok &= check(
        all(90 <= orientation(rectangle(b, 8.0, 2.0))[0] < 270 for b in np.arange(0, 360, 7)),
        'tous les azimuts dans [90, 270['
    )

    print()
    print('RESULTAT :', 'tout passe' if ok else 'AU MOINS UN ECHEC')

    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
