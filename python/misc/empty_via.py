#!/usr/bin/env python

"""
Fabrique un VIA de tuiles vides couvrant une region.

Pourquoi ce script
------------------
download.py telecharge les tuiles listees par un VIA ; walonmap.py couvre
une region mais exige un reseau et rend des detections. Il manquait le
chainon : partir d'un contour et rendre la liste des tuiles, sans modele
et sans annotation.

C'est ce qu'il faut pour fournir des negatifs. La revue du 2026-08-12 a
montre que le reseau se trompe sur ce qu'il n'a jamais vu -- conteneurs
d'un depot, hangars a bac acier, ombrieres de parking. Ni Liege ville ni
la Californie residentielle n'en contiennent. Le geste utile est donc
d'aller chercher des tuiles de zoning industriel et de les declarer
vides.

A ne pas confondre avec un defaut du chargeur : VIADataset montre deja
des negatifs au reseau (alt=1 tire une vignette aleatoire par vignette
positive, et le mode -special livre les tuiles entieres, vides comprises).
Ce qui manque est la couverture, pas le mecanisme.

Le fichier produit ne doit PAS etre fusionne dans via_liege_city.json :
le fold se calcule sur les cles triees puis melangees, donc y ajouter des
tuiles redistribuerait le fold 0 et ferait perdre l'etalon de mesure.
Il se passe a train.py par -negatives, qui le verse entierement a
l'entrainement.

Entree
------
Un GeoJSON de geometrie Polygon brute (pas une FeatureCollection), le
meme format que le -p de walonmap.py :

    {"type": "Polygon", "coordinates": [[[lon, lat], [lon, lat], ...]]}

geojson.io en produit d'un rectangle trace a la souris ; il suffit de
copier la geometrie.

Usage
-----
    cd python
    python misc/empty_via.py -p ../resources/walonmap/zoning.geojson \\
        -o ../resources/walonmap/via_negatifs.json

    python misc/download.py -i ../resources/walonmap/via_negatifs.json \\
        -d ../products/liege

Puis verifier a l'oeil, en pleine resolution, qu'aucune tuile ne porte de
panneau -- et retirer du VIA celles qui en portent.

Le dossier de destination est bien ../products/liege, celui des tuiles
deja annotees : VIADataset ne connait qu'un seul dossier (le -p de
train.py) et ecarte en silence les fichiers qu'il n'y trouve pas. Des
negatifs ranges ailleurs donneraient un entrainement identique sans
aucun message. Les noms (row, col) sont des coordonnees, donc uniques :
aucune collision a craindre.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import via as VIA

from walonmap import Contour
from wms import WMS


def main():
    parser = argparse.ArgumentParser(description='Liste les tuiles couvrant un contour')
    parser.add_argument('-p', '--polygon', required=True, help='fichier GeoJSON (geometrie Polygon)')
    parser.add_argument('-o', '--output', required=True, help='VIA de sortie')
    parser.add_argument('-t', '--tile', default='', help='prefixe des noms de tuile')
    args = parser.parse_args()

    if not os.path.exists(args.polygon):
        sys.exit('Introuvable : ' + args.polygon)

    # La geometrie des tuiles ne depend pas du millesime : un meme (row, col)
    # rend le meme rectangle au sol, seule l'URL change. Le service par
    # defaut suffit donc a nommer les tuiles, et -vintage se choisit plus
    # tard, au telechargement.
    wm = WMS()

    with open(args.polygon, 'r') as f:
        geojson = json.load(f)

    if 'coordinates' not in geojson:
        sys.exit("GeoJSON sans 'coordinates' : attendu une geometrie Polygon, "
                 "pas une Feature ni une FeatureCollection")

    # GeoJSON ordonne (longitude, latitude) ; wgs_to_tile attend l'inverse.
    contour = Contour([
        wm.wgs_to_tile(point[1], point[0])
        for point in geojson['coordinates'][0]
    ])

    via = {
        '{}{}_{}.jpg'.format(args.tile, row, col): []
        for row, col in contour
    }

    VIA.dump(via, args.output)

    print('{} tuiles dans le contour'.format(len(via)))
    print('VIA ecrit : {}'.format(args.output))
    print()
    print('Etape suivante :')
    print('  python misc/download.py -i {} -d ../products/liege'.format(args.output))

    return 0


if __name__ == '__main__':
    sys.exit(main())
