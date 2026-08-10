#!/usr/bin/env python

"""
Summarizing WalOnMap VIA files

Azimut
------
La direction visee par une installation est celle de la pente du toit.
Les rangees de panneaux suivent le faitage, la pente descend
perpendiculairement : vue du ciel, la direction cherchee est donc celle
du PETIT cote du rectangle englobant. La projection verticale raccourcit
encore ce cote (d'un facteur cos(inclinaison)), ce qui va dans le meme
sens.

La version d'origine ne gardait de `cv2.minAreaRect` que son `angle`, en
jetant le couple `(w, h)`. Or cet angle ne dit pas lequel des deux cotes
il decrit : l'orientation n'etait donc connue que modulo 90 degres, et la
formule de repli `180 + angle if angle > -45 else 270 + angle` tassait
toute la colonne dans une bande de 90 degres autour du sud -- mesure sur
les 1043 polygones annotes de Liege : [135,9 ; 225,0]. Conserver `(w, h)`
leve l'ambiguite.

Restent deux limites, irreductibles a partir d'une seule vue nadir :

  - un rectangle ne distingue pas une pente de son opposee (l'azimut est
    connu modulo 180 degres). On retient celle qui regarde le sud, ce qui
    est le bon pari sous nos latitudes mais se trompe sur les rares
    installations orientees au nord ;
  - sur une installation quasi carree, le petit cote n'est pas defini de
    facon significative. L'option -diagnostics ajoute une colonne
    d'elongation (grand cote / petit cote) qui permet de reperer -- et
    d'ecarter -- ces cas en aval. Mesure sur les memes annotations :
    elongation mediane 2,46, superieure a 1,2 pour 91 % des installations.

Le CSV par defaut garde exactement ses quatre colonnes d'origine
(latitude, longitude, area, azimuth) : `wallonia_grid` les valide a
l'identique et refuse le fichier a la moindre colonne supplementaire.
"""

###########
# Imports #
###########

import numpy as np


#############
# Constants #
#############

_TOL = 1e-3


#############
# Functions #
#############

def parse(imagename):
	return tuple(
		map(
			int,
			imagename.split('.')[0].split('_')[-2:]
		)
	)


def orientation(box):
	'''Azimut de la pente d'une installation et fiabilite de l'estimation.

	`box` donne les quatre coins du rectangle englobant, en (est, nord).
	Renvoie l'azimut en degres depuis le nord (0 = nord, 90 = est) et
	l'elongation du rectangle, d'autant plus proche de 1 que l'azimut est
	arbitraire.

	Le calcul part des coins plutot que de l'angle de `minAreaRect` : la
	convention de cet angle a change entre versions d'OpenCV, et les
	coordonnees sont ici orientees vers le nord alors que la fonction les
	suppose orientees vers le bas de l'image, ce qui en inverse le signe.
	Les coins, eux, ne mentent pas.
	'''
	sides = box[[1, 2]] - box[[0, 1]]  # deux cotes adjacents
	lengths = np.hypot(sides[:, 0], sides[:, 1])

	short = sides[lengths.argmin()]

	# atan2(est, nord) : releve au compas, 0 = nord, 90 = est
	azimuth = np.degrees(np.arctan2(short[0], short[1])) % 360

	# Modulo 180 degres pres, on retient la pente qui regarde le sud.
	# Intervalle semi-ouvert : une installation exactement plein est ou
	# plein ouest a ses deux candidats dans [90, 270], et le choix se
	# ferait alors au gre de l'ordre des coins. On tranche pour l'est.
	if not 90 <= azimuth < 270:
		azimuth = (azimuth + 180) % 360

	longest, shortest = lengths.max(), lengths.min()
	elongation = longest / shortest if shortest > 0 else np.inf

	return azimuth, elongation


########
# Main #
########

if __name__ == '__main__':
	# Imports
	import argparse
	import cv2
	import csv
	import numpy as np
	import os

	import via as VIA
	from evaluate import surface
	from walonmap import _WALONMAP as wm

	# Arguments
	parser = argparse.ArgumentParser(description='Summarize a WalOnMap VIA file')
	parser.add_argument('-i', '--input', default='../products/json/walonmap.json', help='input VIA file')
	parser.add_argument('-o', '--output', default='../products/csv/summary.csv', help='output csv file')
	parser.add_argument('-diagnostics', default=False, action='store_true', help='add the elongation column')
	args = parser.parse_args()

	# VIA
	via = VIA.load(args.input)

	# Panels
	if os.path.dirname(args.output):
		os.makedirs(os.path.dirname(args.output), exist_ok=True)

	header = ['latitude', 'longitude', 'area', 'azimuth']

	if args.diagnostics:
		header.append('elongation')

	with open(args.output, 'w', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(header)

		for imagename, polygons in via.items():
			row, col = parse(imagename)

			for polygon in polygons:
				panel = np.array([
					wm.tile_to_xy(row + y / wm.tile_height, col + x / wm.tile_width)
					for x, y in polygon
				])
				x, y = panel.mean(axis=0)

				# Recentre avant de passer a OpenCV, qui calcule en
				# float32 : en Lambert 72 absolu les coordonnees montent
				# a 2,4e8 mm, magnitude ou le pas de float32 est de
				# 16 mm. Toute la geometrie etait donc quantifiee a
				# 16 mm avant d'etre mesuree (0,15 deg d'erreur mediane
				# sur l'azimut, 0,14 % sur la surface). Une fois
				# recentre, le pas retombe sous le micrometre.
				panel = ((panel - (x, y)) / _TOL).astype(int)

				area = surface(panel) * (_TOL ** 2)

				lat, lon = wm.xy_to_wgs(x, y)
				azimuth, elongation = orientation(cv2.boxPoints(cv2.minAreaRect(panel)))

				# Pas `row` : c'est deja la ligne de la tuile courante
				record = [
					round(lat, 6),
					round(lon, 6),
					round(area, 2),
					round(azimuth, 2)
				]

				if args.diagnostics:
					record.append(round(elongation, 3))

				writer.writerow(record)
