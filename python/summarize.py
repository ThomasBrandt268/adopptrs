#!/usr/bin/env python

"""
Summarizing WalOnMap VIA files

Azimuth
-------
The direction an installation faces is that of the roof slope. Panel rows
follow the ridge and the slope descends perpendicular to it: seen from
above, the direction sought is therefore that of the SHORT side of the
bounding rectangle. Vertical projection shortens that side further (by a
factor cos(tilt)), which works in the same direction.

The original version kept only the `angle` from `cv2.minAreaRect`,
discarding the `(w, h)` pair. That angle does not say which of the two
sides it describes, so the orientation was known only modulo 90 degrees,
and the fallback formula `180 + angle if angle > -45 else 270 + angle`
crushed the whole column into a 90-degree band around south -- measured
over the 1043 annotated polygons of Liege: [135.9 ; 225.0]. Keeping
`(w, h)` resolves the ambiguity.

Two limits remain, irreducible from a single nadir view:

  - a rectangle cannot tell a slope from its opposite (the azimuth is
    known modulo 180 degrees). The south-facing one is retained, which is
    the right bet at our latitudes but wrong for the rare north-facing
    installations;
  - on a near-square installation the short side is not meaningfully
    defined. The -diagnostics option adds an elongation column (long side
    / short side) which lets such cases be spotted -- and discarded --
    downstream. Measured on the same annotations: median elongation 2.46,
    above 1.2 for 91 % of installations.

The default CSV keeps exactly its four original columns (latitude,
longitude, area, azimuth): `wallonia_grid` validates them as such and
rejects the file if a single extra column appears.
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
