"""
Remplacement WMS de la classe WMTS d'ADOPPTRS.

Contexte
--------
Le geoportail wallon a retire l'extension WMTS de ses services ORTHO
(verifie sur les millesimes 2018 a 2024 : supportedExtensions = WMSServer
uniquement, tileInfo absent, GetCapabilities WMTS -> HTTP 400).

wmts.py lisait toute sa geometrie depuis le document GetCapabilities au
moment de la construction de l'objet. Comme cet endpoint est mort, la
classe ne peut plus etre instanciee du tout -- pas seulement get_tile().

Ce module fige donc les constantes du TileMatrix "15" du TileMatrixSet
"default028mm" (relevees sur un instantane archive des capabilities) et
reimplemente la recuperation d'image via GetMap WMS, en conservant
exactement la meme interface publique que WMTS.

Constantes du niveau 15
-----------------------
    CRS              EPSG:31370 (Lambert 72 belge, unites en metres)
    ScaleDenominator 472.4711830375448
    TopLeftCorner    -35872700.0 ; 41422700.0
    TileWidth        512
    TileHeight       512
    MatrixWidth      533986
    MatrixHeight     611260

    pixel_span = 472.4711830375448 * 0.28e-3 * 1.0 = 0.13229193125051254 m/px
    tile_span  = 512 * 0.13229193125051254         = 67.73346880026242 m

Usage
-----
Dans walonmap.py et misc/download.py, remplacer

    from wmts import WMTS
    _WALONMAP = WMTS(wmts='...', layer='...', tms='default028mm', tm='15')

par

    from wms import WMS
    _WALONMAP = WMS()

Le reste du code (tile_to_xy, xy_to_tile, wgs_to_tile, get_tile,
tile_width, tile_height, ...) est inchange.
"""

import io

import pyproj as pp
import requests


_WGS84 = 'EPSG:4326'
_PIXEL_SIZE = 0.28e-3  # hypothese OGC : 0.28 mm par pixel de rendu

# --- Constantes figees du TileMatrix "15" / default028mm --------------------

_CRS = 'EPSG:31370'
_SCALE_DENOMINATOR = 472.4711830375448
_TOP_LEFT_X = -35872700.0
_TOP_LEFT_Y = 41422700.0
_TILE_WIDTH = 512
_TILE_HEIGHT = 512
_MATRIX_WIDTH = 533986
_MATRIX_HEIGHT = 611260

# Le geoportail expose un service par millesime, tous sous le meme motif.
# La liste complete se lit sur
# https://geoservices.wallonie.be/arcgis/rest/services/IMAGERIE?f=pjson
# et va de ORTHO_1971 a ORTHO_2024 (relevee le 2026-08-11).
_SERVICE_TEMPLATE = (
    'https://geoservices.wallonie.be/arcgis/services'
    '/IMAGERIE/{}/MapServer/WMSServer'
)
_DEFAULT_VINTAGE = 'ORTHO_2018'
_DEFAULT_WMS = _SERVICE_TEMPLATE.format(_DEFAULT_VINTAGE)
_DEFAULT_LAYER = '0'
_DEFAULT_FORMAT = 'image/jpeg'


def service_url(vintage):
    """URL WMS d'un millesime, depuis son nom de service (ex. 'ORTHO_2024').

    Changer de millesime ne change que cette URL : la geometrie des tuiles
    est calculee ici, pas servie par le geoportail (cf. get_tile, qui
    demande un GetMap sur une bbox au sol). Deux millesimes demandes au
    meme (row, col) rendent donc exactement le meme rectangle de terrain,
    au meme pas de pixel -- c'est ce qui rend leur comparaison directe.

    Attention : identique ne veut pas dire equivalent. Un millesime dont
    la resolution native est plus grossiere que nos 0,132 m/px sera
    reechantillonne par le serveur, et l'image rendue sera molle.
    """
    return _SERVICE_TEMPLATE.format(vintage)


class _Layer:
    """Stub minimal : certains scripts accedent a wm.layer.id."""

    def __init__(self, identifier):
        self.id = identifier


class WMS:
    """
    Interface compatible WMTS, servie par des requetes GetMap WMS.

    Les indices (row, col) restent ceux du quadrillage WMTS d'origine,
    ce qui garantit que les 661 tuiles annotees de via_liege_city.json
    restent alignees au pixel pres avec les images recuperees ici.
    """

    def __init__(
        self,
        wms=None,
        layer=_DEFAULT_LAYER,
        fmt=_DEFAULT_FORMAT,
        timeout=90,
        session=None,
        vintage=None,
    ):
        # WMS() reste ORTHO_2018 ; WMS(vintage='ORTHO_2024') suffit a
        # changer de millesime. Les deux ensemble seraient ambigus, donc
        # refuses plutot qu'arbitres en silence.
        if wms is None:
            self.vintage = vintage or _DEFAULT_VINTAGE
            wms = service_url(self.vintage)
        elif vintage is not None:
            raise ValueError('wms et vintage sont exclusifs')
        else:
            self.vintage = None
        self.url = wms
        self.layer = _Layer(layer)
        self.fmt = fmt
        self.timeout = timeout
        self.session = session if session is not None else requests.Session()

        # CRS
        self.crs = pp.CRS.from_user_input(_CRS)
        self.epsg = ':'.join(self.crs.to_authority())

        self._from_wgs = pp.Transformer.from_crs(_WGS84, self.crs)
        self._to_wgs = pp.Transformer.from_crs(self.crs, _WGS84)

        self.unit = self.crs.coordinate_system.axis_list[0].unit_name
        self.ucf = self.crs.coordinate_system.axis_list[0].unit_conversion_factor

        # Tile span(s)
        self.pixel_size = _PIXEL_SIZE
        self.scale = _SCALE_DENOMINATOR
        self.pixel_span = self.scale * self.pixel_size * self.ucf

        self.tile_width = _TILE_WIDTH
        self.tile_height = _TILE_HEIGHT

        self.tile_span_x = self.tile_width * self.pixel_span
        self.tile_span_y = self.tile_height * self.pixel_span

        # Domain
        self.matrix_width = _MATRIX_WIDTH
        self.matrix_height = _MATRIX_HEIGHT

        self.x_min, self.y_max = _TOP_LEFT_X, _TOP_LEFT_Y
        self.x_max = self.x_min + self.matrix_width * self.tile_span_x
        self.y_min = self.y_max - self.matrix_height * self.tile_span_y

    # --- Recuperation d'image ---------------------------------------------

    def tile_bbox(self, row, col):
        """BBox (minx, miny, maxx, maxy) de la tuile, dans self.crs."""
        left, top = self.tile_to_xy(row, col)
        return (left, top - self.tile_span_y, left + self.tile_span_x, top)

    def get_tile(self, row, col):
        """
        Renvoie un objet lisible par PIL.Image.open(), comme le faisait
        owslib.wmts.gettile().

        WMS 1.1.1 est utilise volontairement : l'ordre des axes de la
        bbox y est toujours (minx, miny, maxx, maxy), alors qu'en 1.3.0
        il depend de la definition du CRS -- source classique d'images
        transposees, silencieuse et penible a diagnostiquer.
        """
        minx, miny, maxx, maxy = self.tile_bbox(row, col)

        params = {
            'service': 'WMS',
            'version': '1.1.1',
            'request': 'GetMap',
            'layers': self.layer.id,
            'styles': '',
            'srs': self.epsg,
            'bbox': '%.6f,%.6f,%.6f,%.6f' % (minx, miny, maxx, maxy),
            'width': str(self.tile_width),
            'height': str(self.tile_height),
            'format': self.fmt,
        }

        r = self.session.get(self.url, params=params, timeout=self.timeout)
        r.raise_for_status()

        # ArcGIS renvoie parfois une ServiceException XML avec un code 200.
        # Sans ce garde-fou, l'erreur ne remonte qu'a l'ouverture PIL, et
        # le "except Exception: continue" de walonmap.py l'avale en silence.
        ctype = r.headers.get('Content-Type', '')
        if not ctype.startswith('image/'):
            raise RuntimeError(
                'Reponse WMS non-image (%s) pour la tuile %d/%d : %s'
                % (ctype, row, col, r.text[:300])
            )

        return io.BytesIO(r.content)

    # --- Conversions (identiques a wmts.py) --------------------------------

    def tile_to_xy(self, row, col):
        x = self.x_min + col * self.tile_span_x
        y = self.y_max - row * self.tile_span_y
        return x, y

    def xy_to_tile(self, x, y, integer=True):
        col = (x - self.x_min) / self.tile_span_x
        row = (self.y_max - y) / self.tile_span_y
        if integer:
            row, col = int(row), int(col)
        return row, col

    def xy_to_wgs(self, x, y):
        return self._to_wgs.transform(x, y)

    def wgs_to_xy(self, lat, lon):
        return self._from_wgs.transform(lat, lon)

    def tile_to_wgs(self, row, col):
        return self.xy_to_wgs(*self.tile_to_xy(row, col))

    def wgs_to_tile(self, lat, lon):
        return self.xy_to_tile(*self.wgs_to_xy(lat, lon))


# --- Auto-test -------------------------------------------------------------

if __name__ == '__main__':
    from PIL import Image

    wm = WMS()

    print('CRS         :', wm.epsg, '| unite :', wm.unit)
    print('pixel_span  : %.11f m/px' % wm.pixel_span)
    print('tile_span   : %.8f m' % wm.tile_span_x)
    print('tile size   : %d x %d px' % (wm.tile_width, wm.tile_height))
    print()

    # Tuile issue de via_liege_city.json
    row, col = 609288, 533063

    x, y = wm.tile_to_xy(row, col)
    print('tuile %d/%d -> Lambert 72 : x=%.1f  y=%.1f' % (row, col, x, y))

    lat, lon = wm.tile_to_wgs(row, col)
    print('              -> WGS84     : %.6f, %.6f' % (lat, lon))

    print('bbox        :', ['%.2f' % v for v in wm.tile_bbox(row, col)])
    print()

    # Aller-retour depuis le CENTRE de la tuile : un coin exact tombe pile
    # sur la frontiere entre deux tuiles, ou l'erreur d'arrondi de la double
    # projection suffit a faire basculer le int() d'une unite.
    cx, cy = x + wm.tile_span_x / 2, y - wm.tile_span_y / 2
    clat, clon = wm.xy_to_wgs(cx, cy)
    r2, c2 = wm.wgs_to_tile(clat, clon)
    print('aller-retour: %d/%d %s' % (r2, c2, 'OK' if (r2, c2) == (row, col) else 'ECHEC'))

    print()
    print('Telechargement de la tuile...')
    img = Image.open(wm.get_tile(row, col)).convert('RGB')
    img.save('tuile_%d_%d.jpg' % (row, col))
    print('IMAGE :', img.size, img.mode, '-> tuile_%d_%d.jpg' % (row, col))

    if img.size != (wm.tile_width, wm.tile_height):
        print('ATTENTION : taille inattendue, verifier les parametres WMS.')