import requests

BASE = ('https://geoservices.wallonie.be/arcgis/rest/services'
        '/HABITAT/TISSU_URBANISE/MapServer')

# Compte les polygones de chaque couche
for lid, name in [(0, 'continu'), (1, 'semi-continu'), (2, 'ouvert'), (3, 'NA')]:
    r = requests.get(BASE + '/%d/query' % lid,
                     params={'where': '1=1', 'returnCountOnly': 'true', 'f': 'json'},
                     timeout=120)
    print('%-14s %s' % (name, r.json().get('count')))

# Un echantillon pour voir la structure reelle
r = requests.get(BASE + '/0/query',
                 params={'where': '1=1', 'resultRecordCount': 1,
                         'outFields': '*', 'returnGeometry': 'false', 'f': 'json'},
                 timeout=120)
print()
print(r.text[:600])