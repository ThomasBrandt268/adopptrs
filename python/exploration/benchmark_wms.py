"""
Benchmark des requetes WMS WalOnMap.

Mesure la latence reelle de recuperation d'une tuile, detecte une
eventuelle degradation du service quand on enchaine les requetes, et
extrapole le temps necessaire pour couvrir une zone donnee.

Le geoportail wallon est un service public mutualise : ce script reste
volontairement modeste (quelques dizaines de requetes, avec une pause
entre chacune). Ne pas augmenter N_SAMPLES sans raison.

Usage
-----
    conda activate adopptrs
    cd python
    python exploration/benchmark_wms.py
"""

import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wms import WMS


# --- Parametres ------------------------------------------------------------

N_SAMPLES = 40          # nombre de tuiles a chronometrer
PAUSE = 0.2             # pause entre requetes (s), par courtoisie
SEED = 0

# Zone d'echantillonnage : autour de Liege, ou la couverture est certaine.
# On tire au hasard dans une fenetre large pour eviter que le serveur ne
# serve des tuiles deja en cache (ce qui fausserait la mesure).
ROW_CENTER, COL_CENTER = 609300, 533100
SPREAD = 300            # +/- 300 tuiles ~ +/- 20 km

# Surfaces de reference (km2)
SURFACES = [
    ('Province de Liege', 3862),
    ('Wallonie', 16901),
]


def human_time(seconds):
    """Formate une duree en unites lisibles."""
    if seconds < 60:
        return '%.1f s' % seconds
    if seconds < 3600:
        return '%.1f min' % (seconds / 60)
    if seconds < 86400:
        return '%.1f h' % (seconds / 3600)
    return '%.1f jours' % (seconds / 86400)


def main():
    wm = WMS()

    tile_area = wm.tile_span_x * wm.tile_span_y  # m2

    print('Tuile : %d x %d px | %.2f x %.2f m | %.0f m2'
          % (wm.tile_width, wm.tile_height,
             wm.tile_span_x, wm.tile_span_y, tile_area))
    print('Resolution : %.4f m/px' % wm.pixel_span)
    print()

    # --- Tirage des tuiles -------------------------------------------------

    random.seed(SEED)
    tiles = set()
    while len(tiles) < N_SAMPLES:
        tiles.add((
            ROW_CENTER + random.randint(-SPREAD, SPREAD),
            COL_CENTER + random.randint(-SPREAD, SPREAD),
        ))
    tiles = sorted(tiles)

    print('Chronometrage de %d tuiles distinctes...' % len(tiles))
    print()

    # --- Mesures -----------------------------------------------------------

    timings = []
    sizes = []
    failures = 0

    for i, (row, col) in enumerate(tiles, 1):
        t0 = time.perf_counter()
        try:
            data = wm.get_tile(row, col).getvalue()
            dt = time.perf_counter() - t0
            timings.append(dt)
            sizes.append(len(data))
            flag = ''
        except Exception as e:
            dt = time.perf_counter() - t0
            failures += 1
            flag = '  ECHEC : %s' % str(e)[:60]

        print('  %2d/%d  %6.0f ms%s' % (i, len(tiles), dt * 1000, flag))
        time.sleep(PAUSE)

    print()

    if not timings:
        sys.exit('Aucune requete reussie -- verifier la connexion ou le service.')

    # --- Statistiques ------------------------------------------------------

    timings_sorted = sorted(timings)
    p95 = timings_sorted[min(int(0.95 * len(timings_sorted)), len(timings_sorted) - 1)]

    print('=== LATENCE ===')
    print('  reussites  : %d / %d' % (len(timings), len(tiles)))
    print('  mediane    : %.0f ms' % (statistics.median(timings) * 1000))
    print('  moyenne    : %.0f ms' % (statistics.mean(timings) * 1000))
    print('  min / max  : %.0f / %.0f ms'
          % (timings_sorted[0] * 1000, timings_sorted[-1] * 1000))
    print('  p95        : %.0f ms' % (p95 * 1000))
    if len(timings) > 1:
        print('  ecart-type : %.0f ms' % (statistics.stdev(timings) * 1000))
    print()
    print('  taille moy : %.0f Ko' % (statistics.mean(sizes) / 1024))
    print()

    # --- Degradation -------------------------------------------------------

    half = len(timings) // 2
    if half >= 3:
        first = statistics.median(timings[:half])
        second = statistics.median(timings[half:])
        ratio = second / first if first > 0 else float('nan')

        print('=== DEGRADATION ===')
        print('  1re moitie : %.0f ms (mediane)' % (first * 1000))
        print('  2e moitie  : %.0f ms (mediane)' % (second * 1000))
        print('  ratio      : %.2f' % ratio)
        if ratio > 1.5:
            print('  -> le service ralentit nettement : throttling probable,')
            print('     augmenter PAUSE et eviter toute parallelisation.')
        elif ratio > 1.15:
            print('  -> leger ralentissement, a surveiller sur un run long.')
        else:
            print('  -> pas de degradation notable sur cet echantillon.')
        print()

    # --- Extrapolation -----------------------------------------------------

    median = statistics.median(timings)

    print('=== EXTRAPOLATION (requetes en serie, sans inference) ===')
    print()
    print('%-22s %14s %14s' % ('Zone', 'tuiles', 'duree'))
    print('-' * 52)

    for name, km2 in SURFACES:
        n_tiles = km2 * 1e6 / tile_area
        print('%-22s %14s %14s'
              % (name, '{:,.0f}'.format(n_tiles).replace(',', ' '),
                 human_time(n_tiles * median)))

    print()
    print('Avec filtrage spatial (hypotheses de reduction) :')
    print()
    print('%-22s %10s %14s %14s' % ('Zone', 'reduction', 'tuiles', 'duree'))
    print('-' * 62)

    for name, km2 in SURFACES:
        n_tiles = km2 * 1e6 / tile_area
        for factor in (5, 10, 20):
            print('%-22s %9dx %14s %14s'
                  % (name if factor == 5 else '', factor,
                     '{:,.0f}'.format(n_tiles / factor).replace(',', ' '),
                     human_time(n_tiles / factor * median)))

    print()
    print('Note : ces durees ne comptent que le reseau. L\'inference du')
    print('modele s\'y ajoute (ordre de la seconde par tuile sur CPU, bien')
    print('moins sur GPU), sauf si les deux sont menes en parallele.')


if __name__ == '__main__':
    main()