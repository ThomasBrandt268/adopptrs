#!/usr/bin/env python

"""
Smoke test du pipeline ADOPPTRS.

Verifie que chaque brique tourne sur l'environnement courant : imports,
passes avant/arriere des quatre modeles, chargement du dataset avec
augmentation, geometrie WMS, et post-traitement (contours, azimut).

C'est le test a lancer en premier apres avoir cree un environnement, en
particulier sur un cluster GPU : il affiche la version de chaque paquet
ainsi que les architectures CUDA compilees dans la roue torch installee,
ce qui est la cause la plus frequente d'un "no kernel image is available
for execution on the device" au premier .to('cuda').

Usage
-----
    conda activate adopptrs
    cd python
    python tests/smoke.py
    python tests/smoke.py --net     # ajoute une vraie requete WMS

Le test reseau est optionnel et ne fait qu'une seule requete : le
geoportail wallon est un service public mutualise.
"""

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_results = []


def check(name):
    """Enregistre le resultat d'une verification sans interrompre les suivantes."""
    def decorator(fn):
        try:
            _results.append(('OK  ', name, fn() or ''))
        except Exception as e:
            _results.append(('FAIL', name, '{}: {}'.format(type(e).__name__, e)))
            if os.environ.get('SMOKE_TRACE'):
                traceback.print_exc()
        return fn
    return decorator


#################
# Verifications #
#################

def run(net=False, data=True):
    @check('versions')
    def _():
        import cv2, numpy, PIL, pyproj, scipy, torch, torchvision
        return ('python={} numpy={} torch={} torchvision={}\n'
                '       opencv={} scipy={} pillow={} pyproj={}').format(
            '.'.join(map(str, sys.version_info[:3])), numpy.__version__,
            torch.__version__, torchvision.__version__,
            cv2.__version__, scipy.__version__, PIL.__version__, pyproj.__version__)

    @check('CUDA')
    def _():
        import torch
        if not torch.cuda.is_available():
            return 'indisponible (CPU) -- normal sur un poste de travail'
        # Une roue torch ne contient que certaines architectures : si celle du
        # GPU n'y est pas, l'erreur ne surgit qu'au premier calcul.
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        archs = torch.cuda.get_arch_list()
        got = 'sm_{}{}'.format(major, minor)
        status = 'OK' if got in archs else 'ABSENTE DE LA ROUE -> reinstaller torch'
        torch.zeros(8, device='cuda').sum().item()  # force l'init d'un kernel
        return '{} ({}) | cuda={} | compilee pour {} -> {}'.format(
            name, got, torch.version.cuda, ' '.join(archs), status)

    for name in ['via', 'criterions', 'dataset', 'models', 'evaluate',
                 'summarize', 'plots', 'wms', 'walonmap']:
        def make(module):
            @check('import ' + module)
            def _():
                __import__(module)
            return _
        make(name)

    # --- Modeles ------------------------------------------------------------

    @check('UNet : passe avant')
    def _():
        import torch
        from models import UNet
        y = UNet(3, 1)(torch.rand(1, 3, 256, 256))
        assert y.shape == (1, 1, 256, 256), y.shape
        return str(tuple(y.shape))

    @check('SegNet : passe avant')
    def _():
        import torch
        from models import SegNet
        y = SegNet(3, 1)(torch.rand(1, 3, 128, 128))
        assert y.shape == (1, 1, 128, 128), y.shape
        return str(tuple(y.shape))

    for model_name in ['MultiTaskUNet', 'MultiTaskSegNet']:
        def make(model_name):
            @check(model_name + ' : passe avant + arriere')
            def _():
                import torch
                import models
                from criterions import MultiTaskLoss
                model = getattr(models, model_name)(3, 1, R=5)
                model.train()
                outputs = model(torch.rand(2, 3, 128, 128))
                loss = MultiTaskLoss(smooth=1., R=5)(
                    outputs, torch.rand(2, 1, 128, 128).round())
                loss.backward()
                return 'seg={} dist={} loss={:.4f}'.format(
                    tuple(outputs[0].shape), tuple(outputs[1].shape), loss.item())
            return _
        make(model_name)

    # --- Donnees ------------------------------------------------------------

    @check('to_mask / to_contours / surface')
    def _():
        import numpy as np
        from dataset import to_mask, to_contours
        from evaluate import bounding, surface
        mask = to_mask((64, 64), [[(10, 10), (10, 40), (40, 40), (40, 10)]])
        contours = to_contours(np.array(mask))
        return 'contours={} surface={:.1f} bbox={}'.format(
            len(contours), surface(contours[0]), tuple(map(int, bounding(contours[0]))))

    if data:
        @check('VIADataset + augmentation + DataLoader')
        def _():
            import via as VIA
            from torch.utils.data import DataLoader
            from dataset import (ColorJitter, RandomFilter, RandomTranspose,
                                 Scale, ToTensor, VIADataset)

            path = os.path.join(_REPO, 'products', 'liege')
            source = os.path.join(_REPO, 'resources', 'walonmap', 'via_liege_city.json')

            via = VIA.load(source)
            via = {k: via[k] for k in list(via)[:4]
                   if os.path.exists(os.path.join(path, k))}

            if not via:
                raise RuntimeError(
                    'aucune tuile dans products/liege -- lancer misc/download.py '
                    "ou relancer avec --no-data")

            dataset = VIADataset(via, path, shuffle=True, alt=1)
            dataset = ToTensor(RandomTranspose(RandomFilter(ColorJitter(Scale(dataset, 2)))))

            total = 0
            for inputs, _targets in DataLoader(dataset, batch_size=2):
                total += inputs.shape[0]
                if total >= 4:
                    break

            return '{} echantillons, batch={}'.format(total, tuple(inputs.shape))

    @check('Contour (pointPolygonTest)')
    def _():
        from walonmap import Contour
        return '{} tuiles dans le carre 5x5'.format(len(Contour([(0, 0), (0, 5), (5, 5), (5, 0)])))

    @check('summarize : parse + azimut')
    def _():
        import cv2
        import numpy as np
        from summarize import parse
        # Rectangle tourne de 30 deg : la convention d'angle de minAreaRect a
        # change entre OpenCV 4.2 et 4.5, et elle pilote l'azimut du CSV final.
        angle_rad = np.deg2rad(30)
        rotation = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                             [np.sin(angle_rad), np.cos(angle_rad)]])
        corners = np.array([[-10., -5.], [-10., 5.], [10., 5.], [10., -5.]])
        _, _, angle = cv2.minAreaRect((corners @ rotation.T + 50).astype(np.float32))
        azimuth = 180 + angle if angle > -45 else 270 + angle
        assert abs(azimuth - 210) < 1e-3, 'azimut={} (attendu 210)'.format(azimuth)
        return 'parse={} azimut={:.2f}'.format(parse('609288_533063.jpg'), azimuth)

    # --- WMS ----------------------------------------------------------------

    @check('WMS : geometrie (hors ligne)')
    def _():
        from wms import WMS
        wm = WMS()
        row, col = 609288, 533063
        x, y = wm.tile_to_xy(row, col)
        # Aller-retour depuis le centre : un coin tombe pile sur la frontiere
        # entre deux tuiles, ou l'arrondi peut faire basculer le int().
        centre = wm.xy_to_wgs(x + wm.tile_span_x / 2, y - wm.tile_span_y / 2)
        assert wm.wgs_to_tile(*centre) == (row, col), 'aller-retour incoherent'
        return 'wgs={:.5f},{:.5f} tuile={:.3f} m'.format(
            *wm.tile_to_wgs(row, col), wm.tile_span_x)

    if net:
        @check('WMS : get_tile (reseau)')
        def _():
            from PIL import Image
            from wms import WMS
            wm = WMS()
            image = Image.open(wm.get_tile(609288, 533063)).convert('RGB')
            assert image.size == (wm.tile_width, wm.tile_height), image.size
            return '{} {}'.format(image.size, image.mode)


########
# Main #
########

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Smoke test du pipeline ADOPPTRS')
    parser.add_argument('--net', action='store_true', help='ajoute une requete WMS reelle')
    parser.add_argument('--no-data', action='store_true', help='saute les tests lisant products/liege')
    args = parser.parse_args()

    run(net=args.net, data=not args.no_data)

    print()
    width = max(len(name) for _, name, _ in _results)
    for status, name, info in _results:
        print('[{}] {:<{w}}  {}'.format(status, name, info, w=width))

    failed = sum(1 for status, _, _ in _results if status == 'FAIL')
    print('\n{}/{} OK'.format(len(_results) - failed, len(_results)))

    if failed and not os.environ.get('SMOKE_TRACE'):
        print('Relancer avec SMOKE_TRACE=1 pour les traces completes.')

    sys.exit(1 if failed else 0)
