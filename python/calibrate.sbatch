#!/usr/bin/env bash
#
# Calibration du seuil de decision de walonmap.py, sur le cluster Alan.
#
# Pendant : evaluate.sbatch mesure un reseau sur un fold californien pour le
# comparer au rapport. Ce script-ci repond a une autre question : ou placer
# le -threshold du reseau qu'on va lancer sur WalOnMap.
#
# Deux differences de fond avec evaluate.sbatch, et ce sont elles qui
# rendent le chiffre transposable a la production :
#
#   - les tuiles de Liege, pas la Californie. Le seuil se calibre sur les
#     images auxquelles il sera applique : meme capteur, meme resolution
#     (0,132 m/px), meme bati. Le modele de production n'a de toute facon
#     aucun fold californien mis de cote (K=0) ;
#   - -negatives. via_liege_city.json compte 322 tuiles sans le moindre
#     panneau sur 661. evaluate.py les ecarte par defaut, donc les faux
#     positifs sur tuile vide ne sont jamais comptes -- or c'est le cas
#     majoritaire en production et c'est la que le seuil se joue.
#
# -min 256 reprend le filtre de walonmap.py, pas le 64 d'evaluate.py : un
# contour que la production jetterait ne doit pas peser dans le seuil.
#
# Usage :
#     cd ~/adopptrs/python
#     sbatch calibrate.sbatch                        # multiunet_020.pth
#     NET=.../multiunet_x2_030.pth sbatch calibrate.sbatch
#     K=5 FOLD=0 sbatch calibrate.sbatch             # sur le seul fold retenu
#     OPENING=1 sbatch calibrate.sbatch              # avec l'ouverture 5x5
#
# K=1 (defaut) prend les 661 tuiles : c'est le bon reglage tant que le
# reseau n'a pas ete affine sur Liege. Des qu'il l'a ete, passer K=5 FOLD=0
# et n'utiliser que le cinquieme laisse de cote a l'entrainement, sinon le
# seuil est lu sur des images deja vues.
#
#SBATCH --job-name=adopptrs-calib
#SBATCH --partition=a5000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# 661 tuiles de 512 px contre 7481 decoupes californiennes en 2 min 45 s
# (job 3896887) : quelques minutes suffisent. Le gros du temps reste la
# morphologie OpenCV sur 21 seuils, donc CPU -- --partition=1080ti fait
# aussi bien si a5000 est prise.
#SBATCH --time=00:20:00
#SBATCH --output=/home/%u/adopptrs/products/logs/%j.log
#SBATCH --export=ALL

set -euo pipefail

#############
# Parametres #
#############

MODEL=${MODEL:-unet}
SCALE=${SCALE:-1}
MIN=${MIN:-256}

# OPENING=0 : chaine de walonmap.py, sans ouverture morphologique. Mesure
# du 2026-08-11 sur le fold 0 (multiunet_liege_030, seuil 1e-4) : sans
# ouverture, TP 108 / FP 14 / FN 30 contre 104 / 12 / 33 avec. L'ouverture
# supprime 2 faux positifs et 4 vraies detections -- elle coute plus
# qu'elle ne rapporte. C'est donc OPENING=0 qui decrit la production, et
# c'est sur cette courbe-la que le seuil doit etre lu.
OPENING=${OPENING:-0}

# K=1 FOLD=0 : i % 1 == 0 pour tout i, donc le jeu entier. C'est le seul
# moyen de tout prendre, K=0 rendant un jeu de validation vide.
K=${K:-1}
FOLD=${FOLD:-0}

EPOCH=${EPOCH:-20}
NAME=${NAME:-multiunet}

MODELS=$HOME/adopptrs/products/models
NET=${NET:-$(printf '%s/%s_%03d.pth' "$MODELS" "$NAME" "$EPOCH")}

DATA=${DATA:-$HOME/adopptrs/products/liege}
VIA=${VIA:-$HOME/adopptrs/resources/walonmap/via_liege_city.json}

EVAL=$HOME/adopptrs/products/eval

# Deux noms distincts : les deux chaines donnent deux courbes, et ecraser
# l'une par l'autre interdirait de les comparer.
if (( OPENING )); then
    OUT=$EVAL/$(basename "$NET" .pth)_liege_ouverture.txt
else
    OUT=$EVAL/$(basename "$NET" .pth)_liege.txt
fi

EXTRA=()
if (( ! OPENING )); then
    EXTRA+=(-no-opening)
fi

################
# Environnement #
################

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate adopptrs

export PYTHONUNBUFFERED=1

cd "$HOME/adopptrs/python"
mkdir -p "$EVAL"

echo "=== job ${SLURM_JOB_ID:-?} sur $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "reseau=$NET echelle=$SCALE min=$MIN ouverture=$OPENING"
echo "tuiles=$DATA (fold $FOLD sur K=$K, tuiles vides comptees)"
echo

test -f "$NET" || { echo "ERREUR : $NET introuvable (entrainement termine ?)"; exit 1; }
test -d "$DATA" || { echo "ERREUR : $DATA introuvable"; exit 1; }
test -f "$VIA" || { echo "ERREUR : $VIA introuvable"; exit 1; }

###############
# Calibration #
###############

python evaluate.py \
    -m "$MODEL" -multitask \
    -n "$NET" \
    -i "$VIA" \
    -p "$DATA" \
    -k "$K" -f "$FOLD" \
    -min "$MIN" \
    -scale "$SCALE" \
    -negatives \
    ${EXTRA[@]+"${EXTRA[@]}"} \
    | tee "$OUT"

echo
echo "=== seuil retenu | $(date) ==="

# -raw : la retouche « precision := 0 quand aucune cible n'est manquee »
# corrige un artefact des seuils bas que -negatives rend deja visible dans
# les faux positifs. L'appliquer ici effacerait des seuils legitimes.
python misc/threshold.py "$OUT" -raw

echo
echo "=== termine | $(date) ==="
