#!/usr/bin/env bash
#
# Evaluation ADOPPTRS sur le cluster Alan (ULiege).
#
# Pendant : train.sbatch avec K=5 met le fold FOLD de cote ; ce script
# l'evalue avec le checkpoint de la derniere epoque, puis compare les
# chiffres obtenus a ceux du rapport (misc/compare_report.py).
#
# Usage :
#     cd ~/adopptrs/python
#     sbatch evaluate.sbatch                    # multiunet_0_020.pth, fold 0
#     FOLD=1 sbatch evaluate.sbatch             # un autre fold
#     EPOCH=10 sbatch evaluate.sbatch           # un checkpoint intermediaire
#     NET=/chemin/vers/reseau.pth sbatch evaluate.sbatch
#
# Suivi :
#     squeue --user $USER
#     tail -f ~/adopptrs/products/logs/<jobid>.log
#
#SBATCH --job-name=adopptrs-eval
#SBATCH --partition=a5000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# L'evaluation est bien plus courte qu'un entrainement : un seul passage
# sur un cinquieme du jeu, sans retropropagation. Compter moins d'une heure.
# Elle est d'ailleurs surtout limitee par le CPU (decodage TIFF, morphologie
# OpenCV sur 21 seuils), d'ou l'interet de la lancer sur une partition libre
# plutot que d'attendre le meilleur GPU : sbatch --partition=1080ti ...
#SBATCH --time=02:00:00
#SBATCH --output=/home/%u/adopptrs/products/logs/%j.log
#SBATCH --export=ALL

set -euo pipefail

#############
# Parametres #
#############

MODEL=${MODEL:-unet}
K=${K:-5}
FOLD=${FOLD:-0}
EPOCH=${EPOCH:-20}
MIN=${MIN:-64}

# Doit correspondre au NAME de train.sbatch, suffixe par le fold.
NAME=${NAME:-multiunet_$FOLD}

MODELS=$HOME/adopptrs/products/models
NET=${NET:-$(printf '%s/%s_%03d.pth' "$MODELS" "$NAME" "$EPOCH")}

DATA=${DATA:-/scratch/users/$USER/california}
VIA=${VIA:-$HOME/adopptrs/products/json/california.json}

EVAL=$HOME/adopptrs/products/eval
OUT=$EVAL/$(basename "$NET" .pth).txt

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
echo "reseau=$NET fold=$FOLD sur K=$K"
echo "donnees=$DATA"
echo

test -f "$NET" || { echo "ERREUR : $NET introuvable (entrainement termine ?)"; exit 1; }
test -d "$DATA" || { echo "ERREUR : $DATA introuvable"; exit 1; }
test -f "$VIA" || { echo "ERREUR : $VIA introuvable"; exit 1; }

##############
# Evaluation #
##############

# tee : la sortie reste lisible dans le log SLURM et laisse un fichier
# stable a comparer. evaluate.py -o l'ecrirait dans le fichier seulement.
python evaluate.py \
    -m "$MODEL" -multitask \
    -n "$NET" \
    -i "$VIA" \
    -p "$DATA" \
    -k "$K" -f "$FOLD" \
    -min "$MIN" \
    | tee "$OUT"

echo
echo "=== comparaison au rapport | $(date) ==="

# Sort en 1 si l'ecart d'AP depasse la tolerance : le job apparait alors
# en FAILED dans sacct, ce qui est bien le signal voulu.
python misc/compare_report.py "$OUT" -m "$(basename "$NAME" | cut -d_ -f1)" -f "$FOLD"

echo
echo "=== termine | $(date) ==="
