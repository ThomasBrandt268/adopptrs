#!/usr/bin/env bash
#
# Entrainement ADOPPTRS sur le cluster Alan (ULiege).
#
# Preparation, une seule fois :
#     mkdir -p ~/adopptrs/products/logs
#
# Usage :
#     cd ~/adopptrs/python
#     EPOCHS=1 sbatch train.sbatch          # essai court : cout d'une epoque
#     K=5 FOLD=0 sbatch train.sbatch        # validation croisee, evaluable
#     SCALE=2 sbatch train.sbatch           # modele de production (rapport)
#
# Suivi :
#     squeue --user $USER
#     tail -f ~/adopptrs/products/logs/<jobid>.log
#     scancel <jobid>
#
# Le modele entraine est Multi-Task U-Net, celui retenu par le rapport. Pour
# SegNet, soumettre avec MODEL=segnet.
#
#SBATCH --job-name=adopptrs
#SBATCH --partition=a5000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# 16 Go = ~2,4x le pic mesure (6,6 Go, job 3896764, jeu californien complet).
# Demander moins fait aussi passer le job plus vite dans la file.
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/adopptrs/products/logs/%j.log
#SBATCH --export=ALL

set -euo pipefail

#############
# Parametres #
#############

# Surchargeables a la soumission : EPOCHS=1 sbatch train.sbatch
MODEL=${MODEL:-unet}
EPOCHS=${EPOCHS:-20}
CHUNK=${CHUNK:-5}
SCALE=${SCALE:-1}
BATCH=${BATCH:-5}

# K=0  : entrainement sur la totalite du jeu (modele de production, celui
#        applique a WalOnMap). evaluate.py n'a alors aucun jeu de validation.
# K=5  : validation croisee, le fold FOLD est mis de cote pour evaluate.py.
#        C'est le protocole du rapport, a utiliser pour mesurer les performances.
K=${K:-0}
FOLD=${FOLD:-0}

# Nom du reseau : suffixe par le fold en validation croisee, sinon les
# differents folds ecraseraient mutuellement leurs checkpoints.
if (( K > 0 )); then
    NAME=${NAME:-multiunet_$FOLD}
else
    NAME=${NAME:-multiunet}
fi

# Les images californiennes vivent sur /scratch : 601 fichiers relus a chaque
# epoque, c'est exactement l'acces aleatoire que /home ne doit pas subir.
DATA=${DATA:-/scratch/users/$USER/california}
VIA=${VIA:-$HOME/adopptrs/products/json/california.json}

MODELS=$HOME/adopptrs/products/models
CSV=$HOME/adopptrs/products/csv/$NAME.csv

################
# Environnement #
################

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate adopptrs

# Sans cela, Python bufferise sa sortie des qu'elle est redirigee vers un
# fichier : le log reste muet pendant des minutes et donne l'impression que
# le job est fige, alors qu'il travaille.
export PYTHONUNBUFFERED=1

cd "$HOME/adopptrs/python"
mkdir -p "$MODELS" "$(dirname "$CSV")"

echo "=== job ${SLURM_JOB_ID:-?} sur $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "modele=$MODEL nom=$NAME epoques=$EPOCHS scale=$SCALE batch=$BATCH"
echo "donnees=$DATA"
echo

test -d "$DATA" || { echo "ERREUR : $DATA introuvable (transfert scp fait ?)"; exit 1; }
test -f "$VIA" || { echo "ERREUR : $VIA introuvable"; exit 1; }

#####################
# Reprise et tranches #
#####################

# train.py n'ecrit un .pth qu'a la derniere epoque de la plage demandee. On
# l'appelle donc par tranches de CHUNK epoques : chaque tranche laisse un
# checkpoint, ce qui rend le job reprenable s'il est interrompu (limite de
# temps atteinte, noeud redemarre...). Relancer le meme sbatch repart alors
# du dernier checkpoint au lieu de tout recalculer.

last=0
for f in "$MODELS/${NAME}"_*.pth; do
    [ -e "$f" ] || continue
    n=$(basename "$f" .pth)
    n=$((10#${n##*_}))  # 10# : "008" est decimal, pas octal
    (( n > last )) && last=$n
done

(( last > 0 )) && echo "Checkpoint trouve a l'epoque $last -> reprise"

resume=$((last + 1))

if (( resume > EPOCHS )); then
    echo "Les $EPOCHS epoques sont deja faites. Rien a calculer."
    exit 0
fi

while (( resume <= EPOCHS )); do
    n=$(( EPOCHS - resume + 1 ))
    (( n > CHUNK )) && n=$CHUNK

    echo "--- epoques $resume a $((resume + n - 1)) sur $EPOCHS ---"

    python train.py \
        -m "$MODEL" -multitask \
        -n "$NAME" \
        -e "$n" \
        -r "$resume" \
        -scale "$SCALE" \
        -batch "$BATCH" \
        -i "$VIA" \
        -p "$DATA" \
        -d "$MODELS" \
        -s "$CSV" \
        -k "$K" -f "$FOLD"

    resume=$((resume + n))
done

echo
echo "=== termine | $(date) ==="
ls -lh "$MODELS/${NAME}"_*.pth
