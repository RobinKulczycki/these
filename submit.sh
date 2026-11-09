#!/bin/bash
# Train <R>/NAME with run.slurm, then evaluate its best.pt on 50,000 validation profiles (-> val_metrics.npz).
#   bash submit.sh NAME [key=value ...]        e.g.  bash submit.sh v4_cutoff $B experiment=cutoff input_drift_mode=lagged
#   EVAL=only bash submit.sh NAME              evaluation of an existing run only (also runs trained by older code)
# Needs D (processed data); R defaults to $GLOBALSCRATCH/data/runs. Both may come from ~/tfec/env.sh.
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f ~/tfec/env.sh ] && . ~/tfec/env.sh
cd "$CODE" && mkdir -p logs
set -euo pipefail
NAME=${1:?usage: bash submit.sh NAME [key=value ...]}; shift
R=${R:-$GLOBALSCRATCH/data/runs}
: "${D:?set D, the processed data folder}"
dep=""
if [ "${EVAL:-}" != only ]; then
  id=$(RUN_NAME=$NAME CKPT_DIR=$R/$NAME PROCESSED_DIR=$D sbatch --parsable -J "$NAME" run.slurm "$@")
  dep="--dependency=afterok:${id%%;*}"
  echo "training $NAME: job ${id%%;*}"
fi
id=$(sbatch --parsable -J "ev_$NAME" $dep --gres=gpu:1 -c 4 --mem=48G -t 04:00:00 -o "logs/eval_$NAME.out" --wrap \
  "[ -f ~/tfec/env.sh ] && . ~/tfec/env.sh; cd $CODE; export WANDB_MODE=offline; \
   python train.py --eval-only --split val --max-profiles 50000 --set processed_dir=$D ckpt_dir=$R/$NAME num_workers=3")
echo "evaluation $NAME: job ${id%%;*} -> $R/$NAME/val_metrics.npz"
