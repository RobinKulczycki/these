#!/bin/bash
# Train $R/NAME with run.slurm, then evaluate its best.pt on 50,000 validation profiles (-> val_metrics.npz).
#   bash submit.sh NAME [key=value ...]        e.g.  bash submit.sh v5_cutoff experiment=cutoff
#   EVAL=only bash submit.sh NAME              evaluation of an existing run only
# Every setting has its default in config.py: a key=value is only needed for what differs from it.
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$CODE/cluster.sh"
cd "$CODE" && mkdir -p logs
set -euo pipefail
NAME=${1:?usage: bash submit.sh NAME [key=value ...]}; shift
dep=""
if [ "${EVAL:-}" != only ]; then
  [ -f "$R/$NAME/last.pt" ] && echo "NOTE: $R/$NAME exists: its training is continued. For a new training, use another name."
  id=$(RUN_NAME=$NAME sbatch --parsable -J "$NAME" run.slurm "$@")
  dep="--dependency=afterok:${id%%;*}"
  echo "training $NAME: job ${id%%;*}"
fi
id=$(sbatch --parsable -J "ev_$NAME" $dep --gres=gpu:1 -c 4 --mem=48G -t 04:00:00 -o "logs/eval_$NAME.out" --wrap \
  ". $CODE/cluster.sh; cd $CODE; export WANDB_MODE=offline; \
   python train.py --eval-only --split val --max-profiles 50000 --set ckpt_dir=$R/$NAME num_workers=3")
echo "evaluation $NAME: job ${id%%;*} -> $R/$NAME/val_metrics.npz"
