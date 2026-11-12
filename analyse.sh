#!/bin/bash
# Every figure and table of the three trained runs, as cluster jobs. Run it once submit.sh has finished:
#   bash analyse.sh [CUTOFF_RUN SPARSE_RUN NEIGHBOURS_RUN]        default: v4_cutoff v4_sparse v4_nb
# Figures -> figs/<run>/, job outputs -> logs/. Needs D (processed data) and R (runs), e.g. from ~/tfec/env.sh.
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f ~/tfec/env.sh ] && . ~/tfec/env.sh
cd "$CODE" && mkdir -p logs
set -euo pipefail
CUT=${1:-v4_cutoff}; SPARSE=${2:-v4_sparse}; NB=${3:-v4_nb}
R=${R:-$GLOBALSCRATCH/data/runs}
: "${D:?set D, the processed data folder}"

job() {  # job NAME "COMMAND" [extra sbatch option] -> job id
  sbatch --parsable -J "$1" --gres=gpu:1 -c 4 --mem=48G -t 04:00:00 -o "logs/$1.out" ${3:-} --wrap \
    "[ -f ~/tfec/env.sh ] && . ~/tfec/env.sh; cd $CODE; export WANDB_MODE=offline; $2"
}
evaluate="python train.py --eval-only --split val --max-profiles 50000 --set ckpt_dir=$R/$NB processed_dir=$D num_workers=3"

job "figs_$CUT"    "python figures.py all v4=$R/$CUT --out figs/$CUT --processed-dir $D"
job "figs_$SPARSE" "python figures.py all v4=$R/$SPARSE --out figs/$SPARSE --processed-dir $D"
job "figs_$NB"     "python figures.py all model=$R/$NB --out figs/$NB --processed-dir $D"
job "layouts_$NB"  "python figures.py layouts $R/$NB --out figs/$NB/layouts --processed-dir $D --every 5 10 15"
one=$(job "eval1_$NB" "$evaluate neighbour_eval_casts=1 --tag 1cast")
four=$(job "eval4_$NB" "$evaluate neighbour_eval_casts=4 --tag 4cast")
job "casts_$NB" "python figures.py compare '1 cast=$R/$NB/val_metrics_1cast.npz' '4 casts=$R/$NB/val_metrics_4cast.npz' --out figs/$NB/casts" \
  "--dependency=afterok:${one%%;*}:${four%%;*}"
echo "submitted: follow with  squeue -u \$USER ; results in $CODE/figs/"
