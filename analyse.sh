#!/bin/bash
# Every figure and table of trained runs, as cluster jobs:      bash analyse.sh RUN [RUN ...]
#   SPLIT=test bash analyse.sh RUN [RUN ...]    the same on the test years (once, at the very end)
# Figures -> figs/RUN/ (figs/RUN_test/), job outputs -> logs/. The split is evaluated first if it has not been
# with this version of the code (no <split>_metrics_climmonth.npz: the climatology baselines are new).
# A "cutoff" or "sparse" run also gets snapshot/: every column of one winter and one summer day, mapped at 30, 75
# and 150 m (truth, reconstruction, error, spread).
# A "neighbours" run also gets, in sub-folders:
#   casts/    evaluated with exactly 1, 2 and 4 given columns
#   layouts/  maps from stations on a grid against Argo-like stations, one station in 3, 5, 10, 15, 20 grid points
#   field/    maps from stations 30, 60 and 120 km apart, and (60 km) with a member's noise smooth over 10, 15, 25,
#             50 km or one draw for the whole sea; each one also from as many stations placed like Argo floats;
#             and maps from 135 Argo-like floats of O2 at 30, 75, 150 m, temperature at 30, 75 m, chlorophyll at 30 m
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$CODE/cluster.sh"
cd "$CODE" && mkdir -p logs
set -euo pipefail
: "${1:?usage: bash analyse.sh RUN [RUN ...]}"
SPLIT=${SPLIT:-val}
# the climatology of the training years (baseline of every evaluation): computed once, by one CPU job, then cached
clim=$(sbatch --parsable -J climatology -c 4 --mem=48G -t 03:00:00 -o logs/climatology.out --wrap \
  ". $CODE/cluster.sh; cd $CODE; python -c 'from config import Config; from dataset import climatology_of; climatology_of(Config())'" | cut -d';' -f1)

job() {  # job NAME "COMMAND" [sbatch option] -> prints the job id
  sbatch --parsable -J "$1" --gres=gpu:1 -c 4 --mem=48G -t 04:00:00 -o "logs/$1.out" ${3:-} --wrap \
    ". $CODE/cluster.sh; cd $CODE; export WANDB_MODE=offline; $2" | cut -d';' -f1
}

for NAME in "$@"; do
  RUN=$R/$NAME
  TAG=$NAME; [ "$SPLIT" = val ] || TAG=${NAME}_$SPLIT  # name of the jobs and of the figure folder
  OUT=figs/$TAG
  [ -f "$RUN/best.pt" ] || { echo "$RUN/best.pt not found: train it first (bash submit.sh $NAME ...)"; exit 1; }
  evaluate="python train.py --eval-only --split $SPLIT --max-profiles 50000 --set ckpt_dir=$RUN num_workers=3"
  figures="--split $SPLIT"
  after=""
  [ -f "$RUN/${SPLIT}_metrics_climmonth.npz" ] || after="--dependency=afterok:$(job "eval_$TAG" "$evaluate" "--dependency=afterok:$clim")"
  job "figs_$TAG" "python figures.py all $NAME=$RUN --out $OUT $figures" "$after" > /dev/null
  experiment=$(python -c "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['cfg']['experiment'])" "$RUN/best.pt")
  if [ "$experiment" != neighbours ]; then
    job "snap_$TAG" "for step in 0 60; do python figures.py snapshot $RUN --out $OUT/snapshot $figures --step \$step --depths 30 75 150; done" > /dev/null
  else
    ids=""
    for k in 1 2 4; do ids="$ids:$(job "eval${k}_$TAG" "$evaluate neighbour_eval_casts=$k --tag ${k}cast" "--dependency=afterok:$clim")"; done
    m=$RUN/${SPLIT}_metrics
    job "casts_$TAG" "python figures.py compare '1 given=${m}_1cast.npz' '2 given=${m}_2cast.npz' '4 given=${m}_4cast.npz' --out $OUT/casts" \
      "--dependency=afterok$ids" > /dev/null
    job "layouts_$TAG" "python figures.py layouts $RUN --out $OUT/layouts $figures --every 3 5 10 15 20" > /dev/null
    field="python figures.py field $RUN --out $OUT/field $figures"
    job "field_$TAG" "for layout in grid argo; do for km in 30 60 120; do $field --layout \$layout --station-km \$km; done; for km in 0 10 15 25 50; do $field --layout \$layout --station-km 60 --noise-km \$km; done; done" > /dev/null
    argo="$field --layout argo --station-km 60 --noise-km 10 --out $OUT/field_depths"
    job "depths_$TAG" "for d in 30 75 150; do $argo --var o2 --depth \$d; done; for d in 30 75; do $argo --var thetao --depth \$d --step 60; done; $argo --var chl --depth 30 --step 60" > /dev/null
  fi
  echo "$NAME ($SPLIT): jobs submitted -> $CODE/$OUT"
done
echo "follow with:  squeue -u \$USER"
