#!/bin/bash
# Every figure and table of trained runs, as cluster jobs:      bash analyse.sh RUN [RUN ...]
#   SPLIT=test bash analyse.sh RUN [RUN ...]    the same on the test years (once, at the very end)
# Figures -> figs/RUN/ (figs/RUN_test/), job outputs -> logs/. The split is evaluated first if its metrics file
# is missing.
# A "neighbours" run also gets, in sub-folders:
#   casts/    evaluated with exactly 1, 2 and 4 given columns
#   layouts/  maps from stations on a grid against Argo-like stations, one station in 3, 5, 10, 15, 20 grid points
#   field/    maps from stations 30, 60 and 120 km apart, and (60 km) with a member's noise smooth over 10, 15, 25,
#             50 km or one draw for the whole sea
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$CODE/cluster.sh"
cd "$CODE" && mkdir -p logs
set -euo pipefail
: "${1:?usage: bash analyse.sh RUN [RUN ...]}"
SPLIT=${SPLIT:-val}

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
  [ -f "$RUN/${SPLIT}_metrics.npz" ] || after="--dependency=afterok:$(job "eval_$TAG" "$evaluate")"
  job "figs_$TAG" "python figures.py all $NAME=$RUN --out $OUT $figures" "$after" > /dev/null
  if [ "$(python -c "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['cfg']['experiment'])" "$RUN/best.pt")" = neighbours ]; then
    ids=""
    for k in 1 2 4; do ids="$ids:$(job "eval${k}_$TAG" "$evaluate neighbour_eval_casts=$k --tag ${k}cast")"; done
    m=$RUN/${SPLIT}_metrics
    job "casts_$TAG" "python figures.py compare '1 given=${m}_1cast.npz' '2 given=${m}_2cast.npz' '4 given=${m}_4cast.npz' --out $OUT/casts" \
      "--dependency=afterok$ids" > /dev/null
    job "layouts_$TAG" "python figures.py layouts $RUN --out $OUT/layouts $figures --every 3 5 10 15 20" > /dev/null
    field="python figures.py field $RUN --out $OUT/field $figures"
    job "field_$TAG" "for km in 30 60 120; do $field --station-km \$km; done; for km in 0 10 15 25 50; do $field --station-km 60 --noise-km \$km; done" > /dev/null
  fi
  echo "$NAME ($SPLIT): jobs submitted -> $CODE/$OUT"
done
echo "follow with:  squeue -u \$USER"
