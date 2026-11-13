#!/bin/bash
# Training job of the run $R/RUN_NAME. Resumes from its last.pt and requeues itself before the time limit.
#   RUN_NAME=v5_cutoff sbatch run.slurm experiment=cutoff
# Extra arguments are config overrides (key=value). Usually submitted through submit.sh.
#SBATCH --job-name=bs-o2
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
set +u; source ./cluster.sh; set -u

RUN_NAME="${RUN_NAME:?set RUN_NAME}"
CKPT_DIR="${R}/${RUN_NAME}"
GPUS="${SLURM_GPUS_ON_NODE:-1}"
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} / GPUS - 1 )); (( WORKERS < 1 )) && WORKERS=1
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_MODE="${WANDB_MODE:-offline}" WANDB_DIR="${CKPT_DIR}" WANDB_RESUME=allow  # then: wandb sync <run>/wandb/offline-run-*
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
export MASTER_ADDR MASTER_PORT
mkdir -p "${CKPT_DIR}" logs
echo "=== $(date) job ${SLURM_JOB_ID} restart ${SLURM_RESTART_COUNT:-0} | ${SLURM_NNODES} node(s) x ${GPUS} GPU(s) | ${CKPT_DIR}"
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'no GPU visible to PyTorch')"

requeue() { (( ${SLURM_RESTART_COUNT:-0} < 20 )) && scontrol requeue "${SLURM_JOB_ID}"; }
trap requeue USR1
srun --kill-on-bad-exit=1 --ntasks-per-node=1 torchrun --nnodes="${SLURM_NNODES}" --nproc_per_node="${GPUS}" \
    --rdzv_id="${SLURM_JOB_ID}" --rdzv_backend=c10d --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" --max_restarts=0 \
    train.py --resume auto --set "ckpt_dir=${CKPT_DIR}" "wandb_run_name=${RUN_NAME}" "num_workers=${WORKERS}" "$@" &
set +e; wait $!; STATUS=$?; set -e
(( STATUS > 128 )) && { wait || true; exit 0; }  # interrupted by USR1: the requeued job continues
exit "${STATUS}"
