#!/bin/bash
#SBATCH -p gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -N 1 -c 32
#SBATCH -t 24:00:00
#SBATCH -J nllb_ft
#SBATCH -A lt200304

set -euo pipefail
module purge

# ---- DeepSpeed/Accelerate comms: use NCCL, not MPI ----
export DEEPSPEED_COMM=nccl
export DEEPSPEED_USE_MPI=0

# If the cluster injects MPI/PMI env vars, scrub them so DS doesn't try MPI
for v in $(env | egrep '^(OMPI_|PMI_)' | cut -d= -f1); do unset "$v"; done

# (Optional) NCCL safety/debug
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

# ---- Paths ----
cd /project/lt200304-dipmt/paweekorn/JOB
WORK_DIR=/project/lt200304-dipmt/paweekorn

echo "[$(date)] Starting GPU Job on ${SLURM_JOB_NODELIST:-node}"
# Use srun to inherit Slurm’s cgroup/CPU binding
srun -N 1 -n 1 --gpus=1 --cpus-per-task=32 python -u nllb_ft.py
echo "[$(date)] GPU Job Finished"
