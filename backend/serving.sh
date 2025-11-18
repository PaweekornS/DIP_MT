#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 32                 # Specify number of nodes and processors per task
#SBATCH --gpus-per-task=1        
#SBATCH --ntasks-per-node=1
#SBATCH -t 1:00:00               # Time limit (hh:mm:ss)
#SBATCH -J api             # Job name
#SBATCH -A lt200304               # Your allocation/account
#SBATCH --output=./logs/api-%j.out

# ===================
# Load modules
# ===================
module purge
module load gcc
module load cudatoolkit

# ปิดของ Cray ที่ชอบกวน extension
module unload perftools-base 2>/dev/null
unset LD_PRELOAD
unset CRAY_LD_LIBRARY_PATH
export CRAYPE_LINK_TYPE=dynamic

# ====================
# Run FastAPI (uvicorn)
# ====================
cd "$APP_DIR"

PORT=${PORT:-8000}
echo "Starting API on port ${PORT} ..."
echo "Node: ${SLURM_JOB_NODELIST:-node}"
echo ""

# ใช้ srun เพื่อให้ผูก resource กับ Slurm
srun uvicorn app:app \
    --host 0.0.0.0 \
    --port "${PORT}"        
