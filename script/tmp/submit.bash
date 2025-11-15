#!/bin/bash
#SBATCH -p gpu
#SBATCH --ntasks=1                # Total tasks
#SBATCH --gpus-per-node=4
#SBATCH -c 32
#SBATCH -t 24:00:00               # Time limit (hh:mm:ss)
#SBATCH -J judge                  # Job name
#SBATCH -A lt200304               # Your allocation/account

# Load modules
module purge
cd /project/lt200304-dipmt/paweekorn/script/tmp

echo "Starting GPU Job"

python format_query.py \
  --dataset /project/lt200304-dipmt/paweekorn/data/unique_goods.csv \
  --model_dir /project/lt200304-dipmt/paweekorn/models/base/gemma3-27b-it \
  --quantization bitsandbytes

echo "GPU Job Finished"