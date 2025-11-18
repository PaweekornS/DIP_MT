#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1 -c 32                 # Specify number of nodes and processors per task
#SBATCH --gpus-per-task=2        
#SBATCH --ntasks-per-node=1
#SBATCH -t 24:00:00               # Time limit (hh:mm:ss)
#SBATCH -J unsloth_ft             # Job name
#SBATCH -A lt200304               # Your allocation/account
#SBATCH --output=./logs/unsloth-%j.out

# Load modules
module purge
cd /project/lt200304-dipmt/paweekorn/JOB/
BASE_DIR=/project/lt200304-dipmt/paweekorn

echo "[$(date)] Starting Fine-tuning Job on ${SLURM_JOB_NODELIST:-node}"

python finetuning.py \
  --train_dataset $BASE_DIR/data/train_40k.csv \
  --test_dataset $BASE_DIR/data/DS01/test_v1.csv \
  --model_dir $BASE_DIR/models/base/gemma3-4b-it \
  --load_in_4bit False \
  --rank 64 \
  --target_modules q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj \

rm -rf $BASE_DIR/JOB/unsloth_training_checkpoints
rm -rf $BASE_DIR/JOB/torch_compile_debug

echo "GPU Job Finished"