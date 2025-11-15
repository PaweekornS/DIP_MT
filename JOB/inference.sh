#!/bin/bash
#SBATCH -p gpu
#SBATCH --ntasks=1                # Total tasks
#SBATCH --gpus-per-node=1
#SBATCH -c 32
#SBATCH -t 2:00:00               # Time limit (hh:mm:ss)
#SBATCH -J inference                 # Job name
#SBATCH -A lt200304               # Your allocation/account
#SBATCH --output=./logs/infer-%j.out

module purge
cd /project/lt200304-dipmt/paweekorn/JOB/
BASE_DIR=/project/lt200304-dipmt/paweekorn


echo "[$(date)] Starting Inference Job on ${SLURM_JOB_NODELIST:-node}"

python vllm-infer.py \
  --dataset $BASE_DIR/data/DS01/test_v1.csv \
  --quantization bitsandbytes \
  --model_dir $BASE_DIR/models/base/gemma3-4b-it \
  --is_rag True \
  --adapter_dir $BASE_DIR/models/adapter/gemma3-4b-it/checkpoint-1242 \
  --save_dir $BASE_DIR/data/infer-result/en2th/gemma3-4b-it_ft+RAG.csv

echo "GPU Job Finished"