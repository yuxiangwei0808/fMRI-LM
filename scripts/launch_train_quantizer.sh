#!/bin/bash

# Resolve the repo root from this script's own location so it runs from anywhere.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Optional: activate a conda env by exporting FMRILM_CONDA_ENV before running.
if [ -n "$FMRILM_CONDA_ENV" ]; then
    conda activate "$FMRILM_CONDA_ENV" 2>/dev/null || source activate "$FMRILM_CONDA_ENV"
fi

# Hugging Face cache; override by exporting HF_HOME before running.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# export CUDA_VISIBLE_DEVICES=0,1,2,3
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  export NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)  # Get number of available GPUs
else
  export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)  # Get number of available GPUs
fi

# Make sure another job doesnt use same port, here using random number
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000)) 
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
export MASTER_ADDR=localhost

echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${NUM_GPUS}

accelerate launch --num_processes=$NUM_GPUS --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=bf16 train_quantizer.py \
 --batch_size=32 \
 --dataset_dir=data/UKB/fmri/TianS3/ \
 --wandb_runname=UKB_robust-VQ_Align-ViT_base-p160 \
 --quantizer=vq \
 --cfg_path=configs/vit_base_p160_newTok.yaml \
 --ckpt_dir=./checkpoints/tokenizer/UKB_robust-VQ-ViT_base-p160 \
 --domain_loss_weight=1 \
 --lm_name=gpt2 \
 --epochs=100 \
 --warmup_epochs=5 \
 --learning_rate=5e-5