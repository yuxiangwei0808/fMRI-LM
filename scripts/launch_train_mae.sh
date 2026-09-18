#!/bin/bash

# Resolve the repo root from this script's own location so it runs from anywhere.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Optional: activate a conda env by exporting FMRILM_CONDA_ENV before running.
if [ -n "$FMRILM_CONDA_ENV" ]; then
    conda activate "$FMRILM_CONDA_ENV" 2>/dev/null || source activate "$FMRILM_CONDA_ENV"
fi
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  export NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
else
  export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000))
export MASTER_ADDR=localhost
export COUNT_NODE=1

# Stage 1, masked-autoencoder variant (mask ratio 0.5), matching the released `mae` tokenizer.
accelerate launch --num_processes=$NUM_GPUS --num_machines=$COUNT_NODE \
  --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=bf16 train_mae.py \
 --dataset_dir=data/UKB/fmri/TianS3/ \
 --cfg_path=configs/vit_base_p160_newTok.yaml \
 --ckpt_dir=./checkpoints/tokenizer_mae/UKB \
 --model_size=base \
 --patch_size=160 \
 --mask_ratio=0.5 \
 --norm=robust \
 --batch_size=32 \
 --epochs=30 \
 --warmup_epochs=5 \
 --learning_rate=5e-5
