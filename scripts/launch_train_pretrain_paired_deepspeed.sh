#!/bin/bash
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

# Resolve the repo root from this script's own location so it runs from anywhere.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Optional: activate a conda env by exporting FMRILM_CONDA_ENV before running.
if [ -n "$FMRILM_CONDA_ENV" ]; then
    conda activate "$FMRILM_CONDA_ENV" 2>/dev/null || source activate "$FMRILM_CONDA_ENV"
fi

# Hugging Face cache; override by exporting HF_HOME before running.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)  # Get number of available GPUs

# Make sure another job doesnt use same port, here using random number
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000))
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
export MASTER_ADDR=localhost
export COUNT_NODE=1
echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${NUM_GPUS}

export TOKENIZERS_PARALLELISM=false
export DS_SKIP_CUDA_CHECK=1


accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=bf16 train_pretrain_paired.py \
 --tokenizer_path=checkpoints/tokenizer/UKB_robust-VQ-ViT_base-p160/ckpt-best.pt \
 --fmri_batch_size=4 \
 --gradient_accumulation_steps=8 \
 --epochs=30 \
 --desc_type=fc,ica \
 --dataset_dir=data/UKB/fmri/TianS3/ \
 --cfg_path=configs/vit_base_p160_newTok.yaml \
 --lm_name=Qwen/Qwen3-0.6B \
 --text_only_weight=0.1 \
 --quantizer=vq \
 --ckpt_postfix=lora_r1_a2_drop.1_qk \
 --deepspeed \
 --zero_stage=2 \
 --save_ckpt \
 --lora_target_modules=q_proj,k_proj \
 --lora_r=1 \
 --lora_alpha=2 \
 --lora_dropout=0.1 \
#  --wandb_log \
#  --resume \
#  --offload_optimizer \
#  --zero_stage=3 \
#  --offload_params \