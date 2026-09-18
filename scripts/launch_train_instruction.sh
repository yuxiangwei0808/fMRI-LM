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

# Stage 2 writes to a timestamped directory:
#   checkpoints/pretrain/<dataset>-<norm>/<desc>_<objectives>_<lm>_<postfix>_<MMDD_HHMMSS>
# Point STAGE2_CKPT at the merged checkpoint from your stage-2 run, e.g.
#   export STAGE2_CKPT=checkpoints/pretrain/UKB-robust/fc_ica_text0.1_f2t1_Qwen3-0.6B_new-lora_r1_a2_drop.1_qk_0415_230152/deepspeed_checkpoint_best_f2t/merged_checkpoint.pt
STAGE2_CKPT="${STAGE2_CKPT:?set STAGE2_CKPT to your stage-2 merged_checkpoint.pt}"
# export CUDA_VISIBLE_DEVICES=4,5,6,7
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
export COUNT_NODE=1
echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${NUM_GPUS}

export TOKENIZERS_PARALLELISM=false

# accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=fp16 train_instruction_open_ended.py \
# accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=fp16 train_instruction_mq.py \
accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=bf16 train_instruction.py \
 --ckpt_dir=checkpoints/instruction/UKB-robust/Qwen3-0.6B \
 --wandb_group=pretrained \
 --cfg_path=configs/vit_base_p160_newTok.yaml \
 --gradient_accumulation_steps=8 \
 --epochs=30 \
 --quantizer=vq \
 --add_src_info \
 --save_ckpt \
 --use_random_prompt \
 --use_allowed_tokens \
 --add_desc \
 --lm_name=Qwen/Qwen3-0.6B \
 --pretrained_ckpt=${STAGE2_CKPT} \
#  --tokenizer_ckpt=checkpoints/tokenizer/UKB_robust-VQ-ViT_base-p160/ckpt-best.pt \
#  --fewshot_samples=10 \
#  --wandb_log \
#  --datasets=UKB,HCP,HCP_Aging,ADNI,ABIDE2,ADHD200 \
#  --resume \