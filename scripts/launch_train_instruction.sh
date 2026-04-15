#!/bin/bash

if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi
export PATH="/sysapps/ubuntu-applications/miniconda/4.12.0/miniconda3/bin:$PATH"
cd ~/playground/BrainFM/public_repos/fMRI-LM

source activate 
conda activate playground
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
 --ckpt_dir=checkpoints/instruction/UKB_robust/VQ-ViT_base_p160_gpt2 \
 --lm_name=Qwen/Qwen3-0.6B \
 --wandb_group=pretrained \
 --cfg_path=configs/vit_base_p160.yaml \
 --gradient_accumulation_steps=8 \
 --epochs=30 \
 --quantizer=vq \
 --add_src_info \
 --save_ckpt \
 --use_random_prompt \
 --use_allowed_tokens \
 --add_desc \
 --lm_name=Qwen/Qwen3-0.6B \
 --pretrained_ckpt=checkpoints/pretrain/UKB-robust/fc_ica_text0.1_f2t1_Qwen3-0.6B_lora_r1_a2_drop.1_qk_0415_143500/deepspeed_checkpoint_best_f2t/merged_checkpoint.pt \
#  --tokenizer_ckpt=checkpoints/tokenizer/UKB_ABCD_HCP_robust/VQ_Align-ViT_base-p160-Qwen3-0.6B/ckpt-best.pt \
#  --fewshot_samples=10 \
#  --wandb_log \
#  --datasets=UKB,HCP,HCP_Aging,ADNI,ABIDE2,ADHD200 \
#  --resume \