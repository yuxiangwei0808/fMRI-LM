#!/bin/bash
export PATH="/sysapps/ubuntu-applications/miniconda/4.12.0/miniconda3/bin:$PATH"
cd ~/playground/BrainFM

source activate 
conda activate playground
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


# python train_pretrain.py \
accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=bf16 train_pretrain.py \
 --tokenizer_path=checkpoints/tokenizer/UKB_ABCD_robust/VQ_Align-ViT_base-p160/ckpt.pt \
 --ckpt_dir=checkpoints/pretrain/UKB_ABCD_robust/VQ_Align-ViT_base-p160-gpt2 \
 --cfg_path=configs/vit_base_gpt2_p160.yaml \
 --wandb_runname=UKB_ABCD_robust-vq-ViT_base-p160-gpt2 \
 --fmri_batch_size=24 \
 --text_batch_size=16 \
 --epochs=50 \
 --quantizer=vq \
 --save_ckpt \
 --resume \
#  --wandb_log \
#  --freeze_llm \