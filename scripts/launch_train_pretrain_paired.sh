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

export TOKENIZERS_PARALLELISM=false
export HF_HOME=/data/users1/ywei/data/cache/

accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=bf16 train_pretrain_paired.py \
 --tokenizer_path=checkpoints/tokenizer/UKB_ABCD_robust/VQ_Align-ViT_base-p160/ckpt.pt \
 --ckpt_dir=checkpoints/pretrain/UKB_ABCD_robust/VQ_Align-ViT_base-p160-gpt2 \
 --wandb_runname=UKB_ABCD_robust-vq-ViT_base-p160-gpt2 \
 --fmri_batch_size=16 \
 --epochs=50 \
 --quantizer=vq \
 --save_ckpt \
 --dataset_dir=data/UKB/fmri/TianS3/,data/ABCD/fmri/TianS3/ \
 --cfg_path=configs/vit_base_gpt2_p160.yaml \
 --lm_name=gpt2 \
 --text_only_weight=1 \
 --fmri_only_weight=0.5 \
 --fmri2text_weight=0 \
#  --desc_type=fc,ica \
#  --wandb_log \
#  --resume \
#  --cfg_path=configs/vit_small_qwen_p160.yaml \
#  --freeze_llm \