#!/bin/bash
export PATH="/sysapps/ubuntu-applications/miniconda/4.12.0/miniconda3/bin:$PATH"
export HF_HOME="/data/users1/ywei/data"
cd ~/playground/BrainFM

source activate 
conda activate playground
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

accelerate launch --num_processes=8 --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=bf16 train_quantizer.py \
 --batch_size=8 \
 --dataset_dir=data/UKB/fmri/TianS3/,data/ABCD/fmri/TianS3/,data/HCP/fmri/TianS3/ \
 --wandb_runname=UKB_ABCD_HCP_robust-vq-vit_base-p160-Qwen3-0.6B \
 --quantizer=vq \
 --cfg_path=configs/vit_base_qwen_p160.yaml \
 --ckpt_dir=./checkpoints/tokenizer/UKB_ABCD_HCP_robust/VQ_Align-ViT_base-p160-Qwen3-0.6B \
 --lm_name=Qwen/Qwen3-0.6B \
#  --resume \
#  --wandb_log \