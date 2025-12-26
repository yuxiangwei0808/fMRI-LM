#!/bin/bash
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

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
export DS_SKIP_CUDA_CHECK=1

# DeepSpeed with ZeRO-2: Offload optimizer states to CPU
# This should allow batch_size=4-8 with Qwen3-4B
accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=bf16 train_pretrain_paired.py \
 --tokenizer_path=checkpoints/tokenizer/UKB_ABCD_HCP_robust/VQ_Align-ViT_base-p32-Qwen3-0.6B/ckpt.pt \
 --ckpt_dir=checkpoints/pretrain/UKB_ABCD_HCP_robust/VQ_Align-ViT_base-p32-Qwen3-0.6B-Contr_F2T-DeepSpeed-delimiter-PEFT_all_8_16_.1-textW.1 \
 --wandb_runname=UKB_ABCD_HCP_robust-vq-ViT_base-p32-Qwen3-0.6B-Contr_F2T-DeepSpeed-delimiter-PEFT_all_8_16_.1-textW.1 \
 --fmri_batch_size=3 \
 --gradient_accumulation_steps=8 \
 --epochs=20 \
 --quantizer=vq \
 --desc_type=fc,ica \
 --save_ckpt \
 --dataset_dir=data/UKB/fmri/TianS3/ \
 --cfg_path=configs/vit_base_qwen_p32.yaml \
 --lm_name=Qwen/Qwen3-0.6B \
 --deepspeed \
 --zero_stage=2 \
 --wandb_log \
 --text_only_weight=0.1 \
 --resume \
#  --offload_optimizer \
#  --zero_stage=3 \
#  --offload_params \