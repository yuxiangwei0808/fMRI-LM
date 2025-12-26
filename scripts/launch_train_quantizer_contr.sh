#!/bin/bash
export PATH="/sysapps/ubuntu-applications/miniconda/4.12.0/miniconda3/bin:$PATH"
cd ~/playground/BrainFM

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

echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${NUM_GPUS}


accelerate launch --num_processes=$NUM_GPUS --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=fp16 train_quantizer_contr.py \
 --batch_size=12 \
 --epochs=50 \
 --dataset_dir=data/UKB/fmri/TianS3/,data/ABCD/fmri/TianS3/ \
 --wandb_runname=UKB_ABCD_robust-contr-soft_siglip_cls_last-vq-vit_small-p160-domainConfuse0.5 \
 --quantizer=vq \
 --cfg_path=configs/vit_small_gpt2_p160.yaml \
 --contr_loss=soft_siglip \
 --fmri_pool_method=cls \
 --text_pool_method=last \
 --contr_weight=1.0 \
 --ckpt_dir=./checkpoints/tokenizer/UKB_ABCD_robust-contr/VQ-ViT_small-p160-soft_siglip_cls_last-domainConfuse0.5 \
 --desc_type=fc,ica \
 --domain_confuse_weight=0.5 \
 --wandb_log \
#  --resume \