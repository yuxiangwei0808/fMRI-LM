export TOKENIZERS_PARALLELISM=false

python eval_zeroshot.py \
  --checkpoint=checkpoints/pretrain/UKB_ABCD_robust/VQ_Align-ViT_base-p160-gpt2-Contr_F2T/ckpt.pt \
  --output_dir=checkpoints/zeroshot/UKB_ABCD_robust-VQ-ViT_base_p160_gpt2-Contr_F2T-UKB_fluidintel_enc \
  --cfg_path=configs/vit_base_gpt2_p160.yaml \
  --lm_name=gpt2 \
  --datasets=UKB \
  --batch_size=32 \
  --use_allowed_tokens \