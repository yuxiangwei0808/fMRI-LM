export TOKENIZERS_PARALLELISM=false

python eval_zeroshot.py \
  --checkpoint=checkpoints/released/fMRI-LM-B-Qwen3-0.6B/fMRI-LM-B-Qwen3-0.6B-instruct.pt \
  --output_dir=checkpoints/zeroshot/fMRI-LM-B-Qwen3-0.6B-UKB_fluidintel_enc \
  --cfg_path=configs/vit_base_p160.yaml \
  --lm_name=Qwen/Qwen3-0.6B \
  --datasets=UKB \
  --batch_size=32 \
  --use_allowed_tokens \