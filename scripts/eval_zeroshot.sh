export TOKENIZERS_PARALLELISM=false

# Zero-shot evaluation runs on a stage-3 (instruction-tuned) checkpoint, which you produce
# with scripts/launch_train_instruction.sh. The released checkpoints are stage 1 and 2 only.
STAGE3_CKPT="${STAGE3_CKPT:?set STAGE3_CKPT to your stage-3 checkpoint, e.g. checkpoints/instruction/UKB-robust/Qwen3-0.6B/<run>/best_avg_classification_ckpt.pt}"

python eval_zeroshot.py \
  --checkpoint="${STAGE3_CKPT}" \
  --output_dir=checkpoints/zeroshot/UKB_fluidintel_enc \
  --cfg_path=configs/vit_base_p160_newTok.yaml \
  --lm_name=Qwen/Qwen3-0.6B \
  --datasets=UKB \
  --batch_size=32 \
  --use_allowed_tokens \