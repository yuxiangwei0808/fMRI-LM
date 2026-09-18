---
library_name: pytorch
pipeline_tag: text-generation
tags:
  - fmri
  - neuroimaging
  - brain-decoding
  - multimodal
  - foundation-model
  - qwen3
---

# fMRI-LM-B (Qwen3-0.6B) — pretrained backbones

Stage-1 tokenizers and stage-2 paired-pretraining checkpoints for **fMRI-LM**, a foundation model
that aligns functional MRI with language.

- Code: <https://github.com/yuxiangwei0808/fMRI-LM>
- Paper: [arXiv:2511.21760](https://arxiv.org/abs/2511.21760)

## Files

Three variants, differing in the objective the stage-1 fMRI tokenizer was trained with.

**`vq-contrastive/`** — vector quantization + SigLIP contrastive alignment.

| file | stage | size |
|---|---|---|
| `vq-contrastive/stage1-tokenizer.pt` | 1 — fMRI tokenizer | 0.95 GiB |
| `vq-contrastive/stage2-pretrain-Qwen3-0.6B.pt` | 2 — paired fMRI-text pretraining | 1.28 GiB |

**`vq-domain/`** — vector quantization + adversarial domain loss.

| file | stage | size |
|---|---|---|
| `vq-domain/stage1-tokenizer.pt` | 1 — fMRI tokenizer | 0.94 GiB |
| `vq-domain/stage2-pretrain-Qwen3-0.6B.pt` | 2 — paired fMRI-text pretraining | 1.28 GiB |

**`mae/`** — masked autoencoding (mask ratio 0.5) + adversarial domain loss.

| file | stage | size |
|---|---|---|
| `mae/stage1-tokenizer.pt` | 1 — fMRI tokenizer | 0.71 GiB |
| `mae/stage2-pretrain-Qwen3-0.6B.pt` | 2 — paired fMRI-text pretraining | 1.28 GiB |

All three were trained on UK Biobank with robust normalisation and Qwen3-0.6B. Stage-2 files are
DeepSpeed checkpoints already merged to a single file.

The MAE stage-1 file loads with `MaskedAutoencoderViT`; the two VQ stage-1 files load with the
`Tokenizer` class. They are not interchangeable.

## Usage

Clone the repo, place a stage-1 file where stage 2 expects it, or a stage-2 file where stage 3
expects it, and run the corresponding script in `scripts/`. Stage 3 reads its parent through
`$STAGE2_CKPT`:

```bash
export STAGE2_CKPT=/path/to/vq-contrastive/stage2-pretrain-Qwen3-0.6B.pt
bash scripts/launch_train_instruction.sh
```

Inputs must be preprocessed as the repo README describes: TR resampled to 2.0 s, 160 timepoints,
450 ROIs (Schaefer-400 + Tian-S3), then robust z-scored per ROI with site-wise variance
normalisation.

Research use only.

## Citation

```bibtex
@article{wei2025fmrilm,
  title   = {fMRI-LM: Towards a Universal Foundation Model for Language-Aligned fMRI Understanding},
  author  = {Wei, Yuxiang and Zhang, Yanteng and Xiao, Xi and Qian, Chengxuan and Wang, Tianyang and Calhoun, Vince D.},
  journal = {arXiv preprint arXiv:2511.21760},
  year    = {2025}
}
```
