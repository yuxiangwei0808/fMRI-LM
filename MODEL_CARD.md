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

# fMRI-LM-B (Qwen3-0.6B)

Instruction-tuned checkpoint for **fMRI-LM**, a foundation model that aligns functional MRI with
language.

- Code: <https://github.com/yuxiangwei0808/fMRI-LM>
- Paper: [arXiv:2511.21760](https://arxiv.org/abs/2511.21760)

## File

`fMRI-LM-B-Qwen3-0.6B-instruct.pt` (2.56 GiB)

Self-contained: the fMRI tokenizer, positional embedding, encode-transform layer and the
Qwen3-0.6B LM with its LoRA adapters, plus `model_args` and the metrics the run recorded for
itself. No separate stage-1 or stage-2 file is needed to evaluate.

## Usage

```python
import torch

ckpt = torch.load("fMRI-LM-B-Qwen3-0.6B-instruct.pt", map_location="cpu", weights_only=False)
print(ckpt["model_args"])          # architecture config used at training time
print(ckpt["validation_results"])  # metrics recorded by the run
state_dict = ckpt["model"]         # keys: llm.*, tokenizer.*, pos_embed, encode_transform_layer.*
```

To evaluate with the repo, place the file at
`checkpoints/released/fMRI-LM-B-Qwen3-0.6B/fMRI-LM-B-Qwen3-0.6B-instruct.pt` and run
`bash scripts/eval_zeroshot.sh`.

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
