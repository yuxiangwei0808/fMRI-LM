# fMRI-LM: LLM-based Brain Foundation Model for fMRI Analysis

Official PyTorch implementation of **fMRI-LM**, a foundation model for analyzing functional magnetic resonance imaging (fMRI) data using large language models.

## Note
Since the work is still in a preliminary phase, codes are not cleaned. The current repo, methods, and results may have significant changes in future versions.

## Overview

BrainFM is a multimodal foundation model that bridges brain imaging data (fMRI) and natural language. The model learns to:
- **Tokenize** fMRI signals into discrete representations using vector quantization
- **Align** brain activity patterns with text descriptions through contrastive learning
- **Generate** textual descriptions from brain signals via instruction tuning
- **Predict** clinical outcomes from fMRI data in zero-shot and few-shot settings

The framework supports multiple large language models (GPT-2, Qwen) and can handle diverse neuroimaging datasets including UK Biobank (UKB), ABCD, HCP, HCP-Aging, ADNI, ABIDE2, and ADHD200.

## Key Features

- ✅ **Multi-stage training pipeline**: Tokenizer pre-training → Paired alignment → Instruction tuning
- ✅ **Flexible quantization schemes**: Vector Quantization (VQ), Finite Scalar Quantization (FSQ)
- ✅ **LoRA-based parameter-efficient fine-tuning** for instruction following
- ✅ **Zero-shot evaluation** on clinical prediction tasks
- ✅ **Multi-dataset support** with dataset-specific preprocessing
- ✅ **Distributed training** with DeepSpeed and Accelerate integration

## Architecture

```
fMRI Signal (N_rois × N_timepoints)
    ↓
Vision Transformer Encoder
    ↓
Vector Quantizer (VQ/FSQ)
    ↓
Projection Layer
    ↓
Large Language Model (GPT-2/Qwen)
    ↓
Text Generation
```

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.7 (for GPU support)

### Dependencies

```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install transformers accelerate peft
pip install numpy pandas polars h5py
pip install scikit-learn scipy
pip install einops flash-attn
pip install omegaconf colorlog tqdm wandb

# Optional: For DeepSpeed training
pip install deepspeed
```

## Project Structure

```
.
├── brain_encoder/           # Vision Transformer encoder for fMRI
│   ├── vision_transformer.py
│   ├── patch_embed.py
├── language_models/         # LLM implementations (Adapt from huggingface's implementations with key modifications to the attention mask)
│   ├── gpt2.py
│   └── qwen3.py
├── quantizers/              # Quantization modules
│   ├── vq.py               # Vector Quantization
│   ├── fsq.py              # Finite Scalar Quantization
├── metrics/                 # Evaluation metrics
├── configs/                 # Model and dataset configurations
│   ├── vit_base_p160.yaml
│   ├── dataset_config.yaml
│   └── ...
├── scripts/                 # Training and evaluation scripts
|   ├── launch_train_quantizer.sh        # Stage 1: Toenizer training (without contrastive learning)
│   ├── launch_train_quantizer_contr.sh  # Stage 1: Tokenizer training
│   ├── launch_train_pretrain_paired_deepspeed.sh  # Stage 2: LLM tuning
│   ├── launch_train_instruction.sh      # Stage 3: Instruction tuning
│   └── eval_zeroshot.sh
├── train_quantizer.py          # Stage 1: Tokenizer training (without contrastive learning)
├── train_quantizer_contr.py    # Stage 1: Tokenizer training
├── train_pretrain_paired.py    # Stage 2: LLM tuning
├── train_instruction.py         # Stage 3: Instruction tuning
├── eval_zeroshot.py            # Zero-shot evaluation
├── model_fmrilm.py             # Main model architecture
├── model_gpt.py                # Multimodal LLM wrapper
├── dataset.py                  # Data loading utilities
└── utils.py                    # Helper functions
```

## Data Preparation

### Expected Data Format

The code expects preprocessed fMRI data in HDF5 format:

```
data/
├── UKB/fmri/TianS3/
│   ├── data_resampled.h5      # fMRI time series
│   ├── normalization_params.npz
│   ├── descriptors_rewritten/  # Text descriptions
│   │   ├── fc_descriptors.csv
│   │   ├── gradient_descriptors.csv
│   │   ├── graph_descriptors.csv
│   │   └── ica_descriptors.csv
│   └── metadata_with_text_medical_gpt.csv
├── ABCD/fmri/TianS3/
├── HCP/fmri/TianS3/
└── ...
```

The descriptors for UKB are provided in `https://gtvault-my.sharepoint.com/:u:/g/personal/ywei355_gatech_edu/IQDWAgF06AlQTrZT2jH6jR0YAer1jrn3Lmf_WfaPWrFAJaw?e=8nkDCG`. Relevant codes to generate descriptors are provided in `nbs_data/get_fmri_discriptor.py`

### HDF5 Data Structure

```python
data_resampled.h5
├── time_series/
│   ├── sample_0: (N_rois, N_timepoints)
│   ├── sample_1: (N_rois, N_timepoints)
│   └── ...
└── metadata/
    ├── subjects: [subject_ids]
    └── sessions: [session_ids]
```

### Configuration

Edit `configs/dataset_config.yaml` to specify prediction targets for each dataset:

```yaml
datasets:
  UKB:
    targets:
      - sex
      - fluidintel_enc
  HCP:
    targets:
      - sex
  ADNI:
    targets:
      - AD
  ABIDE2:
    targets:
      - ASD
```

## Training Pipeline

### Stage 1: Tokenizer Pre-training with Contrastive Alignment

Train the fMRI tokenizer to align brain signals with text embeddings:

```bash
bash scripts/launch_train_quantizer.sh
```

**Key arguments:**
- `--quantizer`: Quantization type (`vq`, `fsq`)
- `--desc_type`: Text descriptor types for alignment (`fc`, `ica`, `gradient`, `graph`)

### Stage 2: Paired Pre-training

Align fMRI tokens with language model using paired fMRI-text data:

```bash
bash scripts/launch_train_pretrain_paired_deepspeed.sh
```

**Key arguments:**
- `--lm_name`: Language model (`gpt2`, `Qwen/Qwen3-0.6B`)
- `--text_only_weight`: Weight for text-only language modeling loss
- `--fmri_only_weight`: Weight for fMRI-only language modeling loss
- `--fmri2text_weight`: Weight for fMRI-to-text generation loss

**Note**
Training with deepspeed requires additional postprocessing of the checkpoint.
If using DeepSpeed, merge sharded checkpoints:

```bash
cd checkpoints/pretrain/model_name/deepspeed_checkpoint
python zero_to_fp32.py . .
cd -
python merge_deepspeed_checkpoint.py \
  --deepspeed_dir checkpoints/pretrain/model_name/deepspeed_checkpoint/
```


### Stage 3: Instruction Tuning

Fine-tune the model for specific downstream tasks:

```bash
bash scripts/launch_train_instruction.sh
```

**Key arguments:**
- `--pretrained_ckpt`: Path to pre-trained checkpoint
- `--add_src_info`: Add dataset source information to prompts
- `--add_desc`: Include text descriptors in input
- `--use_allowed_tokens`: Constrain generation to valid answer tokens
- `--use_random_prompt`: Use diverse prompt paraphrases

## Evaluation

### Zero-shot Evaluation

Evaluate pre-trained models on classification tasks without fine-tuning:

```bash
bash scripts/eval_zeroshot.sh
```

**Key arguments:**
- `--checkpoint`: Path to model checkpoint
- `--datasets`: Comma-separated list of datasets to evaluate
- `--batch_size`: Batch size for inference

## Model Checkpoints

**Stage 3 (instruction-tuned, released model)** — `fMRI-LM-B (Qwen3-0.6B)`, on Hugging Face:
<https://huggingface.co/stanjsx/fMRI-LM-B-Qwen3-0.6B>. See `MODEL_CARD.md` for the full card. The
checkpoint is self-contained (fMRI tokenizer + LM + LoRA), so evaluation needs only that one file.

**Stage 1 (tokenizer) and stage 2 (paired pretraining)** checkpoints:
`https://drive.google.com/drive/folders/1vGN12_bCg4CY2d7AodLw163TuP1QKlkG?usp=drive_link`

### Layout expected by the scripts

```
checkpoints/
├── tokenizer/UKB_robust-VQ-ViT_base-p160/ckpt-best.pt     # stage 1 output / stage 2 input
├── pretrain/<dataset>-<norm>/<run>_<MMDD_HHMMSS>/         # stage 2 output (auto-named)
│   └── deepspeed_checkpoint_best_f2t/merged_checkpoint.pt # -> export STAGE2_CKPT=this
├── instruction/UKB-robust/Qwen3-0.6B/                     # stage 3 output
└── released/fMRI-LM-B-Qwen3-0.6B/                         # downloaded release, used by eval
    └── fMRI-LM-B-Qwen3-0.6B-instruct.pt
```

Stage 2 auto-generates its own timestamped output directory, so `scripts/launch_train_instruction.sh`
reads the stage-2 checkpoint from `$STAGE2_CKPT`; export it before running that script.

### Released model results

Recorded by the run itself and stored inside the checkpoint, so these describe the single released
file. Accuracy / ROC-AUC (%). The paper's Table 3 reports validation numbers, so that column is the
comparable one.

| task | validation | test | paper fMRI-LM-B(G) | paper fMRI-LM-B(Q) |
|---|---|---|---|---|
| UKB-sex | 93.33 / 93.21 | 92.90 / 92.76 | 94.89 / 94.90 | 94.45 / 94.67 |
| HCP-sex | **84.82 / 84.46** | 82.59 / 81.85 | 82.38 / 83.06 | 83.04 / 85.22 |
| HCP_Aging-sex | 84.38 / 85.71 | 85.71 / 84.78 | — | — |
| ADNI-AD | 65.00 / 58.12 | 71.02 / 64.39 | 77.92 / 79.91 | 85.27 / 81.02 |
| ADHD200-ADHD | 71.43 / 68.57 | 77.68 / 73.63 | 75.06 / 77.14 | 78.57 / 79.48 |
| ABIDE2-ASD | **78.12 / 76.08** | 63.33 / 63.57 | 73.44 / 73.02 | 76.56 / 76.22 |

This run post-dates the camera-ready version. It improves on both published rows for **HCP-sex** and
**ABIDE2-ASD**, and is **below** the paper on UKB-sex, ADNI-AD and ADHD200 — markedly so on ADNI-AD.
Single seed (1337); the paper's parenthesised values are standard deviations over seeds. The stage-2
checkpoint this run started from was not retained, so the scripts reproduce the recipe rather than
this exact artifact.

To publish or re-publish the release:

```bash
python scripts/upload_to_hf.py --repo-id stanjsx/fMRI-LM-B-Qwen3-0.6B \
  --src <stage-3 run directory> --dry-run
```

## Supported Datasets

The code supports the following neuroimaging datasets:

- **UK Biobank (UKB)**: 30-70 years old adults
- **ABCD**: 9-10 years old children
- **HCP**: Young adults aged 22-35
- **HCP-Aging**: Older adults aged 36-100
- **ADNI**: Alzheimer's disease research cohort
- **ADHD200**: Children and adolescents from 7-21
- **ABIDE2**: Autism research, ages 5-64

## Citations
Wei Y, Zhang Y, Xiao X, et al. fMRI-LM: Towards a Universal Foundation Model for Language-Aligned fMRI Understanding[J]. arXiv preprint arXiv:2511.21760, 2025.

## Acknowledgments

This codebase builds upon:
- [NeuroLM](https://github.com/935963004/NeuroLM) - Initial framework for brain-language modeling
- [BrainJEPA](https://github.com/Eric-LRL/Brain-JEPA) - Vision transformer for brain imaging
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - Language model implementations