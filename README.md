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
- ✅ **Flexible quantization schemes**: Vector Quantization (VQ), Finite Scalar Quantization (FSQ), TiTok
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
Vector Quantizer (VQ/FSQ/TiTok)
    ↓
Projection Layer
    ↓
Large Language Model (GPT-2/Qwen)
    ↓
Text Generation / Classification
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
│   └── titok_models_vanilla.py
├── language_models/         # LLM implementations (Adapt from huggingface's implementations with key modifications to the attention mask)
│   ├── gpt2.py
│   └── qwen3.py
├── quantizers/              # Quantization modules
│   ├── vq.py               # Vector Quantization
│   ├── fsq.py              # Finite Scalar Quantization
│   └── titok/              # TiTok tokenizer
├── metrics/                 # Evaluation metrics
├── configs/                 # Model and dataset configurations
│   ├── vit_base_qwen_p160.yaml
│   ├── dataset_config.yaml
│   └── ...
├── scripts/                 # Training and evaluation scripts
│   ├── launch_train_quantizer_contr.sh
│   ├── launch_train_pretrain_paired.sh
│   ├── launch_train_instruction.sh
│   └── eval_zeroshot.sh
├── train_quantizer_contr.py    # Stage 1: Tokenizer training
├── train_pretrain_paired.py    # Stage 2: Paired alignment
├── train_instruction.py         # Stage 3: Instruction tuning
├── eval_zeroshot.py            # Zero-shot evaluation
├── model_mindlm.py             # Main model architecture
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
bash scripts/launch_train_quantizer_contr.sh
```

Or run directly:

```bash
accelerate launch train_quantizer_contr.py \
  --dataset_dir data/UKB/fmri/TianS3/,data/ABCD/fmri/TianS3/ \
  --ckpt_dir checkpoints/tokenizer/VQ_Align-ViT_base-p160 \
  --cfg_path configs/vit_base_qwen_p160.yaml \
  --quantizer vq \
  --desc_type fc,ica \
  --epochs 50 \
  --batch_size 16 \
  --save_ckpt
```

**Key arguments:**
- `--quantizer`: Quantization type (`vq`, `fsq`, `titok`)
- `--desc_type`: Text descriptor types for alignment (`fc`, `ica`, `gradient`, `graph`)
- `--clip_loss_type`: Contrastive loss (`clip`, `soft_clip`, `siglip`)

### Stage 2: Paired Pre-training

Align fMRI tokens with language model using paired fMRI-text data:

```bash
bash scripts/launch_train_pretrain_paired.sh
```

Or run directly:

```bash
accelerate launch --mixed_precision=bf16 train_pretrain_paired.py \
  --tokenizer_path checkpoints/tokenizer/VQ_Align-ViT_base-p160/ckpt.pt \
  --ckpt_dir checkpoints/pretrain/VQ-ViT_base-p160-qwen0.6B \
  --cfg_path configs/vit_base_qwen_p160.yaml \
  --lm_name Qwen/Qwen3-0.6B \
  --dataset_dir data/UKB/fmri/TianS3/,data/ABCD/fmri/TianS3/ \
  --quantizer vq \
  --epochs 50 \
  --fmri_batch_size 16 \
  --text_only_weight 1.0 \
  --fmri_only_weight 0.5 \
  --save_ckpt
```

**Key arguments:**
- `--lm_name`: Language model (`gpt2`, `Qwen/Qwen3-0.6B`)
- `--text_only_weight`: Weight for text-only language modeling loss
- `--fmri_only_weight`: Weight for fMRI-only language modeling loss
- `--fmri2text_weight`: Weight for fMRI-to-text generation loss

### Stage 3: Instruction Tuning

Fine-tune the model for specific downstream tasks:

```bash
bash scripts/launch_train_instruction.sh
```

Or run directly:

```bash
accelerate launch --mixed_precision=bf16 train_instruction.py \
  --pretrained_ckpt checkpoints/pretrain/VQ-ViT_base-p160-qwen0.6B/ckpt.pt \
  --ckpt_dir checkpoints/instruction/VQ-ViT_base-qwen0.6B-classification \
  --cfg_path configs/vit_base_qwen_p160.yaml \
  --lm_name Qwen/Qwen3-0.6B \
  --quantizer vq \
  --global_fmri_batch_size 64 \
  --gradient_accumulation_steps 8 \
  --epochs 20 \
  --add_src_info \
  --add_desc \
  --use_allowed_tokens \
  --save_ckpt
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

Or run directly:

```bash
python eval_zeroshot.py \
  --checkpoint checkpoints/pretrain/VQ-ViT_base-p160-qwen0.6B/ckpt.pt \
  --output_dir results/zeroshot/ \
  --cfg_path configs/vit_base_qwen_p160.yaml \
  --lm_name Qwen/Qwen3-0.6B \
  --datasets UKB,ABCD,HCP \
  --batch_size 32 \
  --use_allowed_tokens
```

**Key arguments:**
- `--checkpoint`: Path to model checkpoint
- `--datasets`: Comma-separated list of datasets to evaluate
- `--batch_size`: Batch size for inference

## Model Checkpoints

Pretrained models will be available at [link to be added].

**Available Models:**
- `VQ-ViT_base-p160-GPT2`: ViT-Base encoder + GPT-2 decoder
- `VQ-ViT_base-p160-Qwen0.6B`: ViT-Base encoder + Qwen3-0.6B decoder
- `VQ-ViT_large-p160-Qwen0.6B`: ViT-Large encoder + Qwen3-0.6B decoder

## Configuration Files

### Model Configuration (`configs/vit_base_qwen_p160.yaml`)

```yaml
model:
  vq_model:
    patch_size: 160           # Temporal patch size
    n_embd: 768              # Embedding dimension
    num_rois: 450            # Number of brain regions
    num_timestamp: 160       # Time points per ROI
    enc_cls: vit_base        # Encoder architecture
    
  lm:
    base_model: Qwen/Qwen3-0.6B
    num_fmri_tokens: 450
    fmri_vocab_size: 8192
    use_fmri_lm_head: true
    add_fmri_delimiter: true
    peft_tune: true          # Enable LoRA
    temperature: 0.7
    top_p: 0.8
```

## Advanced Usage

### Using DeepSpeed

For large-scale distributed training:

```bash
bash scripts/launch_train_pretrain_paired_deepspeed.sh
```

Configure DeepSpeed settings in `configs/deepspeed_zero2.json` or `configs/deepspeed_zero3.json`.

### Custom Datasets

To add a new dataset:

1. Prepare data in HDF5 format following the structure above
2. Add dataset info to `dataset.py`:
   ```python
   DATASET_INFO = {
       'MyDataset': 'Description of the cohort...',
   }
   ```
3. Update `configs/dataset_config.yaml`:
   ```yaml
   datasets:
     MyDataset:
       targets:
         - target1
         - target2
   ```

### Merge DeepSpeed Checkpoints

If using DeepSpeed, merge sharded checkpoints:

```bash
python merge_deepspeed_checkpoint.py \
  --checkpoint_dir checkpoints/pretrain/model_name/deepspeed_checkpoint/ \
  --output_file checkpoints/pretrain/model_name/merged_checkpoint.pt
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