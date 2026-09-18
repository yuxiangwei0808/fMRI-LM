"""
Adapted from https://github.com/935963004/NeuroLM
"""

import os
import time
import math
import argparse
import logging
import sys
import json
import wandb
from tqdm import tqdm
from omegaconf import OmegaConf
import copy
import colorlog
from datetime import datetime

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed

from model_fmrilm import fMRILM
from quantizers import Tokenizer
from model_gpt import MultimodalConfig
from dataset import get_fmri_data, fMRITextDataset
from utils import combine_attn_mask
from metrics.text_generation import TextGenerationMetrics
import checkpoint_naming as ckpt_naming


def setup_logging(log_level=logging.INFO, log_file=None):
    """Setup logging configuration"""
    formatters = {
        'detailed': logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ),
        'colored': colorlog.ColoredFormatter(
            '%(log_color)s%(levelname)s%(reset)s: %(message)s',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red',
            }
        )
    }
    
    # Create logger
    logger = logging.getLogger('train_pretrain')
    logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler with colors
    console_handler = colorlog.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatters['colored'])
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatters['detailed'])
        logger.addHandler(file_handler)
    
    return logger


def validate_paths(args, logger):
    """Validate that required paths exist"""
    required_paths = {
        'tokenizer_path': args.tokenizer_path,
        # 'dataset_dir': args.dataset_dir,
    }
    
    for name, path in required_paths.items():
        if not os.path.exists(path):
            logger.error(f"Required path does not exist: {name} = {path}")
            raise FileNotFoundError(f"Path not found: {path}")
    
    # Check text data directory
    text_data_dir = 'data/text/openwebtext'
    for split in ['train.bin', 'val.bin']:
        text_file = os.path.join(text_data_dir, split)
        if not os.path.exists(text_file):
            logger.warning(f"Text data file not found: {text_file}")
    
    logger.info("All required paths validated successfully")


def simplify_lm_name(lm_name):
    """Simplify language model name for file/dir naming"""
    # Extract just the model name after the last '/'
    if '/' in lm_name:
        lm_name = lm_name.split('/')[-1]
    # Remove version numbers and make it more compact
    # E.g., "Qwen3-0.6B" -> "Qwen3-0.6B", "gpt2-medium" -> "gpt2-medium"
    return lm_name


def auto_generate_names(args):
    """Auto-generate checkpoint directory and wandb run name based on training configuration."""
    data_norm = ckpt_naming.data_norm_name(args.dataset_dir, args.norm)
    desc_str = '_'.join(args.desc_type) if isinstance(args.desc_type, list) else args.desc_type
    
    objectives = []
    if args.fmri_only_weight > 0:
        objectives.append(f"fmri{ckpt_naming.format_float(args.fmri_only_weight)}")
    if args.text_only_weight > 0:
        objectives.append(f"text{ckpt_naming.format_float(args.text_only_weight)}")
    if args.fmri2text_weight > 0:
        objectives.append(f"f2t{ckpt_naming.format_float(args.fmri2text_weight)}")
    if args.text2fmri_weight > 0:
        objectives.append(f"t2f{ckpt_naming.format_float(args.text2fmri_weight)}")
    
    obj_str = '_'.join(objectives) if objectives else "noobj"
    base_name = ckpt_naming.sanitize_component(f"{desc_str}_{obj_str}")
    run_name = ckpt_naming.append_postfix_and_timestamp(base_name, postfix=args.ckpt_postfix, add_timestamp=True)
    ckpt_dir = ckpt_naming.pretrain_ckpt_dir(data_norm, args.lm_name, run_name)
    
    return ckpt_dir, run_name


def save_args_to_json(args, save_dir):
    """Save all arguments to a JSON file"""
    args_dict = vars(args)
    json_path = os.path.join(save_dir, 'training_args.json')
    
    # Convert any non-serializable objects to strings
    serializable_dict = {}
    for key, value in args_dict.items():
        try:
            json.dumps(value)
            serializable_dict[key] = value
        except (TypeError, ValueError):
            serializable_dict[key] = str(value)
    
    with open(json_path, 'w') as f:
        json.dump(serializable_dict, f, indent=4, sort_keys=True)
    
    return json_path


def prepare_tokenizer_config(cfg):
    if OmegaConf.select(cfg, "model.vq_model") is not None:
        raise ValueError(
            "Legacy model.vq_model configs are no longer supported in this stage. "
            "Use the unified Tokenizer config with model.quantize_mode instead."
        )
    if OmegaConf.select(cfg, "model") is None or OmegaConf.select(cfg, "lm") is None:
        raise ValueError("Tokenizer configs must contain top-level model and lm sections.")
    for key in ("num_rois", "num_timestamp", "quantize_mode", "vit_enc_patch_size"):
        if OmegaConf.select(cfg, f"model.{key}") is None:
            raise ValueError(f"Tokenizer config is missing model.{key}.")
    if OmegaConf.select(cfg, "model.image_size") is None:
        cfg.model.image_size = [int(cfg.model.num_rois), int(cfg.model.num_timestamp)]
    if OmegaConf.select(cfg, "model.gate_attention") is None and OmegaConf.select(cfg, "model.gate_attn") is not None:
        cfg.model.gate_attention = cfg.model.gate_attn
    return cfg


def strip_compile_prefix(state_dict):
    state_dict = dict(state_dict)
    for prefix in ('_orig_mod.', 'module.'):
        for key in list(state_dict.keys()):
            if key.startswith(prefix):
                state_dict[key[len(prefix):]] = state_dict.pop(key)
    return state_dict


def checkpoint_model_state(checkpoint):
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        return checkpoint['model']
    return checkpoint


def strip_prefix_if_present(state_dict, prefix):
    prefix = prefix.rstrip('.') + '.'
    stripped = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    return stripped if stripped else None


def map_mae_encoder_to_tokenizer_state(state_dict, expected):
    candidates = [state_dict]
    for prefix in ('mae', 'encoder', 'backbone'):
        stripped = strip_prefix_if_present(state_dict, prefix)
        if stripped is not None:
            candidates.append(stripped)

    best_mapped = {}
    for candidate in candidates:
        mapped = {}
        for key, value in candidate.items():
            new_key = None
            new_value = value
            if key.startswith('patch_embed.') or key.startswith('blocks.'):
                new_key = f'encoder.{key}'
            elif key.startswith('norm.'):
                new_key = f"encoder.ln_post.{key[len('norm.'):]}"
            elif key == 'pos_embed':
                new_key = 'encoder.pos_embed'
            elif key == 'cls_token':
                new_key = 'encoder.cls_embed'
                if value.ndim == 3 and value.shape[1] == 1:
                    new_value = value.squeeze(1)

            if new_key in expected and expected[new_key].shape == new_value.shape:
                mapped[new_key] = new_value

        if len(mapped) > len(best_mapped):
            best_mapped = mapped
    return best_mapped


def load_tokenizer_state(tokenizer, checkpoint, logger):
    state_dict = strip_compile_prefix(checkpoint_model_state(checkpoint))
    expected = tokenizer.state_dict()
    candidates = [('raw', state_dict)]
    for prefix in (
        'quantizer',
        'fmri_model.quantizer',
        'module.quantizer',
        'module.fmri_model.quantizer',
        'tokenizer',
        'module.tokenizer',
    ):
        stripped = strip_prefix_if_present(state_dict, prefix)
        if stripped is not None:
            candidates.append((prefix, stripped))

    mae_mapped = map_mae_encoder_to_tokenizer_state(state_dict, expected)
    if mae_mapped:
        candidates.append(('mae_encoder', mae_mapped))

    def score(candidate):
        _, sd = candidate
        return sum(k in expected and expected[k].shape == v.shape for k, v in sd.items())

    best_name, best = max(candidates, key=score)
    filtered = {k: v for k, v in best.items() if k in expected and expected[k].shape == v.shape}
    if not filtered:
        raise ValueError("No tokenizer checkpoint keys matched the unified Tokenizer state dict.")

    msg = tokenizer.load_state_dict(filtered, strict=False)
    tokenizer._checkpoint_source = best_name
    tokenizer._loaded_from_mae_encoder = best_name == 'mae_encoder'
    logger.info(
        f"Tokenizer state loaded from {best_name}: matched={len(filtered)}, "
        f"missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}"
    )
    if best_name == 'mae_encoder':
        logger.info(
            "Loaded MAE encoder weights into Tokenizer.encoder; "
            "Tokenizer decoder and quantizer parameters remain initialized from cfg."
        )
    else:
        core_missing = [k for k in msg.missing_keys if k.startswith(('encoder.', 'quantize.'))]
        if core_missing:
            logger.warning(f"Tokenizer core missing keys after load: {core_missing[:10]}")
    return msg


class TokenizerEncoderForLM(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.patch_embed = encoder.patch_embed
        self.num_patches = encoder.num_patches
        self.embed_dim = encoder.width

    def forward(self, x):
        encoder_param = next(self.encoder.parameters(), None)
        if encoder_param is not None:
            x = x.to(device=encoder_param.device, dtype=encoder_param.dtype)
        features, _ = self.encoder(x)
        return features.permute(0, 2, 1)


def main(args):
    # Auto-generate checkpoint directory and wandb run name if using defaults.
    # We must ensure all processes agree on the *same* directory name.
    # datetime.now() called independently on each process produces slightly
    # different timestamps, leading to each process creating its own directory.
    #
    # Strategy: rank-0 (LOCAL_RANK == 0) generates the name and writes it to a
    # temporary file; every other rank waits for that file and reads from it.
    # This works regardless of whether torch.distributed is already initialised.
    if args.ckpt_dir == 'tmp':
        import tempfile, time as _time

        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))

        if world_size == 1:
            # Single-process: just generate normally
            args.ckpt_dir, args.wandb_runname = auto_generate_names(args)
        else:
            # Use a shared temp file whose path is deterministic across processes.
            # We derive it from the job's master port (set by torchrun/accelerate)
            # so concurrent jobs on the same node don't collide.
            master_port = os.environ.get('MASTER_PORT', 'default')
            sync_file = os.path.join(tempfile.gettempdir(), f'brainfm_ckpt_sync_{master_port}.txt')

            if local_rank == 0:
                ckpt_dir, wandb_runname = auto_generate_names(args)
                # Write atomically so other ranks don't read a partial file
                tmp_path = sync_file + '.tmp'
                with open(tmp_path, 'w') as f:
                    f.write(f'{ckpt_dir}\n{wandb_runname}\n')
                os.replace(tmp_path, sync_file)
                args.ckpt_dir = ckpt_dir
                args.wandb_runname = wandb_runname
            else:
                # Poll until rank 0 has written the file (up to 60 s)
                deadline = _time.time() + 60
                while not os.path.exists(sync_file):
                    if _time.time() > deadline:
                        raise TimeoutError(
                            f"Timed out waiting for rank-0 to write checkpoint sync file: {sync_file}"
                        )
                    _time.sleep(0.1)
                with open(sync_file, 'r') as f:
                    lines = f.read().splitlines()
                args.ckpt_dir = lines[0]
                args.wandb_runname = lines[1]
    
    # Always sync wandb_runname with ckpt_dir basename
    if args.wandb_runname == 'pretrain':
        args.wandb_runname = os.path.basename(args.ckpt_dir)
    
    # Setup logging
    os.makedirs(args.ckpt_dir, exist_ok=True)
    log_file = os.path.join(args.ckpt_dir, 'training.log') if args.ckpt_dir else None
    logger = setup_logging(log_file=log_file)
    
    # Save arguments to JSON file (includes LoRA hyperparams from CLI overrides)
    json_path = save_args_to_json(args, args.ckpt_dir)
    tokenizer_run_dir = ckpt_naming.checkpoint_run_dir(args.tokenizer_path)
    lineage_path = ckpt_naming.write_lineage_json(
        args.ckpt_dir,
        stage='pretrain',
        checkpoint_dir=args.ckpt_dir,
        tokenizer_checkpoint=args.tokenizer_path,
        tokenizer_run=os.path.basename(tokenizer_run_dir),
        cfg_path=args.cfg_path,
        dataset_dir=args.dataset_dir,
        data_norm=ckpt_naming.data_norm_name(args.dataset_dir, args.norm),
        lm_name=args.lm_name,
        run_name=os.path.basename(args.ckpt_dir),
        args=vars(args),
    )
    logger.info(f"Training arguments saved to: {json_path}")
    logger.info(f"Lineage metadata saved to: {lineage_path}")
    logger.info(f"Checkpoint directory: {args.ckpt_dir}")
    logger.info(f"W&B run name: {args.wandb_runname}")
    
    # Validate paths
    try:
        validate_paths(args, logger)
    except FileNotFoundError as e:
        logger.error(f"Path validation failed: {e}")
        raise
    
    # Initialize Accelerator
    try:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        
        # Setup DeepSpeed if specified
        deepspeed_plugin = None
        if args.deepspeed:
            from accelerate import DeepSpeedPlugin
            deepspeed_plugin = DeepSpeedPlugin(
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                gradient_clipping=args.grad_clip,
                zero_stage=args.zero_stage,
                offload_optimizer_device="cpu" if args.offload_optimizer else "none",
                offload_param_device="cpu" if args.offload_params else "none",
                zero3_init_flag=args.zero_stage == 3,
                zero3_save_16bit_model=True,
            )
            logger.info(f"DeepSpeed ZeRO Stage {args.zero_stage} enabled with optimizer offload: {args.offload_optimizer}, param offload: {args.offload_params}")
        
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            log_with="wandb" if args.wandb_log else None,
            deepspeed_plugin=deepspeed_plugin,
            kwargs_handlers=[ddp_kwargs] if not args.deepspeed else None,
            # project_dir="wandb" if args.wandb_log else None,
        )
        logger.info(f"Accelerator initialized with device: {accelerator.device}")
    except Exception as e:
        logger.error(f"Failed to initialize accelerator: {e}")
        raise
    
    # Set seed for reproducibility
    set_seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")
    
    # Enable optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logger.info("TF32 optimizations enabled")
    
    # Check if we're the main process
    if accelerator.is_main_process:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        logger.info(f"Checkpoint directory created: {args.ckpt_dir}")
        
        if args.wandb_log:
            run = wandb.init(project=args.wandb_project, name=args.wandb_runname, dir='./wandb', config=vars(args))
            run.log_code('.')

            artifact = wandb.Artifact(
                name="config", 
                type="config",
                description="Configuration file for model"
            )
            artifact.add_file(local_path=args.cfg_path, name=args.cfg_path)
            run.log_artifact(artifact)

    # Text data loader
    data_dir = 'data/text/openwebtext'
    
    # Cache memmap to avoid repeated file loading
    _text_data_cache = {}
    
    def get_batch(split, num_token=768, tokenizer=None):
        if split == 'train':
            data_file = os.path.join(data_dir, 'train.bin')
        else:
            data_file = os.path.join(data_dir, 'val.bin')
        
        if tokenizer is not None:
            vocab_size = tokenizer.vocab_size
        else:
            vocab_size = 50257  # default GPT-2 vocab size
        
        # Use cached memmap if available
        if data_file not in _text_data_cache:
            if not os.path.exists(data_file):
                logger.warning(f"Text data file not found: {data_file}, creating dummy data")
                dummy_data = np.random.randint(0, vocab_size, size=num_token * args.text_batch_size * 2, dtype=np.uint16)
            else:
                dummy_data = np.memmap(data_file, dtype=np.uint16, mode='r')
            _text_data_cache[data_file] = dummy_data
        else:
            dummy_data = _text_data_cache[data_file]
        
        data_len = len(dummy_data)
        if data_len <= num_token:
            logger.warning(f"Data length ({data_len}) is too small, using minimum required size")
            data_len = num_token + 1
            dummy_data = np.random.randint(0, vocab_size, size=data_len, dtype=np.uint16)

        ix = torch.randint(data_len - num_token, (args.text_batch_size,))
        # Pre-allocate tensors and use slice indexing
        x = torch.empty((args.text_batch_size, num_token), dtype=torch.long)
        y = torch.empty((args.text_batch_size, num_token), dtype=torch.long)
        for idx, i in enumerate(ix):
            x[idx] = torch.from_numpy(dummy_data[i:i+num_token].astype(np.int64))
            y[idx] = torch.from_numpy(dummy_data[i+1:i+1+num_token].astype(np.int64))
        return x, y

    # Load tokenizer config
    cfg = prepare_tokenizer_config(OmegaConf.load(args.cfg_path))
    quantizer_cfg = cfg.model
    lm_cfg = copy.deepcopy(cfg.lm)
    patch_size = int(quantizer_cfg.vit_enc_patch_size)

    try:
        dataset_train, dataset_val = get_fmri_data(args.dataset_dir,
                                                   data_cls=fMRITextDataset,
                                                   train_ratio=1,
                                                   val_ratio=0.1,
                                                   norm=args.norm, 
                                                   GPT_training=True, 
                                                   patch_size=patch_size,
                                                   next_time_mask=True,
                                                   descriptor_types=args.desc_type, 
                                                   lm_name=args.lm_name,
                                                   max_len=600,
                                                   )
        logger.info(f"Loaded datasets - Train: {len(dataset_train)}, Val: {len(dataset_val)}")
    except Exception as e:
        logger.error(f"Failed to load EEG datasets: {e}")
        raise

    # Create data loaders (no need for manual DistributedSampler)
    try:
        # Use a manual sampler for better control over RNG state with DeepSpeed
        # This avoids RNG state synchronization issues
        if args.deepspeed:
            # For DeepSpeed, use a sampler with a fixed generator to avoid RNG sync issues
            train_sampler = torch.utils.data.RandomSampler(
                dataset_train,
                generator=torch.Generator().manual_seed(args.seed)
            )
            data_loader_train = torch.utils.data.DataLoader(
                dataset_train,
                batch_size=args.fmri_batch_size,
                num_workers=16,
                pin_memory=True,
                drop_last=True,
                sampler=train_sampler
            )
        else:
            data_loader_train = torch.utils.data.DataLoader(
                dataset_train,
                batch_size=args.fmri_batch_size,
                num_workers=16,
                pin_memory=True,
                drop_last=True,
                shuffle=True
            )
        
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=int(1.5 * args.fmri_batch_size),
            num_workers=16,
            pin_memory=True,
            drop_last=True if args.deepspeed else False,  # drop_last for DeepSpeed to keep batch counts consistent across ranks
            shuffle=False
        )
        logger.info(f"Data loaders created - Train batches: {len(data_loader_train)}, Val batches: {len(data_loader_val)}")
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        raise

    try:
        tokenizer_ckpt_path = args.tokenizer_path
        logger.info(f"Loading tokenizer from: {tokenizer_ckpt_path}")
        tokenizer_checkpoint = torch.load(tokenizer_ckpt_path, map_location='cpu', weights_only=False)
        logger.info("Tokenizer checkpoint loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load tokenizer checkpoint: {e}")
        raise

    tokenizer = Tokenizer(cfg)
    load_tokenizer_state(tokenizer, tokenizer_checkpoint, logger)
    if getattr(tokenizer, '_loaded_from_mae_encoder', False) and (args.fmri_only_weight > 0 or args.text2fmri_weight > 0):
        logger.warning(
            "MAE checkpoints only provide encoder weights. fMRI-token prediction objectives "
            "will use the cfg-initialized tokenizer quantizer/codebook."
        )
    tokenizer_encoder = TokenizerEncoderForLM(copy.deepcopy(tokenizer.encoder))

    tokenizer.eval()
    logger.info("Tokenizer set to evaluation mode")
    
    # Clean up memory
    tokenizer_checkpoint = None
    logger.info("Tokenizer checkpoint memory cleaned up")


    # Model initialization
    # Check for both DeepSpeed and standard checkpoints
    ckpt_path = os.path.join(args.ckpt_dir, 'ckpt.pt')
    deepspeed_latest_dir = os.path.join(args.ckpt_dir, 'deepspeed_checkpoint_latest')
    deepspeed_ckpt_dirs = [d for d in os.listdir(args.ckpt_dir) if d.startswith('deepspeed_checkpoint_epoch_') and os.path.isdir(os.path.join(args.ckpt_dir, d))] if os.path.exists(args.ckpt_dir) else []
    
    if args.deepspeed and os.path.isdir(deepspeed_latest_dir) and args.resume:
        # Prefer the "latest" rolling checkpoint
        deepspeed_ckpt_path = deepspeed_latest_dir
        init_from = 'resume_deepspeed'
        logger.info(f"Resuming DeepSpeed training from latest checkpoint: {deepspeed_ckpt_path}")
    elif args.deepspeed and len(deepspeed_ckpt_dirs) > 0 and args.resume:
        # Backward compat: fall back to epoch-numbered directories
        latest_epoch = max([int(d.split('_')[-1]) for d in deepspeed_ckpt_dirs if d.split('_')[-1].isdigit()])
        deepspeed_ckpt_path = os.path.join(args.ckpt_dir, f'deepspeed_checkpoint_epoch_{latest_epoch}')
        init_from = 'resume_deepspeed'
        logger.info(f"Resuming DeepSpeed training from checkpoint: {deepspeed_ckpt_path}")
    elif os.path.exists(ckpt_path) and args.resume:
        init_from = 'resume'
        logger.info(f"Resuming training from checkpoint: {ckpt_path}")
    else:
        init_from = 'llm'
        logger.info("Initializing from pretrained weights")

    num_tokens = tokenizer_encoder.num_patches
    num_chans = int(quantizer_cfg.num_rois)

    iter_num = 0
    best_f2t_loss = float('inf')
    n_embd = tokenizer_encoder.embed_dim

    fmri_vocab_size = tokenizer.codebook_size if hasattr(tokenizer, 'codebook_size') else 0
    if fmri_vocab_size:
        if lm_cfg.fmri_vocab_size != fmri_vocab_size:
            logger.warning(f"Overriding lm.fmri_vocab_size from {lm_cfg.fmri_vocab_size} to tokenizer codebook size {fmri_vocab_size}")
        lm_cfg.fmri_vocab_size = fmri_vocab_size
    lm_cfg.num_fmri_tokens = num_tokens
    lm_cfg.base_model = args.lm_name

    # Override LoRA hyperparameters from CLI args (if provided)
    if args.no_lora:
        lm_cfg.peft_tune = False
    if args.lora_r is not None:
        lm_cfg.lora_r = args.lora_r
    if args.lora_alpha is not None:
        lm_cfg.lora_alpha = args.lora_alpha
    if args.lora_dropout is not None:
        lm_cfg.lora_dropout = args.lora_dropout
    if args.lora_target_modules is not None:
        lm_cfg.lora_target_modules = args.lora_target_modules

    # Convert to plain dict for consistent serialization in checkpoints
    model_args = OmegaConf.to_container(lm_cfg, resolve=True)

    try:
        if init_from == 'resume_deepspeed':
            # For DeepSpeed, load metadata first to get model_args
            metadata_path = os.path.join(deepspeed_ckpt_path, 'metadata.pt')
            if os.path.exists(metadata_path):
                metadata = torch.load(metadata_path, map_location='cpu', weights_only=False)
                ckpt_model_args = metadata['model_args']
                # Ensure it's a plain dict (older checkpoints may contain OmegaConf DictConfig)
                if hasattr(ckpt_model_args, '_metadata') or hasattr(ckpt_model_args, '_iter_ex'):
                    ckpt_model_args = OmegaConf.to_container(ckpt_model_args, resolve=True)
                model_args = dict(ckpt_model_args)
                iter_num = metadata['iter_num']
                start_epoch = metadata['epoch'] + 1
                best_f2t_loss = metadata.get('best_f2t_loss', float('inf'))
                logger.info(f"Loaded metadata from DeepSpeed checkpoint: epoch {start_epoch}, iter {iter_num}")
            else:
                raise Exception("No metadata found in DeepSpeed checkpoint")
            
            # Create model (weights will be loaded by accelerator.load_state later)
            # IMPORTANT: Disable peft_tune temporarily to avoid double-application in __init__
            gptconf = MultimodalConfig(**model_args)
            use_peft = model_args.get('peft_tune', False)
            temp_peft_flag = gptconf.peft_tune
            gptconf.peft_tune = False
            
            model = fMRILM(gptconf, tokenizer_encoder, args.tune_tokenizer, args.freeze_llm,
                            num_rois=quantizer_cfg.num_rois, n_embd=n_embd)
            
            # For DeepSpeed, we need to apply LoRA BEFORE prepare() if checkpoint has it
            # so the model structure matches when accelerator.load_state() is called
            if use_peft:
                logger.info("Checkpoint has LoRA - applying LoRA before prepare() to match structure")
                model.llm.apply_lora()
            
            # Restore flag for future operations
            gptconf.peft_tune = temp_peft_flag
            logger.info("Model created for DeepSpeed checkpoint loading")
            
        elif init_from == 'resume':
            logger.info(f"Resuming training from {args.ckpt_dir}")
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            ckpt_model_args = checkpoint['model_args']
            # Ensure it's a plain dict (older checkpoints may contain OmegaConf DictConfig)
            if hasattr(ckpt_model_args, '_metadata') or hasattr(ckpt_model_args, '_iter_ex'):
                ckpt_model_args = OmegaConf.to_container(ckpt_model_args, resolve=True)
            model_args = dict(ckpt_model_args)
            
            gptconf = MultimodalConfig(**model_args)
            # Check if the checkpoint was trained with PEFT
            use_peft = model_args.get('peft_tune', False)
            
            # IMPORTANT: Always create model with peft_tune=False to avoid __init__ application
            temp_peft_flag = gptconf.peft_tune
            gptconf.peft_tune = False
            
            model = fMRILM(gptconf, tokenizer_encoder, args.tune_tokenizer, args.freeze_llm, 
                            num_rois=quantizer_cfg.num_rois, n_embd=n_embd)

            # If checkpoint has LoRA, apply it BEFORE loading to match structure
            if use_peft:
                logger.info("Checkpoint has LoRA - applying LoRA before loading to match structure")
                model.llm.apply_lora()

            state_dict = checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
            # Load state dict
            model.load_state_dict(state_dict, strict=True)
            
            # Restore peft_tune flag for future operations
            gptconf.peft_tune = temp_peft_flag
            
            iter_num = checkpoint['iter_num']
            start_epoch = checkpoint['epoch'] + 1
            
            # Load best f2t loss if available
            if 'best_f2t_loss' in checkpoint:
                best_f2t_loss = checkpoint['best_f2t_loss']
            elif 'f2t_loss' in checkpoint:
                best_f2t_loss = checkpoint['f2t_loss']
            
            logger.info(f"Resumed from epoch {start_epoch}, iteration {iter_num}")
            if use_peft:
                logger.info("Checkpoint was trained with PEFT/LoRA - LoRA weights loaded")
        elif init_from.startswith('llm'):
            logger.info(f"Initializing from pretrained weights: {lm_cfg.base_model}")
            # gptconf = GPTConfig(**model_args)
            gptconf = MultimodalConfig(**model_args)
            model = fMRILM(gptconf, tokenizer_encoder, args.tune_tokenizer, args.freeze_llm,
                            num_rois=quantizer_cfg.num_rois, n_embd=n_embd)
            start_epoch = 0
            logger.info("Model initialized from LLM")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise
     
    if lm_cfg.get('peft_tune', False):
        assert model.llm.base_model.peft_config is not None, "LoRA config not found in model for PEFT tuning"

    num_params = model.get_num_params()
    vocab_size = model.llm.original_vocab_size
    logger.info(f'Model parameters: {num_params:,} ({num_params/1e6:.2f}M)')

    # Optimizer
    try:
        optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (args.beta1, args.beta2), accelerator.device.type)
        if init_from == 'resume':
            # For non-DeepSpeed, load optimizer state here
            # checkpoint variable is guaranteed to exist from the 'resume' block above
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("Optimizer state loaded from checkpoint")
        logger.info("Optimizer configured successfully")
    except Exception as e:
        logger.error(f"Failed to configure optimizer: {e}")
        raise

    # Prepare everything with accelerator
    try:
        # When using DeepSpeed, cannot prepare multiple models with same accelerator
        # Only prepare the main model, keep tokenizer separate
        if args.deepspeed:
            model, optimizer, data_loader_train, data_loader_val = accelerator.prepare(
                model, optimizer, data_loader_train, data_loader_val
            )
            # Move tokenizer to device manually (not wrapped by accelerator)
            tokenizer = tokenizer.to(accelerator.device)
            tokenizer.eval()
            logger.info("DeepSpeed: Model prepared, tokenizer moved to device separately")
            
            # Load DeepSpeed checkpoint after preparing
            if init_from == 'resume_deepspeed':
                accelerator.load_state(deepspeed_ckpt_path)
                logger.info(f"DeepSpeed checkpoint loaded from {deepspeed_ckpt_path}")
        else:
            model, tokenizer, optimizer, data_loader_train, data_loader_val = accelerator.prepare(
                model, tokenizer, optimizer, data_loader_train, data_loader_val
            )
            if accelerator.num_processes > 1:
                model._set_static_graph()
                logger.warning('set model with static computational graph')
            # Move tokenizer to device
            tokenizer = tokenizer.to(accelerator.device)
            tokenizer.eval()
            logger.info("Model and tokenizer prepared with accelerator")
        
        logger.info(f"Tokenizer on device: {accelerator.device}")
    except Exception as e:
        logger.error(f"Failed to prepare with accelerator: {e}")
        raise

    # Learning rate scheduler: cosine with optional linear warmup
    try:
        total_steps = args.epochs * len(data_loader_train) // args.gradient_accumulation_steps
        num_training_steps_per_epoch = len(data_loader_train)
        if args.warmup_epochs > 0:
            warmup_steps = args.warmup_epochs * num_training_steps_per_epoch // args.gradient_accumulation_steps
            warmup_scheduler = LinearLR(optimizer, start_factor=1e-2, total_iters=warmup_steps)
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=args.min_lr)
            lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
        else:
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.min_lr)

        if init_from == 'resume' and 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            logger.info("LR scheduler state loaded from checkpoint")
            del checkpoint  # free up memory (moved here from optimizer block)

        logger.info(f"Learning rate scheduler configured - steps per epoch: {num_training_steps_per_epoch}, warmup epochs: {args.warmup_epochs}")
    except Exception as e:
        logger.error(f"Failed to configure learning rate scheduler: {e}")
        raise

    # Get text tokenizer - handle both DeepSpeed and DDP wrapping
    if args.deepspeed:
        # DeepSpeed wraps model in DeepSpeedEngine
        text_tokenizer = model.module.llm.tokenizer if hasattr(model, 'module') else model.llm.tokenizer
    elif accelerator.num_processes == 1:
        text_tokenizer = model.llm.tokenizer
    else:
        text_tokenizer = model.module.llm.tokenizer

    # Training loop
    try:
        X_text, Y_text = get_batch('train', tokenizer=text_tokenizer)
        X_text, Y_text = X_text.to(accelerator.device), Y_text.to(accelerator.device)
        logger.info("Initial text batch loaded")
    except Exception as e:
        logger.error(f"Failed to load initial text batch: {e}")
        raise
        
    def compute_forward_pass(X_fmri, fmri_gpt_mask, text_input_ids, text_attention_mask, X_text, Y_text, Y_fmri=None):
        """Helper function to compute forward pass for all training objectives"""
        log_fmri, log_text, log_f2t = None, None, None
        loss_fmri, loss_text, loss_f2t = 0, 0, 0
        
        if args.fmri_only_weight > 0:
            assert Y_fmri is not None, "Y_fmri must be provided for fmri_only training"
            loss_fmri, log_fmri, _ = model(X_fmri, Y_fmri, None, None, fmri_gpt_mask)
        
        if args.text_only_weight > 0:
            loss_text, log_text, _ = model(None, None, X_text, Y_text)
        
        if args.fmri2text_weight > 0:  # fMRI -> text, description (NOT semantic QA)
            Y_text_input_ids = text_input_ids.clone()
            # make all padded token -1
            Y_text_input_ids[text_attention_mask == 0] = -1
            Y_text_input_ids[:, :-1] = Y_text_input_ids[:, 1:].clone()  # next token prediction
            Y_fmri_f2t = torch.full((X_fmri.size(0), num_tokens), fill_value=-1-vocab_size, device=accelerator.device)
            
            # Pass raw masks as tuple; _prepare_inputs will handle combining and interleaving
            loss_f2t, log_f2t, _ = model(X_fmri, Y_fmri_f2t, text_input_ids, Y_text_input_ids, (fmri_gpt_mask, text_attention_mask))
        
        if args.text2fmri_weight > 0:
            raise Exception("text2fmri not supported yet")
        
        # Compute total loss
        loss = args.fmri_only_weight * loss_fmri + args.text_only_weight * loss_text + args.fmri2text_weight * loss_f2t
        
        return loss, log_fmri, log_text, log_f2t
    
    local_iter_num = 0
    best_sbert_sim = 0.
    logger.info(f"Starting training from epoch {start_epoch} to {args.epochs}")
    progress_bar = tqdm(range(start_epoch, args.epochs), desc="Epochs", disable=not accelerator.is_main_process)

    for epoch in progress_bar:
        model.train()
        
        epoch_log, log2 = {}, None
        for step, batch in tqdm(enumerate(data_loader_train), disable=not accelerator.is_main_process, total=len(data_loader_train), desc="Training"):
            X_fmri, fmri_gpt_mask, text_input_ids, text_attention_mask = batch
            fmri_gpt_mask = fmri_gpt_mask.to(X_fmri.dtype)
            Y_fmri = None

            if args.fmri_only_weight > 0:
                with torch.no_grad():
                    Y_fmri = torch.full((X_fmri.size(0), num_tokens),  
                                        fill_value=-1 if gptconf.use_fmri_lm_head else -1-vocab_size,
                                        device=accelerator.device)
                    
                    # Handle both DeepSpeed (not wrapped) and DDP (wrapped with .module)
                    if args.deepspeed or accelerator.num_processes == 1:
                        codebook_indices = tokenizer.get_codebook_indices(X_fmri)
                    else:
                        codebook_indices = tokenizer.module.get_codebook_indices(X_fmri)

                    # next timestamp prediction; use the ROIs from the prev timestamp to predict ROIs for the next timestamp
                    # TODO this assume a 1v1 correspondance between the ROI from adjacent timestamp, try generalize this?
                    if num_tokens == num_chans:
                        Y_fmri = codebook_indices
                    else:
                        for i in range(len(codebook_indices)):
                            Y_fmri[i, :num_tokens - num_chans] = codebook_indices[i, num_chans:num_tokens]

            # DeepSpeed handles gradient accumulation internally, don't use accelerator.accumulate()
            if args.deepspeed:
                # Forward pass
                loss, log_fmri, log_text, log_f2t = compute_forward_pass(
                    X_fmri, fmri_gpt_mask, text_input_ids, text_attention_mask, X_text, Y_text, Y_fmri
                )

                # Check for NaN or infinite loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Invalid loss detected at epoch {epoch}, step {step}: {loss.item()}")
                    raise ValueError("Invalid loss detected")

                # Backward pass - DeepSpeed handles gradient accumulation
                accelerator.backward(loss)
                
                # Gradient clipping and optimizer step
                if args.grad_clip != 0.0:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
            else:
                # Standard training with accelerator.accumulate() for non-DeepSpeed
                with accelerator.accumulate(model):
                    # Forward pass
                    loss, log_fmri, log_text, log_f2t = compute_forward_pass(
                        X_fmri, fmri_gpt_mask, text_input_ids, text_attention_mask, X_text, Y_text, Y_fmri
                    )

                    # Check for NaN or infinite loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"Invalid loss detected at epoch {epoch}, step {step}: {loss.item()}")
                        raise ValueError("Invalid loss detected")

                    # Backward pass
                    accelerator.backward(loss)
                    
                    # Gradient clipping
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.grad_clip != 0.0:
                            accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                        
                        optimizer.step()
                        optimizer.zero_grad()
                        lr_scheduler.step()
            
            if args.text_only_weight > 0:
                # Get next text batch
                X_text, Y_text = get_batch('train', tokenizer=text_tokenizer)
                X_text, Y_text = X_text.to(accelerator.device), Y_text.to(accelerator.device)
            
            # Aggregate logs - use whichever loss objective is active
            active_logs = [log for log in [log_fmri, log_text, log_f2t] if log is not None]
            if len(active_logs) > 0:
                total_loss = sum(log.get('train/loss', 0) for log in active_logs)
                total_accuracy = sum(log.get('train/accuracy', 0) for log in active_logs) / len(active_logs)
            else:
                total_loss = 0
                total_accuracy = 0

            log = {
                'total_loss': total_loss, 
                'fmri_loss': log_fmri['train/loss'] if log_fmri else 0,
                'text_loss': log_text['train/loss'] if log_text else 0,
                'f2t_loss': log_f2t['train/loss'] if log_f2t else 0,
                'fmri_acc': log_fmri['train/accuracy'] if log_fmri else 0,
                'text_acc': log_text['train/accuracy'] if log_text else 0,
                'f2t_acc': log_f2t['train/accuracy'] if log_f2t else 0,
            }
            if epoch_log == {}: 
                epoch_log = log
            else: 
                epoch_log = {k: epoch_log[k] + log[k] if not torch.is_tensor(log[k]) else epoch_log[k] + log[k].detach().item() for k in log}

            # Logging
            if (iter_num + 1) % args.log_interval == 0:
                
                # logger.info(f"Epoch {epoch + 1} Step [{step + 1}/{num_training_steps_per_epoch}]: "
                #             f"Total Loss {total_loss:.4f}, fMRI Loss {log['fmri_loss']:.4f}, Text Loss {log['text_loss']:.4f}, "
                #             f"LR {optimizer.param_groups[0]['lr']:.2e}")
                
                if args.wandb_log and accelerator.is_main_process:
                    run.log({
                        "iter": iter_num,
                        "train_iter/total_loss": total_loss,
                        "train_iter/fmri_loss": log['fmri_loss'],
                        "train_iter/text_loss": log['text_loss'],
                        "train_iter/f2t_loss": log['f2t_loss'],
                        "train_iter/fmri_accuracy": log['fmri_acc'],
                        "train_iter/text_accuracy": log['text_acc'],
                        "train_iter/f2t_accuracy": log['f2t_acc'],
                        "train_iter/lr": optimizer.param_groups[0]['lr']
                    })

            iter_num += 1
            local_iter_num += 1

        # gather logs from processes
        epoch_log = {k: torch.tensor(v, device=accelerator.device) / (step + 1) for k, v in log.items()}
        epoch_log = accelerator.gather(epoch_log)
        epoch_log = {k: epoch_log[k].mean().item() for k in epoch_log}
        progress_bar.set_description(f"Epoch {epoch + 1}: fmri_loss - {epoch_log['fmri_loss']:.4f}, text_loss - {epoch_log['text_loss']:.4f}")

        if args.wandb_log and accelerator.is_main_process:
            run.log({
                "epoch": epoch + 1,
                "train/total_loss": epoch_log['total_loss'],
                "train/fmri_loss": epoch_log['fmri_loss'],
                "train/text_loss": epoch_log['text_loss'],
                "train/fmri_accuracy": epoch_log['fmri_acc'],
                "train/text_accuracy": epoch_log['text_acc'],
                "lr": optimizer.param_groups[0]['lr'],
            })

        # Validation
        if (epoch + 1) % args.val_interval == 0:
            val_start_time = time.time()
            sbert_sim, cider, all_metrics = evaluate(model, tokenizer, data_loader_val, accelerator, args, logger, vocab_size, num_chans, num_tokens, gptconf)
            val_time = time.time() - val_start_time
            
            if accelerator.is_main_process:
                logger.info('=' * 50)
                logger.info(f"Epoch {epoch + 1} Validation Results:")
                logger.info(f"  BLEU-1: {all_metrics.get('BLEU-1', 0):.4f}")
                logger.info(f"  BLEU-2: {all_metrics.get('BLEU-2', 0):.4f}")
                logger.info(f"  BLEU-3: {all_metrics.get('BLEU-3', 0):.4f}")
                logger.info(f"  BLEU-4: {all_metrics.get('BLEU-4', 0):.4f}")
                logger.info(f"  CIDEr: {cider:.4f}")
                logger.info(f"  BERTScore F1: {all_metrics.get('BERTScore_f1', 0):.4f}")
                logger.info(f"  SBERT Similarity: {sbert_sim:.4f}")
                logger.info(f"  ROUGE-L F1: {all_metrics.get('ROUGE-L_f1', 0):.4f}")
                logger.info(f"  Validation time: {val_time:.2f}s")
                logger.info('=' * 50)
            
            if args.wandb_log and accelerator.is_main_process:
                run.log({
                    "epoch": epoch + 1,
                    "val/BLEU-1": all_metrics.get('BLEU-1', 0),
                    "val/BLEU-2": all_metrics.get('BLEU-2', 0),
                    "val/BLEU-3": all_metrics.get('BLEU-3', 0),
                    "val/BLEU-4": all_metrics.get('BLEU-4', 0),
                    "val/CIDEr": cider,
                    "val/BERTScore_precision": all_metrics.get('BERTScore_precision', 0),
                    "val/BERTScore_recall": all_metrics.get('BERTScore_recall', 0),
                    "val/BERTScore_f1": all_metrics.get('BERTScore_f1', 0),
                    "val/SBERT_similarity": sbert_sim,
                    "val/ROUGE-L_precision": all_metrics.get('ROUGE-L_precision', 0),
                    "val/ROUGE-L_recall": all_metrics.get('ROUGE-L_recall', 0),
                    "val/ROUGE-L_f1": all_metrics.get('ROUGE-L_f1', 0),
                })
            
        # Save checkpoint
        # Ensure all processes are synchronized before checkpoint saving
        # This is safe because it's OUTSIDE any conditional blocks
        accelerator.wait_for_everyone()
        
        if args.save_ckpt:
            if args.deepspeed:
                # For DeepSpeed, use accelerator.save_state() which handles ZeRO stages correctly
                # This will save the full model state including optimizer and scheduler states
                # accelerator.save_state() internally handles synchronization across all processes
                # Overwrite the same "latest" directory each epoch to avoid disk bloat
                checkpoint_dir = os.path.join(args.ckpt_dir, 'deepspeed_checkpoint_latest')
                accelerator.save_state(checkpoint_dir)
                
                # Only main process saves metadata
                if accelerator.is_main_process:
                    # Save additional metadata that accelerator.save_state doesn't save
                    metadata = {
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'epoch': epoch,
                        'total_loss': epoch_log['total_loss'],
                        'fmri_loss': epoch_log['fmri_loss'],
                        'text_loss': epoch_log['text_loss'],
                        'f2t_loss': epoch_log['f2t_loss'],
                        'fmri_acc': epoch_log['fmri_acc'],
                        'text_acc': epoch_log['text_acc'],
                        'f2t_acc': epoch_log['f2t_acc'],
                        'best_f2t_loss': best_f2t_loss,
                    }
                    metadata_path = os.path.join(checkpoint_dir, 'metadata.pt')
                    torch.save(metadata, metadata_path)
                    logger.info(f"DeepSpeed checkpoint saved: {checkpoint_dir}")

                    if lm_cfg.get('peft_tune', False):
                        unwrapped_model = accelerator.unwrap_model(model)
                        lora_dir = os.path.join(checkpoint_dir, 'lora_adapter')
                        unwrapped_model.llm.save_pretrained_lora(lora_dir)
                        logger.info(f"LoRA adapter saved: {lora_dir}")
                                
                # Save periodic checkpoints
                if (epoch + 1) % args.save_ckpt_freq == 0:
                    epoch_checkpoint_dir = os.path.join(args.ckpt_dir, f'deepspeed_checkpoint_epoch_{epoch}_saved')
                    accelerator.save_state(epoch_checkpoint_dir)
                    if accelerator.is_main_process:
                        torch.save(metadata, os.path.join(epoch_checkpoint_dir, 'metadata.pt'))
                        logger.info(f"DeepSpeed epoch checkpoint saved: {epoch_checkpoint_dir}")
                
                # Save best checkpoint based on f2t_loss (only when f2t training is enabled)
                if args.fmri2text_weight > 0:
                    current_f2t_loss = epoch_log['f2t_loss']
                    if current_f2t_loss < best_f2t_loss:
                        best_f2t_loss = current_f2t_loss
                        if accelerator.is_main_process:
                            metadata['best_f2t_loss'] = best_f2t_loss
                        best_f2t_checkpoint_dir = os.path.join(args.ckpt_dir, f'deepspeed_checkpoint_best_f2t')
                        accelerator.save_state(best_f2t_checkpoint_dir)
                        if accelerator.is_main_process:
                            torch.save(metadata, os.path.join(best_f2t_checkpoint_dir, 'metadata.pt'))
                            logger.info(f"Best f2t DeepSpeed checkpoint saved (f2t_loss={best_f2t_loss:.4f}): {best_f2t_checkpoint_dir}")
                
                # Save best checkpoint based on SBERT similarity score
                if (epoch + 1) % args.val_interval == 0 and sbert_sim > best_sbert_sim:
                    best_sbert_sim = sbert_sim
                    best_checkpoint_dir = os.path.join(args.ckpt_dir, f'deepspeed_checkpoint_best_sbert')
                    accelerator.save_state(best_checkpoint_dir)
                    if accelerator.is_main_process:
                        torch.save(metadata, os.path.join(best_checkpoint_dir, 'metadata.pt'))
                        logger.info(f"Best DeepSpeed checkpoint saved (SBERT Similarity={sbert_sim:.4f}): {best_checkpoint_dir}")
            else:
                # Standard checkpoint saving for non-DeepSpeed training
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    checkpoint = {
                        'model': unwrapped_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'epoch': epoch,
                        'total_loss': epoch_log['total_loss'],
                        'fmri_loss': epoch_log['fmri_loss'],
                        'text_loss': epoch_log['text_loss'],
                        'f2t_loss': epoch_log['f2t_loss'],
                        'fmri_acc': epoch_log['fmri_acc'],
                        'text_acc': epoch_log['text_acc'],
                        'f2t_acc': epoch_log['f2t_acc'],
                        'best_f2t_loss': best_f2t_loss,
                    }
                    checkpoint_path = os.path.join(args.ckpt_dir, f'ckpt.pt')
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Checkpoint saved: {checkpoint_path}")
                    
                    # Save LoRA adapter separately if using PEFT
                    if lm_cfg.get('peft_tune', False):
                        lora_dir = os.path.join(args.ckpt_dir, 'lora_adapter')
                        unwrapped_model.llm.save_pretrained_lora(lora_dir)

                    if (epoch + 1) % args.save_ckpt_freq == 0:
                        epoch_checkpoint_path = os.path.join(args.ckpt_dir, f'ckpt-{epoch}.pt')
                        torch.save(checkpoint, epoch_checkpoint_path)
                        logger.info(f"Epoch checkpoint saved: {epoch_checkpoint_path}")
                        
                        # Save LoRA adapter for epoch checkpoint
                        if lm_cfg.get('peft_tune', False):
                            lora_dir = os.path.join(args.ckpt_dir, f'lora_adapter_epoch_{epoch}')
                            unwrapped_model.llm.save_pretrained_lora(lora_dir)

                    # Save best checkpoint based on f2t_loss (only when f2t training is enabled)
                    if args.fmri2text_weight > 0:
                        current_f2t_loss = epoch_log['f2t_loss']
                        if current_f2t_loss < best_f2t_loss:
                            best_f2t_loss = current_f2t_loss
                            checkpoint['best_f2t_loss'] = best_f2t_loss
                            best_f2t_checkpoint_path = os.path.join(args.ckpt_dir, f'ckpt-best_f2t.pt')
                            torch.save(checkpoint, best_f2t_checkpoint_path)
                            logger.info(f"Best f2t checkpoint saved (f2t_loss={best_f2t_loss:.4f}): {best_f2t_checkpoint_path}")
                            
                            # Save LoRA adapter for best f2t checkpoint
                            if lm_cfg.get('peft_tune', False):
                                lora_dir = os.path.join(args.ckpt_dir, 'lora_adapter_best_f2t')
                                unwrapped_model.llm.save_pretrained_lora(lora_dir)

                    # Save best checkpoint based on SBERT similarity score
                    if (epoch + 1) % args.val_interval == 0 and sbert_sim > best_sbert_sim:
                        best_sbert_sim = sbert_sim
                        best_checkpoint_path = os.path.join(args.ckpt_dir, f'ckpt-best_sbert.pt')
                        torch.save(checkpoint, best_checkpoint_path)
                        logger.info(f"Best checkpoint saved (SBERT Similarity={sbert_sim:.4f}): {best_checkpoint_path}")
                        
                        # Save LoRA adapter for best sbert checkpoint
                        if lm_cfg.get('peft_tune', False):
                            lora_dir = os.path.join(args.ckpt_dir, 'lora_adapter_best_sbert')
                            unwrapped_model.llm.save_pretrained_lora(lora_dir)

    if args.wandb_log and accelerator.is_main_process:
        run.finish()
        logger.info("W&B tracking ended")

@torch.no_grad()
def evaluate(model, tokenizer, dataloader, accelerator, args, logger, vocab_size, num_chans, num_tokens, gptconf):
    """Evaluate with fMRI -> text
    We will need to use left padding due to autoregressive generation and constraint of KV cache
    """
    model.eval()
    
    # Initialize text generation metrics
    text_metrics = TextGenerationMetrics(device=accelerator.device)
    
    # Get the text tokenizer from the model - handle DeepSpeed wrapping
    if args.deepspeed:
        text_tokenizer = model.module.llm.tokenizer if hasattr(model, 'module') else model.llm.tokenizer
    elif accelerator.num_processes == 1:
        text_tokenizer = model.llm.tokenizer
    else:
        text_tokenizer = model.module.llm.tokenizer
    
    # Lists to store generated and reference texts
    all_generated_texts = []
    all_reference_texts = []
    
    logger.info("Starting validation...")
    
    for batch_idx, batch in enumerate(dataloader):
        X_fmri, fmri_gpt_mask, text_input_ids, text_attention_mask, reference_texts = batch
        # Let accelerator handle mixed precision casting automatically
        fmri_gpt_mask = fmri_gpt_mask.to(X_fmri.dtype)

        # Pass raw masks as tuple; _prepare_inputs will handle combining and interleaving
        attention_mask = (fmri_gpt_mask, text_attention_mask)
        
        Y_text_input_ids = text_input_ids.clone()
        # make all padded token -1
        Y_text_input_ids[text_attention_mask == 0] = -1
        Y_text_input_ids[:, :-1] = Y_text_input_ids[:, 1:].clone()

        with accelerator.autocast():
            # Generate texts - handle DeepSpeed wrapping
            if args.deepspeed:
                # DeepSpeed always wraps in .module
                if hasattr(model, 'module'):
                    generated_texts = model.module.generate(X_fmri, text_input_ids, attention_mask, max_new_tokens=200, text_gen=True, accelerator=accelerator)
                else:
                    generated_texts = model.generate(X_fmri, text_input_ids, attention_mask, max_new_tokens=200, text_gen=True, accelerator=accelerator)
            elif accelerator.num_processes == 1:
                generated_texts = model.generate(X_fmri, text_input_ids, attention_mask, max_new_tokens=200, text_gen=True)
            else:
                generated_texts = model.module.generate(X_fmri, text_input_ids, attention_mask, max_new_tokens=200, text_gen=True, accelerator=accelerator)
            
            # remove the prefix part in generated_texts
            cleaned_generated_texts = []
            for i in range(len(generated_texts)):
                gen_text = generated_texts[i].split(' ')
                ans_idx = [i for i, w in enumerate(gen_text) if w.lower().startswith('answer:')][0]
                cleaned_generated_texts.append(' '.join(gen_text[ans_idx + 1:]).strip())
                
            # Collect all texts across batches
            all_generated_texts.extend(cleaned_generated_texts)
            all_reference_texts.extend(reference_texts)

    
    # Gather texts from all processes (only main process gets the full data)
    if accelerator.num_processes > 1:
        all_generated_texts = accelerator.gather_for_metrics(all_generated_texts)
        all_reference_texts = accelerator.gather_for_metrics(all_reference_texts)
    
    # Compute metrics only on main process
    metrics = {}
    if accelerator.is_main_process:
        logger.info(f"Computing metrics on {len(all_generated_texts)} samples...")
        
        # Compute all text generation metrics
        metrics = text_metrics.compute_all_metrics(
            hypotheses=all_generated_texts,
            references=all_reference_texts,
            sbert_batch_size=32
        )
        
        logger.info("Validation Metrics:")
        logger.info(f"  BLEU-1: {metrics.get('BLEU-1', 0):.4f}")
        logger.info(f"  BLEU-2: {metrics.get('BLEU-2', 0):.4f}")
        logger.info(f"  BLEU-3: {metrics.get('BLEU-3', 0):.4f}")
        logger.info(f"  BLEU-4: {metrics.get('BLEU-4', 0):.4f}")
        logger.info(f"  CIDEr: {metrics.get('CIDEr', 0):.4f}")
        logger.info(f"  BERTScore F1: {metrics.get('BERTScore_f1', 0):.4f}")
        logger.info(f"  SBERT Similarity: {metrics.get('SBERT_similarity', 0):.4f}")
        logger.info(f"  ROUGE-L F1: {metrics.get('ROUGE-L_f1', 0):.4f}")
    
    # Synchronize all processes before returning to training
    model.train()
    accelerator.wait_for_everyone()
    
    # Return the main metrics (only main process has real values, others return 0)
    return metrics.get('SBERT_similarity', 0.0), metrics.get('CIDEr', 0.0), metrics

def get_args():
    def list_of_strs(arg):
        return arg.split(',')
    parser = argparse.ArgumentParser('NeuroLM training script', add_help=False)
    parser.add_argument('--dataset_dir', default=['data/UKB/fmri/TianS3/'], type=list_of_strs, help='path to the dataset directory')
    parser.add_argument('--norm', type=str, default='robust', help='normalization method')
    parser.add_argument('--tokenizer_path', default='checkpoints/tokenizer/default/UKB-robust/tok_fsq-vit_base_p160_newTok/ckpt-best.pt', help='path where tokenizer is')
    parser.add_argument('--ckpt_dir', default='tmp', help='path where to save. If set to "tmp" (default), will auto-generate based on training config')
    parser.add_argument('--ckpt_postfix', default='', help='optional postfix to add to checkpoint directory name')
    parser.add_argument('--resume', default=False, action='store_true', help='resume from the latest checkpoint')
    parser.add_argument('--log_interval', default=15, type=int)
    parser.add_argument('--wandb_log', default=False, action='store_true')
    parser.add_argument('--wandb_project', default='BrainFM_pretrain')
    parser.add_argument('--wandb_runname', default='pretrain', help='W&B run name. If set to "pretrain" (default), will auto-generate based on training config')

    # training args
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--fmri_batch_size', default=1, type=int)
    parser.add_argument('--text_batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--warmup_epochs', default=0, type=int)
    parser.add_argument('--save_ckpt', default=False, action=argparse.BooleanOptionalAction, help='whether to save checkpoints')
    parser.add_argument('--save_ckpt_freq', default=15, type=int)
    parser.add_argument('--tune_tokenizer', action='store_true', help='whether to finetune the tokenizer during training', default=False)
    parser.add_argument('--freeze_llm', action='store_true', help='whether to freeze the LLM during training', default=False)

    parser.add_argument('--desc_type', type=list_of_strs, default=['fc', 'ica'], )
    parser.add_argument('--lm_name', type=str, default='Qwen/Qwen3-0.6B', help='language model name')

    parser.add_argument('--fmri_only_weight', type=float, help='weight for fMRI -> fMRI NTP objective', default=0)
    parser.add_argument('--text_only_weight', type=float, help='weight for text -> text NTP objective', default=0)
    parser.add_argument('--fmri2text_weight', type=float, help='weight for fMRI -> text NTP objective', default=1)
    parser.add_argument('--text2fmri_weight', type=float, help='weight for text -> fMRI NTP objective', default=0)

    parser.add_argument('--cfg_path', type=str, default='configs/vit_base_p160_newTok.yaml', help='path to the tokenizer config file',)

    # LoRA hyperparameters (override YAML defaults when provided)
    parser.add_argument('--no_lora', action='store_true', default=False, help='disable LoRA and fine-tune the entire model')
    parser.add_argument('--lora_r', type=int, default=None, help='LoRA rank (default: use YAML config)')
    parser.add_argument('--lora_alpha', type=int, default=None, help='LoRA alpha scaling factor (default: use YAML config)')
    parser.add_argument('--lora_dropout', type=float, default=None, help='LoRA dropout rate (default: use YAML config)')
    parser.add_argument('--lora_target_modules', type=list_of_strs, default=None,
                        help='Comma-separated list of target modules for LoRA, e.g. "q_proj,v_proj" (default: use YAML config)')

    parser.add_argument('--learning_rate', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='clip gradients at this value, or disable if == 0.0')
    parser.add_argument('--decay_lr', default=True, action='store_false')
    parser.add_argument('--seed', default=1337, type=int)
    parser.add_argument('--val_interval', default=100, type=int, help='number of epochs between validations')

    # DeepSpeed optimization args
    parser.add_argument('--deepspeed', action='store_true', default=False, help='use DeepSpeed for training')
    parser.add_argument('--zero_stage', type=int, default=2, choices=[0, 1, 2, 3], help='DeepSpeed ZeRO optimization stage')
    parser.add_argument('--offload_optimizer', action='store_true', default=False, help='offload optimizer states to CPU')
    parser.add_argument('--offload_params', action='store_true', default=False, help='offload model parameters to CPU (only for ZeRO-3)')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)
