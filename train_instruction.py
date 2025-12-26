import os
import time
import math
import argparse
import logging
import sys
import wandb
from tqdm import tqdm
from omegaconf import OmegaConf
import copy
import colorlog
import json
from collections import OrderedDict
from datetime import datetime
import shutil

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from sklearn.metrics import accuracy_score, roc_auc_score

from model_mindlm import MindLM
from quantizers import *
from model_gpt import MultimodalConfig
from dataset import get_fmri_data_inst
from utils import get_metrics, get_allowed_token_id


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
        'pretrained_path': args.pretrained_ckpt,
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


def create_timestamped_dir(base_dir, add_timestamp=True):
    """Create a timestamped directory and return the path"""
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_dir = f"{base_dir}_{timestamp}"
    else:
        timestamped_dir = base_dir
    
    os.makedirs(timestamped_dir, exist_ok=True)
    return timestamped_dir


def save_configurations(ckpt_dir, args, cfg_path, logger):
    """Save all configurations to the checkpoint directory"""
    config_dir = os.path.join(ckpt_dir, 'configs')
    os.makedirs(config_dir, exist_ok=True)
    
    # Save command-line arguments
    args_file = os.path.join(config_dir, 'args.json')
    args_dict = vars(args)
    with open(args_file, 'w') as f:
        json.dump(args_dict, f, indent=2)
    logger.info(f"Arguments saved to: {args_file}")
    
    # Save command-line string (reconstructed)
    cmd_file = os.path.join(config_dir, 'command.txt')
    cmd_str = "python train_instruction.py " + " \\\n  ".join([
        f"--{k}={v}" if not isinstance(v, bool) else (f"--{k}" if v else f"--no-{k}")
        for k, v in args_dict.items()
    ])
    with open(cmd_file, 'w') as f:
        f.write(cmd_str)
    logger.info(f"Command saved to: {cmd_file}")
    
    # Copy the YAML configuration file
    if os.path.exists(cfg_path):
        yaml_dest = os.path.join(config_dir, os.path.basename(cfg_path))
        shutil.copy2(cfg_path, yaml_dest)
        logger.info(f"YAML config copied to: {yaml_dest}")
    
    # Save the actual shell script command if available from environment
    shell_cmd_file = os.path.join(config_dir, 'shell_command.sh')
    try:
        # Try to get the actual command that was run
        import subprocess
        result = subprocess.run(['ps', '-p', str(os.getppid()), '-o', 'args='], 
                              capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            with open(shell_cmd_file, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write("# Parent process command:\n")
                f.write(result.stdout.strip() + "\n")
            logger.info(f"Shell command saved to: {shell_cmd_file}")
    except Exception as e:
        logger.debug(f"Could not save shell command: {e}")
    
    # Save a summary file with key information
    summary_file = os.path.join(config_dir, 'experiment_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Experiment Summary\n")
        f.write(f"=" * 80 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Checkpoint Directory: {ckpt_dir}\n")
        f.write(f"Config File: {cfg_path}\n")
        f.write(f"\n" + "=" * 80 + "\n")
        f.write(f"Key Arguments:\n")
        f.write(f"  Datasets: {args.datasets}\n")
        f.write(f"  Quantizer: {args.quantizer}\n")
        f.write(f"  Batch Size (fMRI): {args.fmri_batch_size}\n")
        f.write(f"  Batch Size (Text): {args.text_batch_size}\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Learning Rate: {args.learning_rate}\n")
        f.write(f"  Pretrained Checkpoint: {args.pretrained_ckpt}\n")
        f.write(f"  Use Random Prompt: {args.use_random_prompt}\n")
        f.write(f"  Add Source Info: {args.add_src_info}\n")
        f.write(f"  Add Description: {args.add_desc}\n")
    logger.info(f"Experiment summary saved to: {summary_file}")
    
    return config_dir


def main(args):
    # Create timestamped checkpoint directory (unless resuming)
    if not args.resume:
        args.ckpt_dir = create_timestamped_dir(args.ckpt_dir, add_timestamp=not args.no_timestamp)
    
    # Setup logging
    os.makedirs(args.ckpt_dir, exist_ok=True)
    log_file = os.path.join(args.ckpt_dir, 'training.log') if args.ckpt_dir and args.save_ckpt else None
    logger = setup_logging(log_file=log_file)
    
    # Save all configurations
    if not args.resume:
        save_configurations(args.ckpt_dir, args, args.cfg_path, logger)
    
    # Initialize Accelerator
    try:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            log_with="wandb" if args.wandb_log else None,
            kwargs_handlers=[ddp_kwargs]
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
            run = wandb.init(project=args.wandb_project, name=args.wandb_runname, dir='./wandb', config=vars(args), group=args.wandb_group)
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
    def get_batch(split, num_token):
        if split == 'train':
            data_file = os.path.join(data_dir, 'train.bin')
        else:
            data_file = os.path.join(data_dir, 'val.bin')
        
        if not os.path.exists(data_file):
            logger.warning(f"Text data file not found: {data_file}, creating dummy data")
            # Create dummy data if file doesn't exist
            dummy_data = np.random.randint(0, 50257, size=num_token * args.text_batch_size * 2, dtype=np.uint16)
        else:
            dummy_data = np.memmap(data_file, dtype=np.uint16, mode='r')
        
        data_len = len(dummy_data)
        if data_len <= num_token:
            logger.warning(f"Data length ({data_len}) is too small, using minimum required size")
            data_len = num_token + 1
            dummy_data = np.random.randint(0, 50257, size=data_len, dtype=np.uint16)

        ix = torch.randint(data_len - num_token, (args.text_batch_size,))
        x = torch.stack([torch.from_numpy((dummy_data[i:i+num_token]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((dummy_data[i+1:i+1+num_token]).astype(np.int64)) for i in ix])
        return x, y

    # Load tokenizer
    model_cfg = OmegaConf.load(args.cfg_path).model
    quantizer_cfg = model_cfg.vq_model
    quantizer_cfg.img_size = (quantizer_cfg.num_rois, quantizer_cfg.num_timestamp)
    lm_cfg = model_cfg.lm
    if lm_cfg.get('base_model') is None:
        OmegaConf.set_struct(lm_cfg, False)  # Allow modification
        lm_cfg.base_model = args.lm_name
        OmegaConf.set_struct(lm_cfg, True)   # Re-enable struct mode

    dataset_target_mapping = None
    dataset_config_dict = None
    if args.dataset_config is not None:
        logger.info(f"Loading dataset configuration from: {args.dataset_config}")
        dataset_config = OmegaConf.load(args.dataset_config)
        dataset_config = OmegaConf.to_container(dataset_config, resolve=True)
            
        # Check if using new format (dict with attributes) or old format (list)
        if 'datasets' in dataset_config:
            raw_datasets = dataset_config['datasets']
            dataset_target_mapping = {}
            dataset_config_dict = {}
            
            for dataset_name, dataset_info in raw_datasets.items():
                # Support both old format (list) and new format (dict with attributes)
                if isinstance(dataset_info, list):
                    # Old format: "UKB": ["sex"]
                    dataset_target_mapping[dataset_name] = dataset_info
                    dataset_config_dict[dataset_name] = {
                        'targets': dataset_info,
                        'is_multi': False
                    }
                elif isinstance(dataset_info, dict):
                    # New format: "UKB": {"targets": ["sex"], "is_multi": false}
                    dataset_target_mapping[dataset_name] = dataset_info['targets']
                    if 'is_multi' not in dataset_info:
                        dataset_info['is_multi'] = False
                    dataset_config_dict[dataset_name] = dataset_info
                else:
                    raise ValueError(f"Invalid format for dataset {dataset_name}")
            
            # Override datasets list if provided in config
            args.datasets = list(dataset_target_mapping.keys())
            logger.info(f"Using datasets from config: {args.datasets}")
            logger.info(f"Dataset configurations: {dataset_config_dict}")

    if args.fmri_batch_size is None:
        args.fmri_batch_size = args.global_fmri_batch_size // accelerator.num_processes // args.gradient_accumulation_steps
    args.global_fmri_batch_size = args.fmri_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    data_loader_train, data_loader_val_test = get_fmri_data_inst(
        args.fmri_batch_size,
        int(1.5 * args.fmri_batch_size),
        args.datasets,
        lm_name=args.lm_name,
        norm='robust',
        patch_size=quantizer_cfg.patch_size,
        next_time_mask=(args.quantizer != 'titok'),
        use_random_prompt=args.use_random_prompt,
        add_source_info=args.add_src_info,
        add_desc=args.add_desc,
        dataset_target_mapping=dataset_target_mapping,
        dataset_config_dict=dataset_config_dict,
        fewshot_samples=args.fewshot_samples,
    )
    logger.info(f"Data loaders created - Train batches: {len(data_loader_train)}")

    if args.quantizer == 'vq':
        quantizer_cls = VQ
    elif args.quantizer == 'fsq':
        quantizer_cls = FSQ_Model
    elif args.quantizer == 'titok':
        quantizer_cls = TiTok

    # Create tokenizer
    try:
        if args.quantizer == 'titok':
            tokenizer = quantizer_cls(quantizer=quantizer_cfg.quantizer,
                num_latent_tokens=quantizer_cfg.num_latent_tokens,
                latent_token_size=quantizer_cfg.latent_token_size,
                model_size=quantizer_cfg.model_size,
                image_size=quantizer_cfg.img_size,
                patch_size=quantizer_cfg.patch_size,
                codebook_size=quantizer_cfg.codebook_size,
                commitment_cost=quantizer_cfg.commitment_cost,
                use_l2_norm=quantizer_cfg.use_l2_norm,
            )
        else:
            tokenizer = quantizer_cls(quantizer_cfg, decoder_out_dim=quantizer_cfg.num_timestamp)

        latent_tokens = None
        if args.quantizer == 'titok':
            latent_tokens = tokenizer.latent_tokens

        tokenizer_encoder = copy.deepcopy(tokenizer.encoder)

        logger.info("Tokenizer checkpoint memory cleaned up")
    except Exception as e:
        logger.error(f"Failed to create and load tokenizer: {e}")
        raise

    # Model initialization
    ckpt_path = os.path.join(args.ckpt_dir, 'ckpt.pt')
    if os.path.exists(ckpt_path) and args.resume:
        init_from = 'resume'
        logger.info(f"Resuming training from checkpoint: {ckpt_path}")
    elif (args.pretrained_ckpt and os.path.exists(args.pretrained_ckpt)):
        # Check if it's a DeepSpeed checkpoint directory
        if os.path.isdir(args.pretrained_ckpt):
            logger.error(f"ERROR: DeepSpeed checkpoint directory detected: {args.pretrained_ckpt}")
            logger.error(f"Please convert it to .pt format first using:")
            logger.error(f"  python convert_deepspeed_checkpoint.py \\")
            logger.error(f"    --deepspeed_dir {args.pretrained_ckpt} \\")
            logger.error(f"    --output_path {args.pretrained_ckpt.rstrip('/')}.pt \\")
            logger.error(f"    --cfg_path {args.cfg_path}")
            raise ValueError("DeepSpeed checkpoints must be converted to .pt format first")
        init_from = 'pretrained'
        logger.info("Initializing from pretrained weights")
    elif (args.tokenizer_ckpt and os.path.exists(args.tokenizer_ckpt)):
        init_from = 'pretrained_tokenizer'
        logger.info("Initializing tokenizer from checkpoint")
    else:
        init_from = 'scratch'
        logger.info("Training from scratch")

    num_tokens = tokenizer.encoder.num_patches
    num_chans = quantizer_cfg.num_rois if args.quantizer != 'titok' else 1  # TiTok uses vanilla NTP since it produes latent tokens

    iter_num = 0
    n_embd = quantizer_cfg.n_embd
    dropout = 0.0
    model_args = copy.deepcopy(lm_cfg)
    
    # Initialize best metrics tracking (separate for regression and classification)
    best_avg_metric_regression = -float('inf')
    best_avg_metric_classification = -float('inf')
    best_metrics_per_dataset = {}
    all_results = {'validation': {}, 'test': {}}

    try:
        if init_from == 'resume':
            logger.info(f"Resuming training from {args.ckpt_dir}")
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model_args = checkpoint['model_args']
            
            # Load best metrics if available (backward compatible with old checkpoints)
            if 'best_avg_metric_regression' in checkpoint:
                best_avg_metric_regression = checkpoint['best_avg_metric_regression']
                logger.info(f"Loaded best average regression metric: {best_avg_metric_regression}")
            elif 'best_avg_metric' in checkpoint:
                # Backward compatibility
                best_avg_metric_regression = checkpoint['best_avg_metric']
                logger.info(f"Loaded best average metric (legacy): {best_avg_metric_regression}")
            
            if 'best_avg_metric_classification' in checkpoint:
                best_avg_metric_classification = checkpoint['best_avg_metric_classification']
                logger.info(f"Loaded best average classification metric: {best_avg_metric_classification}")
            elif 'best_avg_metric' in checkpoint:
                # Backward compatibility
                best_avg_metric_classification = checkpoint['best_avg_metric']
                logger.info(f"Loaded best average metric (legacy): {best_avg_metric_classification}")
            
            # Try to load previous results
            results_dir = os.path.join(args.ckpt_dir, 'results')
            if os.path.exists(results_dir):
                all_results_file = os.path.join(results_dir, 'all_results.json')
                best_metrics_file = os.path.join(results_dir, 'best_metrics.json')
                
                if os.path.exists(all_results_file):
                    try:
                        with open(all_results_file, 'r') as f:
                            all_results = json.load(f)
                        logger.info("Loaded previous validation/test results")
                    except Exception as e:
                        logger.warning(f"Could not load previous results: {e}")
                
                if os.path.exists(best_metrics_file):
                    try:
                        with open(best_metrics_file, 'r') as f:
                            best_summary = json.load(f)
                            best_metrics_per_dataset = best_summary.get('best_metrics_per_dataset', {})
                            main._best_epoch_regression = best_summary.get('best_epoch_regression', 0)
                            main._best_epoch_classification = best_summary.get('best_epoch_classification', 0)
                            # Backward compatibility
                            if 'best_epoch' in best_summary and not main._best_epoch_regression:
                                main._best_epoch_regression = best_summary['best_epoch']
                                main._best_epoch_classification = best_summary['best_epoch']
                        logger.info("Loaded best metrics tracking")
                    except Exception as e:
                        logger.warning(f"Could not load best metrics: {e}")
            
            gptconf = MultimodalConfig(**model_args)
            model = MindLM(gptconf, tokenizer_encoder, args.tune_tokenizer, num_rois=quantizer_cfg.num_rois, n_embd=n_embd, eeg_vocab_size=tokenizer.codebook_size, latent_tokens=latent_tokens)

            state_dict = checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
            iter_num = checkpoint['iter_num']
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"Resumed from epoch {start_epoch}, iteration {iter_num}")
        elif init_from == 'pretrained':
            logger.info(f"Initializing from pretrained weights")
            checkpoint = torch.load(args.pretrained_ckpt, map_location='cpu', weights_only=False)
            model_args.update(checkpoint['model_args'])
            # gptconf = GPTConfig(**model_args)
            gptconf = MultimodalConfig(**model_args)
            
            # Check if pretrained model was trained with PEFT/LoRA
            pretrained_use_peft = model_args.get('peft_tune', False)
            
            # IMPORTANT: Always create model with peft_tune=False first to avoid double-application
            # We'll manually apply LoRA after if needed
            temp_peft_flag = gptconf.peft_tune
            gptconf.peft_tune = False
            
            model = MindLM(gptconf, tokenizer_encoder, False, num_rois=quantizer_cfg.num_rois, n_embd=n_embd, eeg_vocab_size=tokenizer.codebook_size, latent_tokens=latent_tokens)
            
            if pretrained_use_peft:
                # Pretrained model has LoRA - apply LoRA to match checkpoint structure before loading
                logger.info("Pretrained checkpoint contains LoRA weights - applying LoRA to match structure")
                model.llm.apply_lora()
            
            # Restore flag for later use (after loading checkpoint)
            gptconf.peft_tune = temp_peft_flag

            state_dict = checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            loading_result = model.load_state_dict(state_dict, strict=False)
            
            if loading_result.unexpected_keys:
                logger.warning(f"Unexpected keys when loading pretrained model: {loading_result.unexpected_keys[:5]}...")
            if loading_result.missing_keys:
                logger.warning(f"Missing keys when loading pretrained model: {loading_result.missing_keys[:5]}...")

            start_epoch = 0
            if pretrained_use_peft:
                logger.info("Model initialized from pretrained model (trained with LoRA) - LoRA weights loaded")
            else:
                logger.info("Model initialized from pretrained model (no LoRA in checkpoint)")
        elif init_from == 'pretrained_tokenizer':
            logger.info(f"Initializing tokenizer from checkpoint")
            checkpoint = torch.load(args.tokenizer_ckpt, map_location='cpu', weights_only=False)
            gptconf = MultimodalConfig(**model_args)

            # load checkpoint for the tokenizer only
            tokenizer_state_dict = checkpoint['model']
            # Clean up state dict
            unwanted_prefix = '_orig_mod.'
            for k,v in list(tokenizer_state_dict.items()):
                if k.startswith(unwanted_prefix):
                    tokenizer_state_dict[k[len(unwanted_prefix):]] = tokenizer_state_dict.pop(k)
            
            try:
                msg = tokenizer.load_state_dict(tokenizer_state_dict, strict=False)
                assert msg.missing_keys == []
                logger.info("Tokenizer loaded successfully from checkpoint")
            except Exception as e:
                all_keys = list(tokenizer_state_dict.keys())
                new_dict = OrderedDict()
                prefix_dict = {'vq': 'VQ', 'fsq': 'FSQ', 'titok': 'TiTok'}
                prefix = prefix_dict[args.quantizer]
                for key in all_keys:
                    if key.startswith(f'{prefix}.'):
                        new_dict[key[len(prefix) + 1:]] = tokenizer_state_dict[key]
                    elif key.startswith('fmri_model.') and not key.startswith('text_model.'):
                        new_dict[key[len('fmri_model.'):]] = tokenizer_state_dict[key]
                msg = tokenizer.load_state_dict(new_dict, strict=False)
                assert msg.missing_keys == []
            
            tokenizer_encoder = copy.deepcopy(tokenizer.encoder)
            model = MindLM(gptconf, tokenizer_encoder, args.tune_tokenizer, num_rois=quantizer_cfg.num_rois, n_embd=n_embd, eeg_vocab_size=tokenizer.codebook_size, latent_tokens=latent_tokens)

            start_epoch = 0
            logger.info("Model tokenizer initialized from checkpoint")
        elif init_from.startswith('scratch'):
            raise NotImplementedError("Training from scratch is not recommended or supported in this script")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise

    checkpoint = None  # free up memory
    del tokenizer

    # apply LoRA if specified in config (but only if not already loaded from pretrained checkpoint)
    if lm_cfg.get('peft_tune', False):
        # Check if model already has LoRA (from pretrained checkpoint)
        already_has_lora = hasattr(model.llm.base_model, 'peft_config')
        freeze_pretrained = lm_cfg.get('freeze_pretrained_lora', False)
        
        if already_has_lora and freeze_pretrained:
            # Scenario: Pretrained has LoRA-A, freeze it and add new LoRA-B
            logger.info("Model has pretrained LoRA - freezing it and adding new LoRA adapter for instruction tuning")
            
            # Freeze the pretrained LoRA adapter
            model.llm.freeze_lora_adapter()
            
            # Add a new LoRA adapter on top
            model.llm.apply_lora(adapter_name="instruction_lora")
            
            logger.info("New LoRA adapter 'instruction_lora' added and set as active")
            logger.info("Only the new LoRA adapter will be trained")
        elif already_has_lora:
            # Scenario: Pretrained has LoRA, continue training it
            assert model.llm.base_model.peft_config is not None, "Pretrained model's LoRA config not found"
            logger.info("Model already has LoRA from pretrained checkpoint - will continue training it")
        else:
            # Scenario: No pretrained LoRA, add fresh LoRA
            logger.info("Applying LoRA for instruction tuning")
            model.llm.apply_lora(adapter_name="instruction_lora")

    num_params = model.get_num_params()
    vocab_size = model.llm.original_vocab_size
    text_tokenizer = model.llm.tokenizer

    args.lm_use_cls_head = lm_cfg.get('use_cls_head', False)
    logger.info(f'Model parameters: {num_params:,} ({num_params/1e6:.2f}M)')

    # Optimizer
    try:
        optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (args.beta1, args.beta2), accelerator.device.type)
        if init_from == 'resume':
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("Optimizer state loaded from checkpoint")
        checkpoint = None  # free up memory
        logger.info("Optimizer configured successfully")
    except Exception as e:
        logger.error(f"Failed to configure optimizer: {e}")
        raise

    # Learning rate scheduler
    if not args.fewshot_samples:
        num_training_steps_per_epoch = len(data_loader_train)
        lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=num_training_steps_per_epoch, T_mult=1, eta_min=args.min_lr
        )
        logger.info(f"Learning rate scheduler configured - steps per epoch: {num_training_steps_per_epoch}")
    
    # Prepare everything with accelerator
    try:
        if not args.fewshot_samples:
            # Normal DDP training - prepare everything
            model, optimizer, data_loader_train, lr_scheduler = accelerator.prepare(
                model, optimizer, data_loader_train, lr_scheduler
            )
            # Only set static graph if NOT using gradient accumulation
            # Static graph is incompatible with gradient accumulation
            if accelerator.num_processes > 1 and args.gradient_accumulation_steps == 1:
                model._set_static_graph()
                logger.warning('set model with static computational graph')
            elif accelerator.num_processes > 1 and args.gradient_accumulation_steps > 1:
                logger.info(f'Using gradient accumulation (steps={args.gradient_accumulation_steps}) - static graph disabled')
        else:
            # Few-shot training - do NOT prepare model/optimizer to avoid DDP synchronization Just move model to the correct device
            model = model.to(accelerator.device)
            logger.info(f"Few-shot mode: model moved to device {accelerator.device} (NO DDP wrapping)")
            logger.info("Few-shot mode: optimizer NOT prepared - single GPU training only")
            
        # Always prepare validation/test loaders for multi-GPU evaluation
        for dataset_target, loaders_dict in data_loader_val_test.items():
            for split in ['val', 'test']:
                loaders_dict[split] = accelerator.prepare(loaders_dict[split])
        
        logger.info("Model and data loaders prepared successfully")
    except Exception as e:
        logger.error(f"Failed to prepare with accelerator: {e}")
        raise

    # Training loop
    X_text_random, Y_text_random = get_batch('train', 1024)
    X_text_random, Y_text_random = X_text_random.to(accelerator.device), Y_text_random.to(accelerator.device)
    logger.info("Initial text batch loaded")

    local_iter_num = 0
    
    logger.info(f"Starting training from epoch {start_epoch} to {args.epochs}")
    progress_bar = tqdm(range(start_epoch, args.epochs), desc="Epochs", disable=not accelerator.is_main_process)

    for epoch in progress_bar:
        model.train()
        
        epoch_log, log2 = {}, None
        preds, targs = [], []

        # In few-shot mode, only main process trains
        if args.fewshot_samples and not accelerator.is_main_process:
            # Non-main processes skip training entirely
            pass
        else:
            for step, batch in tqdm(enumerate(data_loader_train), desc="Training Steps", disable=not accelerator.is_main_process, total=len(data_loader_train)):
                X_fmri, X_text, Y_text, gpt_mask, Y = batch
                X_fmri = X_fmri.float()
                gpt_mask = gpt_mask.to(X_fmri.dtype)
                Y_fmri = torch.full((X_fmri.size(0), num_tokens), fill_value=-1-vocab_size, dtype=torch.long, device=X_fmri.device)  # no loss on fMRI tokens

                if X_fmri.device != accelerator.device:
                    X_fmri, Y_fmri, X_text, Y_text, gpt_mask = X_fmri.to(accelerator.device), Y_fmri.to(accelerator.device), X_text.to(accelerator.device), Y_text.to(accelerator.device), gpt_mask.to(accelerator.device)

                with accelerator.accumulate(model):
                    loss1, log1, logits = model(X_fmri, Y_fmri, X_text, Y_text, gpt_mask, Y=Y)
                    # loss2, log2, _ = model(None, None, X_text_random, Y_text_random)

                    loss = loss1 + loss2 if log2 is not None else loss1

                    if lm_cfg.get('use_cls_head', False):
                        if lm_cfg.get('num_classes', 2) >= 2:  # classification
                            preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
                        # TODO how to make this more generalizable?
                        else:  # regression
                            preds.extend(logits.squeeze(-1).cpu().detach().numpy().tolist())
                        targs.extend(Y.cpu().numpy().tolist())

                    # Check for NaN or infinite loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"Invalid loss detected at epoch {epoch}, step {step}: {loss.item()}")
                        raise ValueError("Invalid loss detected")

                    # Backward pass
                    if args.fewshot_samples:
                        # Few-shot mode: use standard PyTorch backward (no DDP)
                        loss.backward()
                    else:
                        # Normal DDP mode: use accelerator's backward
                        accelerator.backward(loss)
                    
                    # Gradient clipping and optimizer step
                    # These should be inside accumulate() context so they only happen
                    # when gradients are actually synchronized
                    if args.grad_clip != 0.0:
                        if args.fewshot_samples:
                            # Few-shot mode: use standard PyTorch gradient clipping
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                        else:
                            # Normal DDP mode: use accelerator's gradient clipping
                            accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                    
                    optimizer.step()
                    
                    if not args.fewshot_samples:
                        lr_scheduler.step()
                    
                    optimizer.zero_grad()
                
                # Get next text batch
                # X_text_random, Y_text_random = get_batch('train', 1024)
                # X_text_random, Y_text_random = X_text_random.to(accelerator.device), Y_text_random.to(accelerator.device)

                total_loss = log1['train/loss'] + log2['train/loss'] if log2 is not None else log1['train/loss']
                fmri_loss = log1['train/loss']
                text_loss = log2['train/loss'] if log2 is not None else 0

                log = {'total_loss': total_loss, 'fmri_loss': fmri_loss, 'text_loss': text_loss, 
                       'fmri_acc': log1['train/accuracy'], 'text_acc': log2['train/accuracy'] if log2 is not None else 0}
                if epoch_log == {}: 
                    epoch_log = log
                else: 
                    epoch_log = {k: epoch_log[k] + log[k] for k in log}

                iter_num += 1
                local_iter_num += 1

        # gather logs from processes
        if args.fewshot_samples:
            if accelerator.is_main_process:
                # In few-shot mode, only main process has epoch_log
                epoch_log = {k: v / (step + 1) for k, v in epoch_log.items()}
                progress_bar.set_description(f"Epoch {epoch + 1}: fmri_loss - {epoch_log['fmri_loss']:.4f}, text_loss - {epoch_log['text_loss']:.4f}")
        else:
            # Normal DDP mode: gather from all processes
            epoch_log = {k: torch.tensor(v, device=accelerator.device, dtype=torch.float64) / (step + 1) for k, v in epoch_log.items()}
            epoch_log = accelerator.gather(epoch_log)
            epoch_log = {k: epoch_log[k].mean().item() for k in epoch_log}
            progress_bar.set_description(f"Epoch {epoch + 1}: fmri_loss - {epoch_log['fmri_loss']:.4f}, text_loss - {epoch_log['text_loss']:.4f}")

        if lm_cfg.get('use_cls_head', False):
            if args.fewshot_samples:
                # In few-shot mode, only main process has predictions
                if accelerator.is_main_process:
                    preds_np = np.array(preds)
                    targs_np = np.array(targs)
                    if lm_cfg.get('num_classes', 2) >= 2:  # classification
                        logger.info(f"Epoch {epoch + 1} Training Accuracy: {accuracy_score(targs_np, preds_np):.4f}, AUC: {roc_auc_score(targs_np, preds_np):.4f}")
                    else:  # regression
                        results = get_metrics(preds_np, targs_np, metrics=['mae', 'pearson'], is_binary=False, is_regression=True)
                        logger.info(f"Epoch {epoch + 1} Training MAE: {results['mae']:.4f}, Pearson: {results['pearson']:.4f}")
            else:
                # Normal DDP mode: gather from all processes
                preds = torch.tensor(np.array(preds), device=accelerator.device)
                targs = torch.tensor(np.array(targs), device=accelerator.device)
                preds, targs = accelerator.gather(preds).cpu().numpy(), accelerator.gather(targs).cpu().numpy()
                if accelerator.is_main_process:
                    if lm_cfg.get('num_classes', 2) >= 2:  # classification
                        logger.info(f"Epoch {epoch + 1} Training Accuracy: {accuracy_score(targs, preds):.4f}, AUC: {roc_auc_score(targs, preds):.4f}")
                    else:  # regression
                        results = get_metrics(preds, targs, metrics=['mae', 'pearson'], is_binary=False, is_regression=True)
                        logger.info(f"Epoch {epoch + 1} Training MAE: {results['mae']:.4f}, Pearson: {results['pearson']:.4f}")
        else:
            if accelerator.is_main_process: logger.info(f"Epoch {epoch + 1} Training Losses: {epoch_log}")

        if args.wandb_log and accelerator.is_main_process:
            run.log({
                "epoch": epoch,
                "train/total_loss": epoch_log['total_loss'],
                "train/fmri_loss": epoch_log['fmri_loss'],
                "train/text_loss": epoch_log['text_loss'],
                "train/fmri_accuracy": epoch_log['fmri_acc'],
                "train/text_accuracy": epoch_log['text_acc'],
                "lr": optimizer.param_groups[0]['lr'],
            })

        # Synchronize model weights in few-shot mode and prepare for evaluation
        if args.fewshot_samples and accelerator.num_processes > 1:
            accelerator.wait_for_everyone()
            
            # Broadcast model weights from main process to all other processes
            if torch.distributed.is_initialized():
                for param in model.parameters():
                    torch.distributed.broadcast(param.data, src=0)
            
            # Now wrap model in DDP for multi-GPU evaluation
            model = accelerator.prepare(model)
        
        # Validation
        val_start_time = time.time()
        epoch_val_results = {}
        epoch_test_results = {}
        
        # Collect all metrics for wandb logging (to log once per epoch)
        wandb_metrics = {}
        
        for data_name in data_loader_val_test:
            allowed_tokens = None
            if args.use_allowed_tokens:
                allowed_tokens = get_allowed_token_id(data_loader_val_test[data_name]['info']['target_name'], text_tokenizer)
                
            results_val = evaluate(model, data_loader_val_test[data_name]['val'], accelerator, args, logger, vocab_size, 
                                data_loader_val_test[data_name]['info'], allowed_tokens=allowed_tokens)
            if accelerator.is_main_process:
                logger.info('=' * 10)
                logger.info(f"Validation results for {data_name}: {results_val}")

            results_test = evaluate(model, data_loader_val_test[data_name]['test'], accelerator, args, logger, vocab_size, 
                                data_loader_val_test[data_name]['info'], allowed_tokens=allowed_tokens)
            if accelerator.is_main_process:
                logger.info('=' * 10)
                logger.info(f"Test results for {data_name}: {results_test}")
            
            # Store results for this epoch
            epoch_val_results[data_name] = results_val
            epoch_test_results[data_name] = results_test
            
            # Update best metrics tracking for this dataset
            if data_name not in best_metrics_per_dataset:
                best_metrics_per_dataset[data_name] = {}
            
            # Update all results storage
            if data_name not in all_results['validation']:
                all_results['validation'][data_name] = {}
                all_results['test'][data_name] = {}
            
            all_results['validation'][data_name][f'epoch_{epoch}'] = results_val
            all_results['test'][data_name][f'epoch_{epoch}'] = results_test

            # Collect metrics for wandb (log all datasets together)
            if args.wandb_log and accelerator.is_main_process:
                for metric in results_val.keys():
                    wandb_metrics[f'val_{data_name}/{metric}'] = results_val[metric]
                    wandb_metrics[f'test_{data_name}/{metric}'] = results_test[metric]
        
        # Calculate average validation metric across datasets and metrics
        # Separate for regression and classification tasks
        is_best_avg_regression = False
        is_best_avg_classification = False
        is_best_list = {k: False for k in epoch_val_results.keys()}  # Track best per dataset
        
        current_avg_metric_regression = -float('inf')
        current_avg_metric_classification = -float('inf')
        
        if accelerator.is_main_process:
            val_metrics_regression = []
            val_metrics_classification = []
            
            for data_name, results in epoch_val_results.items():
                # Determine if this dataset is regression or classification
                data_info = data_loader_val_test[data_name]['info']
                is_regression = data_info.get('is_regression', False)
                
                for metric_name, metric_value in results.items():
                    # Only consider numerical metrics (not strings or other types)
                    if isinstance(metric_value, (int, float)) and not math.isnan(metric_value):
                        if is_regression:
                            # For regression, negate MAE/MSE so higher is better for comparison
                            if metric_name.lower() in ['mae', 'mse', 'rmse']:
                                val_metrics_regression.append(-metric_value)
                            else:
                                val_metrics_regression.append(metric_value)
                        else:
                            # For classification, metrics like accuracy are already higher=better
                            val_metrics_classification.append(metric_value)
                
                # Update best metrics per dataset
                if best_metrics_per_dataset[data_name] == {}:
                    best_metrics_per_dataset[data_name] = epoch_val_results[data_name]
                    is_best_list[data_name] = True
                    if 'accuracy' in best_metrics_per_dataset[data_name]:
                        logger.info(f"New best validation acc for {data_name}: {best_metrics_per_dataset[data_name]['accuracy']}")
                    else:
                        logger.info(f"New best validation metric for {data_name}: {best_metrics_per_dataset[data_name].get('mae', 'N/A')}")
                elif 'accuracy' in best_metrics_per_dataset[data_name] and best_metrics_per_dataset[data_name]['accuracy'] < epoch_val_results[data_name]['accuracy']:
                    is_best_list[data_name] = True
                    best_metrics_per_dataset[data_name] = epoch_val_results[data_name]
                    logger.info(f"New best validation acc for {data_name}: {best_metrics_per_dataset[data_name]['accuracy']}")
                elif 'mae' in best_metrics_per_dataset[data_name] and best_metrics_per_dataset[data_name]['mae'] > epoch_val_results[data_name]['mae']:
                    is_best_list[data_name] = True
                    best_metrics_per_dataset[data_name] = epoch_val_results[data_name]
                    logger.info(f"New best validation MAE for {data_name}: {best_metrics_per_dataset[data_name]['mae']}")
            
            # Compute average metrics for regression and classification separately
            if val_metrics_regression:
                current_avg_metric_regression = np.mean(val_metrics_regression)
                is_best_avg_regression = current_avg_metric_regression > best_avg_metric_regression
                
                if is_best_avg_regression:
                    best_avg_metric_regression = current_avg_metric_regression
                    logger.info(f"New best average REGRESSION validation metric: {best_avg_metric_regression:.4f}")
            
            if val_metrics_classification:
                current_avg_metric_classification = np.mean(val_metrics_classification)
                is_best_avg_classification = current_avg_metric_classification > best_avg_metric_classification
                
                if is_best_avg_classification:
                    best_avg_metric_classification = current_avg_metric_classification
                    logger.info(f"New best average CLASSIFICATION validation metric: {best_avg_metric_classification:.4f}")
            
            # Add average metrics to wandb logging
            if args.wandb_log and wandb_metrics:
                if val_metrics_regression:
                    wandb_metrics['val/avg_metric_regression'] = current_avg_metric_regression
                    wandb_metrics['val/best_avg_metric_regression'] = best_avg_metric_regression
                if val_metrics_classification:
                    wandb_metrics['val/avg_metric_classification'] = current_avg_metric_classification
                    wandb_metrics['val/best_avg_metric_classification'] = best_avg_metric_classification
        
        # Log all validation/test metrics together in a single wandb call
        if args.wandb_log and accelerator.is_main_process and wandb_metrics:
            wandb.log(wandb_metrics)
        
        accelerator.wait_for_everyone()
        
        # Unwrap model in few-shot mode after evaluation
        if args.fewshot_samples and accelerator.num_processes > 1:
            model = accelerator.unwrap_model(model)
        
        # Save checkpoint
        if accelerator.is_main_process and args.save_ckpt:
            checkpoint = {
                'model': accelerator.unwrap_model(model).state_dict() if not args.fewshot_samples else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'epoch': epoch,
                'validation_results': epoch_val_results,
                'test_results': epoch_test_results,
                'best_avg_metric_regression': best_avg_metric_regression,
                'best_avg_metric_classification': best_avg_metric_classification
            }
            
            # Save best model checkpoint
            if (epoch + 1) % args.save_ckpt_freq == 0:
                epoch_checkpoint_path = os.path.join(args.ckpt_dir, f'ckpt-{epoch + 1}.pt')
                torch.save(checkpoint, epoch_checkpoint_path)
                logger.info(f"Epoch checkpoint saved: {epoch_checkpoint_path}")

            # Save best regression checkpoint
            if is_best_avg_regression:
                best_regression_checkpoint_path = os.path.join(args.ckpt_dir, 'best_avg_regression_ckpt.pt')
                torch.save(checkpoint, best_regression_checkpoint_path)
                logger.info(f"Best REGRESSION checkpoint saved: {best_regression_checkpoint_path}")

            # Save best classification checkpoint
            if is_best_avg_classification:
                best_classification_checkpoint_path = os.path.join(args.ckpt_dir, 'best_avg_classification_ckpt.pt')
                torch.save(checkpoint, best_classification_checkpoint_path)
                logger.info(f"Best CLASSIFICATION checkpoint saved: {best_classification_checkpoint_path}")

            for data_name, is_best in is_best_list.items():
                if is_best:
                    best_data_checkpoint_path = os.path.join(args.ckpt_dir, f'best_{data_name}_ckpt.pt')
                    torch.save(checkpoint, best_data_checkpoint_path)
                    logger.info(f"Best checkpoint for {data_name} saved: {best_data_checkpoint_path}")
            
            # Save detailed results to JSON files
            results_dir = os.path.join(args.ckpt_dir, 'results')
            os.makedirs(results_dir, exist_ok=True)
            
            # Save epoch-specific results
            epoch_results_file = os.path.join(results_dir, f'epoch_{epoch}_results.json')
            epoch_results = {
                'epoch': epoch,
                'validation': epoch_val_results,
                'test': epoch_test_results,
                'avg_validation_metric_regression': current_avg_metric_regression if current_avg_metric_regression != -float('inf') else None,
                'avg_validation_metric_classification': current_avg_metric_classification if current_avg_metric_classification != -float('inf') else None
            }
            
            with open(epoch_results_file, 'w') as f:
                json.dump(epoch_results, f, indent=2)
            logger.info(f"Epoch results saved: {epoch_results_file}")
            
            # Save cumulative results
            all_results_file = os.path.join(results_dir, 'all_results.json')
            with open(all_results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            # Save best metrics summary
            best_metrics_file = os.path.join(results_dir, 'best_metrics.json')
            best_summary = {
                'best_avg_metric_regression': best_avg_metric_regression if best_avg_metric_regression != -float('inf') else None,
                'best_avg_metric_classification': best_avg_metric_classification if best_avg_metric_classification != -float('inf') else None,
                'best_metrics_per_dataset': best_metrics_per_dataset,
                'best_epoch_regression': epoch if is_best_avg_regression else getattr(main, '_best_epoch_regression', 0),
                'best_epoch_classification': epoch if is_best_avg_classification else getattr(main, '_best_epoch_classification', 0)
            }
            if is_best_avg_regression:
                main._best_epoch_regression = epoch
            if is_best_avg_classification:
                main._best_epoch_classification = epoch
            
            with open(best_metrics_file, 'w') as f:
                json.dump(best_summary, f, indent=2)
        accelerator.wait_for_everyone()

    if args.wandb_log and accelerator.is_main_process:
        run.finish()
        logger.info("W&B tracking ended")
    accelerator.end_training()


default_preds = {'sex': ' Male', 'ADHD': ' Control', 'ASD': ' Control', 'age': ' 30'}

def get_pred(pred_string, dataset_info):
    # get the next word after `Answer: `
    pred_words = pred_string.split(' ')
    pred_words = [w.strip() for w in pred_words if w.strip() != '']
    ans_idx = [i for i, w in enumerate(pred_words) if w.lower().startswith('answer:')][0]
    
    default_pred = default_preds.get(dataset_info['target_name'], ' CN')
    
    try:
        pred = pred_words[min(ans_idx + 1, len(pred_words) - 1)]
    except IndexError:
        print('Index out of range!', pred_words)
        pred = default_pred
    if not dataset_info['is_regression']:
        pred = dataset_info['label_dic'].get(pred, 0.)
    else:
        pred = float(pred)
    return pred

@torch.no_grad()
def evaluate(model, dataloader, accelerator, args, logger, vocab_size, data_info, allowed_tokens=None):
    """Evaluate the model on validation data with proper DDP support""" 
    model.eval()
    all_preds, all_targets = [], []    
    
    for batch_idx, batch in enumerate(dataloader):
        X_fmri, X_text, label, gpt_mask = batch
        X_fmri = X_fmri.float()
        assert args.quantizer != 'titok'
        gpt_mask = gpt_mask.to(X_fmri.dtype)        

        if not args.lm_use_cls_head:
            # Use autocast for inference to match training precision
            with accelerator.autocast():
                if accelerator.num_processes == 1:
                    text = model.generate(X_fmri, X_text, gpt_mask, max_new_tokens=4, text_gen=True, allowed_tokens=allowed_tokens)
                else:
                    text = model.module.generate(X_fmri, X_text, gpt_mask, max_new_tokens=4, text_gen=True, allowed_tokens=allowed_tokens, accelerator=accelerator)

            batch_preds = []
            for i, t in enumerate(text):
                pred = get_pred(t, data_info)
                if not data_info['is_binary'] and not data_info['is_regression']:
                    pred = torch.eye(data_info['num_classes'])[pred]
                batch_preds.append(pred)
        else:
            # Use autocast for inference to match training precision
            with accelerator.autocast():
                _, _, logits = model(X_fmri, None, X_text, None, gpt_mask, Y=label)
            if data_info['is_regression']:
                batch_preds = logits[:, 0].cpu().numpy().tolist()
            else:
                batch_preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
        
        all_preds.extend(batch_preds)
        all_targets.extend(label.cpu().numpy())
    
    # Convert to tensors for gathering
    assert len(all_preds) > 0
    if data_info['is_binary']:
        preds_tensor = torch.tensor(all_preds, dtype=torch.float32, device=accelerator.device)
    else:
        preds_tensor = torch.stack([torch.tensor(p, dtype=torch.float32) for p in all_preds]).to(accelerator.device)
    targets_tensor = torch.tensor(all_targets, dtype=torch.long, device=accelerator.device)

    # Gather predictions and targets from all processes
    gathered_preds = accelerator.gather(preds_tensor)
    gathered_targets = accelerator.gather(targets_tensor)
    
    # Compute metrics only on main process to avoid redundant computation
    if accelerator.is_main_process:
        # Convert back to numpy for metrics computation
        final_preds = gathered_preds.cpu().numpy()
        final_targets = gathered_targets.cpu().numpy()
        
        results = get_metrics(final_preds, final_targets, data_info['metrics'], data_info['is_binary'], data_info['is_regression'])
    else:
        results = {}
    
    model.train()
    return results


def get_args():
    def list_of_strs(arg):
        return arg.split(',')
    parser = argparse.ArgumentParser('NeuroLM training script', add_help=False)

    parser.add_argument('--datasets', type=list_of_strs, default=['UKB'],  help='list of dataset names to use for training')
    parser.add_argument('--dataset_config', type=str, default='configs/dataset_config.yaml', help='path to dataset configuration YAML file that defines datasets and their targets')
    parser.add_argument('--add_src_info', default=False, action='store_true', help='whether to add source dataset info to the prompt')
    parser.add_argument('--add_desc', default=False, action='store_true', help='whether to add subject medical descriptions to the prompt')
    parser.add_argument('--fewshot_samples', default=0, type=int, help='number of few-shot samples to include in the prompt')

    parser.add_argument('--pretrained_ckpt', default='checkpoints/pretrain/UKB_ABCD_HCP_robust/VQ_Align-ViT_base-p32-Qwen3-0.6B-Contr_F2T-DeepSpeed-delimiter-PEFT_all_8_16_.1-textW.1/deepspeed_checkpoint_best_f2t/merged_checkpoint.pt')
    parser.add_argument('--tokenizer_ckpt', default='')
    parser.add_argument('--ckpt_dir', default='tmp', help='path where to save, empty for no saving')
    parser.add_argument('--no_timestamp', default=False, action='store_true', help='disable automatic timestamp suffix for checkpoint directory')
    parser.add_argument('--resume', default=False, action='store_true', help='resume from the latest checkpoint')
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--wandb_log', default=False, action='store_true')
    parser.add_argument('--wandb_project', default='BrainFM_instruction', type=str)
    parser.add_argument('--wandb_runname', default='tmp', type=str)
    parser.add_argument('--wandb_group', default=None, type=str)

    # training args
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--fmri_batch_size', default=1, type=int)
    parser.add_argument('--global_fmri_batch_size', default=None, type=int, help='global batch size for fMRI across all GPUs')
    parser.add_argument('--text_batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--save_ckpt', default=False, action=argparse.BooleanOptionalAction, help='whether to save checkpoints')
    parser.add_argument('--save_ckpt_freq', default=10, type=int)
    parser.add_argument('--tune_tokenizer', action='store_true', help='whether to finetune the tokenizer during training', default=False)
    parser.add_argument('--use_random_prompt', action='store_true', help='whether to use random prompt during training/inference', default=False)

    parser.add_argument('--use_allowed_tokens', action='store_true', help='whether to restrict the generation to allowed tokens', default=False)

    parser.add_argument('--quantizer', type=str, default='vq')
    parser.add_argument('--lm_name', type=str, default='Qwen/Qwen3-0.6B', help='name of the language model to use')
    parser.add_argument('--cfg_path', type=str, default='configs/vit_base_qwen_p32.yaml', help='path to the model config file',)
    parser.add_argument('--lm_use_cls_head', type=bool, default=False, help='direct do prediction from hidden states of LM, update by the model cfg')

    parser.add_argument('--learning_rate', type=float, default=6e-4, metavar='LR',
                        help='learning rate (default: 6e-4)')
    parser.add_argument('--min_lr', type=float, default=6e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-1,
                        help='weight decay (default: 1e-1)')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='clip gradients at this value, or disable if == 0.0')
    parser.add_argument('--decay_lr', default=True, action='store_false')
    parser.add_argument('--seed', default=1337, type=int)

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)