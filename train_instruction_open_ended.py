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
from openai import OpenAI

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from sklearn.metrics import accuracy_score, roc_auc_score
from concurrent.futures import ThreadPoolExecutor, as_completed

from model_mindlm import MindLM
from quantizers import *
from model_gpt import MultimodalConfig
from dataset import get_fmri_data_open
from utils import get_metrics, get_allowed_token_id

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com",)

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


def llm_evaluate(pred_str: str, all_targets: dict, max_retries: int = 3):
    """
    Use LLM to evaluate if predicted text contains correct field values.
    
    Args:
        pred_str: Generated text from the model
        all_targets: Dictionary of target field-value pairs
        max_retries: Maximum number of retry attempts if fields are missing
    
    Returns:
        Dictionary with field names as keys and True/False as values indicating correctness
    """
    system_prompt = """You are a strict evaluator for medical/demographic predictions. Given a reference answer with field-value pairs and a model's generated answer, evaluate whether each field in the reference is correctly predicted in the model answer.

Rules:
1. Mark each field as "correct" (true) or "incorrect" (false)
2. A field is correct if the model answer contains the same information, even if worded differently
3. Be lenient with paraphrasing but strict with factual correctness
4. If a field is not mentioned at all in the model answer, mark it as false
5. Return ONLY valid JSON with field names as keys and boolean values
6. IMPORTANT: You MUST include ALL fields from the reference in your output

Example:
Reference: {"sex": "Male", "fluid_intelligence_higher_than_usual": "False", "disease": "CN", "age_group": "senior"}
Model answer: "The subject appears to be a senior male with no signs of disease (cognitive normal), and show higher than usual fluid intelligence."

Output:
{"sex": true, "fluid_intelligence_higher_than_usual": false, "disease": true, "age_group": true}
Now evaluate:"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Reference: {json.dumps(all_targets)}\nModel answer: {pred_str}\n\nOutput JSON only:"}
                ],
                temperature=0,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Check if all target fields are present
            missing_fields = [field for field in all_targets.keys() if field not in result]
            
            if not missing_fields:
                # All fields present, return result
                return result
            else:
                # Some fields are missing
                if attempt < max_retries - 1:
                    # Retry with explicit mention of missing fields
                    print(f"Attempt {attempt + 1}: Missing fields {missing_fields}, retrying...")
                    system_prompt += f"\n\nWARNING: In your previous response, you missed these fields: {missing_fields}. Please include ALL fields in your response."
                else:
                    # Last attempt failed, fill missing fields with False
                    print(f"After {max_retries} attempts, still missing fields: {missing_fields}. Filling with False.")
                    for field in missing_fields:
                        result[field] = False
                    return result
            
        except json.JSONDecodeError as e:
            # JSON parsing failed
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1}: JSON decode error: {e}, retrying...")
                continue
            else:
                print(f"JSON decode failed after {max_retries} attempts: {e}")
                return {field: False for field in all_targets.keys()}
                
        except Exception as e:
            # Other exceptions (API errors, etc.)
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1}: API error: {e}, retrying...")
                time.sleep(1)  # Brief pause before retry
                continue
            else:
                print(f"LLM evaluation failed after {max_retries} attempts: {e}")
                return {field: False for field in all_targets.keys()}
    
    # Should not reach here, but just in case
    return {field: False for field in all_targets.keys()}


def llm_evaluate_batch(pred_texts: list, target_dicts: list, max_workers: int = 16):
    """
    Evaluate multiple predictions in parallel using ThreadPoolExecutor.
    
    Args:
        pred_texts: List of predicted text strings
        target_dicts: List of target dictionaries
        max_workers: Maximum number of concurrent threads
    
    Returns:
        List of evaluation result dictionaries
    """
    field_correctness = []
    
    # Use ThreadPoolExecutor for concurrent API calls
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(llm_evaluate, pred_text, target_dict): idx 
            for idx, (pred_text, target_dict) in enumerate(zip(pred_texts, target_dicts))
        }
        
        # Collect results in order
        results = [None] * len(pred_texts)
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Error evaluating sample {idx}: {e}")
                # Return all False for failed evaluations
                results[idx] = {field: False for field in target_dicts[idx].keys()}
        
        field_correctness = results
    
    return field_correctness


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
    if lm_cfg.base_model is None: lm_cfg = args.lm_name

    data_loader_train, data_loader_val_test = get_fmri_data_open(
        args.fmri_batch_size,
        int(1.5 * args.fmri_batch_size),
        args.datasets,
        lm_name=lm_cfg.base_model,
        norm='robust',
        patch_size=quantizer_cfg.patch_size,
        next_time_mask=(args.quantizer != 'titok'),
        use_random_prompt=args.use_random_prompt,
        add_source_info=args.add_src_info,
        add_desc=args.add_desc,
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
    model_args = lm_cfg
    
    # Initialize best metrics tracking for open-ended generation
    best_avg_metric_regression = -float('inf')  # Reusing variable name for compatibility
    best_metrics_per_dataset = {}
    all_results = {'validation': {}, 'test': {}}

    try:
        if init_from == 'resume':
            logger.info(f"Resuming training from {args.ckpt_dir}")
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model_args = checkpoint['model_args']
            
            # Load best metrics if available (backward compatible with old checkpoints)
            if 'best_avg_metric' in checkpoint:
                best_avg_metric_regression = checkpoint['best_avg_metric']
                logger.info(f"Loaded best average metric: {best_avg_metric_regression}")
            elif 'best_avg_metric_regression' in checkpoint:
                best_avg_metric_regression = checkpoint['best_avg_metric_regression']
                logger.info(f"Loaded best average metric (legacy): {best_avg_metric_regression}")
            
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
            
            # Temporarily disable peft_tune to avoid double wrapping when loading
            temp_peft_flag = gptconf.peft_tune
            gptconf.peft_tune = False

            model = MindLM(gptconf, tokenizer_encoder, False, num_rois=quantizer_cfg.num_rois, n_embd=n_embd, eeg_vocab_size=tokenizer.codebook_size, latent_tokens=latent_tokens)

            state_dict = checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            loading_result = model.load_state_dict(state_dict, strict=False)
            
            # Restore the peft_tune flag
            gptconf.peft_tune = temp_peft_flag
            
            if loading_result.unexpected_keys:
                logger.warning(f"Unexpected keys when loading pretrained model: {loading_result.unexpected_keys[:5]}...")
            if loading_result.missing_keys:
                logger.warning(f"Missing keys when loading pretrained model: {loading_result.missing_keys[:5]}...")

            start_epoch = 0
            if pretrained_use_peft:
                logger.info("Model initialized from pretrained model (trained with LoRA)")
            else:
                logger.info("Model initialized from pretrained model")
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
            logger.info("Training from scratch (with pretrained LLM)")
            gptconf = MultimodalConfig(**model_args)

            model = MindLM(gptconf, tokenizer_encoder, args.tune_tokenizer, num_rois=quantizer_cfg.num_rois, n_embd=n_embd, eeg_vocab_size=tokenizer.codebook_size, latent_tokens=latent_tokens)
            start_epoch = 0
            logger.info("Model created from scratch")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise

    checkpoint = None  # free up memory
    del tokenizer

    # apply LoRA if specified in config
    if lm_cfg.get('peft_tune', False):
        model.llm.apply_lora()

    num_params = model.get_num_params()
    vocab_size = model.llm.original_vocab_size

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
    num_training_steps_per_epoch = len(data_loader_train)
    lr_scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=num_training_steps_per_epoch, T_mult=1, eta_min=args.min_lr
    )
    logger.info(f"Learning rate scheduler configured - steps per epoch: {num_training_steps_per_epoch}")
    
    # Prepare everything with accelerator
    try:
        model, optimizer, data_loader_train, lr_scheduler = accelerator.prepare(
            model, optimizer, data_loader_train, lr_scheduler
        )
        for dataset_target, loaders_dict in data_loader_val_test.items():
            for split in ['val', 'test']:
                loaders_dict[split] = accelerator.prepare(loaders_dict[split])
        if accelerator.num_processes > 1:
            model._set_static_graph()
            logger.warning('set model with static computational graph')
        logger.info("Model and optimizers prepared with accelerator")
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
        for step, batch in enumerate(data_loader_train):
            X_fmri, X_text, Y_text, gpt_mask, Y = batch
            X_fmri = X_fmri.float()
            gpt_mask = gpt_mask.to(X_fmri.dtype)
            Y_fmri = torch.full((X_fmri.size(0), num_tokens), fill_value=-1-vocab_size, dtype=torch.long, device=X_fmri.device)  # no loss on fMRI tokens

            with accelerator.accumulate(model):
                loss1, log1, logits = model(X_fmri, Y_fmri, X_text, Y_text, gpt_mask, Y=Y)
                # loss2, log2, _ = model(None, None, X_text_random, Y_text_random)

                loss = loss1 + loss2 if log2 is not None else loss1

                if lm_cfg.get('use_cls_head', False):
                    preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
                    targs.extend(Y.cpu().numpy().tolist())

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

            # if (iter_num + 1) % args.log_interval == 0:
                # logger.info(f"Epoch {epoch + 1} Step [{step + 1}/{num_training_steps_per_epoch}]: "
                #             f"Total Loss {total_loss:.4f}, EEG Loss {fmri_loss:.4f}, Text Loss {text_loss:.4f}, "
                #             f"LR {optimizer.param_groups[0]['lr']:.2e}")
                
                # if args.wandb_log and accelerator.is_main_process:
                #     run.log({
                #         "iter": iter_num,
                #         "train_iter/total_loss": total_loss,
                #         "train_iter/fmri_loss": fmri_loss,
                #         "train_iter/text_loss": text_loss,
                #         "train_iter/fmri_accuracy": log1['train/accuracy'],
                #         "train_iter/text_accuracy": log2['train/accuracy'],
                #         "train_iter/lr": optimizer.param_groups[0]['lr']
                #     })

            iter_num += 1
            local_iter_num += 1

        # gather logs from processes
        epoch_log = {k: torch.tensor(v, device=accelerator.device, dtype=torch.float64) / (step + 1) for k, v in epoch_log.items()}
        epoch_log = accelerator.gather(epoch_log)
        epoch_log = {k: epoch_log[k].mean().item() for k in epoch_log}
        progress_bar.set_description(f"Epoch {epoch + 1}: fmri_loss - {epoch_log['fmri_loss']:.4f}, text_loss - {epoch_log['text_loss']:.4f}")

        if lm_cfg.get('use_cls_head', False):
            preds = torch.tensor(np.array(preds), device=accelerator.device)
            targs = torch.tensor(np.array(targs), device=accelerator.device)
            preds, targs = accelerator.gather(preds).cpu().numpy(), accelerator.gather(targs).cpu().numpy()
            if accelerator.is_main_process: logger.info(f"Epoch {epoch + 1} Training Accuracy: {accuracy_score(targs, preds):.4f}, AUC: {roc_auc_score(targs, preds):.4f}")
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

        # Validation (only run every eval_freq epochs, and always on the last epoch)
        run_evaluation = ((epoch + 1) % args.eval_freq == 0) or (epoch == args.epochs - 1)
        
        if run_evaluation:
            val_start_time = time.time()
            epoch_val_results = {}
            epoch_test_results = {}
            
            # Collect all metrics for wandb logging (to log once per epoch)
            wandb_metrics = {}
            
            for data_name in data_loader_val_test:
                allowed_tokens = None  # should not use allow tokens
                    
                results_val = evaluate(model, data_loader_val_test[data_name]['val'], accelerator, args, logger, vocab_size, 
                                    data_loader_val_test[data_name]['info'], allowed_tokens=allowed_tokens, llm_workers=args.llm_eval_workers)
                if accelerator.is_main_process:
                    logger.info('=' * 10)
                    logger.info(f"Validation results for {data_name}: {results_val}")

                results_test = evaluate(model, data_loader_val_test[data_name]['test'], accelerator, args, logger, vocab_size, 
                                    data_loader_val_test[data_name]['info'], allowed_tokens=allowed_tokens, llm_workers=args.llm_eval_workers)
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
            
            # Calculate average validation metric across datasets
            # For open-ended generation, we use overall_accuracy and avg_field_accuracy
            is_best_avg = False
            is_best_list = {k: False for k in epoch_val_results.keys()}  # Track best per dataset
            
            current_avg_metric = -float('inf')
            
            if accelerator.is_main_process:
                val_metrics = []
                
                for data_name, results in epoch_val_results.items():
                    # Use avg_field_accuracy as the primary metric for open-ended generation
                    if 'avg_field_accuracy' in results:
                        val_metrics.append(results['avg_field_accuracy'])
                    elif 'overall_accuracy' in results:
                        val_metrics.append(results['overall_accuracy'])
                    
                    # Update best metrics per dataset
                    if data_name not in best_metrics_per_dataset or best_metrics_per_dataset[data_name] == {}:
                        best_metrics_per_dataset[data_name] = results
                        is_best_list[data_name] = True
                        logger.info(f"New best validation for {data_name}: Overall Acc={results.get('overall_accuracy', 0):.4f}, Avg Field Acc={results.get('avg_field_accuracy', 0):.4f}")
                    else:
                        # Compare using avg_field_accuracy
                        current_metric = results.get('avg_field_accuracy', results.get('overall_accuracy', 0))
                        best_metric = best_metrics_per_dataset[data_name].get('avg_field_accuracy', best_metrics_per_dataset[data_name].get('overall_accuracy', 0))
                        
                        if current_metric > best_metric:
                            is_best_list[data_name] = True
                            best_metrics_per_dataset[data_name] = results
                            logger.info(f"New best validation for {data_name}: Overall Acc={results.get('overall_accuracy', 0):.4f}, Avg Field Acc={results.get('avg_field_accuracy', 0):.4f}")
                
                # Compute average metric across all datasets
                if val_metrics:
                    current_avg_metric = np.mean(val_metrics)
                    is_best_avg = current_avg_metric > best_avg_metric_regression  # Reuse the variable for simplicity
                    
                    if is_best_avg:
                        best_avg_metric_regression = current_avg_metric
                        logger.info(f"New best average validation metric: {best_avg_metric_regression:.4f}")
                
                # Add average metrics to wandb logging
                if args.wandb_log and wandb_metrics:
                    if val_metrics:
                        wandb_metrics['val/avg_metric'] = current_avg_metric
                        wandb_metrics['val/best_avg_metric'] = best_avg_metric_regression
            
            # Log all validation/test metrics together in a single wandb call
            if args.wandb_log and accelerator.is_main_process and wandb_metrics:
                wandb.log(wandb_metrics)
        else:
            # Skip evaluation for this epoch
            if accelerator.is_main_process:
                logger.info(f"Skipping evaluation for epoch {epoch + 1} (eval_freq={args.eval_freq})")
            epoch_val_results = {}
            epoch_test_results = {}
            is_best_avg = False
            is_best_list = {}
        
        accelerator.wait_for_everyone()
        
        # Save checkpoint
        if accelerator.is_main_process and args.save_ckpt:
            checkpoint = {
                'model': accelerator.unwrap_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'epoch': epoch,
                'validation_results': epoch_val_results,
                'test_results': epoch_test_results,
                'best_avg_metric': best_avg_metric_regression,  # Using this for open-ended generation
            }
            
            # Save best model checkpoint
            if (epoch + 1) % args.save_ckpt_freq == 0:
                epoch_checkpoint_path = os.path.join(args.ckpt_dir, f'ckpt-{epoch + 1}.pt')
                torch.save(checkpoint, epoch_checkpoint_path)
                logger.info(f"Epoch checkpoint saved: {epoch_checkpoint_path}")

            # Save best model checkpoint based on average metric (only if evaluation was run)
            if run_evaluation and is_best_avg:
                best_checkpoint_path = os.path.join(args.ckpt_dir, 'best_avg_ckpt.pt')
                torch.save(checkpoint, best_checkpoint_path)
                logger.info(f"Best average checkpoint saved: {best_checkpoint_path}")

            # Save best checkpoint per dataset (only if evaluation was run)
            if run_evaluation:
                for data_name, is_best in is_best_list.items():
                    if is_best:
                        best_data_checkpoint_path = os.path.join(args.ckpt_dir, f'best_{data_name}_ckpt.pt')
                        torch.save(checkpoint, best_data_checkpoint_path)
                        logger.info(f"Best checkpoint for {data_name} saved: {best_data_checkpoint_path}")
            
            # Save detailed results to JSON files (only if evaluation was run)
            if run_evaluation:
                results_dir = os.path.join(args.ckpt_dir, 'results')
                os.makedirs(results_dir, exist_ok=True)
                
                # Save epoch-specific results
                epoch_results_file = os.path.join(results_dir, f'epoch_{epoch}_results.json')
                epoch_results = {
                    'epoch': epoch,
                    'validation': epoch_val_results,
                    'test': epoch_test_results,
                    'avg_validation_metric': current_avg_metric if current_avg_metric != -float('inf') else None,
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
                    'best_avg_metric': best_avg_metric_regression if best_avg_metric_regression != -float('inf') else None,
                    'best_metrics_per_dataset': best_metrics_per_dataset,
                    'best_epoch': epoch if is_best_avg else getattr(main, '_best_epoch', 0),
                }
                if is_best_avg:
                    main._best_epoch = epoch
                
                with open(best_metrics_file, 'w') as f:
                    json.dump(best_summary, f, indent=2)
        accelerator.wait_for_everyone()

    if args.wandb_log and accelerator.is_main_process:
        run.finish()
        logger.info("W&B tracking ended")
    accelerator.end_training()


def get_pred(pred_string, dataset_info):
    # get the next word after `Answer: `
    pred_words = pred_string.split(' ')
    pred_words = [w.strip() for w in pred_words if w.strip() != '']
    ans_idx = pred_words.index('Answer:')

    return ' '.join(pred_words[ans_idx + 1:])

@torch.no_grad()
def evaluate(model, dataloader, accelerator, args, logger, vocab_size, data_info, allowed_tokens=None, llm_workers=16):
    """Evaluate the model on validation data with LLM-based evaluation for open-ended answers""" 
    model.eval()
    
    # Store predictions and targets as strings/dicts for LLM evaluation
    all_pred_texts = []
    all_target_dicts = []
    
    # Each process evaluates its own data shard (already split by distributed sampler)
    for batch_idx, batch in enumerate(dataloader):
        X_fmri, X_text, targets_batch, gpt_mask = batch
        X_fmri = X_fmri.float()
        assert args.quantizer != 'titok'
        gpt_mask = gpt_mask.to(X_fmri.dtype)        

        if not args.lm_use_cls_head:
            # Generate text responses
            with accelerator.autocast():
                if accelerator.num_processes == 1:
                    generated_texts = model.generate(X_fmri, X_text, gpt_mask, max_new_tokens=100, text_gen=True, allowed_tokens=allowed_tokens)
                else:
                    generated_texts = model.module.generate(X_fmri, X_text, gpt_mask, max_new_tokens=100, text_gen=True, allowed_tokens=allowed_tokens, accelerator=accelerator)

            # Extract predictions from generated text
            for i, text in enumerate(generated_texts):
                try:
                    pred_text = get_pred(text, data_info)
                except:
                    pred_text = text  # Use full text if extraction fails
                
                all_pred_texts.append(pred_text)
        else:
            raise NotImplementedError("Classification head not supported for open-ended generation")
        
        # Handle targets_batch: convert from dict of lists to list of dicts
        if isinstance(targets_batch, dict):
            # Check if it's a dict of lists (collated format)
            # Get the first key to check the structure
            first_key = next(iter(targets_batch.keys()))
            first_value = targets_batch[first_key]
            
            if isinstance(first_value, (list, tuple)):
                # Dict of lists format: {'sex': ['Female', 'Male'], 'age': [25, 30]}
                # Convert to list of dicts: [{'sex': 'Female', 'age': 25}, {'sex': 'Male', 'age': 30}]
                batch_size = len(first_value)
                for i in range(batch_size):
                    sample_dict = {key: targets_batch[key][i] for key in targets_batch.keys()}
                    all_target_dicts.append(sample_dict)
            else:
                # Single sample dict format
                all_target_dicts.append(targets_batch)
        elif isinstance(targets_batch, list):
            # Already a list of dicts
            all_target_dicts.extend(targets_batch)
        else:
            raise ValueError(f"targets_batch should be a dict or list, got {type(targets_batch)}")
        
    # Each process performs LLM evaluation on its own data shard using parallel processing
    if accelerator.is_main_process:
        logger.info(f"Starting LLM evaluation for {len(all_pred_texts)} samples using {llm_workers} workers...")
    
    # Use parallel batch evaluation instead of sequential
    field_correctness = llm_evaluate_batch(all_pred_texts, all_target_dicts, max_workers=llm_workers)
    
    if accelerator.is_main_process:
        logger.info(f"LLM evaluation completed for {len(field_correctness)} samples")
    
    # Compute per-process metrics
    if len(field_correctness) == 0:
        # No samples in this process
        num_samples = 0
        overall_correct = 0
        field_correct_counts = {}
        all_fields_list = []
    else:
        num_samples = len(field_correctness)
        
        # All samples have the same fields, so just get from first sample
        all_fields_list = sorted(list(field_correctness[0].keys()))
        
        # Count correct predictions per field
        field_correct_counts = {}
        for field in all_fields_list:
            correct_count = sum(1 for result in field_correctness if result.get(field, False))
            field_correct_counts[field] = correct_count
        
        # Count samples where all fields are correct
        overall_correct = sum(1 for result in field_correctness if all(result.values()))
    
    # Convert counts to tensors for gathering
    num_samples_tensor = torch.tensor(num_samples, dtype=torch.long, device=accelerator.device)
    overall_correct_tensor = torch.tensor(overall_correct, dtype=torch.long, device=accelerator.device)
    
    # Gather counts from all processes
    gathered_num_samples = accelerator.gather(num_samples_tensor).sum().item()
    gathered_overall_correct = accelerator.gather(overall_correct_tensor).sum().item()
    
    # For field-level metrics, create tensor of counts (all processes have same fields)
    if len(all_fields_list) > 0:
        field_counts_list = [field_correct_counts[field] for field in all_fields_list]
        # Add batch dimension to ensure proper gathering (shape: [1, num_fields])
        field_counts_tensor = torch.tensor(field_counts_list, dtype=torch.long, device=accelerator.device).unsqueeze(0)
    else:
        # Empty tensor for processes with no samples - need to match the expected size
        # Assume 3 fields based on the data (sex, age_group, fluidintel_enc)
        field_counts_tensor = torch.tensor([[0, 0, 0]], dtype=torch.long, device=accelerator.device)
        if len(all_fields_list) == 0:
            # Populate all_fields_list for consistency
            all_fields_list = ['age_group', 'fluidintel_enc', 'sex']  # Sorted order
    
    # Gather field counts from all processes and sum
    # After gathering, shape will be [num_processes, num_fields]
    gathered_field_counts = accelerator.gather(field_counts_tensor)
    
    # Sum across processes (dimension 0)
    if gathered_field_counts.dim() > 1:
        gathered_field_counts = gathered_field_counts.sum(dim=0).squeeze()  # Sum and remove batch dim
    
    # Compute final metrics on main process
    if accelerator.is_main_process:
        if gathered_num_samples == 0:
            results = {
                'overall_accuracy': 0.0,
                'avg_field_accuracy': 0.0,
                'num_samples': 0,
            }
        else:
            # Overall accuracy
            overall_accuracy = gathered_overall_correct / gathered_num_samples
            
            # Field-level accuracies
            field_accuracies = {}
            if len(all_fields_list) > 0 and gathered_field_counts.numel() > 0:
                for i, field in enumerate(all_fields_list):
                    field_count = gathered_field_counts[i].item()
                    field_acc = field_count / gathered_num_samples
                    field_accuracies[f'{field}_accuracy'] = field_acc
            
            # Average field accuracy
            avg_field_accuracy = sum(field_accuracies.values()) / len(field_accuracies) if field_accuracies else 0.0
            
            results = {
                'overall_accuracy': overall_accuracy,
                'avg_field_accuracy': avg_field_accuracy,
                'num_samples': gathered_num_samples,
                **field_accuracies
            }
            
            logger.info(f"Evaluation complete: {gathered_num_samples} samples evaluated")
            logger.info(f"Overall accuracy: {overall_accuracy:.4f}, Avg field accuracy: {avg_field_accuracy:.4f}")
    else:
        results = {}
    
    model.train()
    return results


def get_args():
    def list_of_strs(arg):
        return arg.split(',')
    parser = argparse.ArgumentParser('NeuroLM training script', add_help=False)

    parser.add_argument('--datasets', type=list_of_strs, default=['UKB'],  help='list of dataset names to use for training')
    parser.add_argument('--add_src_info', default=True, action=argparse.BooleanOptionalAction, help='whether to add source dataset info to the prompt')
    parser.add_argument('--add_desc', default=False, action='store_true', help='whether to add subject medical descriptions to the prompt')

    parser.add_argument('--pretrained_ckpt', default='')
    parser.add_argument('--tokenizer_ckpt', default='checkpoints/tokenizer/UKB_robust-contr_sem/VQ-ViT_small-p32-clip_cls_last/ckpt.pt')
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
    parser.add_argument('--fmri_batch_size', default=2, type=int)
    parser.add_argument('--text_batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--save_ckpt', default=False, action=argparse.BooleanOptionalAction, help='whether to save checkpoints')
    parser.add_argument('--save_ckpt_freq', default=10, type=int)
    parser.add_argument('--eval_freq', default=10, type=int, help='evaluation frequency (in epochs)')
    parser.add_argument('--llm_eval_workers', default=32, type=int, help='number of parallel workers for LLM evaluation')
    parser.add_argument('--tune_tokenizer', action='store_true', help='whether to finetune the tokenizer during training', default=False)
    parser.add_argument('--use_random_prompt', action='store_true', help='whether to use random prompt during training/inference', default=False)

    parser.add_argument('--use_allowed_tokens', action='store_true', help='whether to restrict the generation to allowed tokens', default=False)

    parser.add_argument('--quantizer', type=str, default='vq')
    parser.add_argument('--lm_name', type=str, default='gpt2', help='name of the language model to use')
    parser.add_argument('--cfg_path', type=str, default='configs/vit_small_gpt2.yaml', help='path to the model config file',)
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