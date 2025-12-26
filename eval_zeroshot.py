"""
Zero-shot Evaluation Script for BrainFM
Adapted from train_instruction.py for inference-only evaluation
"""

import os
import argparse
import logging
import sys
import json
from datetime import datetime
import copy
import colorlog
from tqdm import tqdm
from collections import OrderedDict

import numpy as np
import torch
from omegaconf import OmegaConf
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed

from model_mindlm import MindLM
from quantizers import *
from model_gpt import MultimodalConfig
from dataset import get_fmri_data_inst
from utils import get_metrics, get_allowed_token_id
from load_pretrained_utils import load_pretrained_model


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
    logger = logging.getLogger('eval_zeroshot')
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


def create_output_dir(base_dir):
    """Create output directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{base_dir}_eval_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_config(output_dir, args, logger):
    """Save evaluation configuration"""
    config_file = os.path.join(output_dir, 'eval_config.json')
    args_dict = vars(args)
    with open(config_file, 'w') as f:
        json.dump(args_dict, f, indent=2)
    logger.info(f"Configuration saved to: {config_file}")


def get_pred(pred_string, dataset_info):
    """Extract prediction from generated text"""
    pred_words = pred_string.split(' ')
    pred_words = [w.strip() for w in pred_words if w.strip() != '']
    
    try:
        ans_idx = pred_words.index('Answer:')
    except ValueError:
        # If 'Answer:' not found, use default
        if dataset_info['target_name'] == 'sex':
            default_pred = ' Male'
        elif dataset_info['target_name'] in ['ADHD', 'ASD']:
            default_pred = ' Control'
        else:
            default_pred = ' CN'
        return dataset_info['label_dic'].get(default_pred, 0.)
    
    if dataset_info['target_name'] == 'sex':
        default_pred = ' Male'
    elif dataset_info['target_name'] in ['ADHD', 'ASD']:
        default_pred = ' Control'
    else:
        default_pred = ' CN'
    
    try:
        pred = pred_words[min(ans_idx + 1, len(pred_words) - 1)]
    except IndexError:
        print('Index out of range!', pred_words)
        pred = default_pred
    
    pred = dataset_info['label_dic'].get(pred, 0.)
    return pred


@torch.no_grad()
def evaluate(model, dataloader, accelerator, args, logger, data_info, allowed_tokens=None):
    """Evaluate the model on data with proper DDP support"""
    model.eval()
    all_preds, all_targets = [], []
    all_generated_texts = []
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", disable=not accelerator.is_main_process)):
        X_fmri, X_text, label, gpt_mask = batch
        X_fmri = X_fmri.float()
        gpt_mask = gpt_mask.to(X_fmri.dtype)
        
        if not args.lm_use_cls_head:
            # Text generation mode
            with accelerator.autocast():
                if accelerator.num_processes == 1:
                    text = model.generate(X_fmri, X_text, gpt_mask, max_new_tokens=4, 
                                        text_gen=True, allowed_tokens=allowed_tokens)
                else:
                    text = model.module.generate(X_fmri, X_text, gpt_mask, max_new_tokens=4, 
                                                text_gen=True, allowed_tokens=allowed_tokens, 
                                                accelerator=accelerator)
            
            batch_preds = []
            for i, t in enumerate(text):
                pred = get_pred(t, data_info)
                if not data_info['is_binary']:
                    pred = torch.eye(data_info['num_classes'])[pred]
                batch_preds.append(pred)
                all_generated_texts.append(t)
        else:
            # Classification head mode
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
    
    # Compute metrics only on main process
    if accelerator.is_main_process:
        final_preds = gathered_preds.cpu().numpy()
        final_targets = gathered_targets.cpu().numpy()
        
        results = get_metrics(final_preds, final_targets, data_info['metrics'], 
                            data_info['is_binary'], data_info['is_regression'])
        
        # Add generated texts to results if available
        if all_generated_texts and not args.lm_use_cls_head:
            results['generated_texts_sample'] = all_generated_texts[:10]  # Save first 10 as sample
    else:
        results = {}
    
    return results


def main(args):
    # Create output directory
    output_dir = create_output_dir(args.output_dir)
    
    # Setup logging
    log_file = os.path.join(output_dir, 'evaluation.log')
    logger = setup_logging(log_file=log_file)
    logger.info("="*80)
    logger.info("Zero-shot Evaluation Script")
    logger.info("="*80)
    
    # Save configuration
    save_config(output_dir, args, logger)
    
    # Validate checkpoint path
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Initialize Accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    logger.info(f"Accelerator initialized with device: {accelerator.device}")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")
    
    # Enable optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Load model configuration
    model_cfg = OmegaConf.load(args.cfg_path).model
    quantizer_cfg = model_cfg.vq_model
    quantizer_cfg.img_size = (quantizer_cfg.num_rois, quantizer_cfg.num_timestamp)
    lm_cfg = model_cfg.lm
    lm_cfg.base_model = args.lm_name
    
    # Load data
    logger.info(f"Loading data for datasets: {args.datasets}")
    _, data_loader_val_test = get_fmri_data_inst(
        args.batch_size,
        args.batch_size,
        args.datasets,
        lm_name=lm_cfg.base_model,
        norm='robust',
        patch_size=quantizer_cfg.patch_size,
        next_time_mask=(args.quantizer != 'titok'),
        use_random_prompt=args.use_random_prompt,
        add_source_info=args.add_src_info,
        add_desc=args.add_desc,
    )
    logger.info(f"Data loaders created for {len(data_loader_val_test)} datasets")
    
    # Create quantizer
    if args.quantizer == 'vq':
        quantizer_cls = VQ
    elif args.quantizer == 'fsq':
        quantizer_cls = FSQ_Model
    elif args.quantizer == 'titok':
        quantizer_cls = TiTok
    else:
        raise ValueError(f"Unknown quantizer: {args.quantizer}")
    
    logger.info(f"Creating {args.quantizer} tokenizer...")
    if args.quantizer == 'titok':
        tokenizer = quantizer_cls(
            quantizer=quantizer_cfg.quantizer,
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
    
    # Load checkpoint using utility function (handles LoRA properly)
    logger.info(f"Loading checkpoint from: {args.checkpoint}")
    n_embd = quantizer_cfg.n_embd
    
    model, checkpoint = load_pretrained_model(
        checkpoint_path=args.checkpoint,
        tokenizer_encoder=tokenizer_encoder,
        tune_tokenizer=False,  # No tuning during evaluation
        freeze_llm=False,
        num_rois=quantizer_cfg.num_rois,
        n_embd=n_embd,
        eeg_vocab_size=tokenizer.codebook_size,
        latent_tokens=latent_tokens,
        lora_adapter_path=args.lora_adapter if hasattr(args, 'lora_adapter') else None,
        device='cpu'  # Will move to GPU after accelerator.prepare
    )
    
    logger.info("Model loaded successfully")
    
    # Get model info
    num_params = model.get_num_params()
    vocab_size = model.llm.original_vocab_size
    model_args = checkpoint.get('model_args', lm_cfg)
    args.lm_use_cls_head = model_args.get('use_cls_head', False)
    
    logger.info(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    logger.info(f"Using classification head: {args.lm_use_cls_head}")
    
    # Prepare model with accelerator
    model = accelerator.prepare(model)
    for dataset_target, loaders_dict in data_loader_val_test.items():
        for split in ['val', 'test']:
            loaders_dict[split] = accelerator.prepare(loaders_dict[split])
    
    if accelerator.num_processes > 1:
        model._set_static_graph()
        logger.warning('Set model with static computational graph')
    
    # Free up memory
    del tokenizer
    del checkpoint
    
    # Evaluation
    logger.info("="*80)
    logger.info("Starting evaluation...")
    logger.info("="*80)
    
    all_results = {'validation': {}, 'test': {}}
    
    for data_name in data_loader_val_test:
        logger.info(f"\nEvaluating dataset: {data_name}")
        logger.info("-"*80)
        
        # Get allowed tokens if specified
        allowed_tokens = None
        if args.use_allowed_tokens:
            if accelerator.num_processes == 1:
                allowed_tokens = get_allowed_token_id(
                    data_loader_val_test[data_name]['info']['target_name'], 
                    model.llm.tokenizer
                )
            else:
                allowed_tokens = get_allowed_token_id(
                    data_loader_val_test[data_name]['info']['target_name'], 
                    model.module.llm.tokenizer
                )
            logger.info(f"Using allowed tokens restriction")
        
        # Evaluate validation set
        if args.eval_val:
            logger.info(f"Evaluating validation set for {data_name}...")
            results_val = evaluate(
                model, 
                data_loader_val_test[data_name]['val'], 
                accelerator, 
                args, 
                logger, 
                data_loader_val_test[data_name]['info'], 
                allowed_tokens=allowed_tokens
            )
            
            if accelerator.is_main_process:
                all_results['validation'][data_name] = results_val
                logger.info(f"Validation results for {data_name}:")
                for metric, value in results_val.items():
                    if metric != 'generated_texts_sample':
                        logger.info(f"  {metric}: {value}")
        
        # Evaluate test set
        if args.eval_test:
            logger.info(f"Evaluating test set for {data_name}...")
            results_test = evaluate(
                model, 
                data_loader_val_test[data_name]['test'], 
                accelerator, 
                args, 
                logger, 
                data_loader_val_test[data_name]['info'], 
                allowed_tokens=allowed_tokens
            )
            
            if accelerator.is_main_process:
                all_results['test'][data_name] = results_test
                logger.info(f"Test results for {data_name}:")
                for metric, value in results_test.items():
                    if metric != 'generated_texts_sample':
                        logger.info(f"  {metric}: {value}")
    
    # Save results
    if accelerator.is_main_process:
        results_file = os.path.join(output_dir, 'evaluation_results.json')
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\nResults saved to: {results_file}")
        
        # Save summary
        summary_file = os.path.join(output_dir, 'evaluation_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("Zero-shot Evaluation Summary\n")
            f.write("="*80 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Datasets: {args.datasets}\n")
            f.write(f"Model parameters: {num_params:,}\n")
            f.write("\n" + "="*80 + "\n")
            
            for split in ['validation', 'test']:
                if all_results[split]:
                    f.write(f"\n{split.upper()} RESULTS:\n")
                    f.write("-"*80 + "\n")
                    for data_name, results in all_results[split].items():
                        f.write(f"\nDataset: {data_name}\n")
                        for metric, value in results.items():
                            if metric != 'generated_texts_sample':
                                f.write(f"  {metric}: {value}\n")
        
        logger.info(f"Summary saved to: {summary_file}")
        
        # Print final summary
        logger.info("\n" + "="*80)
        logger.info("EVALUATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Results file: {results_file}")
        logger.info(f"Summary file: {summary_file}")
    
    accelerator.end_training()


def get_args():
    def list_of_strs(arg):
        return arg.split(',')
    
    parser = argparse.ArgumentParser('Zero-shot evaluation script for BrainFM', add_help=False)
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--lora_adapter', type=str, default=None,
                       help='Path to LoRA adapter directory (optional, auto-detected if not provided)')
    parser.add_argument('--cfg_path', type=str, required=True,
                       help='Path to model config file (YAML)')
    
    # Dataset arguments
    parser.add_argument('--datasets', type=list_of_strs, default=['UKB'],
                       help='Comma-separated list of dataset names')
    parser.add_argument('--add_src_info', default=False, action='store_true',
                       help='Add source dataset info to the prompt')
    parser.add_argument('--add_desc', default=False, action='store_true',
                       help='Add subject medical descriptions to the prompt')
    
    # Evaluation arguments
    parser.add_argument('--eval_val', default=True, action=argparse.BooleanOptionalAction,
                       help='Evaluate validation set')
    parser.add_argument('--eval_test', default=True, action=argparse.BooleanOptionalAction,
                       help='Evaluate test set')
    parser.add_argument('--batch_size', default=16, type=int,
                       help='Batch size for evaluation')
    
    # Model arguments
    parser.add_argument('--quantizer', type=str, default='vq',
                       choices=['vq', 'fsq', 'titok'],
                       help='Quantizer type')
    parser.add_argument('--use_random_prompt', action='store_true', default=False,
                       help='Use random prompt during inference')
    parser.add_argument('--use_allowed_tokens', action='store_true', default=False,
                       help='Restrict generation to allowed tokens')
    parser.add_argument('--lm_name', type=str, default='gpt2',
                       help='Language model name')
    
    # Output arguments
    parser.add_argument('--output_dir', default='eval_results',
                       help='Base directory for output')
    
    # Other arguments
    parser.add_argument('--seed', default=1337, type=int,
                       help='Random seed')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
