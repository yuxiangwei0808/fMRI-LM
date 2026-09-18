#!/usr/bin/env python3
"""
Merge DeepSpeed checkpoint components into a standard PyTorch checkpoint.

This script combines:
1. pytorch_model.bin OR sharded pytorch_model-*.bin files (model weights from zero_to_fp32.py)
2. metadata.pt (model_args and training metrics)

Into a single .pt file that can be loaded with standard PyTorch code.

Handles both:
- Single file: pytorch_model.bin (for smaller models <2GB)
- Sharded files: pytorch_model-00001-of-00002.bin, pytorch_model-00002-of-00002.bin, etc. (for larger models)

Usage:
    # First run zero_to_fp32.py to create pytorch_model.bin or sharded files:
    cd <deepspeed_checkpoint_dir>
    python zero_to_fp32.py . .
    
    # Then merge with metadata:
    python merge_deepspeed_checkpoint.py \
        --deepspeed_dir checkpoints/pretrain/deepspeed_checkpoint_best_f2t \
        --output_path checkpoints/pretrain/best_f2t.pt
"""

import argparse
import os
import json
import glob
import torch
from collections import OrderedDict


def load_sharded_checkpoint(checkpoint_dir):
    """
    Load checkpoint from sharded model files.
    
    For large models, PyTorch saves weights across multiple files:
    - pytorch_model-00001-of-00002.bin
    - pytorch_model-00002-of-00002.bin
    - pytorch_model.bin.index.json (maps parameter names to shard files)
    """
    index_file = os.path.join(checkpoint_dir, 'pytorch_model.bin.index.json')
    
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Shard index file not found: {index_file}")
    
    # Load the index to understand the sharding structure
    with open(index_file, 'r') as f:
        index = json.load(f)
    
    print(f"Found sharded checkpoint with {len(index['weight_map'])} parameters")
    
    # Get unique shard files
    shard_files = sorted(set(index['weight_map'].values()))
    print(f"Loading {len(shard_files)} shard files:")
    
    # Load all shards and merge
    state_dict = OrderedDict()
    for shard_file in shard_files:
        shard_path = os.path.join(checkpoint_dir, shard_file)
        print(f"  Loading {shard_file}...")
        shard_state = torch.load(shard_path, map_location='cpu', weights_only=False)
        state_dict.update(shard_state)
        del shard_state  # Free memory
    
    print(f"Successfully merged {len(state_dict)} weight tensors from shards")
    return state_dict


def merge_checkpoint(args):
    """Merge pytorch_model.bin (or sharded files) and metadata.pt into a single checkpoint"""
    
    print(f"Loading checkpoint from: {args.deepspeed_dir}")
    
    # Check for required files
    pytorch_model_path = os.path.join(args.deepspeed_dir, 'pytorch_model.bin')
    pytorch_model_index = os.path.join(args.deepspeed_dir, 'pytorch_model.bin.index.json')
    metadata_path = os.path.join(args.deepspeed_dir, 'metadata.pt')
    
    # Detect if we have sharded or single file checkpoint
    is_sharded = os.path.exists(pytorch_model_index)
    has_single_file = os.path.exists(pytorch_model_path)
    
    if not is_sharded and not has_single_file:
        raise FileNotFoundError(
            f"No model weights found at: {args.deepspeed_dir}\n"
            f"Please run zero_to_fp32.py first:\n"
            f"  cd {args.deepspeed_dir}\n"
            f"  python zero_to_fp32.py . ."
        )
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.pt not found at: {metadata_path}")
    
    # Load model weights - handle both sharded and single file
    if is_sharded:
        print("Detected sharded checkpoint (large model)...")
        state_dict = load_sharded_checkpoint(args.deepspeed_dir)
    else:
        print("Loading model weights from single pytorch_model.bin...")
        state_dict = torch.load(pytorch_model_path, map_location='cpu', weights_only=False)
        print(f"  Loaded {len(state_dict)} weight tensors")
    
    # Load metadata
    print("Loading metadata from metadata.pt...")
    metadata = torch.load(metadata_path, map_location='cpu', weights_only=False)
    print(f"  Metadata keys: {list(metadata.keys())}")
    
    # Create merged checkpoint
    checkpoint = {
        'model': state_dict,
        'model_args': metadata.get('model_args'),
        'iter_num': metadata.get('iter_num', 0),
        'epoch': metadata.get('epoch', 0),
        'total_loss': metadata.get('total_loss', 0),
        'fmri_loss': metadata.get('fmri_loss', 0),
        'text_loss': metadata.get('text_loss', 0),
        'f2t_loss': metadata.get('f2t_loss', 0),
        'fmri_acc': metadata.get('fmri_acc', 0),
        'text_acc': metadata.get('text_acc', 0),
        'f2t_acc': metadata.get('f2t_acc', 0),
        'best_f2t_loss': metadata.get('best_f2t_loss', float('inf')),
    }
    
    # Display info
    print("\nCheckpoint information:")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Iteration: {checkpoint['iter_num']}")
    print(f"  Total Loss: {checkpoint['total_loss']:.4f}")
    print(f"  fMRI Loss: {checkpoint['fmri_loss']:.4f}")
    print(f"  Text Loss: {checkpoint['text_loss']:.4f}")
    print(f"  F2T Loss: {checkpoint['f2t_loss']:.4f}")
    print(f"  Best F2T Loss: {checkpoint['best_f2t_loss']:.4f}")
    
    if checkpoint['model_args']:
        model_args = checkpoint['model_args']
        print(f"\nModel configuration:")
        print(f"  Base model: {model_args.get('base_model', 'N/A')}")
        print(f"  PEFT/LoRA: {model_args.get('peft_tune', False)}")
        if hasattr(model_args, 'n_layer'):
            print(f"  Layers: {model_args.get('n_layer', 'N/A')}")
        if hasattr(model_args, 'n_embd'):
            print(f"  Embedding dim: {model_args.get('n_embd', 'N/A')}")
    
    # Save merged checkpoint
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    print(f"\nSaving merged checkpoint to: {args.output_path}")
    print("  (This may take a few minutes for large models...)")
    torch.save(checkpoint, args.output_path)
    
    # Display file size
    file_size_mb = os.path.getsize(args.output_path) / 1024 / 1024
    file_size_gb = file_size_mb / 1024
    
    if file_size_gb >= 1:
        print(f"✓ Successfully saved checkpoint ({file_size_gb:.2f} GB)")
    else:
        print(f"✓ Successfully saved checkpoint ({file_size_mb:.2f} MB)")
    
    print(f"\nYou can now load this checkpoint with:")
    print(f"  checkpoint = torch.load('{args.output_path}')")
    print(f"  model.load_state_dict(checkpoint['model'])")
    
    # Cleanup info for sharded checkpoints
    if is_sharded:
        print(f"\nNote: Original sharded files are still in {args.deepspeed_dir}")
        print(f"      You can delete them to save disk space if desired.")
    print(f"  model_args = checkpoint['model_args']")


def main():
    parser = argparse.ArgumentParser(
        description='Merge DeepSpeed checkpoint components into standard PyTorch format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example workflow:
  1. First, convert DeepSpeed checkpoint to pytorch_model.bin:
     cd checkpoints/pretrain/deepspeed_checkpoint_best_f2t
     python zero_to_fp32.py . .
  
  2. Then merge with metadata:
     python merge_deepspeed_checkpoint.py \\
         --deepspeed_dir checkpoints/pretrain/deepspeed_checkpoint_best_f2t \\
         --output_path checkpoints/pretrain/best_f2t.pt
        """
    )
    parser.add_argument(
        '--deepspeed_dir', 
        type=str, 
        required=True,
        help='Path to DeepSpeed checkpoint directory (must contain pytorch_model.bin and metadata.pt)'
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        help='Output path for merged .pt file',
        default=None,
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.deepspeed_dir):
        raise FileNotFoundError(f"DeepSpeed checkpoint directory not found: {args.deepspeed_dir}")
    
    if args.output_path is None:
        args.output_path = os.path.join(args.deepspeed_dir, 'merged_checkpoint.pt')
    
    if os.path.exists(args.output_path):
        response = input(f"Output file {args.output_path} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Merge cancelled.")
            return
    
    merge_checkpoint(args)


if __name__ == '__main__':
    main()
