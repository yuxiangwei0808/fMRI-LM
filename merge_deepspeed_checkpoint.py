#!/usr/bin/env python3
"""
Merge DeepSpeed checkpoint components into a standard PyTorch checkpoint.

This script combines:
1. pytorch_model.bin (model weights from zero_to_fp32.py)
2. metadata.pt (model_args and training metrics)

Into a single .pt file that can be loaded with standard PyTorch code.

Usage:
    # First run zero_to_fp32.py to create pytorch_model.bin:
    cd <deepspeed_checkpoint_dir>
    python zero_to_fp32.py . .
    
    # Then merge with metadata:
    python merge_deepspeed_checkpoint.py \
        --deepspeed_dir checkpoints/pretrain/deepspeed_checkpoint_best_f2t \
        --output_path checkpoints/pretrain/best_f2t.pt
"""

import argparse
import os
import torch


def merge_checkpoint(args):
    """Merge pytorch_model.bin and metadata.pt into a single checkpoint"""
    
    print(f"Loading checkpoint from: {args.deepspeed_dir}")
    
    # Check for required files
    pytorch_model_path = os.path.join(args.deepspeed_dir, 'pytorch_model.bin')
    metadata_path = os.path.join(args.deepspeed_dir, 'metadata.pt')
    
    if not os.path.exists(pytorch_model_path):
        raise FileNotFoundError(
            f"pytorch_model.bin not found at: {pytorch_model_path}\n"
            f"Please run zero_to_fp32.py first:\n"
            f"  cd {args.deepspeed_dir}\n"
            f"  python zero_to_fp32.py . ."
        )
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.pt not found at: {metadata_path}")
    
    # Load model weights
    print("Loading model weights from pytorch_model.bin...")
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
    torch.save(checkpoint, args.output_path)
    
    # Display file size
    file_size_mb = os.path.getsize(args.output_path) / 1024 / 1024
    print(f"✓ Successfully saved checkpoint ({file_size_mb:.2f} MB)")
    print(f"\nYou can now load this checkpoint with:")
    print(f"  checkpoint = torch.load('{args.output_path}')")
    print(f"  model.load_state_dict(checkpoint['model'])")
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
