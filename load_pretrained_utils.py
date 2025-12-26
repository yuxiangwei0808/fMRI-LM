"""
Utility functions for loading pretrained models with or without LoRA adapters.
"""
import torch
import os
from model_mindlm import MindLM
from model_gpt import MultimodalConfig


def load_pretrained_model(checkpoint_path, tokenizer_encoder, tune_tokenizer=False, 
                          freeze_llm=False, num_rois=450, n_embd=768, 
                          eeg_vocab_size=8192, latent_tokens=None, 
                          lora_adapter_path=None, device='cuda'):
    """
    Load a pretrained model with proper handling of LoRA adapters.
    
    Args:
        checkpoint_path: Path to the main checkpoint file (.pt)
        tokenizer_encoder: The tokenizer encoder
        tune_tokenizer: Whether to tune the tokenizer
        freeze_llm: Whether to freeze the LLM
        num_rois: Number of ROIs
        n_embd: Embedding dimension
        eeg_vocab_size: EEG vocabulary size
        latent_tokens: Latent tokens for TiTok
        lora_adapter_path: Path to LoRA adapter directory (optional)
                          If None, will try to find it automatically
        device: Device to load model to
    
    Returns:
        model: Loaded model
        checkpoint: The full checkpoint dict (for accessing metadata)
    
    Usage:
        # Option 1: Load model with full checkpoint (includes LoRA if trained with it)
        model, ckpt = load_pretrained_model(
            checkpoint_path='checkpoints/model/ckpt.pt',
            tokenizer_encoder=tokenizer.encoder,
            num_rois=450,
            n_embd=768,
            eeg_vocab_size=8192
        )
        
        # Option 2: Load base model then add LoRA adapter separately
        model, ckpt = load_pretrained_model(
            checkpoint_path='checkpoints/model/ckpt.pt',
            tokenizer_encoder=tokenizer.encoder,
            lora_adapter_path='checkpoints/model/lora_adapter',
            num_rois=450,
            n_embd=768,
            eeg_vocab_size=8192
        )
    """
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_args = checkpoint['model_args']
    
    # Create config
    gptconf = MultimodalConfig(**model_args)
    use_peft = model_args.get('peft_tune', False)
    
    # Check if loading with separate LoRA adapter
    if lora_adapter_path is None and use_peft:
        # Try to find LoRA adapter automatically
        checkpoint_dir = os.path.dirname(checkpoint_path)
        auto_lora_path = os.path.join(checkpoint_dir, 'lora_adapter')
        if os.path.exists(auto_lora_path):
            lora_adapter_path = auto_lora_path
            print(f"Found LoRA adapter automatically at: {lora_adapter_path}")
    
    # Decide loading strategy
    if lora_adapter_path and os.path.exists(lora_adapter_path):
        # Strategy 1: Load base model from checkpoint + LoRA adapter separately
        print("Loading with separate LoRA adapter...")
        
        # Create model without LoRA first
        temp_peft_flag = gptconf.peft_tune
        gptconf.peft_tune = False
        
        model = MindLM(gptconf, tokenizer_encoder, tune_tokenizer, freeze_llm,
                      num_rois=num_rois, n_embd=n_embd, 
                      eeg_vocab_size=eeg_vocab_size, latent_tokens=latent_tokens)
        
        # Load base model weights (without LoRA)
        state_dict = checkpoint['model']
        # Filter out LoRA-related keys if they exist
        base_state_dict = {k: v for k, v in state_dict.items() 
                          if 'lora_' not in k and 'modules_to_save' not in k}
        
        # Handle _orig_mod prefix
        unwanted_prefix = '_orig_mod.'
        for k, v in list(base_state_dict.items()):
            if k.startswith(unwanted_prefix):
                base_state_dict[k[len(unwanted_prefix):]] = base_state_dict.pop(k)
        
        model.load_state_dict(base_state_dict, strict=False)
        
        # Apply LoRA and load adapter weights
        if temp_peft_flag:
            model.llm.apply_lora()
            model.llm.load_lora_adapter(lora_adapter_path)
            print(f"✓ Loaded base model + LoRA adapter from {lora_adapter_path}")
        
    else:
        # Strategy 2: Load full state dict (includes LoRA weights if present)
        print("Loading full model state dict...")
        
        # Create model WITHOUT applying LoRA (will load full merged state)
        temp_peft_flag = gptconf.peft_tune
        gptconf.peft_tune = False
        
        model = MindLM(gptconf, tokenizer_encoder, tune_tokenizer, freeze_llm,
                      num_rois=num_rois, n_embd=n_embd, 
                      eeg_vocab_size=eeg_vocab_size, latent_tokens=latent_tokens)
        
        # Load full state dict
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if use_peft:
            print(f"✓ Loaded model trained with LoRA (full state dict)")
            if missing_keys:
                print(f"  Warning: Missing keys: {missing_keys[:5]}...")  # Show first 5
            if unexpected_keys:
                print(f"  Warning: Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
        else:
            print(f"✓ Loaded model (no LoRA)")
    
    # Move to device
    model = model.to(device)
    
    # Print summary
    print(f"Model loaded successfully!")
    print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - Total params: {model.get_num_params():,}")
    print(f"  - PEFT enabled: {use_peft}")
    
    return model, checkpoint


def save_model_with_lora(model, checkpoint_dict, save_dir, save_lora_separately=True):
    """
    Save model checkpoint with proper LoRA handling.
    
    Args:
        model: The model to save (unwrapped)
        checkpoint_dict: Dictionary containing optimizer state, metrics, etc.
        save_dir: Directory to save checkpoint
        save_lora_separately: Whether to save LoRA adapter separately
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save full state dict
    checkpoint_dict['model'] = model.state_dict()
    checkpoint_path = os.path.join(save_dir, 'ckpt.pt')
    torch.save(checkpoint_dict, checkpoint_path)
    print(f"Saved checkpoint to: {checkpoint_path}")
    
    # Check if model uses LoRA
    if hasattr(model, 'llm') and hasattr(model.llm, 'base_model'):
        if hasattr(model.llm.base_model, 'peft_config'):
            if save_lora_separately:
                lora_dir = os.path.join(save_dir, 'lora_adapter')
                model.llm.save_pretrained_lora(lora_dir)
                print(f"Saved LoRA adapter to: {lora_dir}")
            else:
                print("LoRA weights included in main checkpoint")


# Example usage
if __name__ == "__main__":
    # Example 1: Load a model trained with LoRA
    """
    model, ckpt = load_pretrained_model(
        checkpoint_path='checkpoints/model/ckpt.pt',
        tokenizer_encoder=tokenizer.encoder,
        num_rois=450,
        n_embd=768,
        eeg_vocab_size=8192,
        device='cuda'
    )
    """
    
    # Example 2: Load with explicit LoRA adapter path
    """
    model, ckpt = load_pretrained_model(
        checkpoint_path='checkpoints/model/ckpt.pt',
        tokenizer_encoder=tokenizer.encoder,
        lora_adapter_path='checkpoints/model/lora_adapter',
        num_rois=450,
        n_embd=768,
        eeg_vocab_size=8192,
        device='cuda'
    )
    """
    pass
