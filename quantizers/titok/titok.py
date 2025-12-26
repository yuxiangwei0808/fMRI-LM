"""Follow TiTok's Vector Quantization"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import inspect
from einops import rearrange
from omegaconf import OmegaConf
from transformers import GPT2LMHeadModel, AutoModelForCausalLM
import warnings

from brain_encoder.titok_models_vanilla import TiTokEncoder, TiTokDecoder
from .vector_quantizer import *
from ..norm_ema_quantizer import NormEMAVectorQuantizer


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class TiTok(nn.Module):
    def __init__(self,
                 quantizer='vq',
                 num_latent_tokens=64,
                 latent_token_size=12,
                 model_size='small',
                 image_size=(450, 490),
                 patch_size=16,
                 codebook_size=8192,
                 commitment_cost=0.25,
                 use_l2_norm=True,
                 ):
        super().__init__()
        self.encoder = TiTokEncoder(num_latent_tokens, latent_token_size, model_size, image_size, patch_size,)
        self.decoder = TiTokDecoder(num_latent_tokens, latent_token_size, model_size, image_size, patch_size,)
        self.quantizer = quantizer
        self.scale = None  # scale reconst loss
        
        self.num_latent_tokens = num_latent_tokens
        self.codebook_size = codebook_size
        scale = self.encoder.width ** -0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.width))
        
        self.apply(self._init_weights)

        if quantizer == 'vq':
            self.quantize = VectorQuantizer(
                codebook_size=codebook_size,
                token_size=latent_token_size,
                commitment_cost=commitment_cost,
                use_l2_norm=use_l2_norm,)
        elif quantizer == 'norm_vq':
            self.quantize = NormEMAVectorQuantizer(codebook_size, latent_token_size, beta=1.0)
        elif quantizer == 'fsq':
            self.quantize = ...

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x):
        batch_size, n, t = x.shape
        z = self.encoder(x, self.latent_tokens)  # B N C
        z = z.transpose(-1, -2).unsqueeze(2)

        if self.quantizer != 'vq': z = z.squeeze(2)
        z_quantized, loss, encoding_indices = self.quantize(z)

        return z_quantized, encoding_indices, loss, z
    
    def decode(self, z_quantized):
        decoded = self.decoder(z_quantized)
        return decoded

    def get_codebook_indices(self, x, **kwargs):
        _, encoding_indices, _, _ = self.encode(x)
        return encoding_indices.view(x.shape[0], -1)

    def forward(self, x, y, **kwargs):
        z_quantized, embed_indices, emb_loss, encoded_features = self.encode(x)
        decoded = self.decode(z_quantized)

        rec_loss = nn.MSELoss()(decoded, y)
        if self.scale is None: self.scale = rec_loss.detach().mean().item() / emb_loss.detach().mean().item()
        rec_loss /= self.scale
        loss = rec_loss + emb_loss

        split="train" if self.training else "val"
        log = {
            f"{split}/rec_raw_loss": rec_loss.detach().mean().item(),
            f"{split}/quant_loss": emb_loss.detach().mean().item(),
            f"{split}/total_loss": loss.detach().mean().item()
        }

        return loss, encoded_features, log
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer


class TiTok_Align(nn.Module):
    def __init__(self,
                 config
                #  text_model_name='gpt2',  # Support for different LLMs, defaults to GPT2 for backward compatibility
                #  text_hidden_dim=None,    # Auto-detect if None
                 ):
        super(TiTok_Align, self).__init__()
        image_size = (config.num_rois, config.num_timestamp)
        self.TiTok = TiTok(
            quantizer=config.quantizer,
            num_latent_tokens=config.num_latent_tokens,
            latent_token_size=config.latent_token_size,
            model_size=config.model_size,
            image_size=image_size,
            patch_size=config.patch_size,
            codebook_size=config.codebook_size,
            commitment_cost=config.commitment_cost,
            use_l2_norm=config.use_l2_norm,
        )
        self.num_tokens = self.TiTok.num_latent_tokens

        # Get the encoder width based on model size
        encoder_width = {
            "small": 512,
            "base": 768,
            "large": 1024,
        }[config.model_size]

        # Load text model and get dimensions
        self.text_model_name = config.text_model_name
        text_model, text_hidden_dim, vocab_size = self._load_text_model(self.text_model_name, config.text_hidden_dim)
        self.text_hidden_dim = text_hidden_dim
        
        # TODO currently used latent_token_size instead of encoder_width becuase original TiTok encoder has a projector at the output
        self.x_proj = nn.Linear(config.latent_token_size, text_hidden_dim)
        self.domain_classifier = nn.Sequential(
                nn.Linear(text_hidden_dim, text_hidden_dim // 3),  # Adaptive intermediate dimension
                nn.GELU(),
                nn.Linear(text_hidden_dim // 3, 2)
            )

        # Extract word embeddings from the text model
        self.wte = nn.Embedding(vocab_size, text_hidden_dim, _freeze=True)
        self.wte.weight.data = self._extract_embeddings(text_model).clone()
        
        self.x_proj.apply(self._init_weights)
        self.domain_classifier.apply(self._init_weights)
    
    def _load_text_model(self, model_name, text_hidden_dim=None):
        """Load text model and extract configuration."""
        if model_name.startswith('gpt2'):
            from transformers import GPT2LMHeadModel
            model = GPT2LMHeadModel.from_pretrained(model_name)
            auto_hidden_dim = model.config.n_embd
            vocab_size = model.config.vocab_size
        elif 'llama' in model_name.lower():
            from transformers import LlamaForCausalLM
            model = LlamaForCausalLM.from_pretrained(model_name, cache_dir='data/cache')
            auto_hidden_dim = model.config.hidden_size
            vocab_size = model.config.vocab_size
        elif 'qwen' in model_name.lower():
            from transformers import Qwen2ForCausalLM
            model = Qwen2ForCausalLM.from_pretrained(model_name, cache_dir='data/cache')
            auto_hidden_dim = model.config.hidden_size
            vocab_size = model.config.vocab_size
        elif 'mistral' in model_name.lower():
            from transformers import MistralForCausalLM
            model = MistralForCausalLM.from_pretrained(model_name, cache_dir='data/cache')
            auto_hidden_dim = model.config.hidden_size
            vocab_size = model.config.vocab_size
        else:
            # Generic approach for other HuggingFace models
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='data/cache')
            # Try different possible attribute names for hidden dimension
            for attr in ['hidden_size', 'n_embd', 'd_model', 'embed_dim']:
                if hasattr(model.config, attr):
                    auto_hidden_dim = getattr(model.config, attr)
                    break
            else:
                raise ValueError(f"Could not determine hidden dimension for model {model_name}")
            vocab_size = model.config.vocab_size
        
        # Use provided dimension or auto-detected one
        final_hidden_dim = text_hidden_dim if text_hidden_dim is not None else auto_hidden_dim
        print(f"Using text model: {model_name}, hidden_dim: {final_hidden_dim}, vocab_size: {vocab_size}")
        
        return model, final_hidden_dim, vocab_size
    
    def _extract_embeddings(self, model):
        """Extract word embeddings from different model architectures."""
        # Try different possible locations for embeddings
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
            # GPT2-style
            return model.transformer.wte.weight.data
        elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            # LLaMA/Mistral-style
            return model.model.embed_tokens.weight.data
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
            # Some other transformer variants
            return model.transformer.wte.weight.data
        elif hasattr(model, 'embed_tokens'):
            # Direct embedding access
            return model.embed_tokens.weight.data
        else:
            # Search through all modules
            for name, module in model.named_modules():
                if isinstance(module, nn.Embedding) and 'embed' in name.lower():
                    print(f"Found embeddings at: {name}")
                    return module.weight.data
            raise ValueError(f"Could not find embeddings in model {type(model)}")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, y=None, alpha=0):
        if y is not None:
            loss, encoder_features, log = self.TiTok(x, y, alpha=alpha)  # B latent_token_size 1 num_latent_token
            encoder_features = encoder_features.squeeze(2).transpose(-1, -2)
            encoder_features = self.x_proj(encoder_features)
            reverse_x = ReverseLayerF.apply(encoder_features, alpha)
            domain_out = self.domain_classifier(reverse_x)
            target = torch.full((domain_out.size(0), domain_out.size(1)), fill_value=0, device=x.device)
            domain_loss = F.cross_entropy(domain_out.view(-1, domain_out.size(-1)), target.view(-1), ignore_index=-1)
            split="train" if self.training else "val"
            log[f'{split}/domain_loss'] = domain_loss.detach().item()
            return loss, domain_loss, log
        else:
            x = self.wte(x).detach()
            domain_out = self.domain_classifier(x)
            domain_loss = F.cross_entropy(domain_out.view(-1, domain_out.size(-1)), torch.ones((x.size(0) * x.size(1),), device=x.device).long(), ignore_index=-1)
            return domain_loss
        
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters 
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer


def create_titok_align_for_llm(llm_name, **titok_kwargs):
    """Factory function to create TiTok_Align for specific LLMs with predefined configurations.
    
    Args:
        llm_name (str): Name of the LLM. Supported: 'gpt2', 'llama-7b', 'llama-13b', 'qwen-7b', etc.
        **titok_kwargs: Additional arguments for TiTok_Align
    
    Returns:
        TiTok_Align: Configured model instance
    
    Example:
        # Use with predefined LLM
        model = create_titok_align_for_llm('llama-7b', model_size='base')
        
        # Use with custom HuggingFace model
        model = create_titok_align_for_llm('microsoft/DialoGPT-medium', model_size='small')
    """
    # Predefined configurations for popular models
    model_configs = {
        'gpt2': {'text_model_name': 'gpt2', 'text_hidden_dim': 768},
        'gpt2-medium': {'text_model_name': 'gpt2-medium', 'text_hidden_dim': 1024},
        'gpt2-large': {'text_model_name': 'gpt2-large', 'text_hidden_dim': 1280},
        'gpt2-xl': {'text_model_name': 'gpt2-xl', 'text_hidden_dim': 1600},
        
        'llama-7b': {'text_model_name': 'meta-llama/Llama-2-7b-hf', 'text_hidden_dim': 4096},
        'llama-13b': {'text_model_name': 'meta-llama/Llama-2-13b-hf', 'text_hidden_dim': 5120},
        'llama-70b': {'text_model_name': 'meta-llama/Llama-2-70b-hf', 'text_hidden_dim': 8192},
        
        'qwen-7b': {'text_model_name': 'Qwen/Qwen2-7B', 'text_hidden_dim': 3584},
        'qwen-14b': {'text_model_name': 'Qwen/Qwen2-72B', 'text_hidden_dim': 8192},
        
        'mistral-7b': {'text_model_name': 'mistralai/Mistral-7B-v0.1', 'text_hidden_dim': 4096},
    }
    
    if llm_name in model_configs:
        config = model_configs[llm_name]
        titok_kwargs.update(config)
        print(f"Using predefined config for {llm_name}: {config}")
    else:
        # For custom model names, try to auto-detect
        titok_kwargs['text_model_name'] = llm_name
        print(f"Using custom model {llm_name}, will auto-detect dimensions")
    
    return TiTok_Align(**titok_kwargs)