"""
Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
Code adapted from Jax version in Appendix A.1
"""

from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor, int32
import torch.nn.functional as F
import inspect
from einops import rearrange, pack, unpack
from transformers import GPT2LMHeadModel

from .vq import ReverseLayerF
from brain_encoder import *

# helper functions

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# tensor helpers

def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()

# main class

class FSQ(Module):
    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks = 1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None
    ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent = False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent = False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out = False)
        self.register_buffer("implicit_codebook", implicit_codebook, persistent = False)

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).tan()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2 # Renormalize to [-1, 1].
        return quantized / half_width
    
    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width
    
    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width
    
    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)
    
    def indices_to_codes(
        self,
        indices: Tensor,
        project_out = True
    ) -> Tensor:
        """Inverse of `codes_to_indices`."""

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        if project_out:
            codes = self.project_out(codes)

        if is_img_or_video:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def forward(self, z: Tensor) -> Tensor:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """

        is_img_or_video = z.ndim >= 4

        # standardize image or video into (batch, seq, dimension)

        if is_img_or_video:
            z = rearrange(z, 'b d ... -> b ... d')
            z, ps = pack_one(z, 'b * d')

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'

        z = self.project_in(z)

        z = rearrange(z, 'b n (c d) -> b n c d', c = self.num_codebooks)

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        codes = rearrange(codes, 'b n c d -> b n (c d)')

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        if is_img_or_video:
            out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')

            indices = unpack_one(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')

        return out, indices


class FSQ_Model(nn.Module):
    def __init__(self,
                 config,
                 levels: List[int] =  [8, 8, 8, 6, 5],
                 embed_dim=128,
                 decoder_out_dim=160,
                 smooth_l1_loss=True,
                 **kwargs
                 ):
        super().__init__()
        print(kwargs)
        enc_cls_dict = {'vit_small': vit_small, 'vit_base': vit_base, 'vit_large': vit_large}
        self.num_rois = config.img_size[0]

        # encoder & decode params
        print('Final encoder config', config)
        enc_cls = enc_cls_dict[config.enc_cls]
        self.encoder = enc_cls(**dict(config))

        print('Final decoder config', config)
        if config.in_chans != embed_dim:
            print(f"Rewrite the in_chans in decoder from {config.in_chans} to {embed_dim}")
            config.in_chans = embed_dim
        self.decoder_raw = enc_cls(**dict(config))

        # FSQ quantizer
        self.quantize = FSQ(levels=levels, dim=embed_dim)
        self.codebook_size = self.quantize.codebook_size

        self.decoder_out_dim = decoder_out_dim

        # task layer
        self.encode_task_layer = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.Tanh(),
            nn.Linear(config.n_embd, embed_dim) # for quantize
        )

        time_patches = self.encoder.patch_embed.num_time_patches  # ROI is kept the same
        self.decode_task_layer_raw = nn.Sequential(
            nn.Linear(config.n_embd * time_patches, config.n_embd),
            nn.Tanh(),
            nn.Linear(config.n_embd, self.decoder_out_dim),
        )

        self.kwargs = kwargs
        
        self.encode_task_layer.apply(self._init_weights)
        self.decode_task_layer_raw.apply(self._init_weights)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def device(self):
        return self.decoder_raw.cls_token.device
    
    def get_number_of_tokens(self):
        return self.quantize.codebook_size

    def get_tokens(self, data, **kwargs):
        _, embed_ind, _ = self.encode(data)
        return embed_ind.view(data.size(0), -1)  # (B, num_token)

    def encode(self, x, **kwargs):
        batch_size, n, t = x.shape
        encoder_features = self.encoder(x, **kwargs)

        with torch.cuda.amp.autocast(enabled=False):
            to_quantizer_features = self.encode_task_layer(encoder_features.type_as(self.encode_task_layer[-1].weight))

        quantize, embed_ind = self.quantize(to_quantizer_features)

        return quantize, embed_ind, encoder_features
        
    def decode(self, quantize, **kwargs):
        # reshape tokens to feature maps for patch embed in decoder
        decoder_features = self.decoder_raw(quantize, **kwargs)
        # TODO try NeuroLM's method that combines t with c
        decoder_features = rearrange(decoder_features, 'b (n t) d -> b n (t d)', n=self.num_rois)
        rec_raw = self.decode_task_layer_raw(decoder_features)
        return rec_raw
    
    def get_codebook_indices(self, x, **kwargs):
        return self.get_tokens(x, **kwargs)
    
    def calculate_rec_loss(self, rec, target):
        rec_loss = self.loss_fn(rec, target)
        return rec_loss

    def forward(self, x, y_raw, **kwargs):
        """
        x: shape [B, N, T]
        """
        quantize, embed_ind, encoder_features = self.encode(x, **kwargs)

        xrec_raw = self.decode(quantize, **kwargs)

        rec_raw_loss = self.calculate_rec_loss(xrec_raw, y_raw)
        # FSQ doesn't have embedding loss like VQ
        loss = rec_raw_loss

        log = {}
        split = "train" if self.training else "val"
        log[f'{split}/rec_raw_loss'] = rec_raw_loss.detach().mean()
        log[f'{split}/quant_loss'] = 0
        log[f'{split}/total_loss'] = loss.detach().mean()

        return loss, encoder_features, log
    
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


class FSQ_Align(nn.Module):
    def __init__(self, 
                 config,
                #  levels: List[int] = [8, 8, 8, 6, 5],
                levels: List[int] = [7, 5, 5, 5, 5],
                 ):
        super(FSQ_Align, self).__init__()
        self.FSQ = FSQ_Model(config, levels=levels, decoder_out_dim=config.img_size[1])
        self.num_tokens = self.FSQ.encoder.num_patches

        self.x_proj = nn.Linear(config.n_embd, 768)
        self.domain_classifier = nn.Sequential(
                nn.Linear(768, 256),  # set as 768 because of GPT2's hidden dim; TODO change this for generalization
                nn.GELU(),
                nn.Linear(256, 2)
            )

        model_hf = GPT2LMHeadModel.from_pretrained('gpt2')
        sd_hf = model_hf.state_dict()
        self.wte = nn.Embedding(50257, 768, _freeze=True)
        self.wte.weight.data = sd_hf['transformer.wte.weight']
        
        self.domain_classifier.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, y_raw=None, alpha=0):
        if y_raw is not None:
            loss, encoder_features, log = self.FSQ(x, y_raw, alpha=alpha)
            encoder_features = self.x_proj(encoder_features)
            # Apply reverse layer for domain adaptation
            reverse_x = ReverseLayerF.apply(encoder_features, alpha)
            domain_out = self.domain_classifier(reverse_x)
            target = torch.full((domain_out.size(0), domain_out.size(1)), fill_value=0, device=x.device)
            domain_loss = F.cross_entropy(domain_out.view(-1, domain_out.size(-1)), target.view(-1), ignore_index=-1)
            split = "train" if self.training else "val"
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
    


if __name__ == '__main__':
    levels = [8,5,5,5] # see 4.1 and A.4.1 in the paper
    quantizer = FSQ(levels)

    x = torch.randn(1, 4, 16, 16) # 4 since there are 4 levels
    xhat, indices = quantizer(x)

    print(xhat.shape)    # (1, 1024, 4) - (batch, seq, dim)
    # print(indices.shape) # (1, 1024)    - (batch, seq)