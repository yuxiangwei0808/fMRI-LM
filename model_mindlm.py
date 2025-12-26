"""
Copied and Adapted from https://github.com/935963004/NeuroLM
"""

import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F

from brain_encoder import *
from model_gpt import MultimodalLLM


class MindLM(nn.Module):

    def __init__(self,
                 GPT_config,
                 tokenizer,
                 tune_tokenizer,
                 freeze_llm=False,
                 num_rois=450,
                 n_embd=768,
                 eeg_vocab_size=8192,
                 latent_tokens=None,
                 ):
        super().__init__()
        self.llm = MultimodalLLM(GPT_config)

        self.tokenizer = tokenizer
        self.tune_tokenizer = tune_tokenizer
        self.latent_tokens = latent_tokens  # Titok latent token

        if not tune_tokenizer:
            for p in self.tokenizer.parameters():
                p.requires_grad = False

        if freeze_llm: # freeze LLM except the fmri_lm_head and LoRA parameters
            assert hasattr(self.llm, 'fmri_lm_head'), "The LLM does not have a fmri_lm_head attribute."
            for name, param in self.llm.named_parameters():
                # Don't freeze fmri_lm_head or LoRA parameters
                if 'fmri_lm_head' not in name and 'lora' not in name.lower():
                    param.requires_grad = False

        # construct input ROI positional embedding as NeuroLM's input_chans
        self.input_chans = torch.LongTensor(torch.arange(num_rois)).unsqueeze(-1)
        self.input_chans =  self.input_chans.repeat(1, self.tokenizer.patch_embed.num_time_patches).flatten()

        self.pos_embed = nn.Embedding(num_rois, self.llm.n_embd)

        # task layer
        self.encode_transform_layer = nn.Sequential(
            nn.Linear(n_embd, self.llm.n_embd),
            nn.GELU(),
        )

        self.encode_transform_layer.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_brain=None, y_brain=None, x_text=None, y_text=None, attention_mask=None, Y=None):
        """
        x_brain: shape [B, N1, T]
        x_text: shape [B, N2]
        """
        if x_brain is not None:
            if self.tune_tokenizer:
                x_brain = self.tokenizer(x_brain, latent_tokens=self.latent_tokens)
            else:
                with torch.no_grad():
                    x_brain = self.tokenizer(x_brain, latent_tokens=self.latent_tokens)
            x_brain = self.encode_transform_layer(x_brain)
            
            # pos embedding on ROI
            # TODO consider remove this either becuase: 1. it's redundant. 2. We use 1D tokenization
            input_chans = self.input_chans.unsqueeze(0).repeat(x_brain.size(0), 1).to(x_brain.device)
            input_chans = input_chans[:, :x_brain.size(1)]  # in case of different length of fMRI tokens
            x_brain += self.pos_embed(input_chans)

        logits, loss, accuracy = self.llm(x_brain, y_brain, x_text, y_text, attention_mask, Y=Y)

        log = {}
        split="train" if self.training else "val"
        if loss is not None:
            log[f'{split}/loss'] = loss.item()
        if accuracy is not None:
            log[f'{split}/accuracy'] = accuracy

        return loss, log, logits
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return n_params

    @torch.no_grad()
    def generate(self, x_brain, x_text, attention_mask=None, max_new_tokens=10, temperature=0.8, top_k=20, top_p=0.8, text_gen=False, accelerator=None, **kwargs):
        if x_brain is not None:
            x_brain = self.tokenizer(x_brain)
            x_brain = self.encode_transform_layer(x_brain)
            
            # pos embedding on ROI
            # TODO consider remove this either becuase: 1. it's redundant. 2. We use 1D tokenization
            input_chans = self.input_chans.unsqueeze(0).repeat(x_brain.size(0), 1).to(x_brain.device)
            x_brain += self.pos_embed(input_chans)
        
        # If accelerator is provided and mixed precision is enabled, use autocast for consistency
        if accelerator is not None and hasattr(accelerator, 'autocast'):
            with accelerator.autocast():
                if text_gen:
                    return self.llm.generate_text(x_brain, x_text, attention_mask, max_new_tokens,
                                                temperature=temperature, top_k=top_k, top_p=top_p, **kwargs)
                else:
                    return self.llm.generate(x_brain, x_text, attention_mask, max_new_tokens, temperature, top_k, top_p, **kwargs)
        else:
            if text_gen:
                return self.llm.generate_text(x_brain, x_text, attention_mask, max_new_tokens,
                                            temperature=temperature, top_k=top_k, top_p=top_p, **kwargs)
            else:
                return self.llm.generate(x_brain, x_text, attention_mask, max_new_tokens, temperature, top_k, top_p, **kwargs)

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