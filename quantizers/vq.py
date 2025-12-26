import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
import inspect
from einops import rearrange
from transformers import GPT2LMHeadModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .norm_ema_quantizer import NormEMAVectorQuantizer
from brain_encoder import vit_small, vit_base, vit_large


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class VQ(nn.Module):
    def __init__(self,
                 config,
                 codebook_size=8192,
                 embed_dim=128,
                 decay=0.99,
                 quantize_kmeans_init=True,
                 decoder_out_dim=160,
                 smooth_l1_loss = False,
                 **kwargs
                 ):
        super().__init__()
        print(kwargs)
        enc_cls_dict = {'vit_small': vit_small, 'vit_base': vit_base, 'vit_large': vit_large}

        self.config = config
        self.num_rois = config.img_size[0]
        self.codebook_size = codebook_size
        self.scale = None  # scale reconst loss

        # encoder & decode params
        print('Final encoder config', config)
        enc_cls = enc_cls_dict[config.enc_cls]
        self.encoder = enc_cls(**dict(config))
        self.enc_embed_dim = self.encoder.embed_dim

        if config.in_chans != embed_dim:
            print(f"Rewrite the in_chans in decoder from {config.in_chans} to {embed_dim}")
            config.in_chans = embed_dim
        print('Final decoder config', config)
        self.decoder_raw = enc_cls(**dict(config))

        self.quantize = NormEMAVectorQuantizer(
            n_embed=codebook_size, embedding_dim=embed_dim, beta=1.0, kmeans_init=quantize_kmeans_init, decay=decay,
        )

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
            
    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {'quantize.embedding.weight', 'decoder.pos_embed', 'decoder.time_embed', 
    #             'encoder.pos_embed', 'encoder.time_embed'}

    @property
    def device(self):
        return self.decoder.cls_token.device
    
    def get_number_of_tokens(self):
        return self.quantize.n_e

    def get_tokens(self, data, **kwargs):
        quantize, embed_ind, loss, _ = self.encode(data)
        return embed_ind.view(data.size(0), -1)  # (B, num_token)

    def encode(self, x, **kwargs):
        batch_size, n, t = x.shape
        
        encoder_features = self.encoder(x, **kwargs)

        with torch.cuda.amp.autocast(enabled=False):
            to_quantizer_features = self.encode_task_layer(encoder_features.type_as(self.encode_task_layer[-1].weight))

        quantize, loss, embed_ind = self.quantize(to_quantizer_features)

        return quantize, embed_ind, loss, encoder_features
        
    def decode(self, quantize, **kwargs):
        # reshape tokens to feature maps for patch embed in decoder
        decoder_features = self.decoder_raw(quantize, **kwargs)
        if self.config.add_cls_token:
            decoder_features = decoder_features[:, 1:, :]  # remove cls token

        # TODO try NeuroLM's method that combines t with c when encoding
        # TODO directly project hidden dim to 1
        decoder_features = rearrange(decoder_features, 'b (n t) d -> b n (t d)', n=self.num_rois)
        rec_raw = self.decode_task_layer_raw(decoder_features)
        return rec_raw
    
    def get_codebook_indices(self, x, **kwargs):
        return self.get_tokens(x, **kwargs)
    
    def calculate_rec_loss(self, rec, target):
        rec_loss = self.loss_fn(rec, target)
        return rec_loss

    def forward(self, x, y_raw=None, return_token=False, **kwargs):
        """
        x: shape [B, N, T]
        """
        quantize, embed_ind, emb_loss, encoder_features = self.encode(x, **kwargs)

        xrec_raw = self.decode(quantize, **kwargs)

        if y_raw is None:
            return None, encoder_features, None
        rec_raw_loss = self.calculate_rec_loss(xrec_raw, y_raw)
        loss = emb_loss + rec_raw_loss

        log = {}
        split="train" if self.training else "val"
        log[f'{split}/quant_loss'] = emb_loss.detach().mean()
        log[f'{split}/rec_raw_loss'] = rec_raw_loss.detach().mean()
        log[f'{split}/total_loss'] = loss.detach().mean()

        if return_token:
            return loss, encoder_features, embed_ind, log
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


class VQ_Align(nn.Module):
    def __init__(self, config, lm_name='gpt2', **kwargs):
        super(VQ_Align, self).__init__()
        self.VQ = VQ(config, decoder_out_dim=config.img_size[1])
        self.num_tokens = self.VQ.encoder.num_patches
        self.enc_embed_dim = self.VQ.enc_embed_dim

        self.lm_name = lm_name
        model_hf = AutoModelForCausalLM.from_pretrained(lm_name)
        self.hidden_dim = model_hf.config.hidden_size

        self.x_proj = nn.Linear(config.n_embd, self.hidden_dim)
        self.domain_classifier = nn.Sequential(
                nn.Linear(self.hidden_dim, 256),
                nn.GELU(),
                nn.Linear(256, 2)
            )

        self.wte = model_hf.get_input_embeddings()
        for p in self.wte.parameters():
            p.requires_grad = False
        
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
            loss, encoder_features, log = self.VQ(x, y_raw, alpha=alpha)
            encoder_features = self.x_proj(encoder_features)
            reverse_x = ReverseLayerF.apply(encoder_features, alpha)
            domain_out = self.domain_classifier(reverse_x)
            target = torch.full((domain_out.size(0), domain_out.size(1)), fill_value=0, device=x.device)
            domain_loss = F.cross_entropy(domain_out.view(-1, domain_out.size(-1)), target.view(-1), ignore_index=-1)
            split="train" if self.training else "val"
            log[f'{split}/domain_loss'] = domain_loss.detach().item()
            return loss, domain_loss, log
        else:
            with torch.no_grad():
                x = self.wte(x).detach()
            domain_out = self.domain_classifier(x)
            domain_loss = F.cross_entropy(domain_out.view(-1, domain_out.size(-1)), torch.ones((x.size(0) * x.size(1),), device=x.device).long(), ignore_index=-1)
            return domain_loss
    
    def forward_domain(self, fmri_embed=None, x_text=None, alpha=0):
        if x_text is None:
            fmri_embed = self.x_proj(fmri_embed)
            reverse_x = ReverseLayerF.apply(fmri_embed, alpha)
            domain_out = self.domain_classifier(reverse_x)
            target = torch.full((domain_out.size(0), domain_out.size(1)), fill_value=0, device=fmri_embed.device)
            domain_loss = F.cross_entropy(domain_out.view(-1, domain_out.size(-1)), target.view(-1), ignore_index=-1)
        else:
            x_text = self.wte(x_text).detach()
            domain_out = self.domain_classifier(x_text)
            domain_loss = F.cross_entropy(domain_out.view(-1, domain_out.size(-1)), torch.ones((x_text.size(0) * x_text.size(1),), device=x_text.device).long(), ignore_index=-1)
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