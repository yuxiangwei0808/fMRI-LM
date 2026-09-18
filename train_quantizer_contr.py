"""
Unified Contrastive Training Script
"""

import os
import time
import argparse
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.autograd import Function
import inspect
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import AutoModel
import math

from quantizers import TokAlign
from dataset import fMRITextDataset
from utils_loss import clip_loss, soft_clip_loss, siglip_loss, soft_siglip_loss
import checkpoint_naming as ckpt_naming

accelerator = None

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class MAPHead(nn.Module):
    def __init__(self, d, n_heads=8):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d))    # learned global query
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.nh = n_heads
        self.dh = d // n_heads

    def forward(self, H, attn_mask=None):
        B, L, d = H.shape
        q = self.q.expand(B, -1, -1)                   # (B,1,d)
        q = self.q_proj(q).view(B, 1, self.nh, self.dh).transpose(1, 2)      # (B,nh,1,dh)
        K = self.k_proj(H).view(B, L, self.nh, self.dh).transpose(1, 2)      # (B,nh,L,dh)
        V = self.v_proj(H).view(B, L, self.nh, self.dh).transpose(1, 2)      # (B,nh,L,dh)
        scores = (q @ K.transpose(-2, -1)) / math.sqrt(self.dh)              # (B,nh,1,L)
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask[:, None, None, :], -1e9)  # mask pads
        w = scores.softmax(dim=-1)
        z = (w @ V).transpose(1, 2).reshape(B, 1, d).squeeze(1)              # (B,d)
        return self.out(z)                                                   # pooled text emb


class ContrastiveWrapper(torch.nn.Module):
    def __init__(self, fmri_model, text_model,
                 proj_embed_dim, fmri_embed_dim, text_embed_dim, 
                 fmri_pool_method, text_pool_method,
                 contr_method='clip', contr_kwargs=None):
        super().__init__()

        self.fmri_model = fmri_model
        self.text_model = text_model
        self.fmri_embed_dim = fmri_embed_dim

        self.contr_method = contr_method
        self.fmri_pool_method = fmri_pool_method
        self.text_pool_method = text_pool_method

        self.fmri_proj = torch.nn.Linear(fmri_embed_dim, proj_embed_dim, bias=False)
        self.text_proj = torch.nn.Linear(text_embed_dim, proj_embed_dim, bias=False)

        if fmri_pool_method == 'mean':
            self.gap = nn.AdaptiveAvgPool1d(1)
        if fmri_pool_method == 'map':  # sigLip attention pooling
            self.map_head = MAPHead(fmri_embed_dim, n_heads=8)
        
        if contr_method in ['clip', 'siglip']:
            self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.contr_kwargs = contr_kwargs or {}

    def _normalize_fmri_embed(self, fmri_embed):
        if fmri_embed.ndim == 4:
            if fmri_embed.shape[2] == 1:
                fmri_embed = fmri_embed.squeeze(2).transpose(-1, -2)
            else:
                fmri_embed = fmri_embed.flatten(2).transpose(1, 2)
        elif fmri_embed.ndim == 3 and fmri_embed.shape[-1] != self.fmri_embed_dim and fmri_embed.shape[1] == self.fmri_embed_dim:
            fmri_embed = fmri_embed.transpose(1, 2)
        return fmri_embed

    def forward_fmri_quantizer(self, X, Y_raw):
        """Run the unified Tokenizer inside TokAlign."""
        if not hasattr(self.fmri_model, 'quantizer'):
            raise TypeError(f"Expected TokAlign, got {type(self.fmri_model).__name__}")
        fmri_quant_loss, fmri_embed, quant_log = self.fmri_model.quantizer(X, Y_raw)
        return fmri_quant_loss, self._normalize_fmri_embed(fmri_embed), quant_log

    @torch.no_grad()
    def forward_text(self, input_ids, attention_mask, output_hidden_states=True):
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=output_hidden_states)

        if self.text_pool_method == 'mean':  # masked mean
            mask = attention_mask.unsqueeze(-1).expand(text_outputs.last_hidden_state.size()).float()
            summed = (mask * text_outputs.last_hidden_state).sum(1)
            count = mask.sum(1).clamp(min=1e-8)
            text_pooled = summed / count
        elif self.text_pool_method == 'last':  # last EOS token
            last_idx = attention_mask.sum(1) - 1  # (B,)
            text_pooled = text_outputs.last_hidden_state[range(len(last_idx)), last_idx]

        return text_pooled

    def forward(self, X=None, Y_raw=None,
                text_input_ids=None, text_attention_mask=None, output_hidden_states=True,
                fmri_embed=None, x_text=None, alpha=0, forward_domain=False):
        if forward_domain:
            return self.forward_domain(fmri_embed=fmri_embed, x_text=x_text, alpha=alpha)
        
        fmri_quant_loss, fmri_embed, quant_log = self.forward_fmri_quantizer(X, Y_raw)
        
        text_embed = self.forward_text(text_input_ids, text_attention_mask, output_hidden_states=output_hidden_states)

        fmri_embed = self._normalize_fmri_embed(fmri_embed)

        # Pool fMRI embeddings
        if self.fmri_pool_method == 'mean':
            fmri_pooled = self.gap(fmri_embed.permute(0, 2, 1)).squeeze(-1)
        elif self.fmri_pool_method == 'map':
            fmri_pooled = self.map_head(fmri_embed)
        else:
            raise ValueError(f"Unknown fMRI pooling method: {self.fmri_pool_method}")

        fmri_proj = self.fmri_proj(fmri_pooled)
        text_proj = self.text_proj(text_embed)

        # Compute contrastive loss
        if self.contr_method == 'clip':
            loss_contr = clip_loss(fmri_proj, text_proj, logit_scale=self.logit_scale)
        elif self.contr_method == 'soft_clip':
            loss_contr = soft_clip_loss(fmri_proj, text_proj, **self.contr_kwargs)
        elif self.contr_method == 'siglip':
            loss_contr = siglip_loss(fmri_proj, text_proj, logit_scale=self.logit_scale)
        elif self.contr_method == 'soft_siglip':
            loss_contr = soft_siglip_loss(fmri_proj, text_proj, **self.contr_kwargs)
        else:
            raise ValueError(f"Unknown contrastive method: {self.contr_method}")

        return fmri_quant_loss, loss_contr, quant_log, fmri_embed

    def forward_domain(self, fmri_embed=None, x_text=None, alpha=0):
        if hasattr(self.fmri_model, 'forward_domain'):
            return self.fmri_model.forward_domain(fmri_embed=fmri_embed, x_text=x_text, alpha=alpha)

        if x_text is None:
            fmri_embed = self._normalize_fmri_embed(fmri_embed)
            fmri_embed = self.fmri_model.x_proj(fmri_embed)
            reverse_x = ReverseLayerF.apply(fmri_embed, alpha)
            domain_out = self.fmri_model.domain_classifier(reverse_x)
            target = torch.zeros(domain_out.size(0) * domain_out.size(1), dtype=torch.long, device=fmri_embed.device)
        else:
            x_text = self.fmri_model.wte(x_text).detach()
            domain_out = self.fmri_model.domain_classifier(x_text)
            target = torch.ones(x_text.size(0) * x_text.size(1), dtype=torch.long, device=x_text.device)
        return nn.functional.cross_entropy(domain_out.view(-1, 2), target)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
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
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer


def init(args):
    global accelerator
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb" if args.wandb_log else None,
        kwargs_handlers=[ddp_kwargs]
    )
    
    # Set seed
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def prepare_model_config(cfg):
    if OmegaConf.select(cfg, "model.vq_model") is not None:
        raise ValueError(
            "Legacy model.vq_model configs are no longer supported. "
            "Use the unified Tokenizer config with model.quantize_mode instead."
        )

    if OmegaConf.select(cfg, "model") is None:
        raise ValueError("Config must contain a model section for the unified Tokenizer.")

    for key in ("num_rois", "num_timestamp", "quantize_mode"):
        if OmegaConf.select(cfg, f"model.{key}") is None:
            raise ValueError(f"Tokenizer config is missing model.{key}.")

    if OmegaConf.select(cfg, "model.image_size") is None:
        cfg.model.image_size = [int(cfg.model.num_rois), int(cfg.model.num_timestamp)]

    if OmegaConf.select(cfg, "model.gate_attention") is None and OmegaConf.select(cfg, "model.gate_attn") is not None:
        cfg.model.gate_attention = cfg.model.gate_attn

    # rope_mode may be present in unified configs, but this tokenizer path intentionally
    # uses the non-RoPE transformer block.
    return cfg, int(cfg.model.image_size[1])


def build_fmri_model(model_cfg, args):
    return TokAlign(model_cfg, lm_name=args.lm_name)

def strip_compile_prefix(state_dict):
    state_dict = dict(state_dict)
    unwanted_prefix = '_orig_mod.'
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    return state_dict


def wrapper_checkpoint_state(model):
    excluded_prefixes = ('text_model.', 'fmri_model.')
    return {k: v for k, v in model.state_dict().items() if not k.startswith(excluded_prefixes)}


def scalar_value(value):
    if torch.is_tensor(value):
        return value.detach().item()
    return float(value)


def main(args):
    global accelerator
    
    init(args)

    cfg = OmegaConf.load(args.cfg_path)
    quantizer_cfg, clip_timepoints = prepare_model_config(cfg)

    quantizer_cfg.model.add_cls_token = False
    
    checkpoint_out_dir = args.ckpt_dir
    if accelerator.is_main_process:
        os.makedirs(checkpoint_out_dir, exist_ok=True)
        ckpt_naming.write_lineage_json(
            checkpoint_out_dir,
            stage='tokenizer',
            objective='contrastive',
            checkpoint_dir=checkpoint_out_dir,
            cfg_path=args.cfg_path,
            dataset_dir=args.dataset_dir,
            data_norm=ckpt_naming.data_norm_name(args.dataset_dir, args.norm),
            lm_name=args.lm_name,
            quantize_mode=ckpt_naming.tokenizer_mode_from_config(args.cfg_path),
            run_name=args.wandb_runname,
            args=vars(args),
        )

    print('prepare dataloader...')
    dataset_train = []
    for path in args.dataset_dir:
        assert os.path.exists(path), f"Dataset path {path} does not exist." 
        dataset = fMRITextDataset(
            file=path,
            descriptor_types=args.desc_type,
            lm_name=args.lm_name,
            norm=args.norm,
            GPT_training=False,
            clip_timepoints=clip_timepoints,
        )
        dataset_train.append(dataset)
    dataset_train = torch.utils.data.ConcatDataset(dataset_train)
    print('finished!')

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=16,
        pin_memory=True,
        shuffle=True
    )

    if os.path.exists(os.path.join(checkpoint_out_dir, 'ckpt.pt')) and args.resume:
        init_from = 'resume'
    else:
        init_from = 'scratch'
    
    # text data loader
    data_dir = 'data/text/openwebtext'
    def get_batch(split, num_tokens):
        if split == 'train':
            data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - num_tokens, (args.text_batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+num_tokens]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+num_tokens]).astype(np.int64)) for i in ix])
        return x, y

    iter_num = 0
    best_loss = float('inf')
    checkpoint = None
    model_fmri = build_fmri_model(quantizer_cfg, args)

    if init_from == 'scratch':
        print("Initializing a new TokAlign model from scratch")
        start_epoch = 0
    elif init_from == 'resume':
        print(f"Resuming training from {checkpoint_out_dir}")
        ckpt_path = os.path.join(checkpoint_out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = strip_compile_prefix(checkpoint['model'])
        model_fmri.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        start_epoch = checkpoint['epoch'] + 1
        if 'best_loss' in checkpoint:
            best_loss = checkpoint['best_loss']
        elif 'log' in checkpoint and 'train/total_loss' in checkpoint['log']:
            best_loss = checkpoint['log']['train/total_loss']
    # Initialize frozen text encoder
    print(f"Loading frozen text encoder: {args.lm_name}")
    model_text = AutoModel.from_pretrained(args.lm_name)
    for param in model_text.parameters():
        param.requires_grad = False
    model_text.eval()

    num_training_steps_per_epoch = len(dataset_train) // args.batch_size // accelerator.num_processes

    model = ContrastiveWrapper(
        fmri_model=model_fmri,
        text_model=model_text,
        proj_embed_dim=args.proj_embed_dim,
        fmri_embed_dim=model_fmri.enc_embed_dim,
        text_embed_dim=model_text.config.hidden_size,
        fmri_pool_method=args.fmri_pool_method,
        text_pool_method=args.text_pool_method,
        contr_method=args.contr_loss,
        contr_kwargs={'temp':0.125, 'alpha_soft':1.0, 'alpha_hard':0.0},
    )

    if init_from == 'resume' and checkpoint is not None and 'wrapper_model' in checkpoint:
        wrapper_state = strip_compile_prefix(checkpoint['wrapper_model'])
        missing_keys, unexpected_keys = model.load_state_dict(wrapper_state, strict=False)
        print(f"Loaded wrapper checkpoint. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")

    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (args.beta1, args.beta2), 'cpu')
    if init_from == 'resume' and checkpoint is not None and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except ValueError as exc:
            print(f"Skipping optimizer state load because parameter groups changed: {exc}")

    lr_scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=num_training_steps_per_epoch, T_mult=1, eta_min=args.min_lr
    )
    checkpoint = None

    model, optimizer, lr_scheduler, data_loader_train = accelerator.prepare(
        model, optimizer, lr_scheduler, data_loader_train
    )

    if args.domain_confuse_weight > 0.0:
        model._set_static_graph()
 
    if getattr(args, 'compile', False):
        print("compiling the model.. (takes a ~minute)")
        model = torch.compile(model)

    if args.wandb_log and accelerator.is_main_process:
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.wandb_runname, dir='./wandb', resume=False)
        run.log_code('.')
        
        artifact = wandb.Artifact(
            name="config", 
            type="config",
            description="Configuration file for model"
        )
        artifact.add_file(local_path=args.cfg_path, name=args.cfg_path)
        run.log_artifact(artifact)

    # training loop
    t0 = time.time()
    local_iter_num = 0

    if args.domain_confuse_weight > 0.0:
        num_tokens = accelerator.unwrap_model(model).fmri_model.num_tokens
        X_text, _ = get_batch('train', num_tokens)
    domain_loss_fmri, domain_loss_text = torch.tensor(0.0), torch.tensor(0.0)

    progress_bar = tqdm(range(start_epoch, args.epochs), desc="Training Progress", disable=not accelerator.is_main_process)
    for epoch in progress_bar:
        epoch_log = {}
        for step, batch in enumerate(data_loader_train):
            X, Y_raw, text_input_ids, text_attention_mask = batch
            X = X.float()
            Y_raw = Y_raw.float()

            log = {}
            with accelerator.autocast():
                fmri_quant_loss, contr_loss, quant_log, fmri_embed = model(
                    X, Y_raw,
                    text_input_ids=text_input_ids, text_attention_mask=text_attention_mask,
                    output_hidden_states=True,
                )
                total_loss = args.quant_weight * fmri_quant_loss + args.contr_weight * contr_loss
                log['train/fmri_loss'] = scalar_value(fmri_quant_loss)
                log['train/rec_raw_loss'] = scalar_value(quant_log['train/rec_raw_loss'])

                # Domain confusion loss (common to both)
                if args.domain_confuse_weight > 0.0:
                    alpha = 2 / (1 + math.exp(-10 * iter_num / args.epochs / num_training_steps_per_epoch)) - 1
                    domain_loss_fmri = model(fmri_embed=fmri_embed, x_text=None, alpha=alpha, forward_domain=True)
                    domain_loss_text = model(fmri_embed=None, x_text=X_text.to(accelerator.device), alpha=alpha, forward_domain=True)
                    total_loss += args.domain_confuse_weight * (domain_loss_fmri + domain_loss_text)
                    X_text, _ = get_batch('train', num_tokens)

                total_loss = total_loss / args.gradient_accumulation_steps
                
                # Update log
                log['train/total_loss'] = scalar_value(total_loss) * args.gradient_accumulation_steps
                log['train/contr_loss'] = scalar_value(contr_loss)
                log['train/domain_loss'] = scalar_value(domain_loss_fmri + domain_loss_text) * args.gradient_accumulation_steps

            if epoch_log == {}: 
                epoch_log = log
            else: 
                epoch_log = {k: epoch_log[k] + log[k] for k in log}

            accelerator.backward(total_loss)
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.grad_clip != 0.0:
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            if accelerator.is_main_process and args.wandb_log:
                wandb_log_dict = {
                    "iter": iter_num,
                    "train_iter/total_loss": log['train/total_loss'],
                    "train_iter/contr_loss": log['train/contr_loss'],
                    "train_iter/domain_loss": log['train/domain_loss'],
                    "train_iter/lr": optimizer.param_groups[0]['lr'],
                }
                wandb_log_dict.update({
                    "train_iter/fmri_loss": log['train/fmri_loss'],
                    "train_iter/rec_loss": log['train/rec_raw_loss'],
                })
                wandb.log(wandb_log_dict)
            
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            iter_num += 1
            local_iter_num += 1

        # gather logs from processes
        epoch_log = {k: torch.as_tensor(v, device=accelerator.device, dtype=torch.float32) / (step + 1) for k, v in epoch_log.items()}
        epoch_log = accelerator.gather(epoch_log)
        epoch_log = {k: epoch_log[k].mean().item() for k in epoch_log}

        # Save checkpoints
        if accelerator.is_main_process:
            progress_bar.set_description(
                f"epoch {epoch}: train loss {epoch_log['train/total_loss']:.4f}, "
                f"rec loss {epoch_log['train/rec_raw_loss']:.4f}, "
                f"fmri loss {epoch_log['train/fmri_loss']:.4f}, "
                f"contr loss {epoch_log['train/contr_loss']:.4f}, "
                f"domain loss {epoch_log['train/domain_loss']:.4f}"
            )

            if args.wandb_log:
                wandb_log_dict = {
                    'epoch': epoch,
                    'train/total_loss': epoch_log['train/total_loss'],
                    'train/contr_loss': epoch_log['train/contr_loss'],
                    'train/domain_loss': epoch_log['train/domain_loss'],
                    'lr': optimizer.param_groups[0]['lr']
                }
                wandb_log_dict.update({
                    'train/rec_loss': epoch_log['train/rec_raw_loss'],
                    'train/fmri_loss': epoch_log['train/fmri_loss'],
                })
                wandb.log(wandb_log_dict)

            current_loss = epoch_log['train/total_loss']
            is_best = current_loss < best_loss
            if is_best:
                best_loss = current_loss

            unwrapped_model = accelerator.unwrap_model(model)
            checkpoint = {
                'model': unwrapped_model.fmri_model.state_dict(),
                'wrapper_model': wrapper_checkpoint_state(unwrapped_model),
                'optimizer': optimizer.state_dict(),
                'conf': quantizer_cfg,
                'iter_num': iter_num,
                'epoch': epoch,
                'log': epoch_log,
                'best_loss': best_loss,
            }
            print(f"saving checkpoint to {checkpoint_out_dir}")
            torch.save(checkpoint, os.path.join(checkpoint_out_dir, f'ckpt.pt'))
        
            if (epoch + 1) % args.save_ckpt_freq == 0:
                print(f"saving checkpoint {epoch} to {checkpoint_out_dir}")
                torch.save(checkpoint, os.path.join(checkpoint_out_dir, f'ckpt-{epoch}.pt'))

            if is_best:
                print(f"saving best checkpoint with loss {best_loss:.4f} to {checkpoint_out_dir}")
                torch.save(checkpoint, os.path.join(checkpoint_out_dir, 'ckpt-best.pt'))

        accelerator.wait_for_everyone()

    accelerator.end_training()


def tokenizer_mode_from_config(cfg_path):
    cfg = OmegaConf.load(cfg_path)
    return OmegaConf.select(cfg, "model.quantize_mode") or "tokenizer"


def generate_run_name(args):
    """Generate checkpoint directory and wandb run name from arguments."""
    data_norm = ckpt_naming.data_norm_name(args.dataset_dir, args.norm)
    cfg_basename = ckpt_naming.config_name(args.cfg_path)
    tokenizer_mode = ckpt_naming.tokenizer_mode_from_config(args.cfg_path)
    desc_str = '_'.join(args.desc_type) if isinstance(args.desc_type, list) else args.desc_type
    
    components = []
    if args.run_prefix:
        components.append(args.run_prefix)
    
    components.append(f"contr_{args.contr_loss}")
    if args.contr_weight > 0.0:
        components.append(f'C{args.contr_weight:.2f}')
        print(args.contr_weight)
    if args.domain_confuse_weight > 0.0:
        components.append(f"D{ckpt_naming.format_float(args.domain_confuse_weight)}")
    if args.quant_weight > 0.0:
        components.append(f'Q{ckpt_naming.format_float(args.quant_weight)}')
    components.append(f"desc_{desc_str}")
    components.append(f"pool_{args.fmri_pool_method}_{args.text_pool_method}")
    components.append(f"tok_{tokenizer_mode}-{cfg_basename}")
    components.append(ckpt_naming.simplify_lm_name(args.lm_name))
    
    if args.domain_confuse_weight > 0.0:
        components.append(f"domain{ckpt_naming.format_float(args.domain_confuse_weight)}")
    
    run_name = ckpt_naming.join_name_parts(*components)
    ckpt_dir = ckpt_naming.tokenizer_ckpt_dir('contrastive', data_norm, run_name)
    
    return ckpt_dir, run_name


def get_args():
    def list_of_strs(arg):
        return arg.split(',')
    
    parser = argparse.ArgumentParser('Tokenizer/Contrastive training script', add_help=False)
    
    parser.add_argument('--run_prefix', default='', type=str, help='Optional prefix for run name')
    parser.add_argument('--ckpt_dir', default=None, type=str, help='path where to save, if not provided will be auto-generated')
    parser.add_argument('--dataset_dir', default=['data/UKB/fmri/TianS3/'], type=list_of_strs, 
                        help='path to the training dataset folder, can be multiple paths separated by comma')
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--wandb_log', default=False, action='store_true')
    parser.add_argument('--wandb_project', default='BrainFM_quantizer')
    parser.add_argument('--wandb_runname', default=None, type=str, help='wandb run name, if not provided will be auto-generated')
    
    # training args
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--text_batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--save_ckpt_freq', default=10, type=int)
    parser.add_argument('--resume', default=False, action='store_true')

    parser.add_argument('--desc_type', type=list_of_strs, default=['fc', 'ica'])
    parser.add_argument('--norm', type=str, default='robust', help='Normalization method for fMRI data')
    parser.add_argument('--lm_name', type=str, default='gpt2', help='language model name')

    parser.add_argument('--proj_embed_dim', type=int, default=768, help='dimension for projection head')
    parser.add_argument('--quant_weight', type=float, default=1.0, help='weight for quantization loss')
    parser.add_argument('--contr_loss', type=str, default='siglip', 
                        choices=['clip', 'soft_clip', 'siglip', 'soft_siglip'],
                        help='contrastive loss type')
    parser.add_argument('--contr_weight', type=float, default=1.0, help='weight for contrastive loss')
    parser.add_argument('--fmri_pool_method', type=str, default='mean', choices=['mean', 'map'],
                        help='pooling method for fMRI embeddings')
    parser.add_argument('--text_pool_method', type=str, default='last', choices=['last', 'mean'],
                        help='pooling method for text embeddings')
    parser.add_argument('--domain_confuse_weight', type=float, default=0.0, 
                        help='weight for domain confusion loss')

    parser.add_argument('--cfg_path', type=str, default='configs/vit_base_p160_newTok.yaml',
                        help='path to model config file')

    parser.add_argument('--learning_rate', type=float, default=1e-4, metavar='LR')
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--grad_clip', type=float, default=0.0)
    parser.add_argument('--decay_lr', default=True, action='store_false')
    parser.add_argument('--seed', default=1337, type=int)
    parser.add_argument('--compile', default=False, action='store_true')

    args = parser.parse_args()
    
    # Auto-generate ckpt_dir and wandb_runname if not provided
    if args.ckpt_dir is None or args.wandb_runname is None:
        auto_ckpt_dir, auto_run_name = generate_run_name(args)
        if args.ckpt_dir is None:
            args.ckpt_dir = auto_ckpt_dir
        if args.wandb_runname is None:
            args.wandb_runname = auto_run_name
    
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)
