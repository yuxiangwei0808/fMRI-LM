"""
by Wei-Bang Jiang
https://github.com/935963004/NeuroLM
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
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import AutoModel
import math

from quantizers import VQ, FSQ_Model, TiTok, VQ_Align
from dataset import fMRITextDataset
from brain_encoder import vit_small, vit_base, vit_large
from utils_loss import clip_loss, soft_clip_loss, siglip_loss, soft_siglip_loss

accelerator = None

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
    # Wrap quantizer model to embeddings from the final hidden state
    def __init__(self, fmri_model, text_model,
                 proj_embed_dim, fmri_embed_dim, text_embed_dim, 
                 fmri_pool_method, text_pool_method,
                 contr_method='clip', contr_kwargs={},
                 domain_confuse_weight=0.0):
        super().__init__()

        self.fmri_model = fmri_model
        self.text_model = text_model

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

        self.contr_kwargs = contr_kwargs
        self.domain_confuse_weight = domain_confuse_weight

    def forward_fmri(self, X, Y_raw, return_tokens):
        if self.domain_confuse_weight > 0.0:
            return self.fmri_model.VQ(X, Y_raw, return_tokens)
        return self.fmri_model(X, Y_raw, return_tokens)

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

    def forward(self, X=None, Y_raw=None, return_tokens=None,
                text_input_ids=None, text_attention_mask=None, output_hidden_states=True,
                fmri_embed=None, x_text=None, alpha=0, forward_domain=False):
        if forward_domain:
            return self.forward_domain(fmri_embed=fmri_embed, x_text=x_text, alpha=alpha)
        
        fmri_quant_loss, fmri_embed, _, quant_log = self.forward_fmri(X, Y_raw, return_tokens=return_tokens)  # (B, N, D_fmri)
        text_embed = self.forward_text(text_input_ids, text_attention_mask, output_hidden_states=output_hidden_states)  # (B, D_text)

        if self.fmri_pool_method == 'mean':
            fmri_pooled = self.gap(fmri_embed.permute(0, 2, 1)).squeeze(-1)  # (B, D_fmri)
        elif self.fmri_pool_method == 'cls':
            fmri_pooled = fmri_embed[:, 0, :]     # (B, D_fmri)
        elif self.fmri_pool_method == 'map':
            fmri_pooled = self.map_head(fmri_embed) # (B, D_fmri)

        fmri_proj = self.fmri_proj(fmri_pooled)   # (B, proj_dim)
        text_proj = self.text_proj(text_embed)     # (B, proj_dim)

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
        assert isinstance(self.fmri_model, VQ_Align), "Domain confusion only implemented for VQ_Align model."
        return self.fmri_model.forward_domain(fmri_embed=fmri_embed, x_text=x_text, alpha=alpha)

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

def main(args):
    global accelerator
    
    init(args)
    
    checkpoint_out_dir = args.ckpt_dir
    if accelerator.is_main_process:  # Replace master_process check
        os.makedirs(checkpoint_out_dir, exist_ok=True)

    print('prepare dataloader...')
    dataset_train = []
    for path in args.dataset_dir:
        assert os.path.exists(path), f"Dataset path {path} does not exist." 
        dataset = fMRITextDataset(file=path, descriptor_types=args.desc_type, lm_name=args.lm_name, norm='robust', GPT_training=False)
        dataset_train.append(dataset)
    dataset_train = torch.utils.data.ConcatDataset(dataset_train)
    print('finished!')

    # Simplified dataloader creation - no need for DistributedSampler
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=16,
        pin_memory=True,
        shuffle=True
    )

    # model init
    quantizer_cfg = OmegaConf.load(args.cfg_path).model.vq_model
    quantizer_cfg.img_size = (quantizer_cfg.num_rois, quantizer_cfg.num_timestamp)
    if args.fmri_pool_method == 'cls': 
        quantizer_cfg.add_cls_token = True  # add cls token
    else:
        quantizer_cfg.add_cls_token = False

    if os.path.exists(os.path.join(checkpoint_out_dir, 'ckpt.pt')) and args.resume:
        init_from = 'resume'
    else:
        init_from = 'scratch'

    if args.quantizer == 'vq':
        quantizer_cls = VQ if not args.domain_confuse_weight else VQ_Align
    elif args.quantizer == 'fsq':
        quantizer_cls = FSQ_Model
    elif args.quantizer == 'titok':
        quantizer_cls = TiTok
    else:
        raise ValueError(f"Unknown quantizer type: {args.quantizer}")
    
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

    # init these up here, can override if init_from='resume'
    iter_num = 0

    if init_from == 'scratch':
        print("Initializing a new model from scratch")
        model_fmri = quantizer_cls(quantizer_cfg)
        start_epoch = 0
    elif init_from == 'resume':
        print(f"Resuming training from {checkpoint_out_dir}")
        ckpt_path = os.path.join(checkpoint_out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location='cpu')  # Load to CPU first

        model_fmri = quantizer_cls(quantizer_cfg)
        state_dict = checkpoint['model']
        
        # Fix state dict keys
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model_fmri.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        start_epoch = checkpoint['epoch'] + 1
    
    # Initialize frozen text encoder for getting text embeddings
    print(f"Loading frozen text encoder: {args.lm_name}")
    model_text = AutoModel.from_pretrained(args.lm_name)
    for param in model_text.parameters():
        param.requires_grad = False
    model_text.eval()

    num_training_steps_per_epoch = len(dataset_train) // args.batch_size // accelerator.num_processes

    # optimizer
    optimizer = model_fmri.configure_optimizers(args.weight_decay, args.learning_rate, (args.beta1, args.beta2), 'cpu')
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    lr_scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=num_training_steps_per_epoch, T_mult=1, eta_min=args.min_lr
    )
    checkpoint = None

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
        domain_confuse_weight=args.domain_confuse_weight,
    )

    # Prepare everything with Accelerate - this replaces DDP wrapping
    model, optimizer, lr_scheduler, data_loader_train = accelerator.prepare(
        model, optimizer, lr_scheduler, data_loader_train
    )

    if args.domain_confuse_weight > 0.0:
        model._set_static_graph()  # for gradient reversal layer, or it will throw error in DDP
 
    # compile the model (optional)
    if getattr(args, 'compile', False):
        print("compiling the model.. (takes a ~minute)")
        model = torch.compile(model)

    # Initialize wandb
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
        num_tokens = model.module.fmri_model.num_tokens
        X_text, _ = get_batch('train', num_tokens)
    domain_loss_fmri, domain_loss_text = torch.tensor(0.0), torch.tensor(0.0)

    progress_bar = tqdm(range(start_epoch, args.epochs), desc="Training Progress", disable=not accelerator.is_main_process)
    for epoch in progress_bar:
        epoch_log = {}
        for step, batch in enumerate(data_loader_train):
            X, Y_raw, text_input_ids, text_attention_mask = batch
            X = X.float()
            Y_raw = Y_raw.float()

            with accelerator.autocast():  # Replace manual autocast context
                fmri_quant_loss, contr_loss, log, fmri_embed = model(
                    X, Y_raw, return_tokens=True,
                    text_input_ids=text_input_ids, text_attention_mask=text_attention_mask, output_hidden_states=True
                )

                total_loss = fmri_quant_loss + args.contr_weight * contr_loss

                if args.domain_confuse_weight > 0.0:
                    alpha = 2 / (1 + math.exp(-10 * iter_num / args.epochs / num_training_steps_per_epoch)) - 1
                    domain_loss_fmri = model(fmri_embed=fmri_embed, x_text=text_input_ids, alpha=alpha, forward_domain=True)
                    domain_loss_text = model(fmri_embed=None, x_text=text_input_ids, alpha=alpha, forward_domain=True)
                    total_loss += args.domain_confuse_weight * (domain_loss_fmri + domain_loss_text)
                    X_text, _ = get_batch('train', num_tokens)

                total_loss = total_loss / args.gradient_accumulation_steps
                
                # Update log
                log['train/contr_loss'] = contr_loss.detach().item()
                log['train/fmri_loss'] = fmri_quant_loss.detach().item()
                log['train/rec_raw_loss'] = log['train/rec_raw_loss'].detach().item()
                log['train/total_loss'] = total_loss.detach().item() * args.gradient_accumulation_steps
                log['train/domain_loss'] = (domain_loss_fmri + domain_loss_text).detach().item() * args.gradient_accumulation_steps

            if epoch_log == {}: 
                epoch_log = log
            else: 
                epoch_log = {k: epoch_log[k] + log[k] for k in log}

            accelerator.backward(total_loss)
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.grad_clip != 0.0:
                    # Accelerate handles gradient clipping across devices
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            if accelerator.is_main_process and args.wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train_iter/total_loss": log['train/total_loss'],
                    "train_iter/fmri_loss": log['train/fmri_loss'],
                    "train_iter/rec_loss": log['train/rec_raw_loss'],
                    "train_iter/contr_loss": log['train/contr_loss'],
                    "train_iter/domain_loss": log['train/domain_loss'],
                    "train_iter/lr": optimizer.param_groups[0]['lr'],
                })
            
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            iter_num += 1
            local_iter_num += 1

        # gather logs from processes
        epoch_log = {k: torch.tensor(v, device=accelerator.device) / (step + 1) for k, v in epoch_log.items()}
        epoch_log = accelerator.gather(epoch_log)
        epoch_log = {k: epoch_log[k].mean().item() for k in epoch_log}

        # Save checkpoints (only on main process)
        if accelerator.is_main_process:
            progress_bar.set_description(
                f"epoch {epoch}: train loss {epoch_log['train/total_loss']:.4f}, "
                f"rec loss {epoch_log['train/rec_raw_loss']:.4f}, "
                f"fmri loss {epoch_log['train/fmri_loss']:.4f}, "
                f"contr loss {epoch_log['train/contr_loss']:.4f}, "
                f"domain loss {epoch_log['train/domain_loss']:.4f}"
            )

            if args.wandb_log:
                wandb.log({
                    'epoch': epoch,
                    'train/total_loss': epoch_log['train/total_loss'],
                    'train/rec_loss': epoch_log['train/rec_raw_loss'],
                    'train/fmri_loss': epoch_log['train/fmri_loss'],
                    'train/contr_loss': epoch_log['train/contr_loss'],
                    'train/domain_loss': epoch_log['train/domain_loss'],
                    'lr': optimizer.param_groups[0]['lr']
                })

            # Use accelerator.unwrap_model to get the original model for saving
            unwrapped_model = accelerator.unwrap_model(model)
            checkpoint = {
                'model': unwrapped_model.fmri_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'conf': quantizer_cfg,
                'iter_num': iter_num,
                'epoch': epoch
            }
            print(f"saving checkpoint to {checkpoint_out_dir}")
            torch.save(checkpoint, os.path.join(checkpoint_out_dir, f'ckpt.pt'))
        
            if (epoch + 1) % args.save_ckpt_freq == 0:
                print(f"saving checkpoint {epoch} to {checkpoint_out_dir}")
                torch.save(checkpoint, os.path.join(checkpoint_out_dir, f'ckpt-{epoch}.pt'))

        # Wait for all processes to finish the epoch
        accelerator.wait_for_everyone()

    # End training
    accelerator.end_training()

def get_args():
    def list_of_strs(arg):
        return arg.split(',')
    parser = argparse.ArgumentParser('VQ training script', add_help=False)
    parser.add_argument('--ckpt_dir', default='./checkpoints/tmp', help='path where to save, empty for no saving')
    parser.add_argument('--dataset_dir', default=['data/UKB/fmri/TianS3/'], type=list_of_strs, help='path to the training dataset folder, can be multiple paths separated by comma')
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--wandb_log', default=False, action='store_true')
    parser.add_argument('--wandb_project', default='BrainFM_quantizer')
    parser.add_argument('--wandb_runname', default='VQ_align')
    
    # training args
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--text_batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--save_ckpt_freq', default=10, type=int)
    parser.add_argument('--resume', default=False, action='store_true',)

    parser.add_argument('--desc_type', type=list_of_strs, default=['fc', 'ica', 'gradient', 'graph'], )
    parser.add_argument('--lm_name', type=str, default='gpt2', help='language model name')

    parser.add_argument('--proj_embed_dim', type=int, default=512, help='dimension for projection head for contrastive learning')
    parser.add_argument('--contr_loss', type=str, default='clip', choices=['clip', 'soft_clip', 'siglip', 'soft_siglip'],
                        help='contrastive loss type for aligning fMRI and text embeddings')
    parser.add_argument('--contr_weight', type=float, default=1.0, help='weight for contrastive loss')
    parser.add_argument('--fmri_pool_method', type=str, default='cls', choices=['cls', 'mean', 'map'],
                        help='pooling method for fMRI embeddings before contrastive head')
    parser.add_argument('--text_pool_method', type=str, default='last', choices=['last', 'mean'],
                        help='pooling method for text embeddings before contrastive head')
    
    parser.add_argument('--domain_confuse_weight', type=float, default=0.0, help='weight for classifier confusion (reverse gradient) loss')

    # quantizer args
    parser.add_argument('--quantizer', type=str, default='vq')
    parser.add_argument('--cfg_path', type=str, default='configs/vit_small_gpt2.yaml', help='path to the model config file',)

    parser.add_argument('--learning_rate', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--grad_clip', type=float, default=0.0,
                        help='clip gradients at this value, or disable if == 0.0')
    parser.add_argument('--decay_lr', default=True, action='store_false')
    parser.add_argument('--seed', default=1337, type=int)

    parser.add_argument('--compile', default=False, action='store_true')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)