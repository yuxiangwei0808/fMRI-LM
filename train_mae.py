"""
MAE (Masked Autoencoder) Training Script for fMRI data
Self-supervised pretraining using masked reconstruction
"""

import os
import math
import time
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.autograd import Function
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import AutoModelForCausalLM
from omegaconf import OmegaConf

from brain_encoder.vit_mae import MaskedAutoencoderViT
from dataset import fMRIDataSet
import checkpoint_naming as ckpt_naming

accelerator = None


TOKENIZER_ENCODER_SPECS = {
    'micro': {'embed_dim': 192, 'depth': 4, 'num_heads': 3},
    'tiny': {'embed_dim': 384, 'depth': 6, 'num_heads': 4},
    'small': {'embed_dim': 512, 'depth': 8, 'num_heads': 8},
    'base': {'embed_dim': 768, 'depth': 12, 'num_heads': 12},
    'large': {'embed_dim': 1024, 'depth': 24, 'num_heads': 16},
    'huge': {'embed_dim': 1280, 'depth': 32, 'num_heads': 16},
    'giant': {'embed_dim': 1408, 'depth': 40, 'num_heads': 16},
}


def _cfg_get(cfg, path, default=None):
    value = OmegaConf.select(cfg, path)
    return default if value is None else value


def _cfg_bool(cfg, path, default=False):
    return bool(_cfg_get(cfg, path, default))


def apply_tokenizer_config(args):
    cfg = OmegaConf.load(args.cfg_path)
    if OmegaConf.select(cfg, 'model') is None:
        raise ValueError('MAE training now expects a unified tokenizer config with a top-level model section.')

    model_cfg = cfg.model
    if OmegaConf.select(cfg, 'model.image_size') is None:
        if OmegaConf.select(cfg, 'model.num_rois') is None or OmegaConf.select(cfg, 'model.num_timestamp') is None:
            raise ValueError('Tokenizer config must define either model.image_size or model.num_rois/model.num_timestamp.')
        model_cfg.image_size = [int(model_cfg.num_rois), int(model_cfg.num_timestamp)]

    if OmegaConf.select(cfg, 'model.vit_enc_patch_size') is None:
        raise ValueError('Tokenizer config is missing model.vit_enc_patch_size.')
    if OmegaConf.select(cfg, 'model.vit_enc_model_size') is None:
        raise ValueError('Tokenizer config is missing model.vit_enc_model_size.')

    args.num_rois = int(args.num_rois or model_cfg.image_size[0])
    args.num_timestamp = int(args.num_timestamp or model_cfg.image_size[1])
    args.patch_size = int(args.patch_size or model_cfg.vit_enc_patch_size)
    args.model_size = args.model_size or str(model_cfg.vit_enc_model_size)
    args.attn_mode = args.attn_mode or str(_cfg_get(cfg, 'model.attn_mode', 'normal'))
    args.drop_path_rate = float(args.drop_path_rate if args.drop_path_rate is not None else _cfg_get(cfg, 'model.drop_path_rate', 0.0))
    args.cls_embed = _cfg_bool(cfg, 'model.add_cls_token', False) if args.cls_embed is None else bool(args.cls_embed)

    args.mlp_ratio = float(_cfg_get(cfg, 'model.mlp_ratio', 4.0))
    args.qkv_bias = _cfg_bool(cfg, 'model.qkv_bias', True)
    args.qk_scale = _cfg_get(cfg, 'model.qk_scale', None)
    args.drop_rate = float(_cfg_get(cfg, 'model.drop_rate', 0.0))
    args.attn_drop_rate = float(_cfg_get(cfg, 'model.attn_drop_rate', 0.0))

    unsupported = []
    if _cfg_bool(cfg, 'model.qk_norm', False):
        unsupported.append('model.qk_norm=True')
    if not _cfg_bool(cfg, 'model.proj_bias', True):
        unsupported.append('model.proj_bias=False')
    if _cfg_bool(cfg, 'model.attn_causal', False):
        unsupported.append('model.attn_causal=True')
    gate_attention = _cfg_get(cfg, 'model.gate_attention', _cfg_get(cfg, 'model.gate_attn', 'none'))
    if gate_attention not in (None, 'none'):
        unsupported.append(f'model.gate_attention={gate_attention}')
    if unsupported:
        raise ValueError('MAE ViT does not support these tokenizer encoder options yet: ' + ', '.join(unsupported))

    args.cfg_basename = os.path.basename(args.cfg_path).replace('.yaml', '')
    args.tokenizer_model_config = OmegaConf.to_container(model_cfg, resolve=True)
    return args


def build_mae_model(args):
    if args.model_size not in TOKENIZER_ENCODER_SPECS:
        raise ValueError(f"Unknown tokenizer encoder model size for MAE: {args.model_size}")
    spec = TOKENIZER_ENCODER_SPECS[args.model_size]
    return MaskedAutoencoderViT(
        img_size=(args.num_rois, args.num_timestamp),
        patch_size=args.patch_size,
        in_chans=1,
        embed_dim=spec['embed_dim'],
        depth=spec['depth'],
        num_heads=spec['num_heads'],
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads,
        mlp_ratio=args.mlp_ratio,
        qkv_bias=args.qkv_bias,
        qk_scale=args.qk_scale,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
        sep_pos_embed=args.sep_pos_embed,
        cls_embed=args.cls_embed,
        norm_pix_loss=args.norm_pix_loss,
        attn_mode=args.attn_mode,
        gradient_checkpointing=args.gradient_checkpointing,
        use_rope=False,
    )


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class MAEDomainAligner(nn.Module):
    """TokAlign-style domain classifier for MAE encoder features."""

    def __init__(self, fmri_embed_dim, lm_name='gpt2'):
        super().__init__()
        model_hf = AutoModelForCausalLM.from_pretrained(lm_name)
        self.hidden_dim = model_hf.config.hidden_size
        self.x_proj = nn.Linear(fmri_embed_dim, self.hidden_dim)
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 2),
        )
        self.wte = model_hf.get_input_embeddings()
        for p in self.wte.parameters():
            p.requires_grad = False
        self.domain_classifier.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward_fmri(self, fmri_embed, alpha):
        fmri_embed = self.x_proj(fmri_embed)
        reverse_x = ReverseLayerF.apply(fmri_embed, alpha)
        domain_out = self.domain_classifier(reverse_x)
        target = torch.zeros(
            domain_out.size(0) * domain_out.size(1),
            dtype=torch.long,
            device=domain_out.device,
        )
        return F.cross_entropy(domain_out.reshape(-1, 2), target)

    def forward_text(self, text_input_ids, text_attention_mask=None):
        with torch.no_grad():
            x_text = self.wte(text_input_ids).detach()
        domain_out = self.domain_classifier(x_text)
        target = torch.ones(
            domain_out.size(0) * domain_out.size(1),
            dtype=torch.long,
            device=domain_out.device,
        )
        loss = F.cross_entropy(domain_out.reshape(-1, 2), target, reduction='none')
        if text_attention_mask is None:
            return loss.mean()
        mask = text_attention_mask.reshape(-1).to(loss.dtype)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    def forward(self, fmri_embed=None, text_input_ids=None, text_attention_mask=None, alpha=0):
        loss = None
        if fmri_embed is not None:
            loss = self.forward_fmri(fmri_embed, alpha=alpha)
        if text_input_ids is not None:
            text_loss = self.forward_text(text_input_ids, text_attention_mask=text_attention_mask)
            loss = text_loss if loss is None else loss + text_loss
        if loss is None:
            raise ValueError("MAEDomainAligner.forward requires fmri_embed or text_input_ids.")
        return loss


def trainable_parameters(*modules):
    params = []
    for module in modules:
        if module is not None:
            params.extend(p for p in module.parameters() if p.requires_grad)
    return params


def strip_compile_prefix(state_dict):
    state_dict = dict(state_dict)
    for prefix in ('_orig_mod.', 'module.'):
        for key in list(state_dict.keys()):
            if key.startswith(prefix):
                state_dict[key[len(prefix):]] = state_dict.pop(key)
    return state_dict


def get_random_text_batch(data, batch_size, num_tokens):
    if len(data) <= num_tokens:
        raise ValueError(f"Random text corpus has {len(data)} tokens, need more than {num_tokens}.")
    ix = torch.randint(len(data) - num_tokens, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + num_tokens]).astype(np.int64)) for i in ix])
    return x


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
    if accelerator.is_main_process and not args.no_save_ckpt:
        os.makedirs(checkpoint_out_dir, exist_ok=True)
        ckpt_naming.write_lineage_json(
            checkpoint_out_dir,
            stage='tokenizer',
            objective='mae',
            checkpoint_dir=checkpoint_out_dir,
            cfg_path=args.cfg_path,
            dataset_dir=args.dataset_dir,
            data_norm=ckpt_naming.data_norm_name(args.dataset_dir, args.norm),
            lm_name=args.lm_name if args.domain_loss_weight > 0.0 else None,
            run_name=args.wandb_runname,
            args=vars(args),
        )

    print('Preparing dataloader...')
    use_domain_alignment = args.domain_loss_weight > 0.0
    dataset_train_list = []
    total_samples = 0
    for path in args.dataset_dir:
        assert os.path.exists(path), f"Dataset path {path} does not exist."
        dataset = fMRIDataSet(
            file=path,
            norm=args.norm if args.norm != 'none' else None,
            GPT_training=False,
            clip_timepoints=args.num_timestamp,
            segment_method=args.segment_method,
            oversample=args.oversample_factor,
        )
        total_samples += len(dataset)
        dataset_train_list.append(dataset)
    dataset_train = torch.utils.data.ConcatDataset(dataset_train_list)
    print(f'Training dataset size: {total_samples}')
    print(f'Example data shape: {dataset_train[0][0].shape}')  # Print shape of one example (X, y)

    # Create dataloader
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=16,
        pin_memory=True,
        shuffle=True,
        drop_last=True
    )

    # Model initialization
    print('Initializing model...')
    print(
        f"MAE encoder from {args.cfg_path}: size={args.model_size}, "
        f"patch={args.patch_size}, image=({args.num_rois}, {args.num_timestamp}), "
        f"attn={args.attn_mode}, drop_path={args.drop_path_rate}"
    )

    # Check if resuming from checkpoint
    if os.path.exists(os.path.join(checkpoint_out_dir, 'ckpt.pt')) and args.resume:
        init_from = 'resume'
    else:
        init_from = 'scratch'

    iter_num = 0
    start_epoch = 0
    best_loss = float('inf')  # Initialize best loss tracking
    checkpoint = None

    if init_from == 'scratch':
        print("Initializing a new model from scratch")
        model = build_mae_model(args)
    elif init_from == 'resume':
        print(f"Resuming training from {checkpoint_out_dir}")
        ckpt_path = os.path.join(checkpoint_out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        model = build_mae_model(args)
        
        state_dict = strip_compile_prefix(checkpoint['model'])
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        start_epoch = checkpoint['epoch'] + 1
        # Load best loss if available
        best_loss = checkpoint.get('best_loss', float('inf'))

    domain_aligner = None
    text_data = None
    if use_domain_alignment:
        print(f"Initializing MAE domain aligner with language model: {args.lm_name}")
        domain_aligner = MAEDomainAligner(model.embed_dim, lm_name=args.lm_name)
        text_train_path = os.path.join(args.text_data_dir, 'train.bin')
        if not os.path.exists(text_train_path):
            raise FileNotFoundError(f"Random text corpus not found: {text_train_path}")
        text_data = np.memmap(text_train_path, dtype=np.uint16, mode='r')
        if init_from == 'resume' and checkpoint is not None and 'domain_aligner' in checkpoint:
            domain_aligner.load_state_dict(strip_compile_prefix(checkpoint['domain_aligner']), strict=False)

    num_training_steps_per_epoch = max(1, total_samples // args.batch_size // accelerator.num_processes)

    # Optimizer
    print('Setting up optimizer...')
    optimizer = torch.optim.AdamW(
        trainable_parameters(model, domain_aligner),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    if init_from == 'resume' and checkpoint is not None and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except ValueError as exc:
            print(f"Skipping optimizer state load because parameter groups changed: {exc}")

    # LR scheduler: linear warmup then cosine decay
    total_steps = max(1, args.epochs * num_training_steps_per_epoch // args.gradient_accumulation_steps)
    if args.warmup_epochs > 0:
        warmup_steps = max(1, args.warmup_epochs * num_training_steps_per_epoch // args.gradient_accumulation_steps)
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-2, total_iters=warmup_steps)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=args.min_lr)
        lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    else:
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.min_lr)

    if init_from == 'resume' and checkpoint is not None and 'lr_scheduler' in checkpoint:
        try:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        except ValueError as exc:
            print(f"Skipping LR scheduler state load because scheduler shape changed: {exc}")

    checkpoint = None  # Free memory

    # Prepare everything with Accelerate
    if domain_aligner is not None:
        model, domain_aligner, optimizer, lr_scheduler, data_loader_train = accelerator.prepare(
            model, domain_aligner, optimizer, lr_scheduler, data_loader_train
        )
    else:
        model, optimizer, lr_scheduler, data_loader_train = accelerator.prepare(
            model, optimizer, lr_scheduler, data_loader_train
        )
    if hasattr(model, "_set_static_graph"):
        model._set_static_graph()
    if domain_aligner is not None and hasattr(domain_aligner, "_set_static_graph"):
        domain_aligner._set_static_graph()

    # Compile the model (optional)
    if args.compile:
        print("Compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    # Initialize wandb
    if args.wandb_log and accelerator.is_main_process:
        import wandb
        run = wandb.init(
            project=args.wandb_project, 
            name=args.wandb_runname, 
            dir='./wandb',
            resume=False,
            config=vars(args)
        )
        run.log_code('.')

    # Training loop
    print('Starting training...')
    t0 = time.time()

    progress_bar = tqdm(
        range(start_epoch, args.epochs), 
        desc="Training Progress", 
        disable=not accelerator.is_main_process
    )
    
    for epoch in progress_bar:
        model.train()
        if domain_aligner is not None:
            domain_aligner.train()
        epoch_log = {}
        
        for step, batch in enumerate(data_loader_train):
            X, _ = batch  # fMRIDataSet returns (X, X) when GPT_training=False
            X = X.float()
            
            # Add channel dimension if needed
            if X.dim() == 3:
                X = X.unsqueeze(1)  # (B, C, V, T)

            with accelerator.autocast():
                # Forward pass - MAE returns (latent, loss)
                latent, mae_loss = model(X, mask_ratio=args.mask_ratio, encoding_only=False)
                domain_loss = X.new_zeros(())
                total_loss = args.mae_weight * mae_loss
                if domain_aligner is not None:
                    alpha = 2 / (1 + math.exp(-10 * iter_num / max(1, args.epochs * num_training_steps_per_epoch))) - 1
                    # Reuse the MAE forward latents. A second DDP forward through the same
                    # model before backward can mark shared parameters ready twice.
                    text_input_ids = get_random_text_batch(text_data, args.text_batch_size, latent.shape[1])
                    domain_loss = domain_aligner(
                        fmri_embed=latent,
                        text_input_ids=text_input_ids.to(accelerator.device),
                        alpha=alpha,
                    )
                    total_loss = total_loss + args.domain_loss_weight * domain_loss

            # NaN detection safeguard
            if torch.isnan(total_loss):
                raise ValueError(
                    f"NaN loss detected at epoch {epoch}, step {step}, iter {iter_num}! "
                    f"Learning rate: {optimizer.param_groups[0]['lr']}"
                )

            # Log
            log = {
                'train/loss': total_loss.detach().item(),
                'train/mae_loss': mae_loss.detach().item(),
                'train/domain_loss': domain_loss.detach().item(),
            }
            
            if not epoch_log:
                epoch_log = log
            else:
                epoch_log = {k: epoch_log[k] + log[k] for k in log}

            # Backward pass with gradient accumulation
            accelerator.backward(total_loss)
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.grad_clip != 0.0:
                    accelerator.clip_grad_norm_(trainable_parameters(model, domain_aligner), args.grad_clip)
                
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                # Log to wandb per iteration
                if accelerator.is_main_process and args.wandb_log and iter_num % args.log_interval == 0:
                    wandb.log({
                        "iter": iter_num,
                        "train_iter/loss": log['train/loss'],
                        "train_iter/mae_loss": log['train/mae_loss'],
                        "train_iter/domain_loss": log['train/domain_loss'],
                        "train_iter/lr": optimizer.param_groups[0]['lr'],
                    })
                
                iter_num += 1

            t1 = time.time()
            t0 = t1

        # Compute epoch averages
        epoch_log = {k: v / (step + 1) for k, v in epoch_log.items()}

        # Gather epoch logs from all processes
        epoch_log_tensor = {k: torch.tensor(v, device=accelerator.device) for k, v in epoch_log.items()}
        epoch_log_tensor = accelerator.gather(epoch_log_tensor)
        epoch_log = {k: epoch_log_tensor[k].mean().item() for k in epoch_log_tensor}

        # Update progress bar
        if accelerator.is_main_process:
            desc = (
                f"epoch {epoch}: train loss {epoch_log['train/loss']:.4f}, "
                f"mae {epoch_log['train/mae_loss']:.4f}, "
                f"domain {epoch_log['train/domain_loss']:.4f}"
            )
            progress_bar.set_description(desc)

            # Log to wandb
            if args.wandb_log:
                log_dict = {
                    'epoch': epoch,
                    'train/loss': epoch_log['train/loss'],
                    'train/mae_loss': epoch_log['train/mae_loss'],
                    'train/domain_loss': epoch_log['train/domain_loss'],
                    'lr': optimizer.param_groups[0]['lr']
                }
                wandb.log(log_dict)

            # Save checkpoint
            if not args.no_save_ckpt:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_aligner = accelerator.unwrap_model(domain_aligner) if domain_aligner is not None else None

                # Create model config dict with essential parameters for loading
                model_config = {
                    'checkpoint_type': 'mae',
                    'cfg_path': args.cfg_path,
                    'cfg_basename': args.cfg_basename,
                    'tokenizer_model_config': args.tokenizer_model_config,
                    'model_size': args.model_size,
                    'patch_size': args.patch_size,
                    'num_rois': args.num_rois,
                    'num_timestamp': args.num_timestamp,
                    'sep_pos_embed': args.sep_pos_embed,
                    'cls_embed': args.cls_embed,
                    'norm_pix_loss': args.norm_pix_loss,
                    'attn_mode': args.attn_mode,
                    'drop_path_rate': args.drop_path_rate,
                    'mlp_ratio': args.mlp_ratio,
                    'qkv_bias': args.qkv_bias,
                    'qk_scale': args.qk_scale,
                    'drop_rate': args.drop_rate,
                    'attn_drop_rate': args.attn_drop_rate,
                    'decoder_embed_dim': args.decoder_embed_dim,
                    'decoder_depth': args.decoder_depth,
                    'decoder_num_heads': args.decoder_num_heads,
                    'in_chans': 1,
                }

                # Create dataset config dict with preprocessing parameters
                dataset_config = {
                    'norm': args.norm,
                    'num_timestamp': args.num_timestamp,
                    'segment_method': args.segment_method,
                    'oversample_factor': args.oversample_factor,
                    'num_rois': args.num_rois,
                    'uses_random_text': use_domain_alignment,
                    'text_data_dir': args.text_data_dir if use_domain_alignment else None,
                }

                checkpoint = {
                    'model': unwrapped_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'iter_num': iter_num,
                    'epoch': epoch,
                    'best_loss': best_loss,
                    'loss': epoch_log['train/loss'],
                    'mae_loss': epoch_log['train/mae_loss'],
                    'domain_loss': epoch_log['train/domain_loss'],
                    'args': vars(args),
                    'model_config': model_config,
                    'dataset_config': dataset_config,
                }
                if unwrapped_aligner is not None:
                    checkpoint['domain_aligner'] = unwrapped_aligner.state_dict()
                    checkpoint['alignment_config'] = {
                        'type': 'domain_confusion_random_text',
                        'lm_name': args.lm_name,
                        'text_data_dir': args.text_data_dir,
                        'text_batch_size': args.text_batch_size,
                        'domain_loss_weight': args.domain_loss_weight,
                    }

                print(f"saving checkpoint to {checkpoint_out_dir}")
                torch.save(checkpoint, os.path.join(checkpoint_out_dir, 'ckpt.pt'))

                if (epoch + 1) % args.save_ckpt_freq == 0:
                    print(f"saving checkpoint {epoch} to {checkpoint_out_dir}")
                    torch.save(checkpoint, os.path.join(checkpoint_out_dir, f'ckpt-{epoch}.pt'))

                # Save best checkpoint
                current_loss = epoch_log['train/loss']
                if current_loss < best_loss:
                    best_loss = current_loss
                    checkpoint['best_loss'] = best_loss
                    print(f"saving best checkpoint with loss {best_loss:.4f} to {checkpoint_out_dir}")
                    torch.save(checkpoint, os.path.join(checkpoint_out_dir, 'ckpt-best.pt'))

        # Rebuild segment mapping for next epoch (for random augmentation)
        if args.rebuild_segment_mapping and args.segment_method in ['random_clip', 'random_sample']:
            # Wait for all processes before rebuilding
            accelerator.wait_for_everyone()
            
            # epoch_seed = args.seed + epoch + 1
            # np.random.seed(epoch_seed)
            # torch.manual_seed(epoch_seed)  # Also sync torch RNG for DataLoader shuffle
            
            # Rebuild segment mapping in each dataset
            for dataset in dataset_train_list:
                dataset.rebuild_segment_mapping()
            
            dataset_train = torch.utils.data.ConcatDataset(dataset_train_list)
            new_data_loader_train = torch.utils.data.DataLoader(
                dataset_train,
                batch_size=args.batch_size,
                num_workers=16,
                pin_memory=True,
                shuffle=True,
                drop_last=True
            )
            data_loader_train = accelerator.prepare(new_data_loader_train)
            
            if accelerator.is_main_process:
                print(f"Rebuilt segment mapping for epoch {epoch + 1}, new dataset size: {len(dataset_train)}")
        accelerator.wait_for_everyone()

    # End training
    print('Training completed!')
    accelerator.end_training()


def generate_run_name(args):
    """Generate checkpoint directory and wandb run name from arguments."""
    data_norm = ckpt_naming.data_norm_name(args.dataset_dir, args.norm)
    components = ['mae']
    
    if args.run_prefix:
        components.append(args.run_prefix)

    components.append(args.cfg_basename)
    components.append(f"mask{int(args.mask_ratio * 100)}")
    
    if args.sep_pos_embed:
        components.append("sep_pos")
    
    if args.cls_embed:
        components.append("cls")
    
    if args.norm_pix_loss:
        components.append("norm_pix")

    if args.domain_loss_weight > 0.0:
        components.append(f"domain{ckpt_naming.format_float(args.domain_loss_weight)}")
        components.append(ckpt_naming.simplify_lm_name(args.lm_name))
    
    run_name = ckpt_naming.join_name_parts(*components)
    ckpt_dir = ckpt_naming.tokenizer_ckpt_dir('mae', data_norm, run_name)
    
    return ckpt_dir, run_name


def get_args():
    def list_of_strs(arg):
        return arg.split(',')
    
    parser = argparse.ArgumentParser('MAE training script', add_help=False)
    
    # Paths
    parser.add_argument('--run_prefix', default='', type=str, 
                        help='Optional prefix for run name and checkpoint directory')
    parser.add_argument('--ckpt_dir', default=None, type=str, 
                        help='Path where to save checkpoints, if not provided will be auto-generated')
    parser.add_argument('--dataset_dir', default=['data/UKB/fmri/TianS3/'], type=list_of_strs, 
                        help='Path to the training dataset folder(s), can be multiple paths separated by comma')
    parser.add_argument('--cfg_path', type=str, default='configs/vit_base_p160_newTok.yaml',
                        help='Unified Tokenizer config used for MAE encoder architecture')
    
    # Logging
    parser.add_argument('--log_interval', default=10, type=int,
                        help='Log training metrics every N iterations')
    parser.add_argument('--wandb_log', default=False, action='store_true',
                        help='Enable wandb logging')
    parser.add_argument('--wandb_project', default='BrainFM_MAE',
                        help='Wandb project name')
    parser.add_argument('--wandb_runname', default=None, type=str, 
                        help='Wandb run name, if not provided will be auto-generated')
    
    # Training hyperparameters
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', default=200, type=int,
                        help='Total number of training epochs')
    parser.add_argument('--warmup_epochs', default=0, type=int,
                        help='Number of warmup epochs for learning rate')
    parser.add_argument('--save_ckpt_freq', default=10, type=int,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', default=False, action='store_true',
                        help='Resume from checkpoint if exists')
    parser.add_argument('--rebuild_segment_mapping', default=False, action='store_true',
                        help='Rebuild segment mapping after each epoch for random methods (default: True)')
    parser.add_argument('--mae_weight', type=float, default=1.0,
                        help='Weight for the masked reconstruction objective')

    # Optional text-domain adversarial alignment, matching train_quantizer.py
    parser.add_argument('--domain_loss_weight', type=float, default=0.0,
                        help='Weight for random-text adversarial domain loss; 0 disables alignment')
    parser.add_argument('--text_batch_size', default=16, type=int,
                        help='Random text batch size for domain alignment')
    parser.add_argument('--text_data_dir', default='data/text/openwebtext', type=str,
                        help='Directory containing openwebtext train.bin/val.bin token files')
    parser.add_argument('--lm_name', type=str, default='gpt2',
                        help='Language model used for frozen token embeddings in domain alignment')

    # Data preprocessing
    parser.add_argument('--norm', type=str, default='robust',
                        help='Normalization method for fMRI data')
    parser.add_argument('--num_rois', type=int, default=None,
                        help='Number of ROIs (spatial dimension)')
    parser.add_argument('--num_timestamp', type=int, default=None,
                        help='Number of time points (temporal dimension)')

    parser.add_argument('--segment_method', type=str, default='clip', choices=['clip', 'random_clip', 'sequential_clip', 'pool'],
                        help='Method for segmenting time series during training')
    parser.add_argument('--oversample_factor', type=float, default=1.0)
    
    # Model architecture
    parser.add_argument('--model_size', type=str, default=None,
                        choices=list(TOKENIZER_ENCODER_SPECS.keys()),
                        help='Override model.vit_enc_model_size from cfg_path')
    parser.add_argument('--patch_size', type=int, default=None,
                        help='Override model.vit_enc_patch_size from cfg_path')
    parser.add_argument('--mask_ratio', type=float, default=0.5,
                        help='Ratio of patches to mask during training')
    parser.add_argument('--sep_pos_embed', default=False, action='store_true',
                        help='Use MAE-only separate positional embeddings; disables clean Tokenizer encoder mapping')
    parser.add_argument('--cls_embed', default=None, action=argparse.BooleanOptionalAction,
                        help='Override model.add_cls_token from cfg_path')
    parser.add_argument('--norm_pix_loss', default=False, action='store_true',
                        help='Normalize pixel values in loss computation')
    parser.add_argument('--attn_mode', type=str, default=None,
                        choices=['normal', 'flash_attn'],
                        help='Override model.attn_mode from cfg_path')
    parser.add_argument('--gradient_checkpointing', default=False, action='store_true',
                        help='Enable gradient checkpointing to save memory')
    parser.add_argument('--drop_path_rate', type=float, default=None,
                        help='Override model.drop_path_rate from cfg_path')
    parser.add_argument('--decoder_embed_dim', type=int, default=512,
                        help='MAE decoder embedding dimension')
    parser.add_argument('--decoder_depth', type=int, default=8,
                        help='MAE decoder depth')
    parser.add_argument('--decoder_num_heads', type=int, default=16,
                        help='MAE decoder attention heads')
    
    # Optimizer
    parser.add_argument('--learning_rate', type=float, default=5e-5, metavar='LR',
                        help='Base learning rate (absolute lr = base_lr * batch_size / 256)')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='Minimum learning rate for cosine scheduler')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Adam beta2')
    parser.add_argument('--grad_clip', type=float, default=0.0,
                        help='Clip gradients at this value, or disable if == 0.0')
    
    # Misc
    parser.add_argument('--seed', default=1337, type=int,
                        help='Random seed')
    parser.add_argument('--compile', default=False, action='store_true',
                        help='Use PyTorch 2.0 compile for faster training')
    parser.add_argument('--no_save_ckpt', default=False, action='store_true',
                        help='Skip checkpoint directory creation and checkpoint saving')

    args = parser.parse_args()
    args = apply_tokenizer_config(args)
    
    # Auto-generate ckpt_dir and wandb_runname if not provided
    if args.ckpt_dir is None or args.wandb_runname is None:
        auto_ckpt_dir, auto_run_name = generate_run_name(args)
        if args.ckpt_dir is None:
            args.ckpt_dir = auto_ckpt_dir
        if args.wandb_runname is None:
            args.wandb_runname = auto_run_name
    
    # Validate patch size
    assert args.num_timestamp % args.patch_size == 0, \
        f"num_timestamp ({args.num_timestamp}) must be divisible by patch_size ({args.patch_size})"
    
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)
