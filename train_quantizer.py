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
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from accelerate import Accelerator, DistributedDataParallelKwargs
import math

from quantizers import TokAlign
from dataset import fMRIDataSet
import checkpoint_naming as ckpt_naming


accelerator = None

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

def prepare_tokenizer_config(cfg):
    if OmegaConf.select(cfg, "model.vq_model") is not None:
        raise ValueError(
            "Legacy model.vq_model quantizer configs are no longer supported here. "
            "Use the unified Tokenizer config with model.quantize_mode instead."
        )

    if OmegaConf.select(cfg, "model") is None:
        raise ValueError("Config must contain a model section for the unified Tokenizer.")

    for key in ("num_rois", "num_timestamp"):
        if OmegaConf.select(cfg, f"model.{key}") is None:
            raise ValueError(f"Tokenizer config is missing model.{key}.")

    if OmegaConf.select(cfg, "model.quantize_mode") is None:
        raise ValueError("Tokenizer config is missing model.quantize_mode.")

    if OmegaConf.select(cfg, "model.image_size") is None:
        cfg.model.image_size = [int(cfg.model.num_rois), int(cfg.model.num_timestamp)]

    if OmegaConf.select(cfg, "model.gate_attention") is None and OmegaConf.select(cfg, "model.gate_attn") is not None:
        cfg.model.gate_attention = cfg.model.gate_attn

    # rope_mode may be present in unified configs, but this tokenizer path intentionally
    # uses the non-RoPE transformer block.
    return cfg, int(cfg.model.image_size[1])

def build_quantizer_model(args, quantizer_cfg):
    return TokAlign(quantizer_cfg, lm_name=args.lm_name)

def main(args):
    global accelerator
    
    init(args)

    cfg = OmegaConf.load(args.cfg_path)
    quantizer_cfg, clip_timepoints = prepare_tokenizer_config(cfg)

    checkpoint_out_dir = args.ckpt_dir
    if accelerator.is_main_process and not args.no_save_ckpt:
        os.makedirs(checkpoint_out_dir, exist_ok=True)
        ckpt_naming.write_lineage_json(
            checkpoint_out_dir,
            stage='tokenizer',
            objective='default',
            checkpoint_dir=checkpoint_out_dir,
            cfg_path=args.cfg_path,
            dataset_dir=args.dataset_dir,
            data_norm=ckpt_naming.data_norm_name(args.dataset_dir, args.norm),
            lm_name=args.lm_name if args.domain_loss_weight > 0.0 else None,
            quantize_mode=ckpt_naming.tokenizer_mode_from_config(args.cfg_path),
            run_name=args.wandb_runname,
            args=vars(args),
        )

    print('prepare dataloader...')
    total_samples = 0
    train_datset = []
    for dataset_path in args.dataset_dir:
        assert os.path.exists(dataset_path), f"Dataset path {dataset_path} does not exist."
        dataset = fMRIDataSet(dataset_path, norm=args.norm, GPT_training=False, clip_timepoints=clip_timepoints)
        total_samples += len(dataset)
        train_datset.append(dataset)
    train_datset = torch.utils.data.ConcatDataset(train_datset)
    data_loader_train = torch.utils.data.DataLoader(train_datset, batch_size=args.batch_size, num_workers=16, pin_memory=True, shuffle=True)
    print('finished!')

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

    if os.path.exists(os.path.join(checkpoint_out_dir, 'ckpt.pt')) and args.resume:
        init_from = 'resume'
    else:
        init_from = 'scratch'

    # init these up here, can override if init_from='resume'
    iter_num = 0
    best_loss = float('inf')

    if init_from == 'scratch':
        print("Initializing a new model from scratch")
        model = build_quantizer_model(args, quantizer_cfg)
        start_epoch = 0
    elif init_from == 'resume':
        print(f"Resuming training from {checkpoint_out_dir}")
        ckpt_path = os.path.join(checkpoint_out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)  # Load to CPU first

        model = build_quantizer_model(args, quantizer_cfg)
        state_dict = checkpoint['model']
        
        # Fix state dict keys
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        start_epoch = checkpoint['epoch'] + 1
        
        # Load best loss if available
        if 'best_loss' in checkpoint:
            best_loss = checkpoint['best_loss']
        elif 'loss' in checkpoint:
            best_loss = checkpoint['loss']

    num_training_steps_per_epoch = total_samples // args.batch_size // accelerator.num_processes
    num_tokens = model.num_tokens

    # optimizer
    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (args.beta1, args.beta2), 'cpu')
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # lr scheduler: cosine with optional linear warmup (active)
    total_steps = args.epochs * num_training_steps_per_epoch // args.gradient_accumulation_steps
    if args.warmup_epochs > 0:
        warmup_steps = args.warmup_epochs * num_training_steps_per_epoch // args.gradient_accumulation_steps
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-2, total_iters=warmup_steps)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=args.min_lr)
        lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    else:
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.min_lr)

    if init_from == 'resume' and 'lr_scheduler' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    
    checkpoint = None

    # Prepare everything with Accelerate - this replaces DDP wrapping
    model, optimizer, lr_scheduler, data_loader_train = accelerator.prepare(
        model, optimizer, lr_scheduler, data_loader_train
    )
    model._set_static_graph()
 
    # compile the model (optional)
    if getattr(args, 'compile', False):
        print("compiling the model.. (takes a ~minute)")
        model = torch.compile(model)

    # Initialize wandb
    if args.wandb_log and accelerator.is_main_process:
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.wandb_runname, dir='./wandb', resume=False, config=vars(args))
        run.log_code('.')
        
        artifact = wandb.Artifact(
            name="config", 
            type="config",
            description="Configuration file for model"
        )
        artifact.add_file(local_path=args.cfg_path, name=args.cfg_path)
        run.log_artifact(artifact)

    # training loop
    X_text, Y_text = get_batch('train', num_tokens)
    t0 = time.time()
    local_iter_num = 0

    progress_bar = tqdm(range(start_epoch, args.epochs), desc="Training Progress", disable=not accelerator.is_main_process)
    for epoch in progress_bar:
        epoch_log = {}
        for step, batch in enumerate(data_loader_train):
            X, Y_raw = batch
            X = X.float()
            Y_raw = Y_raw.float()

            with accelerator.autocast():  # Replace manual autocast context
                alpha = 2 / (1 + math.exp(-10 * iter_num / args.epochs / num_training_steps_per_epoch)) - 1

                loss, domain_loss, log = model(X, Y_raw, alpha)
                domain_loss2 = model(X_text.to(accelerator.device))
            
                loss = (loss + (domain_loss + domain_loss2) * args.domain_loss_weight) / args.gradient_accumulation_steps

            # Check log values for NaN
            for key, value in log.items():
                if isinstance(value, (int, float)) and math.isnan(value):
                    raise ValueError(f"NaN detected in log['{key}'] at epoch {epoch}, step {step}, iter {iter_num}")
                elif torch.is_tensor(value) and torch.isnan(value).any():
                    raise ValueError(f"NaN detected in log['{key}'] at epoch {epoch}, step {step}, iter {iter_num}")
            # NaN detection safeguard
            if torch.isnan(loss):
                error_msg = (
                    f"NaN loss detected at epoch {epoch}, step {step}, iter {iter_num}!\n"
                    f"Loss components: loss={loss.item()}, domain_loss={domain_loss.item()}, "
                    f"domain_loss2={domain_loss2.item()}\n"
                    f"Log: {log}\n"
                    f"Learning rate: {optimizer.param_groups[0]['lr']}\n"
                    f"Alpha: {alpha}"
                )
                raise ValueError(error_msg)
            
            if epoch_log == {}: 
                epoch_log = log
            else: 
                epoch_log = {k: epoch_log[k] + log[k] for k in log}

            # Use accelerator.backward instead of loss.backward()
            accelerator.backward(loss)
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Check for NaN gradients before clipping
                for name, param in model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        raise ValueError(
                            f"NaN gradient detected in parameter '{name}' at epoch {epoch}, "
                            f"step {step}, iter {iter_num}"
                        )
                
                if args.grad_clip != 0.0:
                    # Accelerate handles gradient clipping across devices
                    accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            # evaluate the loss and log (only on main process)
            # if (iter_num + 1) % args.log_interval == 0 and accelerator.is_main_process:
            #     print(f"epoch {epoch} step [{step + 1}/{num_training_steps_per_epoch}]: train loss {log['train/total_loss']:.4f}, raw loss {log['train/rec_raw_loss']:.4f}, quant loss {log['train/quant_loss']:.4f} \
            #         domain loss {log['train/domain_loss'] + domain_loss2.item():.4f}")

            if accelerator.is_main_process and args.wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train_iter/total_loss": log['train/total_loss'],
                    "train_iter/rec_loss": log['train/rec_raw_loss'],
                    "train_iter/quant_loss": log['train/quant_loss'],
                    "train_iter/domain_loss": log['train/domain_loss'] + domain_loss2.item(),
                    "train_iter/lr": optimizer.param_groups[0]['lr'],
                })
            
            X_text, Y_text = get_batch('train', num_tokens)
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            iter_num += 1
            local_iter_num += 1

        # gather logs from processes
        epoch_log = {k: torch.as_tensor(v, device=accelerator.device, dtype=torch.float32) / (step + 1) for k, v in epoch_log.items()}
        epoch_log = accelerator.gather(epoch_log)
        epoch_log = {k: epoch_log[k].mean().item() for k in epoch_log}

        # Save checkpoints (only on main process)
        if accelerator.is_main_process:
            progress_bar.set_description(f"epoch {epoch}: {epoch_log}")

            if args.wandb_log:
                wandb.log({
                    'epoch': epoch,
                    'train/total_loss': epoch_log['train/total_loss'],
                    'train/rec_loss': epoch_log['train/rec_raw_loss'],
                    'train/quant_loss': epoch_log['train/quant_loss'],
                    'train/domain_loss': epoch_log['train/domain_loss'],
                    'lr': optimizer.param_groups[0]['lr']
                })

            # Use accelerator.unwrap_model to get the original model for saving
            if not args.no_save_ckpt:
                current_loss = epoch_log['train/total_loss']
                is_best = current_loss < best_loss
                if is_best:
                    best_loss = current_loss

                unwrapped_model = accelerator.unwrap_model(model)
                checkpoint = {
                    'model': unwrapped_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
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
                    torch.save(checkpoint, os.path.join(checkpoint_out_dir, f'ckpt-best.pt'))

        # Wait for all processes to finish the epoch
        accelerator.wait_for_everyone()

    # End training
    accelerator.end_training()


def tokenizer_mode_from_config(cfg_path):
    cfg = OmegaConf.load(cfg_path)
    return OmegaConf.select(cfg, "model.quantize_mode") or "tokenizer"


def dataset_name_from_paths(dataset_dirs):
    dataset_name = []
    for dataset in dataset_dirs:
        if 'UKB' in dataset:
            dataset_name.append('UKB')
        elif 'HCP' in dataset:
            dataset_name.append('HCP')
        elif 'ABCD' in dataset:
            dataset_name.append('ABCD')
        else:
            dataset_name.append('custom')
    return '_'.join(dataset_name)


def generate_run_name(args):
    """Generate checkpoint directory and wandb run name from arguments."""
    data_norm = ckpt_naming.data_norm_name(args.dataset_dir, args.norm)
    cfg_basename = ckpt_naming.config_name(args.cfg_path)
    tokenizer_mode = ckpt_naming.tokenizer_mode_from_config(args.cfg_path)

    components = []
    if args.run_prefix:
        components.append(args.run_prefix)

    components.append(f"tok_{tokenizer_mode}-{cfg_basename}")

    if args.domain_loss_weight > 0.0:
        components.append(f"domain{ckpt_naming.format_float(args.domain_loss_weight)}")
        components.append(ckpt_naming.simplify_lm_name(args.lm_name))

    run_name = ckpt_naming.join_name_parts(*components)
    ckpt_dir = ckpt_naming.tokenizer_ckpt_dir('default', data_norm, run_name)
    print(f"Auto-generated run name: {run_name}")

    return ckpt_dir, run_name

def get_args():
    def list_of_strs(arg):
        return arg.split(',')
    parser = argparse.ArgumentParser('Tokenizer/TokAlign training script', add_help=False)
    parser.add_argument('--run_prefix', default='', type=str, help='Optional prefix for run name')
    parser.add_argument('--ckpt_dir', default=None, type=str, help='path where to save, if not provided will be auto-generated')
    parser.add_argument('--dataset_dir', default=['data/UKB/fmri/TianS3/'], type=list_of_strs, help='path to the dataset h5 file')
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--wandb_log', default=False, action='store_true')
    parser.add_argument('--wandb_project', default='BrainFM_quantizer')
    parser.add_argument('--wandb_runname', default=None, type=str, help='wandb run name, if not provided will be auto-generated')
    
    # training args
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--text_batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--warmup_epochs', default=0, type=int)
    parser.add_argument('--save_ckpt_freq', default=10, type=int)
    parser.add_argument('--resume', default=False, action='store_true',)

    # tokenizer args
    parser.add_argument('--cfg_path', type=str, default='configs/vit_base_p160_newTok.yaml', help='path to the tokenizer config file',)
    parser.add_argument('--lm_name', type=str, default='gpt2', help='name of the language model to use')

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

    parser.add_argument('--norm', type=str, default='robust')
    parser.add_argument('--domain_loss_weight', type=float, default=1.0, help='weight for the domain loss')

    parser.add_argument('--compile', default=False, action='store_true')
    parser.add_argument('--no_save_ckpt', default=False, action='store_true', help='skip checkpoint directory creation and checkpoint saving')

    args = parser.parse_args()

    # Auto-generate ckpt_dir and wandb_runname if not provided, matching train_quantizer_contr.py.
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