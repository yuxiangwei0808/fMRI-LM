#!/usr/bin/env python3
"""Publish the released fMRI-LM checkpoints to a Hugging Face model repo.

The model card in MODEL_CARD.md becomes the repo's README.md.

Usage:
    python scripts/upload_to_hf.py --repo-id <user>/fMRI-LM-B-Qwen3-0.6B \
        --src /path/to/instruction_run_dir --dry-run

Drop --dry-run to actually upload. Authentication comes from `hf auth login`
or the HF_TOKEN environment variable.
"""
import argparse
import os
import sys

# (source filename in --src, filename published in the HF repo)
# Published release: stage-1 tokenizer and stage-2 pretraining, three variants.
# Keys are paths under --src (the checkpoint store root); values are paths in the repo.
RELEASE_FILES = {
    "vq-contrastive": [
        ("tokenizer/contrastive/UKB-robust/contr_siglip-desc_fc_ica-pool_mean_mean-tok_vq-vit_base_p160_newTok-Qwen3-0.6B/ckpt-best.pt",
         "vq-contrastive/stage1-tokenizer.pt"),
        ("pretrain/UKB-robust/Qwen3-0.6B/fc_ica_f2t1_Nocontr-lora_r1_a2_drop.1_qk_20260616_173356/"
         "deepspeed_checkpoint_best_f2t/merged_checkpoint.pt",
         "vq-contrastive/stage2-pretrain-Qwen3-0.6B.pt"),
    ],
    "vq-domain": [
        ("tokenizer/default/UKB-robust/normsweep0615-tok_vq-vit_base_p160_newTok-domain1-Qwen3-0.6B/ckpt-best.pt",
         "vq-domain/stage1-tokenizer.pt"),
        ("pretrain/UKB-robust/Qwen3-0.6B/fc_gradient_text0.1_f2t1_lora_r1_a2_drop.1_qk_vqdomain1_descsweep0619_robust_vqdomain1/"
         "deepspeed_checkpoint_best_f2t/merged_checkpoint.pt",
         "vq-domain/stage2-pretrain-Qwen3-0.6B.pt"),
    ],
    "mae": [
        ("tokenizer_mae/UKB/mae-vit_base_p160_newTok-mask50-domain1-Qwen3-0.6B/ckpt-best.pt",
         "mae/stage1-tokenizer.pt"),
        ("pretrain/UKB-robust/Qwen3-0.6B/fc_ica_f2t1_MAE_domain-lora_r1_a2_drop.1_qk_0610_182333/"
         "deepspeed_checkpoint_best_f2t/merged_checkpoint.pt",
         "mae/stage2-pretrain-Qwen3-0.6B.pt"),
    ],
}
# Training state is stripped from stage-1 files before upload.
DROP_KEYS = {"optimizer", "lr_scheduler"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True,
                    help="target model repo, e.g. yuxiangwei0808/fMRI-LM-B-Qwen3-0.6B")
    ap.add_argument("--src", required=True,
                    help="checkpoint store root that the paths above are relative to")
    ap.add_argument("--private", action="store_true",
                    help="create the repo private (default: public)")
    ap.add_argument("--variant", choices=sorted(RELEASE_FILES) + ["all"], default="all",
                    help="which variant to publish (default: all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be uploaded and exit")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    card = os.path.join(repo_root, "MODEL_CARD.md")
    if not os.path.isfile(card):
        sys.exit(f"model card not found: {card}")

    variants = sorted(RELEASE_FILES) if args.variant == "all" else [args.variant]
    names = [pair for v in variants for pair in RELEASE_FILES[v]]

    files, missing, total = [], [], 0
    for src_name, repo_name in names:
        path = os.path.join(args.src, src_name)
        if os.path.isfile(path):
            size = os.path.getsize(path)
            files.append((path, repo_name, size))
            total += size
        else:
            missing.append(src_name)

    if missing:
        print("WARNING: not found in --src, skipping:", ", ".join(missing), file=sys.stderr)
    if not files:
        sys.exit("nothing to upload")

    print(f"target repo : {args.repo_id} ({'private' if args.private else 'public'})")
    print(f"model card  : {card} -> README.md")
    print(f"checkpoints : {len(files)} files, {total / 2**30:.1f} GiB")
    for _, name, size in files:
        print(f"  {name:46} {size / 2**30:6.2f} GiB")

    if args.dry_run:
        print("\n--dry-run: nothing uploaded")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model",
                    private=args.private, exist_ok=True)
    api.upload_file(path_or_fileobj=card, path_in_repo="README.md",
                    repo_id=args.repo_id, repo_type="model")
    print("uploaded README.md")
    import tempfile, torch
    with tempfile.TemporaryDirectory() as tmp:
        for path, name, size in files:
            if "stage1" in name:
                ck = torch.load(path, map_location="cpu", weights_only=False)
                ck = {k: v for k, v in ck.items() if k not in DROP_KEYS}
                path = os.path.join(tmp, name.replace("/", "_"))
                torch.save(ck, path)
                size = os.path.getsize(path)
            print(f"uploading {name} ({size / 2**30:.2f} GiB) ...", flush=True)
            api.upload_file(path_or_fileobj=path, path_in_repo=name,
                            repo_id=args.repo_id, repo_type="model")
    print(f"\ndone: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
