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
RELEASE_FILES = [
    ("best_avg_classification_ckpt.pt", "fMRI-LM-B-Qwen3-0.6B-instruct.pt"),
]
# Per-task checkpoints, published only with --include-per-task.
PER_TASK_FILES = [
    ("best_UKB-sex_ckpt.pt",       "fMRI-LM-B-Qwen3-0.6B-best-UKB-sex.pt"),
    ("best_HCP-sex_ckpt.pt",       "fMRI-LM-B-Qwen3-0.6B-best-HCP-sex.pt"),
    ("best_HCP_Aging-sex_ckpt.pt", "fMRI-LM-B-Qwen3-0.6B-best-HCP_Aging-sex.pt"),
    ("best_ADNI-AD_ckpt.pt",       "fMRI-LM-B-Qwen3-0.6B-best-ADNI-AD.pt"),
    ("best_ABIDE2-ASD_ckpt.pt",    "fMRI-LM-B-Qwen3-0.6B-best-ABIDE2-ASD.pt"),
    ("best_ADHD200-ADHD_ckpt.pt",  "fMRI-LM-B-Qwen3-0.6B-best-ADHD200-ADHD.pt"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True,
                    help="target model repo, e.g. yuxiangwei0808/fMRI-LM-B-Qwen3-0.6B")
    ap.add_argument("--src", required=True,
                    help="directory holding the stage-3 checkpoints")
    ap.add_argument("--private", action="store_true",
                    help="create the repo private (default: public)")
    ap.add_argument("--include-per-task", action="store_true",
                    help="also upload the six per-task best checkpoints (adds ~15.4 GiB)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be uploaded and exit")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    card = os.path.join(repo_root, "MODEL_CARD.md")
    if not os.path.isfile(card):
        sys.exit(f"model card not found: {card}")

    names = list(RELEASE_FILES)
    if args.include_per_task:
        names += PER_TASK_FILES

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
        print(f"  {name:34} {size / 2**30:6.2f} GiB")

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
    for path, name, size in files:
        print(f"uploading {name} ({size / 2**30:.2f} GiB) ...", flush=True)
        api.upload_file(path_or_fileobj=path, path_in_repo=name,
                        repo_id=args.repo_id, repo_type="model")
    print(f"\ndone: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
