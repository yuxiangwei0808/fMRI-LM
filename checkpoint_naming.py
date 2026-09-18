"""Shared checkpoint naming helpers for the staged BrainFM training pipeline."""

import json
import os
import re
from datetime import datetime

from omegaconf import OmegaConf


TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
CHECKPOINT_SUFFIXES = (".pt", ".pth", ".bin", ".safetensors")
NORMALIZATION_NAMES = (
    "robust_roi",
    "std_roi",
    "roi_frame",
    "robust",
    "std",
    "frame",
    "roi",
    "none",
)


def sanitize_component(value):
    value = str(value).strip()
    value = value.replace(os.sep, "-").replace("/", "-")
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"[^A-Za-z0-9_.+-]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-_") or "none"


def format_float(value):
    return f"{float(value):g}"


def join_name_parts(*parts, sep="-"):
    return sep.join(sanitize_component(part) for part in parts if part not in (None, "", False))


def checkpoint_timestamp():
    return datetime.now().strftime(TIMESTAMP_FORMAT)


def append_postfix_and_timestamp(base_name, postfix="", add_timestamp=True):
    parts = [base_name]
    if postfix:
        parts.append(sanitize_component(str(postfix).strip("_")))
    if add_timestamp:
        parts.append(checkpoint_timestamp())
    return "_".join(part for part in parts if part)


def config_name(cfg_path):
    return sanitize_component(os.path.splitext(os.path.basename(cfg_path))[0])


def tokenizer_mode_from_config(cfg_path):
    cfg = OmegaConf.load(cfg_path)
    return sanitize_component(OmegaConf.select(cfg, "model.quantize_mode") or "tokenizer")


def simplify_lm_name(lm_name):
    return sanitize_component(str(lm_name).rstrip("/").split("/")[-1])


def dataset_name_from_paths(dataset_dirs):
    names = []
    known = (
        ("HCP_Aging", "HCP-Aging"),
        ("HCP-Aging", "HCP-Aging"),
        ("ADHD200", "ADHD200"),
        ("ABIDE2", "ABIDE2"),
        ("ABCD", "ABCD"),
        ("ADNI", "ADNI"),
        ("UKB", "UKB"),
        ("HCP", "HCP"),
        ("ADHD", "ADHD"),
        ("ABIDE", "ABIDE"),
    )
    for dataset in dataset_dirs:
        dataset_str = str(dataset)
        name = None
        for needle, label in known:
            if needle in dataset_str:
                name = label
                break
        if name is None:
            name = os.path.basename(dataset_str.rstrip("/")) or "custom"
        name = sanitize_component(name)
        if name not in names:
            names.append(name)
    return "_".join(names) if names else "data"


def data_norm_name(dataset_dirs, norm=None):
    data_name = dataset_name_from_paths(dataset_dirs)
    if norm and norm != "none":
        return f"{data_name}-{sanitize_component(norm)}"
    return data_name


def tokenizer_ckpt_dir(objective, data_norm, run_name):
    return os.path.join("checkpoints", "tokenizer", sanitize_component(objective), sanitize_component(data_norm), sanitize_component(run_name))


def pretrain_ckpt_dir(data_norm, lm_name, run_name):
    return os.path.join("checkpoints", "pretrain", sanitize_component(data_norm), simplify_lm_name(lm_name), sanitize_component(run_name))


def checkpoint_run_dir(checkpoint_path):
    run_dir = os.path.normpath(checkpoint_path)
    if os.path.isfile(run_dir) or os.path.basename(run_dir).endswith(CHECKPOINT_SUFFIXES):
        run_dir = os.path.dirname(run_dir)
    if os.path.basename(run_dir).startswith("deepspeed_checkpoint"):
        run_dir = os.path.dirname(run_dir)
    return run_dir


def normalization_from_checkpoint_path(checkpoint_path):
    """Return a path-encoded normalization name, independent of path depth."""
    components = re.split(r"[\\/]", os.path.normpath(str(checkpoint_path)))
    for component in components:
        for norm in NORMALIZATION_NAMES:
            if component == norm or component.endswith(f"-{norm}"):
                return norm
    return None


def derive_instruction_ckpt_base(pretrained_ckpt):
    run_dir = checkpoint_run_dir(pretrained_ckpt)
    parts = run_dir.split(os.sep)
    for idx, part in enumerate(parts):
        if part == "pretrain":
            parts[idx] = "instruction"
            return os.sep.join(parts)
    raise ValueError(
        "Cannot derive instruction checkpoint directory because pretrained_ckpt "
        f"does not contain a 'pretrain' path component: {pretrained_ckpt}. "
        "Pass --ckpt_dir explicitly to override auto-derivation."
    )


def resolve_instruction_ckpt_dir(args, default_postfix=""):
    if args.ckpt_dir == "tmp":
        pretrained_ckpt = getattr(args, "pretrained_ckpt", "")
        if pretrained_ckpt:
            base_dir = derive_instruction_ckpt_base(pretrained_ckpt)
        else:
            data_norm = data_norm_name(getattr(args, "datasets", []), getattr(args, "norm", None))
            lm_name = getattr(args, "lm_name", "lm")
            base_dir = os.path.join("checkpoints", "instruction", data_norm, simplify_lm_name(lm_name), "scratch")
    else:
        base_dir = args.ckpt_dir
    postfix = getattr(args, "ckpt_postfix", "") or default_postfix
    return append_postfix_and_timestamp(base_dir, postfix=postfix, add_timestamp=not args.no_timestamp)


def serializable(value):
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        if isinstance(value, dict):
            return {str(k): serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [serializable(v) for v in value]
        return str(value)


def write_lineage_json(save_dir, **metadata):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "lineage.json")
    payload = {k: serializable(v) for k, v in metadata.items()}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path
