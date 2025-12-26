from typing import Dict, List, Optional
import numpy as np
import torch
import sklearn.metrics as sklearn_metrics

from metrics import binary_metrics_fn, multiclass_metrics_fn

def get_metrics(output, target, metrics, is_binary, is_regression=False):
    if is_binary:
        if 'roc_auc' not in metrics or sum(target) * (len(target) - sum(target)) != 0:  # to prevent all 0 or all 1 and raise the AUROC error
            results = binary_metrics_fn(
                target,
                output,
                metrics=metrics
            )
        else:
            results = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
    elif is_regression:
        results = {
            "mse": sklearn_metrics.mean_squared_error(target, output),
            "mae": sklearn_metrics.mean_absolute_error(target, output),
            "r2": sklearn_metrics.r2_score(target, output),
            "pearson": np.corrcoef(target, output)[0, 1],
        }
    else:
        results = multiclass_metrics_fn(
            target, output, metrics=metrics
        )
    return results

def get_allowed_token_id(target_name, tokenizer):
    if target_name == 'sex':
        return [tokenizer.encode(phrase, add_special_tokens=False) for phrase in [' Male', ' Female']]
    elif target_name == 'age':
        return [tokenizer.encode(f' {i}', add_special_tokens=False) for i in range(10, 101)]
    elif target_name == 'fluidintel':
        return [tokenizer.encode(f' {i}', add_special_tokens=False) for i in range(0, 16)]
    elif target_name == 'AD':
        # return [tokenizer.encode(phrase, add_special_tokens=False) for phrase in [' CN', ' AD', ' MCI']]
        return [tokenizer.encode(phrase, add_special_tokens=False) for phrase in [' CN', ' AD']]
    elif target_name == 'AsymAD':
        return [tokenizer.encode(phrase, add_special_tokens=False) for phrase in [' CN', ' AM']]
    elif target_name == 'ASD':
        return [tokenizer.encode(phrase, add_special_tokens=False) for phrase in [' Control', ' Autism']]
    elif target_name == 'ADHD':
        return [tokenizer.encode(phrase, add_special_tokens=False) for phrase in [' Control', ' ADHD']]
    elif target_name in ['fluidintel_enc', 'fluidcomp_enc', 'VIQ_enc', 'PIQ_enc', 'flanker_enc']:
        return [tokenizer.encode(phrase, add_special_tokens=False) for phrase in [' lower', ' mean', ' higher']]
    elif target_name in ['av45_enc', 'apoe4_enc']:
        return [tokenizer.encode(phrase, add_special_tokens=False) for phrase in [' negative', ' positive']]
    else:
        raise ValueError(f"Unknown target_name: {target_name}")


def gather_voxel_within_parcel(fmri_img, atlas_img, max_voxels):
    """Gather voxel time series within each parcel up to max_voxels."""
    fmri_data = fmri_img.get_fdata()
    atlas_data = atlas_img.get_fdata()
    
    labels = np.unique(atlas_data)
    labels = labels[labels != 0]  # Exclude background
    
    all_time_series = []
    
    for label in labels:
        mask = atlas_data == label
        voxel_time_series = fmri_data[mask]
        
        assert voxel_time_series.shape[0] <= max_voxels
        if voxel_time_series.shape[0] < max_voxels:
            # Pad with -1 if fewer than max_voxels
            padding = np.full((max_voxels - voxel_time_series.shape[0], voxel_time_series.shape[1]), -1)
            voxel_time_series = np.vstack([voxel_time_series, padding])
        
        all_time_series.append(voxel_time_series)

    return np.array(all_time_series)  # Shape: (n_parcels, max_voxels, T)

def reconstruct_fmri_from_parcel(processed_data, atlas_img):
    """Reconstruct original fMRI image from processed parcel data."""
    V, max_voxel, T = processed_data.shape
    atlas_data = atlas_img.get_fdata()
    reconstructed_volume = np.zeros(atlas_data.shape + (processed_data.shape[-1],))

    labels = np.unique(atlas_data)
    labels = labels[labels != 0]  # Exclude background
    voxel_mappings = {}
    for label_idx, label in enumerate(labels):
        mask = atlas_data == label
        coords = np.argwhere(mask)
        n_actual_voxels = np.sum(mask)
        
        voxel_mappings[label] = {
            'parcel_idx': label_idx,
            'coords': coords,
            'n_actual_voxels': n_actual_voxels
        }
    
    for label, mapping_info in voxel_mappings.items():
        parcel_idx = mapping_info['parcel_idx']
        coords = mapping_info['coords']
        n_actual_voxels = mapping_info['n_actual_voxels']
        
        # Get time series for this parcel (excluding padded values)
        parcel_data = processed_data[parcel_idx, :n_actual_voxels]
        
        # Place values back in original spatial locations
        for i, coord in enumerate(coords):
            reconstructed_volume[tuple(coord)] = parcel_data[i]

    return reconstructed_volume


def enlarge_causal_mat(mat, pos):
    # mat: (B, L, L), enlarge a causal attention matrix by inserting a row and column at position pos
    B, L, _ = mat.shape
    new_mat = torch.zeros((B, L + 1, L +1), device=mat.device)
    new_mat[:, :pos, :pos] = mat[:, :pos, :pos]  # top-left
    new_mat[:, :pos, pos+1:] = mat[:, :pos, pos:]  # top-right
    new_mat[:, pos+1:, :pos] = mat[:, pos:, :pos]  # bottom-left
    new_mat[:, pos+1:, pos+1:] = mat[:, pos:, pos:]  # bottom-right
    new_mat[:, pos, :pos + 1] = 1  # causal structure
    return new_mat


def combine_attn_mask(fmri_mask, text_mask):
    # Combine fmri_mask (B, L_fmri, L_fmri) and text_mask (B, L_text) or (B, L_text, L_text) into a single attention mask

    B, L_fmri, _ = fmri_mask.shape
    L_text = text_mask.shape[1]
    
    # Allocate the combined mask once
    new_mask = torch.zeros((B, L_fmri + L_text, L_fmri + L_text), device=fmri_mask.device)
    
    # Copy fmri mask
    new_mask[:, :L_fmri, :L_fmri] = fmri_mask
    
    # Create text-to-text causal mask with padding in-place
    # This is memory efficient: text_mask.unsqueeze(1) * text_mask.unsqueeze(2) creates (B, L_text, L_text)
    # then multiply with causal mask using tril in-place
    text_region = new_mask[:, L_fmri:, L_fmri:]
    if text_mask.dim() == 2:
        text_region.copy_(text_mask.unsqueeze(1) * text_mask.unsqueeze(2))
        text_region.tril_()  # Apply causal mask in-place
    else:
        text_region.copy_(text_mask)
    
    # Allow non-padded text tokens to attend to all fmri tokens
    if text_mask.dim() == 2:
        new_mask[:, L_fmri:, :L_fmri] = text_mask.unsqueeze(2)  # (B, L_text, 1) broadcasts to (B, L_text, L_fmri)
    else:
        # find non-padded text tokens
        text_nonpad = text_mask.sum(dim=1) > 0  # (B, L_text)
        new_mask[:, L_fmri:, :L_fmri] = text_nonpad.unsqueeze(2).float()  # (B, L_text, 1) broadcasts to (B, L_text, L_fmri)
    
    return new_mask

def select_few_shot_indices(dataset, train_inds, fewshot_samples):
    # Get labels for train_inds
    labels = []

    # TODO it may be possible that I failed to consider the missing fields when generating banlanced indices (since there are many targets)
    train_inds = [i for i in train_inds if i in dataset.inds]
    train_subjs, train_sess = [dataset.subjs[i] for i in train_inds], [dataset.sess[i] for i in train_inds]
    labels = [dataset.subject_mapping[(subj, sess)]['Y'] for subj, sess in zip(train_subjs, train_sess)]
    
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    samples_per_class = min(1, fewshot_samples // n_classes)
    selected_indices = []
    
    for ul in unique_labels:
        class_indices = np.where(labels == ul)[0]
        if len(class_indices) < samples_per_class:
            selected_class_indices = class_indices
        else:
            selected_class_indices = np.random.choice(class_indices, samples_per_class, replace=False)
        selected_indices.extend(selected_class_indices.tolist())
    
    return [train_inds[i] for i in selected_indices]
