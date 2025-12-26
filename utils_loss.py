import torch
import torch.nn as nn
import torch.nn.functional as F

def clip_loss(image_embeds: torch.Tensor,
              text_embeds: torch.Tensor,
              logit_scale: torch.Tensor | float = None) -> tuple[torch.Tensor, dict]:
    """
    Compute the symmetric CLIP contrastive loss (InfoNCE-style).

    Args:
        image_embeds: (B, D) image features.
        text_embeds:  (B, D) text  features. Must have same batch size.
        logit_scale:  either a scalar (float) temperature inverse, or a 1-element
                      learnable tensor storing log(logit_scale). If None, uses the
                      CLIP default of ~1/0.07.

    Returns:
        loss: scalar tensor (averaged image->text and text->image CE).
        extras: dict with logits and accuracies for monitoring.

    Notes:
        - Features are L2-normalized before similarity.
        - CLIP trains a learnable logit_scale (initialized to log(1/0.07)),
          often clamped to avoid numerical issues.
    """
    assert image_embeds.shape == text_embeds.shape
    B, D = image_embeds.shape

    # Normalize to get cosine similarities
    img = F.normalize(image_embeds, dim=-1)
    txt = F.normalize(text_embeds, dim=-1)

    # Temperature / logit scale
    if logit_scale is None:
        # fixed scale ~ 1/0.07
        logit_scale_val = 1.0 / 0.07
    elif isinstance(logit_scale, torch.Tensor) and logit_scale.numel() == 1:
        # logit_scale is stored as log() in many implementations
        logit_scale_val = torch.clamp(logit_scale.exp(), max=100.0).item()
    else:
        # plain float / tensor scalar passed directly
        logit_scale_val = float(logit_scale)

    # Similarity logits
    # (B, D) @ (D, B) -> (B, B)
    logits_per_image = logit_scale_val * img @ txt.t()
    logits_per_text  = logits_per_image.t()

    # Ground-truth “matching” is diagonal (i-th image matches i-th text)
    targets = torch.arange(B, device=logits_per_image.device)

    # Symmetric cross-entropy
    loss_i2t = F.cross_entropy(logits_per_image, targets)
    loss_t2i = F.cross_entropy(logits_per_text,  targets)
    loss = (loss_i2t + loss_t2i) / 2
    return loss


def soft_clip_loss(preds, targs, temp=0.125, alpha_soft=1.0, alpha_hard=0.0):
    """
    preds : (B, D) predicted embeddings (e.g., brain->text side)
    targs : (B, D) target text embeddings (teacher side)
    temp  : temperature (like CLIP's 1/τ; here lower = sharper)
    alpha_soft : weight for soft-label loss
    alpha_hard : weight for standard hard CLIP (one-hot) loss
    """
    # L2-normalize (cosine sims)
    p = F.normalize(preds, dim=-1)
    t = F.normalize(targs, dim=-1)

    # Student cross-modal logits (image/brain -> text) and symmetric (text -> preds)
    S_stu = (p @ t.T) / temp          # (B, B)
    S_stu_T = S_stu.T                 # (B, B)

    # Teacher intra-modal logits (soft targets from text-text self-similarity)
    with torch.no_grad():
        S_tea = (t @ t.T) / temp      # (B, B)
        P = S_tea.softmax(dim=-1)     # teacher probs row-wise
        P_T = P.T                     # symmetric teacher (same if S_tea is symmetric)

    # Soft loss = symmetric KL (batchmean)
    loss_soft = (
        F.kl_div(F.log_softmax(S_stu, dim=-1),   P,  reduction="batchmean")
      + F.kl_div(F.log_softmax(S_stu_T, dim=-1), P_T, reduction="batchmean")
    ) / 2

    loss = alpha_soft * loss_soft

    # Optional: blend in standard hard CLIP (one-hot) for stability
    if alpha_hard > 0:
        targets = torch.arange(S_stu.size(0), device=S_stu.device)
        loss_hard = (F.cross_entropy(S_stu, targets) + F.cross_entropy(S_stu_T, targets)) / 2
        loss = loss + alpha_hard * loss_hard

    return loss


def siglip_loss(
    image_embeds: torch.Tensor,
    text_embeds:  torch.Tensor,
    logit_scale: torch.Tensor | float = 1.0 / 0.07,
    pos_weight: float | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    Sigmoid loss for language-image pretraining (SigLIP).

    Args:
        image_embeds: (B, D) image features
        text_embeds:  (B, D) text  features (paired with images by index)
        temp:         temperature τ (lower -> sharper); logits = (cosine / τ)
        pos_weight:   optional positive-class weight for BCE balancing.
                      If None, we use (B-1), which balances 1 positive vs B-1 negatives per row/col.

    Returns:
        loss: scalar tensor
        extras: metrics for monitoring
    """
    assert image_embeds.shape == text_embeds.shape
    B, D = image_embeds.shape

    # Normalize for cosine similarity (CLIP/SigLIP convention)
    img = F.normalize(image_embeds, dim=-1)
    txt = F.normalize(text_embeds,  dim=-1)

    # Student logits (all pairs)
    logits = (img @ txt.t()) * logit_scale  # (B, B)

    # Hard binary labels: 1 on diagonal, 0 elsewhere
    target = torch.eye(B, device=logits.device)

    # Balance positives vs negatives (1 vs B-1 per row/col)
    if pos_weight is None:
        pos_weight = float(B - 1)
    pos_weight_t = torch.tensor(pos_weight, device=logits.device)

    # BCE on image->text and text->image (symmetric)
    # Using 'mean' reduction over all entries
    loss_i2t = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight_t, reduction="mean"
    )
    loss_t2i = F.binary_cross_entropy_with_logits(
        logits.t(), target, pos_weight=pos_weight_t, reduction="mean"
    )

    loss = (loss_i2t + loss_t2i) / 2
    return loss


def soft_siglip_loss(
    preds: torch.Tensor,           # (B, D) student embeddings (e.g., images or brain)
    targs: torch.Tensor,           # (B, D) teacher modality (e.g., texts)
    temp_student: float = 0.07,    # τ_s for student logits
    temp_teacher: float = 0.07,    # τ_t for teacher similarity -> prob
    alpha_soft: float = 1.0,       # weight for soft distillation term
    alpha_hard: float = 0.1,       # optionally mix in vanilla SigLIP
    clamp_diag_to_one: bool = True, # ensure P_ii≈1 for stronger positive signal
    **kwargs,
) -> torch.Tensor:
    """
    Soft-SigLIP: BCE-with-logits against *soft* targets from intra-modal teacher sims.

    preds : (B, D) student embeddings (e.g., image/brain)
    targs : (B, D) teacher embeddings (e.g., text)
    """
    assert preds.shape == targs.shape
    B, D = preds.shape

    # Normalize
    p = F.normalize(preds, dim=-1)
    t = F.normalize(targs, dim=-1)

    # Student cross-modal logits (both directions)
    S = (p @ t.t()) / temp_student      # (B, B)
    S_T = S.t()

    # Teacher intra-modal similarities -> soft probabilities via sigmoid
    with torch.no_grad():
        T_sim = (t @ t.t()) / temp_teacher  # (B, B)
        P = torch.sigmoid(T_sim)            # in [0,1], element-wise
        if clamp_diag_to_one:
            # Nudge diagonal to 1; off-diagonals remain soft
            P.fill_diagonal_(1.0)
        P_T = P.t()

    # Balanced BCE with soft labels
    # (Optionally balance positives vs negatives per row/col)
    pos_weight = torch.tensor(float(B - 1), device=S.device)

    loss_soft = 0.5 * (
        F.binary_cross_entropy_with_logits(S,   P,   pos_weight=pos_weight, reduction="mean") +
        F.binary_cross_entropy_with_logits(S_T, P_T, pos_weight=pos_weight, reduction="mean")
    )

    loss = alpha_soft * loss_soft

    # Optional hard mix-in (stability / calibration)
    if alpha_hard > 0:
        I = torch.eye(B, device=S.device)
        loss_hard = 0.5 * (
            F.binary_cross_entropy_with_logits(S,   I, pos_weight=pos_weight, reduction="mean") +
            F.binary_cross_entropy_with_logits(S_T, I, pos_weight=pos_weight, reduction="mean")
        )
        loss = loss + alpha_hard * loss_hard

    return loss