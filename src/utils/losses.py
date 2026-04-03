"""
Loss functions for MaskDistill token distillation.

Implements the core MaskDistill loss: L = (1/|M|) * sum_{i in M} SmoothL1(pred_i, LN(teacher_i))

Supports two encoder modes:
  - Dense mode (use_mask_tokens=True): Student encodes all patches using learnable
    mask tokens for masked positions. Loss is computed only on masked positions
    (where mask==True).
  - Sparse mode (use_mask_tokens=False): Student encodes only visible (unmasked)
    patches. The student predictions and teacher features are already aligned
    (both contain only visible patches), so loss is a simple mean over all outputs.

Reference: "A Unified View of Masked Image Modeling" (arXiv:2210.10615)
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, Tuple


def apply_normalization(tensor: torch.Tensor, method: str = "variance") -> torch.Tensor:
    """Apply normalization to teacher features (or optionally to student predictions).

    The default "variance" method applies LayerNorm without learnable affine parameters:
        output = (x - mean) / sqrt(var + eps)

    This is the standard normalization used in MaskDistill and related MIM frameworks
    for normalizing CLIP teacher features before computing the distillation loss.

    Args:
        tensor: Input tensor of any shape. Normalization is applied along the last
                dimension (feature dimension). Typically [B, N, D].
        method: Normalization method to apply:
            - "variance": LayerNorm without affine (default). Normalizes each feature
              vector to zero mean and unit variance.
            - "l2": L2 normalization. Projects each feature vector onto the unit sphere.
            - "none": No normalization (pass-through).

    Returns:
        Normalized tensor with same shape as input.
    """
    if method == "variance":
        mean = tensor.mean(dim=-1, keepdim=True)
        var = tensor.var(dim=-1, keepdim=True)
        # Handle degenerate case where variance is NaN (e.g., single element)
        var = torch.where(torch.isnan(var), torch.ones_like(var), var)
        return (tensor - mean) / (var + 1.0e-6) ** 0.5
    elif method == "l2":
        return F.normalize(tensor, p=2, dim=-1)
    elif method == "none":
        return tensor
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def masked_smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute Smooth L1 (Huber) loss on masked patch positions only.

    Computes element-wise Smooth L1 loss between ``pred`` and ``target``, then
    averages only over positions where ``mask == True``. Positions where
    ``mask == False`` contribute zero to the loss.

    Args:
        pred: Student predictions [B, N, D].
        target: Teacher targets [B, N, D] (should already be normalized).
        mask: Boolean or binary mask [B, N]. True (1) indicates positions where
              loss should be computed (i.e., masked patches in dense mode).
        beta: Threshold for the transition between L1 and L2 regions in Smooth L1.
              Default 2.0 (MaskDistill paper uses 1.0 or 2.0).
        eps: Small constant to avoid division by zero when no positions are masked.

    Returns:
        Scalar loss averaged over all masked elements across the batch.
    """
    assert mask.shape == pred.shape[:2], (
        f"Mask shape {mask.shape} must match pred spatial dims {pred.shape[:2]}"
    )
    assert pred.shape == target.shape, (
        f"Pred shape {pred.shape} must match target shape {target.shape}"
    )

    # Handle empty tensors gracefully
    if pred.numel() == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    # Expand mask to feature dimension: [B, N] -> [B, N, D]
    mask_expanded = mask.unsqueeze(-1).expand_as(pred)

    with torch.autocast(
        device_type="cuda" if pred.device.type == "cuda" else "cpu", enabled=True
    ):
        loss = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")

        masked_loss = loss * mask_expanded
        result = masked_loss.sum() / (mask_expanded.sum() + eps)

        # Fallback to float32 if mixed precision produces non-finite result
        if not torch.isfinite(result):
            result = masked_loss.float().sum() / (mask_expanded.float().sum() + eps)

        return result


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute MSE (L2) loss on masked patch positions only.

    Same masking semantics as :func:`masked_smooth_l1_loss`, but uses squared
    error. Averages per-patch (mean over feature dim first, then over masked
    patches).

    Args:
        pred: Student predictions [B, N, D].
        target: Teacher targets [B, N, D].
        mask: Boolean or binary mask [B, N]. True (1) = compute loss here.
        eps: Small constant to avoid division by zero.

    Returns:
        Scalar loss averaged over masked patches.
    """
    assert mask.shape == pred.shape[:2], (
        f"Mask shape {mask.shape} must match pred spatial dims {pred.shape[:2]}"
    )
    assert pred.shape == target.shape, (
        f"Pred shape {pred.shape} must match target shape {target.shape}"
    )

    # Expand mask: [B, N] -> [B, N, D]
    mask_expanded = mask.unsqueeze(-1).expand_as(pred)

    with torch.autocast(
        device_type="cuda" if pred.device.type == "cuda" else "cpu", enabled=True
    ):
        loss = (pred - target) ** 2

        # Per-patch mean over features first, then average over masked patches
        loss = loss.mean(dim=-1)  # [B, N]
        mask_2d = mask_expanded[:, :, 0]  # [B, N]
        result = (loss * mask_2d).sum() / (mask_2d.sum() + eps)

        if not torch.isfinite(result):
            result = (loss.float() * mask_2d.float()).sum() / (
                mask_2d.float().sum() + eps
            )

        return result


def compute_loss(
    pred_tok: torch.Tensor,
    teacher_features: torch.Tensor,
    mask: torch.Tensor,
    cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the MaskDistill token distillation loss.

    This is the core loss of MaskDistill:
        L = (1/|M|) * sum_{i in M} Loss(H(S(I_masked))_i, N(T(I_full))_i)

    where H is the distillation head, S is the student encoder, T is the CLIP
    teacher, N is target normalization, and M is the set of masked positions.

    Dense vs. Sparse mode:
        In **dense mode** (``use_mask_tokens=True``), the student processes all
        patches (replacing masked ones with learnable mask tokens). Both
        ``pred_tok`` and ``teacher_features`` have shape [B, N+1, D] (N patches
        + CLS token). The loss is computed only on masked positions using the
        ``mask`` tensor.

        In **sparse mode** (``use_mask_tokens=False``), the student only encodes
        visible (unmasked) patches. Both ``pred_tok`` and ``teacher_features``
        arrive pre-aligned with shape [B, num_visible+1, D]. The mask is not
        needed for indexing since all student outputs correspond to positions
        that should contribute to the loss. The loss is a simple mean.

    Args:
        pred_tok: Student predictions after distillation head [B, N+1, D] in
                  dense mode or [B, num_visible+1, D] in sparse mode. Includes
                  CLS token at position 0.
        teacher_features: CLIP teacher features [B, N+1, D] in dense mode or
                          [B, num_visible+1, D] in sparse mode (already aligned
                          with student in sparse mode). Includes CLS at position 0.
        mask: Boolean mask [B, N] where True indicates masked patches. In dense
              mode this selects which patches to compute loss on. In sparse mode
              this is only used to determine the encoding mode (via config).
        cfg: Configuration dictionary. Relevant keys:

            - ``model.student.use_mask_tokens`` (bool): Dense (True) or sparse
              (False) encoder mode. Default True.
            - ``losses.normalize_targets`` (bool): Whether to normalize teacher
              features. Default True.
            - ``losses.normalize_predictions`` (bool): Whether to normalize
              student predictions. Default False.
            - ``losses.normalization_method`` (str): Normalization method for
              targets/predictions. Default "variance".
            - ``losses.head.type`` (str): Loss function type. One of "smooth_l1"
              (default), "l1", or "l2".
            - ``losses.head.beta`` (float): Beta for Smooth L1. Default 1.0.
            - ``losses.head_loss_weight`` (float): Weight for head loss. Default 1.0.

    Returns:
        loss: Scalar loss value (weighted).
        loss_dict: Dictionary with loss components for logging:
            - ``"L_head"``: The raw distillation loss value (before weighting).
            - ``"total_loss"``: The final weighted loss.
    """
    losses_cfg = cfg.get("losses", {})
    model_cfg = cfg.get("model", {}).get("student", {})

    # Determine encoder mode
    use_mask_tokens = model_cfg.get("use_mask_tokens", True)

    # Normalization settings
    norm_method = losses_cfg.get("normalization_method", "variance")

    # Normalize teacher features if enabled
    if losses_cfg.get("normalize_targets", True):
        teacher_features = apply_normalization(teacher_features, norm_method)

    # Normalize student predictions if enabled (typically False)
    if losses_cfg.get("normalize_predictions", False):
        pred_tok = apply_normalization(pred_tok, norm_method)

    # Remove CLS token (position 0) -- loss is computed on patch tokens only
    pred_patches = pred_tok[:, 1:, :]  # [B, N, D] or [B, num_visible, D]
    target_patches = teacher_features[:, 1:, :]

    # Get loss type configuration
    head_cfg = losses_cfg.get("head", {})
    loss_type = head_cfg.get("type", "smooth_l1")
    beta = head_cfg.get("beta", 1.0)

    # Compute loss based on encoder mode
    if use_mask_tokens:
        # Dense mode: compute loss on masked positions only (mask==True)
        if loss_type == "smooth_l1":
            L_head = masked_smooth_l1_loss(pred_patches, target_patches, mask, beta=beta)
        elif loss_type == "l1":
            L_head = _masked_l1_loss(pred_patches, target_patches, mask)
        elif loss_type == "l2":
            L_head = masked_mse_loss(pred_patches, target_patches, mask)
        else:
            raise ValueError(
                f"Unknown head loss type: {loss_type}. Must be 'smooth_l1', 'l1', or 'l2'."
            )
    else:
        # Sparse mode: student only encoded visible patches, predictions and
        # teacher features are already aligned. Compute loss over all outputs.
        if loss_type == "smooth_l1":
            L_head = F.smooth_l1_loss(pred_patches, target_patches, beta=beta, reduction="mean")
        elif loss_type == "l1":
            L_head = F.l1_loss(pred_patches, target_patches, reduction="mean")
        elif loss_type == "l2":
            L_head = F.mse_loss(pred_patches, target_patches, reduction="mean")
        else:
            raise ValueError(
                f"Unknown head loss type: {loss_type}. Must be 'smooth_l1', 'l1', or 'l2'."
            )

    # Apply loss weight
    head_weight = losses_cfg.get("head_loss_weight", 1.0)
    total_loss = head_weight * L_head

    loss_dict = {
        "L_head": L_head.item(),
        "total_loss": total_loss.item(),
    }

    return total_loss, loss_dict


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _masked_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute L1 (MAE) loss on masked positions only.

    Same contract as :func:`masked_smooth_l1_loss` but with absolute error.
    """
    assert mask.shape == pred.shape[:2]
    assert pred.shape == target.shape

    mask_expanded = mask.unsqueeze(-1).expand_as(pred)

    with torch.autocast(
        device_type="cuda" if pred.device.type == "cuda" else "cpu", enabled=True
    ):
        loss = torch.abs(pred - target)
        masked_loss = loss * mask_expanded
        result = masked_loss.sum() / (mask_expanded.sum() + eps)

        if not torch.isfinite(result):
            result = masked_loss.float().sum() / (mask_expanded.float().sum() + eps)

        return result
