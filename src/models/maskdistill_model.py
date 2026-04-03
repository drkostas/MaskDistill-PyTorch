"""
MaskDistill Model — unified wrapper for student ViT + distillation head.

Implements the MaskDistill framework:
    MIM = L(N(T(I_full)), H(S(I_masked)))

where:
    T = frozen CLIP teacher (external, not wrapped here)
    S = student ViT encoder (sparse mode, masked patches dropped)
    H = distillation head (linear projection to teacher dim)
    N = layer normalization on teacher features
    L = Smooth-L1 loss on masked token positions

Reference: "A Unified View of Masked Image Modeling" (arXiv:2210.10615)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

from .vision_transformer import VisionTransformerMIM


class MaskDistillModel(nn.Module):
    """
    MaskDistill model: student ViT encoder + linear distillation head.

    Supports both dense mode (BEiT-style, mask tokens replace masked patches)
    and sparse mode (MAE-style, masked patches dropped). The mode is
    controlled by use_mask_tokens in config. The distill_head projects
    student token features to match the CLIP teacher's embedding dimension.
    """

    def __init__(self, cfg: Dict[str, Any], mask_generator=None):
        """
        Args:
            cfg: Full configuration dictionary. Expected keys:
                 - model.student.*  (ViT architecture params)
                 - model.teacher.embed_dim  (target projection dim)
                 - losses.use_head_loss (default True)
            mask_generator: A BlockMaskingGenerator or RandomMaskingGenerator
                            instance used to produce masks when none are
                            passed to forward().
        """
        super().__init__()
        self.cfg = cfg
        self.student = build_student(cfg)
        self.mask_generator = mask_generator

        # Distillation head: project student embed_dim -> teacher embed_dim
        self.distill_head: Optional[nn.Module] = None
        losses_cfg = cfg.get("losses", {})
        if losses_cfg.get("use_head_loss", True):
            in_dim = cfg["model"]["student"]["embed_dim"]
            out_dim = cfg.get("model", {}).get("teacher", {}).get("embed_dim", in_dim)
            self.distill_head = nn.Linear(in_dim, out_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        img: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_generator=None,
        teacher=None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass: generate mask -> encode visible tokens -> project.

        Args:
            img:            Input images [B, 3, H, W].
            mask:           Pre-generated boolean mask [B, N] (True = masked).
                            If None, self.mask_generator is used.
            mask_generator: Legacy override; ignored if mask is provided.
            teacher:        Unused here; accepted for call-site compatibility.

        Returns:
            pred_tok:    Distillation head output [B, N_vis+1, D_teacher]
                         (includes CLS token at position 0), or None if
                         distill_head is disabled.
            mask_out:    The boolean mask that was applied [B, N].
            ids_restore: Indices to unshuffle / restore full patch order
                         [B, N], needed for loss computation.
        """
        # ----- Mask generation ----------------------------------------
        if mask is None:
            mask = self._generate_mask(img)

        # ----- Student forward (sparse) -------------------------------
        # VisionTransformerMIM in sparse mode returns (x, ids_restore)
        # x shape: [B, N_vis + 1, embed_dim]  (visible tokens + CLS)
        z, ids_restore = self.student(img, mask, mask_generator=mask_generator)

        # ----- Distillation head --------------------------------------
        pred_tok = None
        if self.distill_head is not None:
            pred_tok = self.distill_head(z)  # [B, N_vis+1, D_teacher]

        return pred_tok, mask, ids_restore

    # ------------------------------------------------------------------
    # Mask generation helpers
    # ------------------------------------------------------------------

    def _generate_mask(self, img: torch.Tensor) -> torch.Tensor:
        """
        Generate a batch of masks using self.mask_generator.

        Supports both block masking (BEiT-style) and random masking.
        For sparse mode with shuffle_patches enabled, consistent
        shuffling is used (all samples share the same shuffle order
        for efficiency).

        Args:
            img: Input images, used only to determine batch size and device.

        Returns:
            mask: Boolean tensor [B, N] where True = masked patch.

        Raises:
            ValueError: If no mask and no mask_generator are available.
        """
        if self.mask_generator is None:
            raise ValueError("Either mask or mask_generator must be provided")

        B = img.shape[0]
        has_shuffle = (
            hasattr(self.mask_generator, "shuffle_patches")
            and self.mask_generator.shuffle_patches
        )

        if has_shuffle:
            # Consistent shuffling: generate one mask to set shuffle/restore
            # indices, then generate the rest of the batch.
            first_mask = self.mask_generator()
            stored_shuffle = self.mask_generator.ids_shuffle
            stored_restore = self.mask_generator.ids_restore

            masks = [first_mask]
            for _ in range(1, B):
                masks.append(self.mask_generator())

            # Restore the first sample's shuffle indices so the whole batch
            # uses the same ordering (more efficient for sparse encoding).
            self.mask_generator.ids_shuffle = stored_shuffle
            self.mask_generator.ids_restore = stored_restore
        else:
            masks = [self.mask_generator() for _ in range(B)]

        return torch.stack(masks, dim=0).to(img.device)

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def no_weight_decay(self) -> set:
        """Parameters that should be excluded from weight decay."""
        return {"student." + k for k in self.student.no_weight_decay()}


# ======================================================================
# Factory functions
# ======================================================================


def build_student(cfg: Dict[str, Any]) -> VisionTransformerMIM:
    """
    Build the student ViT from configuration.

    Supports both dense mode (BEiT-style, mask tokens replace masked patches)
    and sparse mode (MAE-style, masked patches dropped).
    """
    s = cfg["model"]["student"]
    use_mask_tokens = s.get("use_mask_tokens", True)

    return VisionTransformerMIM(
        img_size=s["img_size"],
        patch_size=s["patch_size"],
        embed_dim=s["embed_dim"],
        depth=s["depth"],
        num_heads=s["num_heads"],
        mlp_ratio=s.get("mlp_ratio", 4.0),
        drop_path_rate=s.get("drop_path_rate", 0.1),
        init_values=s.get("init_values", 0.1),
        use_abs_pos_emb=s.get("use_abs_pos_emb", False),
        use_sincos_pos_emb=s.get("use_sincos_pos_emb", False),
        use_shared_rel_pos_bias=s.get("use_shared_rel_pos_bias", True),
        use_rel_pos_bias=s.get("use_rel_pos_bias", False),
        use_mask_tokens=use_mask_tokens,
    )


def build_maskdistill_model(
    cfg: Dict[str, Any],
    mask_generator=None,
) -> MaskDistillModel:
    """
    Build a MaskDistillModel from a configuration dictionary.

    Args:
        cfg:            Full config dict with model.student, model.teacher, losses.
        mask_generator: Optional block or random masking generator.

    Returns:
        Configured MaskDistillModel instance.
    """
    return MaskDistillModel(cfg, mask_generator=mask_generator)
