__all__ = ["build_teacher", "build_student"]

from typing import Dict, Any

from .vision_transformer import VisionTransformerMIM


def build_student(cfg: Dict[str, Any]) -> VisionTransformerMIM:
    """Builds the student ViT model from config."""
    student_cfg = cfg["model"]["student"]

    use_mask_tokens = student_cfg.get("use_mask_tokens", True)

    # Sparse mode requires absolute position embeddings (variable sequence length)
    if not use_mask_tokens:
        default_abs_pos = True
        default_sincos_pos = False
        default_rel_pos = False
        default_shared_rel = False
    else:
        # Dense mode can use relative position embeddings
        default_abs_pos = False
        default_sincos_pos = False
        default_rel_pos = False
        default_shared_rel = True

    model = VisionTransformerMIM(
        img_size=student_cfg["img_size"],
        patch_size=student_cfg["patch_size"],
        embed_dim=student_cfg["embed_dim"],
        depth=student_cfg["depth"],
        num_heads=student_cfg["num_heads"],
        mlp_ratio=student_cfg.get("mlp_ratio", 4.0),
        drop_path_rate=student_cfg.get("drop_path_rate", 0.1),
        init_values=student_cfg.get("init_values", 0.1),
        use_abs_pos_emb=student_cfg.get("use_abs_pos_emb", default_abs_pos),
        use_sincos_pos_emb=student_cfg.get("use_sincos_pos_emb", default_sincos_pos),
        use_shared_rel_pos_bias=student_cfg.get("use_shared_rel_pos_bias", default_shared_rel),
        use_rel_pos_bias=student_cfg.get("use_rel_pos_bias", default_rel_pos),
        use_mask_tokens=use_mask_tokens,
    )
    return model


def build_teacher(cfg: Dict[str, Any]):
    """Builds the frozen CLIP teacher model. Import is lazy to avoid slow CLIP loading at import time."""
    from .clip_teacher import ClipTeacher
    teacher_cfg = cfg["model"]["teacher"]
    return ClipTeacher(
        model_name=teacher_cfg["name"],
        layer_extraction=teacher_cfg.get("layer_extraction", "last"),
        num_layers_to_extract=teacher_cfg.get("num_layers_to_extract", 1),
    )
