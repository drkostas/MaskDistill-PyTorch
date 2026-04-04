"""
MaskDistill Classification Demo
Upload an image -> top-5 ImageNet predictions.
Supports: Linear Probe (76.3%) and Finetuned (when available).
"""

import os
import sys
import json
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from models.vision_transformer import VisionTransformerMIM

# ImageNet labels
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
print("Loading ImageNet labels...")
labels = json.loads(urllib.request.urlopen(LABELS_URL).read().decode())

repo_id = "drkostas/MaskDistill-ViT-Base"

# ── Backbone (shared) ────────────────────────────────────────────────────────
print("Building backbone...")
backbone = VisionTransformerMIM(
    img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
    init_values=0.1, drop_path_rate=0.0,
    use_abs_pos_emb=False, use_shared_rel_pos_bias=True, use_mask_tokens=True,
)

print("Loading pretrained weights...")
pretrain_path = hf_hub_download(repo_id, "pretrain_vit_base_student_only.pth")
ckpt = torch.load(pretrain_path, map_location="cpu", weights_only=False)
state = ckpt["model"]  # Already stripped to student keys
backbone.load_state_dict(state, strict=False)
backbone.eval()
print(f"Backbone: {sum(p.numel() for p in backbone.parameters())/1e6:.0f}M params")

# ── Linear Probe Head ────────────────────────────────────────────────────────
BEST_LAYER = 11
print("Loading linear probe head...")
lp_path = hf_hub_download(repo_id, "linprobe_vit_base_ep90.pth.tar")
lp_ckpt = torch.load(lp_path, map_location="cpu", weights_only=False)
linear_head = nn.Linear(1536, 1000)
linear_head.weight.data = lp_ckpt["state_dict"][f"module.linear.{BEST_LAYER}.weight"]
linear_head.bias.data = lp_ckpt["state_dict"][f"module.linear.{BEST_LAYER}.bias"]
linear_head.eval()
print("Linear probe loaded (76.3% top-1)")

# ── Finetuned Head (loaded if available) ─────────────────────────────────────
# ── Finetuned Model (uses modeling_finetune.VisionTransformer architecture) ──
finetune_model = None
try:
    ft_path = hf_hub_download(repo_id, "finetune_vit_base_ep100.pth")
    print("Loading finetuned checkpoint...")

    # The finetuned model was trained with modeling_finetune.py's VisionTransformer
    # which has different attention (fused qkv.bias) and uses fc_norm for mean pooling.
    # We import it from the downstream code bundled in the Space.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "downstream"))
    from modeling_finetune import VisionTransformer as FineTuneViT

    finetune_model = FineTuneViT(
        img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4.0, qkv_bias=True, init_values=0.1, drop_path_rate=0.1,
        use_shared_rel_pos_bias=True, use_mean_pooling=True, num_classes=1000,
    )
    ft_ckpt = torch.load(ft_path, map_location="cpu", weights_only=False)
    ft_state = ft_ckpt.get("model", ft_ckpt)
    ft_clean = {k.replace("module.", ""): v for k, v in ft_state.items()}
    msg = finetune_model.load_state_dict(ft_clean, strict=False)
    finetune_model.eval()
    print(f"Finetuned model loaded (84.8% top-1), {len(msg.missing_keys)} missing, {len(msg.unexpected_keys)} unexpected")
    if msg.missing_keys:
        print(f"  Missing: {msg.missing_keys[:5]}")
    if msg.unexpected_keys:
        print(f"  Unexpected: {msg.unexpected_keys[:5]}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Finetuned checkpoint not available: {e}")
    finetune_model = None

# ── Available models ─────────────────────────────────────────────────────────
MODELS = ["Linear Probe (76.3% top-1)"]
if finetune_model is not None:
    MODELS.insert(0, "Finetuned (84.8% top-1)")

# ── Transform ────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def classify(image, model_choice):
    if image is None:
        return {}

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        if "Linear Probe" in model_choice:
            features = backbone.get_intermediate_layers(img, use_last_norm=True)
            feat = features[BEST_LAYER]
            cls_token = feat[:, 0]
            patch_avg = feat[:, 1:].mean(dim=1)
            combined = torch.cat([cls_token, patch_avg], dim=-1)
            logits = linear_head(combined)
        else:
            logits = finetune_model(img)

        probs = F.softmax(logits, dim=-1)[0]

    top5_probs, top5_idx = probs.topk(5)
    return {labels[i]: p.item() for p, i in zip(top5_probs, top5_idx)}


demo = gr.Interface(
    fn=classify,
    inputs=[
        gr.Image(type="pil", label="Upload an image"),
        gr.Dropdown(choices=MODELS, value=MODELS[0], label="Model"),
    ],
    outputs=gr.Label(num_top_classes=5, label="Top-5 Predictions"),
    title="MaskDistill Image Classification",
    description=(
        "Classify images using [MaskDistill](https://arxiv.org/abs/2210.10615) ViT-Base/16 "
        "pretrained on ImageNet-1K. "
        "Select a model from the dropdown. "
        "[Weights](https://huggingface.co/drkostas/MaskDistill-ViT-Base) | "
        "[GitHub](https://github.com/drkostas/MaskDistill-PyTorch)"
    ),
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
