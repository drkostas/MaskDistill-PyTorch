"""
MaskDistill Classification Demo
Upload an image -> top-5 ImageNet predictions using linear probe (76.3% top-1).
"""

import os
import sys
import json
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download

# Add src to path for our ViT implementation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from models.vision_transformer import VisionTransformerMIM

# ImageNet labels
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"

print("Loading ImageNet labels...")
labels = json.loads(urllib.request.urlopen(LABELS_URL).read().decode())

print("Downloading checkpoints...")
repo_id = "drkostas/MaskDistill-ViT-Base"
pretrain_path = hf_hub_download(repo_id, "pretrain_vit_base_ep290.pth")
linprobe_path = hf_hub_download(repo_id, "linprobe_vit_base_ep90.pth.tar")

print("Building model...")
# Build backbone (matches pretrain checkpoint)
backbone = VisionTransformerMIM(
    img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
    init_values=0.1, drop_path_rate=0.0,
    use_abs_pos_emb=False, use_shared_rel_pos_bias=True,
    use_mask_tokens=True,
)

# Load pretrained backbone weights
ckpt = torch.load(pretrain_path, map_location="cpu", weights_only=False)
state = {k.replace("module.student.", ""): v for k, v in ckpt["model"].items()
         if k.startswith("module.student.")}
backbone.load_state_dict(state, strict=False)
backbone.eval()
print(f"Backbone loaded ({sum(p.numel() for p in backbone.parameters())/1e6:.0f}M params)")

# Load linear probe head (layer 9 is best at 76.4%, layer 11 at 76.3%)
# Use layer 11 (last layer, standard choice)
BEST_LAYER = 11
linprobe_ckpt = torch.load(linprobe_path, map_location="cpu", weights_only=False)
linear_weight = linprobe_ckpt["state_dict"][f"module.linear.{BEST_LAYER}.weight"]  # [1000, 1536]
linear_bias = linprobe_ckpt["state_dict"][f"module.linear.{BEST_LAYER}.bias"]  # [1000]

linear_head = nn.Linear(1536, 1000, bias=True)
linear_head.weight.data = linear_weight
linear_head.bias.data = linear_bias
linear_head.eval()
print(f"Linear head loaded (layer {BEST_LAYER}, {linear_weight.shape})")

transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def classify(image):
    if image is None:
        return {}

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        # Get intermediate features from specific layer
        features = backbone.get_intermediate_layers(img, use_last_norm=False)
        feat = features[BEST_LAYER]  # [1, N+1, 768]

        # BEiT2 linear probe protocol: concat CLS + avg pooled patches
        cls_token = feat[:, 0]  # [1, 768]
        patch_avg = feat[:, 1:].mean(dim=1)  # [1, 768]
        combined = torch.cat([cls_token, patch_avg], dim=-1)  # [1, 1536]

        logits = linear_head(combined)  # [1, 1000]
        probs = F.softmax(logits, dim=-1)[0]

    top5_probs, top5_idx = probs.topk(5)
    return {labels[i]: p.item() for p, i in zip(top5_probs, top5_idx)}


print("Ready!")

demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Label(num_top_classes=5, label="Top-5 Predictions"),
    title="MaskDistill Image Classification (76.3% top-1)",
    description=(
        "Classify images using a ViT-Base/16 pretrained with "
        "[MaskDistill](https://arxiv.org/abs/2210.10615) + linear probe on ImageNet-1K. "
        "Achieves **76.3% top-1** accuracy. "
        "[Weights](https://huggingface.co/drkostas/MaskDistill-ViT-Base) | "
        "[GitHub](https://github.com/drkostas/MaskDistill-PyTorch)"
    ),
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
