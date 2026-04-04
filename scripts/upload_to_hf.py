#!/usr/bin/env python3
"""
Upload MaskDistill checkpoints to HuggingFace Hub.

Usage:
    # Login first
    huggingface-cli login

    # Upload all checkpoints
    python scripts/upload_to_hf.py --username YOUR_HF_USERNAME

    # Upload specific checkpoint
    python scripts/upload_to_hf.py --username YOUR_HF_USERNAME --checkpoint pretrain
"""

import argparse
import os
import torch
from huggingface_hub import HfApi, create_repo

CHECKPOINTS = {
    "pretrain": {
        "path": "output/pretrain/checkpoint-epoch0290.pth",
        "filename": "pretrain_vit_base_ep290.pth",
        "description": "MaskDistill ViT-Base/16 pretrained (300 epochs, block masking 40%)",
    },
    "semseg": {
        "path": "output/semseg/best_mIoU_iter_160000.pth",
        "filename": "semseg_upernet_ade20k_160k.pth",
        "description": "UPerNet semantic segmentation on ADE20K (160K iter, 52.6 mIoU)",
    },
    "detection": {
        "path": "output/detection/epoch_12.pth",
        "filename": "detection_maskrcnn_coco_12ep.pth",
        "description": "Mask R-CNN object detection on COCO (12 epochs, 44.4 bbox mAP)",
    },
    # These will be added after training completes:
    # "finetune": {
    #     "path": "...",
    #     "filename": "finetune_vit_base_ep100.pth",
    #     "description": "ViT-Base/16 finetuned on ImageNet-1K (100 epochs)",
    # },
    # "linprobe": {
    #     "path": "...",
    #     "filename": "linprobe_vit_base_ep90.pth",
    #     "description": "ViT-Base/16 linear probe on ImageNet-1K (90 epochs)",
    # },
}

MODEL_CARD = """---
license: apache-2.0
tags:
  - vision
  - self-supervised-learning
  - masked-image-modeling
  - knowledge-distillation
  - vit
datasets:
  - ILSVRC/imagenet-1k
  - 1aurent/ADE20K
  - detection-datasets/coco
metrics:
  - accuracy
  - mIoU
  - mAP
pipeline_tag: image-classification
---

# MaskDistill ViT-Base/16

**The first open-source PyTorch implementation of MaskDistill with pre-trained weights.**

This model was trained using the [MaskDistill-PyTorch](https://github.com/{username}/MaskDistill-PyTorch) codebase, reproducing the method from ["A Unified View of Masked Image Modeling"](https://arxiv.org/abs/2210.10615).

## Model Description

MaskDistill learns visual representations by distilling knowledge from a frozen CLIP ViT-B/16 teacher into a ViT-Base student through masked image modeling. The student learns to predict the teacher's features for masked patches using Smooth L1 loss.

- **Architecture**: ViT-Base/16 (86M params)
- **Teacher**: CLIP ViT-B/16 (frozen)
- **Pretraining**: 300 epochs on ImageNet-1K
- **Masking**: Block masking at 40%, dense encoding with shared relative position bias

## Results

| Evaluation | Result |
|-----------|--------|
| k-NN (k=10) | **75.6%** top-1 |
| Linear Probe | **76.3%** top-1 |
| Sem. Seg. (ADE20K, UPerNet) | **52.6** mIoU |
| Obj. Det. (COCO, Mask R-CNN) | **44.4** bbox mAP |
| Inst. Seg. (COCO, Mask R-CNN) | **40.1** segm mAP |

## Available Checkpoints

| File | Description |
|------|------------|
| `pretrain_vit_base_ep290.pth` | Pretrained ViT-Base (300 epochs) |
| `linprobe_vit_base_ep90.pth.tar` | Linear probe (90 epochs, 76.3% top-1) |
| `semseg_upernet_ade20k_160k.pth` | UPerNet on ADE20K (52.6 mIoU) |
| `detection_maskrcnn_coco_12ep.pth` | Mask R-CNN on COCO (44.4 mAP) |

## Usage

```python
import torch
from src.models.vision_transformer import VisionTransformerMIM

# Load pretrained model
model = VisionTransformerMIM(
    img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
    use_shared_rel_pos_bias=True, use_mask_tokens=True,
)
ckpt = torch.load("pretrain_vit_base_ep290.pth", map_location="cpu")
state = {{k.replace("module.student.", ""): v for k, v in ckpt["model"].items() if "student" in k}}
model.load_state_dict(state, strict=False)
```

See the [GitHub repo](https://github.com/{username}/MaskDistill-PyTorch) for full training and evaluation code.

## Citation

```bibtex
@article{{hou2022unified,
  title={{A Unified View of Masked Image Modeling}},
  author={{Hou, Zhenda and Sun, Fei and Chen, Yun-Hao and Yuan, Jia-Hong and Yu, Jia-Mu}},
  journal={{arXiv preprint arXiv:2210.10615}},
  year={{2022}}
}}
```
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", default="drkostas", help="HuggingFace username")
    parser.add_argument("--checkpoint", default=None, help="Specific checkpoint to upload (pretrain/semseg/detection/finetune/linprobe)")
    parser.add_argument("--repo-name", default="MaskDistill-ViT-Base", help="HF repo name")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded without uploading")
    args = parser.parse_args()

    repo_id = f"{args.username}/{args.repo_name}"
    api = HfApi()

    if not args.dry_run:
        # Create repo if it doesn't exist
        try:
            create_repo(repo_id, repo_type="model", exist_ok=True)
            print(f"Repository {repo_id} ready")
        except Exception as e:
            print(f"Warning: {e}")

        # Upload model card
        card = MODEL_CARD.format(username=args.username)
        api.upload_file(
            path_or_fileobj=card.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"Uploaded model card to {repo_id}")

    # Upload checkpoints
    to_upload = {args.checkpoint: CHECKPOINTS[args.checkpoint]} if args.checkpoint else CHECKPOINTS

    for name, info in to_upload.items():
        path = info["path"]
        if not os.path.exists(path):
            print(f"SKIP {name}: {path} not found")
            continue

        size_mb = os.path.getsize(path) / 1e6
        print(f"{'[DRY RUN] ' if args.dry_run else ''}Upload {name}: {path} ({size_mb:.0f} MB) → {info['filename']}")

        if not args.dry_run:
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=info["filename"],
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"  Uploaded {info['filename']} to {repo_id}")

    print(f"\nDone! View at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
