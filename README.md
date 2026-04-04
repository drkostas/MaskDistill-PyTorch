# MaskDistill-PyTorch

**Unofficial PyTorch implementation and reproduction of ["A Unified View of Masked Image Modeling"](https://arxiv.org/abs/2210.10615) (MaskDistill, arXiv 2022).**

No official code or pre-trained weights were ever released for this paper. This repository provides a clean from-scratch implementation with **verified reproduced results** and **open pre-trained checkpoints** for all downstream tasks.

**[Pre-trained Weights on HuggingFace](https://huggingface.co/drkostas/MaskDistill-ViT-Base)**

## Key Results (ViT-B/16, ImageNet-1K)

| Evaluation | Paper | Ours |
|------------|:-----:|:----:|
| Finetuning (top-1) | 85.3% | **84.8%** |
| Sem. Seg. (mIoU, ADE20K) | 53.8 | **52.6** |
| k-NN (k=10) | — | **75.6%** |
| Linear Probe | — | **76.3%** |
| Obj. Det. (bbox mAP, COCO) | — | **44.4** |
| Inst. Seg. (segm mAP, COCO) | — | **40.1** |

The paper reports finetuning and semantic segmentation for ViT-Base. We additionally evaluate k-NN, linear probe, object detection, and instance segmentation.

## Overview

MaskDistill combines masked image modeling with knowledge distillation from CLIP:

1. **Mask** 40% of the input image patches (block masking)
2. **Replace** masked patches with learnable mask tokens (dense encoding, BEiT-style)
3. **Encode** all patches through a ViT-Base student with shared relative position bias
4. **Project** student features through a distillation head
5. **Distill** by minimizing smooth L1 loss between student predictions and frozen CLIP ViT-B/16 teacher features on masked positions

```
                     ┌──────────────┐
  Full Image ──────> │ CLIP Teacher │──── Teacher Features (frozen)
                     │  (ViT-B/16)  │           │
                     └──────────────┘           │
                                          Smooth L1 Loss
                                         (masked positions)
                     ┌──────────────┐           │
  Masked Image ───-> │   Student    │──── Student Predictions
   (mask tokens)     │  (ViT-Base)  │     (via distill head)
                     └──────────────┘
```

An alternative sparse encoding mode (MAE-style, drop masked patches) is also supported via `pretrain_random75.yaml`.

## Installation

```bash
git clone https://github.com/drkostas/MaskDistill-PyTorch.git
cd MaskDistill-PyTorch

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Requirements:** Python 3.10–3.12, PyTorch 2.1+, CUDA 11.8+.

For downstream evaluation (semantic segmentation and object detection), also install:
```bash
# Requires Python ≤3.12 (mmcv-full doesn't support 3.13+)
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
pip install mmsegmentation==0.30.0 mmdetection==2.28.2
```

## Setup

### 1. Dataset Paths

Download [ImageNet-1K](https://image-net.org/) and organize as:
```
/path/to/imagenet/
├── train/
│   ├── n01440764/
│   └── ...
└── val/
    ├── n01440764/
    └── ...
```

Update data paths in config files:
- **Pretrain configs** (`configs/pretrain*.yaml`): Set `data.data_path`, `data.train_dir`, `data.val_dir`
- **Semseg config** (`src/downstream/segmentation/configs/maskdistill/upernet_maskdistill_test_ade20k.py`): Set `data_root` to your [ADE20K](http://sceneparsing.csail.mit.edu/) path
- **Detection config** (`src/downstream/detection/configs/_base_/datasets/coco_instance.py`): Set `data_root` to your [COCO](https://cocodataset.org/) path

### 2. SLURM Configuration (for cluster users)

All scripts in `scripts/` have placeholders you must configure for your cluster:
```bash
#SBATCH -A YOUR_ACCOUNT       # Your SLURM account
#SBATCH --qos=YOUR_QOS        # Your QoS (e.g., normal, high)
#SBATCH --partition=YOUR_PARTITION  # Your partition (e.g., gpu, a100)
```

Also uncomment and adjust the module loads:
```bash
# module load cuda   # Uncomment and set your CUDA module
# module load cudnn  # Uncomment and set your cuDNN module
```

### 3. Weights & Biases (optional)

W&B logging is enabled by default. Set your entity in `configs/pretrain.yaml`:
```yaml
wandb_meta:
  entity: your-wandb-entity  # or null to use default
```

To disable W&B: `WANDB_MODE=disabled python -m src.train ...`

## Pre-trained Weights

All checkpoints are available on [HuggingFace](https://huggingface.co/drkostas/MaskDistill-ViT-Base):

| Checkpoint | Result | Download |
|-----------|--------|----------|
| ViT-B/16 pretrain (300 ep) | 75.6% k-NN | [pretrain_vit_base_ep290.pth](https://huggingface.co/drkostas/MaskDistill-ViT-Base/resolve/main/pretrain_vit_base_ep290.pth) |
| ViT-B/16 finetuned (100 ep) | 84.8% top-1 | [finetune_vit_base_ep100.pth](https://huggingface.co/drkostas/MaskDistill-ViT-Base/resolve/main/finetune_vit_base_ep100.pth) |
| ViT-B/16 linear probe (90 ep) | 76.3% top-1 | [linprobe_vit_base_ep90.pth.tar](https://huggingface.co/drkostas/MaskDistill-ViT-Base/resolve/main/linprobe_vit_base_ep90.pth.tar) |
| UPerNet semseg (160K iter) | 52.6 mIoU | [semseg_upernet_ade20k_160k.pth](https://huggingface.co/drkostas/MaskDistill-ViT-Base/resolve/main/semseg_upernet_ade20k_160k.pth) |
| Mask R-CNN det (12 ep) | 44.4 mAP | [detection_maskrcnn_coco_12ep.pth](https://huggingface.co/drkostas/MaskDistill-ViT-Base/resolve/main/detection_maskrcnn_coco_12ep.pth) |

## Usage

### Pretraining

```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 -m src.train --cfg configs/pretrain.yaml

# SLURM cluster
sbatch scripts/pretrain.sh                         # default: configs/pretrain.yaml
sbatch scripts/pretrain.sh configs/pretrain_small.yaml  # or any config
```

Configs available: `pretrain.yaml` (ViT-Base, paper default), `pretrain_random75.yaml` (MAE-style), `pretrain_tiny.yaml`, `pretrain_small.yaml`, `pretrain_large.yaml`, `pretrain_huge.yaml`.

### k-NN Evaluation

```bash
# Direct
python -m src.eval_knn --cfg configs/pretrain.yaml \
    --weights_folder output/pretrain/<run_folder> --epoch 290

# SLURM
sbatch scripts/eval_knn.sh output/pretrain/<run_folder> 290
```

### Linear Probe

```bash
# SLURM (recommended — needs 4 GPUs, ~1 day for 90 epochs)
sbatch scripts/linprobe.sh /path/to/pretrain_checkpoint.pth /path/to/imagenet

# Direct (single node)
cd src/downstream && torchrun --nproc_per_node=4 run_linear_eval.py \
    --pretrained_weights /path/to/pretrain_checkpoint.pth \
    --model_filter_name "module.student." \
    --data_path /path/to/imagenet --rel_pos_bias --epochs 90
```

### Finetuning

```bash
# SLURM (recommended — needs 4 GPUs, ~1-2 days for 100 epochs)
sbatch scripts/finetune.sh /path/to/pretrain_checkpoint.pth /path/to/imagenet

# Direct (single node)
cd src/downstream && torchrun --nproc_per_node=4 run_class_finetuning.py \
    --finetune /path/to/pretrain_checkpoint.pth \
    --model_filter_name "module.student." \
    --data_path /path/to/imagenet --rel_pos_bias \
    --batch_size 128 --epochs 100 --lr 5e-4 --layer_decay 0.65
```

### Semantic Segmentation

See [downstream/segmentation/README.md](src/downstream/segmentation/README.md) for UPerNet evaluation on ADE20K.

```bash
# SLURM eval (requires mmsegmentation)
sbatch scripts/eval_semseg.sh /path/to/semseg_checkpoint.pth /path/to/ADEChallengeData2016
```

### Object Detection

See [downstream/detection/README.md](src/downstream/detection/README.md) for Mask R-CNN evaluation on COCO.

```bash
# SLURM eval (requires mmdetection)
sbatch scripts/eval_detection.sh /path/to/detection_checkpoint.pth /path/to/coco
```

## Configuration

Key parameters in `configs/pretrain.yaml`:

```yaml
model:
  student:
    use_mask_tokens: true     # Dense mode (BEiT-style, mask tokens)
    use_shared_rel_pos_bias: true  # Shared relative position bias

mask:
  mask_type: "block"          # "block" or "random"
  mask_ratio: 0.40            # Fraction of patches to mask

losses:
  head:
    type: "smooth_l1"         # Loss function
    beta: 1.0                 # Smooth L1 beta
  normalize_targets: true     # LayerNorm on teacher features
```

## Project Structure

```
MaskDistill-PyTorch/
├── configs/
│   ├── pretrain.yaml              # ViT-Base, block masking 40% (paper default)
│   ├── pretrain_random75.yaml     # ViT-Base, random masking 75% (MAE-style)
│   ├── pretrain_tiny.yaml         # ViT-Tiny scale
│   ├── pretrain_small.yaml        # ViT-Small scale
│   ├── pretrain_large.yaml        # ViT-Large scale
│   └── pretrain_huge.yaml         # ViT-Huge scale
├── src/
│   ├── models/
│   │   ├── vision_transformer.py  # ViT student encoder
│   │   ├── clip_teacher.py        # Frozen CLIP teacher wrapper
│   │   └── maskdistill_model.py   # Unified model (student + masking + head)
│   ├── utils/
│   │   ├── losses.py              # Smooth L1 distillation loss
│   │   ├── masking_generator.py   # Block and random masking
│   │   ├── optim_factory.py       # AdamW + cosine scheduler
│   │   └── viz.py                 # Training + attention visualizations
│   ├── data/
│   │   ├── loader.py              # ImageNet data loading
│   │   └── transforms.py          # Augmentation pipeline
│   ├── downstream/
│   │   ├── run_class_finetuning.py  # End-to-end finetuning
│   │   ├── run_linear_eval.py       # Linear probe evaluation
│   │   ├── segmentation/            # UPerNet on ADE20K (mmseg)
│   │   └── detection/               # Mask R-CNN on COCO (mmdet)
│   ├── train.py                   # Pretraining script
│   └── eval_knn.py                # k-NN evaluation
├── scripts/                       # SLURM submission scripts
└── tests/
```

## Training Details

| Hyperparameter | Value |
|---------------|-------|
| Architecture | ViT-Base/16 (student) + CLIP ViT-B/16 (teacher) |
| Epochs | 300 |
| Batch size | 2048 (global) |
| Optimizer | AdamW (beta1=0.9, beta2=0.999) |
| Learning rate | 1.5e-3 (peak), cosine decay to 1e-5 |
| Weight decay | 0.05 |
| Warmup | 10 epochs |
| Gradient clipping | 3.0 |
| Precision | BFloat16 mixed precision |
| Encoding | Dense (BEiT-style, mask tokens replace masked patches) |
| Position Embedding | Shared relative position bias |
| Masking | Block-wise, 40% ratio |
| Loss | Smooth L1 (beta=1.0) on LayerNorm'd CLIP features |

## Comparison with Existing Implementations

The [official UniMIM repository](https://github.com/microsoft/unilm/tree/master/unimim) has not yet released code or checkpoints.

An [existing community reimplementation](https://github.com/bwconrad/masked-distillation) by bwconrad provides pretraining code using PyTorch Lightning. Our implementation builds on a different codebase and additionally provides pre-trained checkpoints, a full downstream evaluation suite, and reproduced results.

### Feature Comparison

| Feature | Official (UniMIM) | Unofficial (bwconrad) | **Ours** |
|---------|:-----------------:|:---------------------:|:--------:|
| Training code | — | Yes (Lightning) | **Yes (PyTorch)** |
| Pre-trained checkpoints | — | — | **[Yes (HuggingFace)](https://huggingface.co/drkostas/MaskDistill-ViT-Base)** |
| k-NN evaluation | — | — | **Yes (75.6%)** |
| Linear probe | — | — | **Yes** |
| Finetuning | — | — | **Yes** |
| Semantic segmentation | — | — | **Yes (52.6 mIoU)** |
| Object detection | — | — | **Yes (44.4 mAP)** |
| Multiple ViT scales | — | Yes | **Yes (Tiny–Huge)** |
| Reproduced results | — | — | **Yes** |

### Design Differences

| Aspect | bwconrad | **Ours** |
|--------|:--------:|:--------:|
| Encoding mode | Dense only (mask tokens) | **Dense + Sparse (MAE-style)** |
| Smooth L1 beta | 2.0 (non-standard) | **1.0 (matches paper/BEiT)** |
| Target normalization | Variance LayerNorm | **Variance LayerNorm (same)** |
| Position bias | Per-layer (timm BEiT default) | **Shared relative (BEiT v2 style)** |
| CLIP teacher output | Patches only (CLS dropped) | **Patches + CLS token** |
| Distillation head | FC or ViT decoder | **FC (paper default)** |
| Downstream: kNN / linprobe / finetune | — | **Full suite included** |
| Downstream: semseg (UPerNet + FPN) | — | **Full mmseg pipeline** |
| Downstream: detection (Mask R-CNN + FPN) | — | **Full mmdet pipeline** |
| Checkpoint format | Lightning `.ckpt` | **Standard PyTorch `.pth`** |
| Distributed training | Lightning DDP | **Native PyTorch DDP + SLURM** |
| W&B integration | — | **Full (metrics, attention viz, masking viz)** |

## Citation

If you find this implementation useful, please cite the original paper:

```bibtex
@article{hou2022unified,
  title={A Unified View of Masked Image Modeling},
  author={Hou, Zhenda and Sun, Fei and Chen, Yun-Hao and Yuan, Jia-Hong and Yu, Jia-Mu},
  journal={arXiv preprint arXiv:2210.10615},
  year={2022}
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [CLIP](https://github.com/openai/CLIP) by OpenAI
- [timm](https://github.com/huggingface/pytorch-image-models) by Ross Wightman
- [BEiT](https://github.com/microsoft/unilm/tree/master/beit) for the ViT architecture with relative position bias
- [MAE](https://github.com/facebookresearch/mae) for the sparse encoding design
