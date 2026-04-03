"""
MaskDistill Classification Demo
Upload an image -> ImageNet classification info.
Full inference available once finetuned checkpoint is uploaded.
"""

import gradio as gr
from PIL import Image

IMAGENET_EXAMPLES = [
    "tabby cat", "golden retriever", "sports car", "espresso", "laptop",
    "soccer ball", "pizza", "sunflower", "lighthouse", "violin",
]


def classify(image):
    if image is None:
        return None, "No image uploaded"

    # Resize
    img = image.resize((224, 224), Image.BILINEAR)

    info = "## MaskDistill Image Classification\n\n"
    info += "**Model**: MaskDistill ViT-Base/16 finetuned on ImageNet-1K\n\n"
    info += "**Training**: 300 epoch pretrain + 100 epoch finetune\n\n"
    info += "**Results**: **75.6% k-NN** | finetuning results coming soon\n\n"
    info += "---\n\n"
    info += "### Status\n\n"
    info += "Finetuning is in progress. Full interactive classification will be "
    info += "available once the finetuned checkpoint is uploaded.\n\n"
    info += "---\n\n"
    info += "### Run locally\n\n"
    info += "```bash\n"
    info += "git clone https://github.com/drkostas/MaskDistill-PyTorch\n"
    info += "cd MaskDistill-PyTorch\n\n"
    info += "# k-NN evaluation (no finetuning needed)\n"
    info += "sbatch scripts/eval_knn.sh checkpoint_folder 290\n\n"
    info += "# Finetuning\n"
    info += "sbatch scripts/finetune.sh pretrain_checkpoint.pth /path/to/imagenet\n"
    info += "```\n\n"
    info += "---\n\n"
    info += "**ImageNet-1K**: 1000 classes, 1.28M training images\n\n"
    info += "Example classes: " + ", ".join(f"`{c}`" for c in IMAGENET_EXAMPLES)

    return img, info


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=[
        gr.Image(type="pil", label="Input (224x224)"),
        gr.Markdown(label="Model Info"),
    ],
    title="MaskDistill Image Classification (ImageNet-1K)",
    description=(
        "Image classification using MaskDistill ViT-Base/16 on ImageNet-1K (1000 classes). "
        "Achieves **75.6% k-NN accuracy**. Finetuning in progress. "
        "[Download checkpoint](https://huggingface.co/drkostas/MaskDistill-ViT-Base) | "
        "[GitHub](https://github.com/drkostas/MaskDistill-PyTorch)"
    ),
)

if __name__ == "__main__":
    demo.launch()
