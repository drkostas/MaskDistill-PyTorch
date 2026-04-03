# Contributing to MaskDistill-PyTorch

Thank you for your interest in contributing!

## Reporting Issues

- **Bug reports**: Include the full error traceback, your PyTorch/CUDA version, and the command you ran.
- **Feature requests**: Describe the use case and expected behavior.

## Pull Requests

1. Fork the repo and create a feature branch from `main`
2. Make your changes and ensure tests pass: `CUDA_VISIBLE_DEVICES="" python tests/test_all.py`
3. Submit a PR with a clear description of what changed and why

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/MaskDistill-PyTorch.git
cd MaskDistill-PyTorch
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Tests

```bash
CUDA_VISIBLE_DEVICES="" python tests/test_all.py
```

All 26 tests should pass. Tests use tiny models and run on CPU — no GPU or ImageNet needed.

## Code Style

- Follow existing patterns in the codebase
- No MoE, decoder, CLS loss, or evolved masking code — this repo is MaskDistill only
- Keep imports at the top of files (except CLIP, which is lazy-loaded for test speed)
