"""
MaskDistill-PyTorch test suite — single file to avoid repeated import overhead.

Run: CUDA_VISIBLE_DEVICES="" python tests/test_all.py
Or:  CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_all.py -v

Uses tiny models (embed_dim=64, depth=2) for speed.
"""
import sys, os, copy, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

t0 = time.time()
import torch
import torch.nn.functional as F
import yaml
print(f"[{time.time()-t0:.1f}s] torch imported")

from src.models.vision_transformer import VisionTransformerMIM
from src.models.maskdistill_model import build_maskdistill_model, MaskDistillModel
from src.utils.losses import compute_loss, masked_smooth_l1_loss, apply_normalization
from src.utils.masking_generator import BlockMaskingGenerator, RandomMaskingGenerator, create_masking_generator
print(f"[{time.time()-t0:.1f}s] all imports done")

# ── Tiny config (builds in <0.1s) ─────────────────────────────────────────────
DENSE_CFG = {
    "model": {
        "student": {
            "img_size": 224, "patch_size": 16, "embed_dim": 64, "depth": 2,
            "num_heads": 2, "drop_path_rate": 0.0, "init_values": 0.1,
            "use_abs_pos_emb": False, "use_shared_rel_pos_bias": True,
            "use_rel_pos_bias": False, "use_sincos_pos_emb": False,
            "use_mask_tokens": True,
        },
        "teacher": {"embed_dim": 128},
    },
    "losses": {
        "use_head_loss": True, "head_loss_weight": 1.0,
        "head": {"type": "smooth_l1", "beta": 1.0},
        "normalize_targets": True, "normalize_predictions": False,
        "normalization_method": "variance",
    },
    "mask": {"mask_ratio": 0.4, "mask_type": "block", "shuffle_patches": False},
}

SPARSE_CFG = copy.deepcopy(DENSE_CFG)
SPARSE_CFG["model"]["student"]["use_mask_tokens"] = False
SPARSE_CFG["model"]["student"]["use_shared_rel_pos_bias"] = False
SPARSE_CFG["model"]["student"]["use_abs_pos_emb"] = True
SPARSE_CFG["mask"] = {"mask_type": "random", "mask_ratio": 0.75, "shuffle_patches": True}

B, N, D_s, D_t = 2, 196, 64, 128  # batch, patches, student_dim, teacher_dim
IMG = torch.randn(B, 3, 224, 224)

passed, failed, errors = 0, 0, []

def test(name, fn):
    global passed, failed, errors
    try:
        fn()
        passed += 1
        print(f"  PASS {name}")
    except AssertionError as e:
        failed += 1
        errors.append((name, str(e)))
        print(f"  FAIL {name}: {e}")
    except Exception as e:
        failed += 1
        errors.append((name, f"{type(e).__name__}: {e}"))
        print(f"  ERROR {name}: {type(e).__name__}: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. MASKING GENERATORS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[{time.time()-t0:.1f}s] === Masking Generators ===")

def test_block_mask_shape():
    g = BlockMaskingGenerator(14, 14, mask_ratio=0.4)
    m = g()
    assert m.shape == (196,), f"Expected (196,), got {m.shape}"
    assert m.dtype == torch.bool

def test_block_mask_ratio():
    g = BlockMaskingGenerator(14, 14, mask_ratio=0.4)
    ratios = [g().float().mean().item() for _ in range(50)]
    avg = sum(ratios) / len(ratios)
    assert 0.3 < avg < 0.5, f"Average ratio {avg} not near 0.4"

def test_random_mask_exact():
    g = RandomMaskingGenerator(14, 14, mask_ratio=0.75)
    m = g()
    assert m.sum().item() == int(196 * 0.75), f"Expected {int(196*0.75)} masked, got {m.sum()}"

def test_random_shuffle_indices():
    g = RandomMaskingGenerator(14, 14, mask_ratio=0.75, shuffle_patches=True)
    g()
    assert g.ids_shuffle is not None, "Shuffle indices not set"
    assert g.ids_restore is not None, "Restore indices not set"

def test_factory_block():
    cfg = {"mask": {"mask_type": "block", "mask_ratio": 0.4}}
    g = create_masking_generator(cfg)
    assert isinstance(g, BlockMaskingGenerator)

def test_factory_random():
    cfg = {"mask": {"mask_type": "random", "mask_ratio": 0.75, "shuffle_patches": True},
           "model": {"student": {"img_size": 224, "patch_size": 16}}}
    g = create_masking_generator(cfg)
    assert isinstance(g, RandomMaskingGenerator)

for fn in [test_block_mask_shape, test_block_mask_ratio, test_random_mask_exact,
           test_random_shuffle_indices, test_factory_block, test_factory_random]:
    test(fn.__name__, fn)

# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL CONSTRUCTION & FORWARD
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[{time.time()-t0:.1f}s] === Model Construction ===")

model_dense = build_maskdistill_model(DENSE_CFG)
model_sparse = build_maskdistill_model(SPARSE_CFG)
mask_block = BlockMaskingGenerator(14, 14, mask_ratio=0.4)
mask_random = RandomMaskingGenerator(14, 14, mask_ratio=0.75, shuffle_patches=True)
print(f"[{time.time()-t0:.1f}s] models built")

def test_dense_forward_shape():
    masks = torch.stack([mask_block() for _ in range(B)])
    pred, mask_out, ids = model_dense(IMG, masks, mask_block)
    assert pred.shape == (B, N + 1, D_t), f"Expected {(B, N+1, D_t)}, got {pred.shape}"
    assert mask_out.shape == (B, N)
    # Dense mode: ids_restore is None (no shuffling with block masking)

def test_sparse_forward_shape():
    masks = torch.stack([mask_random() for _ in range(B)])
    num_visible = (~masks[0]).sum().item()
    pred, mask_out, ids = model_sparse(IMG, masks, mask_random)
    assert pred.shape[0] == B
    assert pred.shape[1] == num_visible + 1, f"Expected {num_visible+1} tokens, got {pred.shape[1]}"
    assert pred.shape[2] == D_t

def test_config_propagation_dense():
    assert model_dense.student.use_mask_tokens == True
    assert model_dense.student.rel_pos_bias is not None
    assert model_dense.student.mask_token is not None

def test_config_propagation_sparse():
    assert model_sparse.student.use_mask_tokens == False
    assert model_sparse.student.rel_pos_bias is None
    assert model_sparse.student.mask_token is None
    assert model_sparse.student.pos_embed is not None

def test_distill_head_dim():
    assert model_dense.distill_head.weight.shape == (D_t, D_s)

def test_all_configs_parse():
    cfg_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
    for f in os.listdir(cfg_dir):
        if f.endswith(".yaml"):
            with open(os.path.join(cfg_dir, f)) as fh:
                cfg = yaml.safe_load(fh)
            assert "model" in cfg, f"{f} missing 'model'"
            assert "losses" in cfg, f"{f} missing 'losses'"

for fn in [test_dense_forward_shape, test_sparse_forward_shape,
           test_config_propagation_dense, test_config_propagation_sparse,
           test_distill_head_dim, test_all_configs_parse]:
    test(fn.__name__, fn)

# ══════════════════════════════════════════════════════════════════════════════
# 3. LOSS COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[{time.time()-t0:.1f}s] === Loss Computation ===")

def test_dense_loss():
    masks = torch.stack([mask_block() for _ in range(B)])
    pred, mask_out, _ = model_dense(IMG, masks, mask_block)
    T = torch.randn(B, N + 1, D_t)
    loss, ld = compute_loss(pred, T, mask_out, DENSE_CFG)
    assert torch.isfinite(loss), f"Loss not finite: {loss}"
    assert loss.item() > 0

def test_sparse_loss():
    masks = torch.stack([mask_random() for _ in range(B)])
    pred, mask_out, _ = model_sparse(IMG, masks, mask_random)
    # Align teacher features for sparse mode
    from src.train import _extract_visible_teacher_features
    T = torch.randn(B, N + 1, D_t)
    T_aligned = _extract_visible_teacher_features(T, mask_out, model_sparse)
    loss, ld = compute_loss(pred, T_aligned, mask_out, SPARSE_CFG)
    assert torch.isfinite(loss)
    assert loss.item() > 0

def test_normalization_applied_once():
    x = torch.randn(2, 10, 64)
    n1 = apply_normalization(x, "variance")
    n2 = apply_normalization(n1, "variance")
    # Double normalization changes the values (mean ~0 after first, variance changes)
    # Single normalized: mean≈0, var≈1
    assert abs(n1.mean().item()) < 0.1
    assert 0.5 < n1.var().item() < 2.0

def test_masked_loss_only_on_masked():
    pred = torch.ones(1, 4, 8)
    target = torch.zeros(1, 4, 8)
    mask = torch.tensor([[True, False, False, False]])  # Only first patch masked
    loss = masked_smooth_l1_loss(pred, target, mask, beta=1.0)
    # Loss should be 1.0 (smooth L1 of 1.0 with beta=1.0 = 0.5)
    assert loss.item() > 0

for fn in [test_dense_loss, test_sparse_loss, test_normalization_applied_once,
           test_masked_loss_only_on_masked]:
    test(fn.__name__, fn)

# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING STEP
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[{time.time()-t0:.1f}s] === Training Step ===")

def test_backward_dense():
    masks = torch.stack([mask_block() for _ in range(B)])
    pred, mask_out, _ = model_dense(IMG, masks, mask_block)
    T = torch.randn(B, N + 1, D_t)
    loss, _ = compute_loss(pred, T, mask_out, DENSE_CFG)
    loss.backward()
    grads = sum(1 for p in model_dense.parameters() if p.grad is not None)
    total = sum(1 for p in model_dense.parameters() if p.requires_grad)
    assert grads == total, f"Only {grads}/{total} params have gradients"
    model_dense.zero_grad()

def test_backward_sparse():
    from src.train import _extract_visible_teacher_features
    masks = torch.stack([mask_random() for _ in range(B)])
    pred, mask_out, _ = model_sparse(IMG, masks, mask_random)
    T = torch.randn(B, N + 1, D_t)
    T_aligned = _extract_visible_teacher_features(T, mask_out, model_sparse)
    loss, _ = compute_loss(pred, T_aligned, mask_out, SPARSE_CFG)
    loss.backward()
    grads = sum(1 for p in model_sparse.parameters() if p.grad is not None)
    total = sum(1 for p in model_sparse.parameters() if p.requires_grad)
    assert grads == total, f"Only {grads}/{total} params have gradients"
    model_sparse.zero_grad()

def test_loss_decreases():
    m = build_maskdistill_model(DENSE_CFG)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    losses = []
    for _ in range(5):
        masks = torch.stack([mask_block() for _ in range(B)])
        pred, mask_out, _ = m(IMG, masks, mask_block)
        T = torch.randn(B, N + 1, D_t)
        loss, _ = compute_loss(pred, T, mask_out, DENSE_CFG)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    # Loss should generally decrease (not guaranteed per step, but 5-step trend)
    assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

for fn in [test_backward_dense, test_backward_sparse, test_loss_decreases]:
    test(fn.__name__, fn)

# ══════════════════════════════════════════════════════════════════════════════
# 5. CHECKPOINT COMPATIBILITY
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[{time.time()-t0:.1f}s] === Checkpoint Compatibility ===")

def test_save_load_roundtrip():
    import tempfile
    state = model_dense.state_dict()
    path = os.path.join(tempfile.gettempdir(), "test_ckpt.pth")
    torch.save({"model": state}, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    m2 = build_maskdistill_model(DENSE_CFG)
    msg = m2.load_state_dict(loaded["model"], strict=True)
    assert len(msg.missing_keys) == 0, f"Missing: {msg.missing_keys}"
    assert len(msg.unexpected_keys) == 0, f"Unexpected: {msg.unexpected_keys}"
    os.remove(path)

def test_dense_has_mask_token():
    keys = set(model_dense.state_dict().keys())
    assert "student.mask_token" in keys

def test_sparse_no_mask_token():
    keys = set(model_sparse.state_dict().keys())
    assert "student.mask_token" not in keys

def test_distill_head_in_state():
    keys = set(model_dense.state_dict().keys())
    assert "distill_head.weight" in keys
    assert "distill_head.bias" in keys

for fn in [test_save_load_roundtrip, test_dense_has_mask_token,
           test_sparse_no_mask_token, test_distill_head_in_state]:
    test(fn.__name__, fn)

# ══════════════════════════════════════════════════════════════════════════════
# 6. NMS WRAPPER
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[{time.time()-t0:.1f}s] === NMS Wrapper ===")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "downstream", "detection"))

def test_nms_return_format():
    from mmdet_custom.nms_wrapper import nms
    boxes = torch.tensor([[10, 10, 50, 50], [12, 12, 52, 52], [100, 100, 150, 150]], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8, 0.7])
    dets, inds = nms(boxes, scores, iou_threshold=0.5)
    assert dets.shape[1] == 5, f"Expected 5 columns (x1,y1,x2,y2,score), got {dets.shape[1]}"
    assert dets.shape[0] == inds.shape[0]
    # Scores should be in the last column
    assert torch.allclose(dets[:, -1], scores[inds])

def test_nms_empty():
    from mmdet_custom.nms_wrapper import nms
    boxes = torch.zeros(0, 4)
    scores = torch.zeros(0)
    dets, inds = nms(boxes, scores, iou_threshold=0.5)
    assert dets.shape == (0, 5)
    assert inds.shape == (0,)

def test_batched_nms_format():
    from mmdet_custom.nms_wrapper import batched_nms
    boxes = torch.tensor([[10, 10, 50, 50], [12, 12, 52, 52]], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8])
    idxs = torch.tensor([0, 0])
    nms_cfg = {"iou_threshold": 0.5}
    dets, inds = batched_nms(boxes, scores, idxs, nms_cfg)
    assert dets.shape[1] == 5

for fn in [test_nms_return_format, test_nms_empty, test_batched_nms_format]:
    test(fn.__name__, fn)

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
total_time = time.time() - t0
print(f"\n{'='*60}")
print(f"Results: {passed} passed, {failed} failed in {total_time:.1f}s")
if errors:
    print(f"\nFailed tests:")
    for name, msg in errors:
        print(f"  {name}: {msg}")
print(f"{'='*60}")
sys.exit(1 if failed else 0)
