"""Step 2: Extract V-JEPA 2 layer-wise features for PEZ reproduction.

Extracts mean-pooled RAW residual-stream features from all layers of V-JEPA 2
for the synthetic ball videos. The paper probes the residual stream, not the
per-layer outputs after the model's final LayerNorm.

Usage:
    /isaac-sim/python.sh /home/solee/pez/step2_extract.py
"""

import json
import os
import sys
import time
import argparse
from contextlib import nullcontext
from glob import glob

import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, "/home/solee/pez")
from constants import (
    CHECKPOINT_DIR,
    DATA_ROOT,
    FEATURES_ROOT,
    IMAGENET_MEAN,
    IMAGENET_STD,
    N_FRAMES,
    VJEPA2_INPUT_SIZE,
    VJEPA2_ROOT,
    VJEPA2_SRC,
)

# V-JEPA 2 imports
sys.path.insert(0, VJEPA2_ROOT)
sys.path.insert(0, VJEPA2_SRC)

MODEL_CONFIGS = {
    "large": {
        "name": "vjepa2_L",
        "factory": "vit_large",
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "checkpoint": os.path.join(CHECKPOINT_DIR, "vitl.pt"),
        "checkpoint_key": "target_encoder",
        "batch_size": 8,
    },
    "giant": {
        "name": "vjepa2_G",
        "factory": "vit_giant_xformers",
        "embed_dim": 1408,
        "depth": 40,
        "num_heads": 22,
        "checkpoint": os.path.join(CHECKPOINT_DIR, "vitg.pt"),
        "checkpoint_key": "target_encoder",
        "batch_size": 4,
    },
}


def forward_raw(self, x, masks=None):
    """Patched forward that returns raw block outputs without final LayerNorm."""
    if masks is not None and not isinstance(masks, list):
        masks = [masks]

    if x.ndim == 4:
        _, _, H, W = x.shape
        T = 1
    elif x.ndim == 5:
        _, _, T, H, W = x.shape
        T = T // self.tubelet_size
    H_patches = H // self.patch_size
    W_patches = W // self.patch_size
    if not self.handle_nonsquare_inputs:
        T = H_patches = W_patches = None

    if not self.use_rope:
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x = self.patch_embed(x)
        x += pos_embed
    else:
        x = self.patch_embed(x)

    if masks is not None:
        from src.masks.utils import apply_masks
        x = apply_masks(x, masks)
        masks = torch.cat(masks, dim=0)

    outs = []
    for i, blk in enumerate(self.blocks):
        x = blk(x, mask=masks, attn_mask=None, T=T, H_patches=H_patches, W_patches=W_patches)
        if self.out_layers is not None and i in self.out_layers:
            outs.append(x.clone())

    if self.out_layers is not None:
        return outs

    if self.norm is not None:
        x = self.norm(x)
    return x


def load_model(model_size, device):
    """Load V-JEPA 2 model and patch it to emit raw residual stream outputs."""
    import models.vision_transformer as vit_module

    cfg = MODEL_CONFIGS[model_size]
    n_layers = cfg["depth"]
    out_layers = list(range(n_layers))

    factory_fn = getattr(vit_module, cfg["factory"])
    model = factory_fn(
        patch_size=16,
        img_size=(VJEPA2_INPUT_SIZE, VJEPA2_INPUT_SIZE),
        num_frames=64,
        tubelet_size=2,
        out_layers=out_layers,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        use_rope=True,
    )

    # Load checkpoint
    state_dict = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=True)
    ckpt_key = cfg["checkpoint_key"]
    if ckpt_key in state_dict:
        state_dict = state_dict[ckpt_key]
    elif "model" in state_dict:
        state_dict = state_dict["model"]

    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("backbone.", "")
        cleaned[k] = v

    msg = model.load_state_dict(cleaned, strict=False)
    if msg.unexpected_keys:
        print(f"  Unexpected keys: {msg.unexpected_keys[:5]}...")
    if msg.missing_keys:
        print(f"  Missing keys: {msg.missing_keys[:5]}...")

    model.__class__.forward = forward_raw
    model = model.to(device).eval()
    print(f"Loaded V-JEPA 2 {model_size}: embed_dim={cfg['embed_dim']}, "
          f"depth={n_layers}, img_size={VJEPA2_INPUT_SIZE}, out_layers=[0..{n_layers-1}], "
          f"raw_residual=True")
    return model, cfg


def get_transform():
    """Preprocessing pipeline: resize to 224, normalize with ImageNet stats."""
    return transforms.Compose([
        transforms.Resize((VJEPA2_INPUT_SIZE, VJEPA2_INPUT_SIZE), antialias=True),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_video_frames(video_dir, transform):
    """Load 16 PNG frames, apply transform, return (16, 3, 224, 224)."""
    frames = []
    for i in range(N_FRAMES):
        path = os.path.join(video_dir, f"frame_{i:02d}.png")
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1)  # (3, H, W) uint8
        tensor = transform(tensor)  # (3, 224, 224) float32 normalized
        frames.append(tensor)
    return torch.stack(frames)  # (16, 3, 224, 224)


def extract_features(model, video_dirs, transform, n_layers, embed_dim, batch_size, device):
    """Extract mean-pooled features for all videos from all layers."""
    n_videos = len(video_dirs)
    all_features = [np.zeros((n_videos, embed_dim), dtype=np.float32) for _ in range(n_layers)]

    token_shape_checked = False

    for batch_start in tqdm(range(0, n_videos, batch_size), desc="  Extracting"):
        batch_end = min(batch_start + batch_size, n_videos)
        batch_clips = []

        for vid_idx in range(batch_start, batch_end):
            frames = load_video_frames(video_dirs[vid_idx], transform)
            clip = frames.permute(1, 0, 2, 3)  # (3, 16, 224, 224)
            batch_clips.append(clip)

        batch = torch.stack(batch_clips).to(device)  # (B, 3, 16, 224, 224)

        autocast_ctx = torch.amp.autocast("cuda") if device.startswith("cuda") else nullcontext()
        with torch.no_grad(), autocast_ctx:
            outputs = model(batch)  # list of n_layers tensors, each (B, tokens, D)

        # Verify token count on first batch
        if not token_shape_checked:
            token_shape = outputs[0].shape
            expected_tokens = 8 * 196  # 8 temporal * 196 spatial = 1568
            print(f"  Token shape: {token_shape} (expected B x {expected_tokens} x {embed_dim})")
            assert token_shape[1] == expected_tokens, (
                f"Token count mismatch: got {token_shape[1]}, expected {expected_tokens}. "
                f"Check input resolution (should be 224x224)."
            )
            token_shape_checked = True

        # Mean-pool and store
        for layer_idx, layer_out in enumerate(outputs):
            pooled = layer_out.mean(dim=1).float().cpu().numpy()  # (B, D)
            all_features[layer_idx][batch_start:batch_end] = pooled

    return all_features


def sanity_checks(features_dir, model_name, n_layers, embed_dim, dataset_name, n_videos):
    """Verify extracted features."""
    print(f"\n  Sanity checks for {model_name}/{dataset_name}:")

    # Shape check
    for l in range(n_layers):
        path = os.path.join(features_dir, f"layer_{l:02d}.npy")
        feat = np.load(path)
        assert feat.shape == (n_videos, embed_dim), \
            f"Layer {l}: expected ({n_videos}, {embed_dim}), got {feat.shape}"

    # NaN/Inf check
    has_nan = False
    for l in range(n_layers):
        feat = np.load(os.path.join(features_dir, f"layer_{l:02d}.npy"))
        if np.isnan(feat).any() or np.isinf(feat).any():
            print(f"  WARNING: Layer {l} has NaN/Inf!")
            has_nan = True
    if not has_nan:
        print(f"  [OK] No NaN/Inf in any layer")

    # Norm check
    norms = []
    for l in range(n_layers):
        feat = np.load(os.path.join(features_dir, f"layer_{l:02d}.npy"))
        norms.append(np.linalg.norm(feat, axis=1).mean())
    print(f"  [OK] Feature norms range: {min(norms):.2f} - {max(norms):.2f}")

    # Inter-layer cosine similarity (sample)
    layers_to_check = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    feats = {}
    for l in layers_to_check:
        f = np.load(os.path.join(features_dir, f"layer_{l:02d}.npy"))
        feats[l] = f.mean(axis=0)  # mean across videos
        feats[l] = feats[l] / (np.linalg.norm(feats[l]) + 1e-8)

    print(f"  Cosine similarities (mean feature vectors):")
    for i, l1 in enumerate(layers_to_check):
        for l2 in layers_to_check[i + 1:]:
            sim = np.dot(feats[l1], feats[l2])
            print(f"    Layer {l1} vs {l2}: {sim:.4f}")


def main():
    print("=" * 60)
    print("PEZ Step 2: V-JEPA 2 Feature Extraction")
    print("=" * 60)

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODEL_CONFIGS.keys()), default=list(MODEL_CONFIGS.keys()))
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    transform = get_transform()

    # Collect video directories for each dataset
    datasets = {}
    for dataset_name in ["velocity", "acceleration"]:
        video_base = os.path.join(DATA_ROOT, dataset_name, "videos")
        video_dirs = sorted(glob(os.path.join(video_base, "*")))
        video_dirs = [d for d in video_dirs if os.path.isdir(d)]
        datasets[dataset_name] = video_dirs
        print(f"{dataset_name}: {len(video_dirs)} videos")

    assert len(datasets["velocity"]) == 392, f"Expected 392, got {len(datasets['velocity'])}"
    assert len(datasets["acceleration"]) == 280, f"Expected 280, got {len(datasets['acceleration'])}"

    extraction_config = {
        "input_size": VJEPA2_INPUT_SIZE,
        "n_frames": N_FRAMES,
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "pooling": "spatiotemporal_mean",
        "feature_source": "raw_residual_stream",
        "expected_tokens": 1568,
        "models": {},
    }

    for model_size in args.models:
        cfg = MODEL_CONFIGS[model_size]
        model_name = cfg["name"]
        n_layers = cfg["depth"]
        embed_dim = cfg["embed_dim"]
        batch_size = cfg["batch_size"]

        print(f"\n{'='*40}")
        print(f"Model: {model_name} ({model_size})")
        print(f"{'='*40}")

        model, _ = load_model(model_size, device)
        start_time = time.time()

        for dataset_name, video_dirs in datasets.items():
            n_videos = len(video_dirs)
            out_dir = os.path.join(FEATURES_ROOT, model_name, dataset_name)
            os.makedirs(out_dir, exist_ok=True)

            print(f"\n  Dataset: {dataset_name} ({n_videos} videos, batch_size={batch_size})")
            features = extract_features(
                model, video_dirs, transform, n_layers, embed_dim, batch_size, device
            )

            # Save per-layer .npy files
            for l in range(n_layers):
                np.save(os.path.join(out_dir, f"layer_{l:02d}.npy"), features[l])

            # Sanity checks
            sanity_checks(out_dir, model_name, n_layers, embed_dim, dataset_name, n_videos)

        elapsed = time.time() - start_time
        print(f"\n  {model_name} total time: {elapsed:.1f}s")

        extraction_config["models"][model_name] = {
            "model_size": model_size,
            "depth": n_layers,
            "embed_dim": embed_dim,
            "checkpoint": cfg["checkpoint"],
            "batch_size": batch_size,
        }

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache()

    # Save config
    with open(os.path.join(FEATURES_ROOT, "extraction_config.json"), "w") as f:
        json.dump(extraction_config, f, indent=2)

    print("\nStep 2 complete!")
    print(f"Output: {FEATURES_ROOT}/")


if __name__ == "__main__":
    main()
