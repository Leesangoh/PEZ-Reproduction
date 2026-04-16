"""Step 2 (raw): Extract V-JEPA 2 layer-wise features WITHOUT final LayerNorm.

The original out_layers implementation applies self.norm() (final LN) to every
layer's output, which smooths inter-layer differences. This version extracts
raw block outputs (residual stream) for more faithful PEZ reproduction.

Usage:
    /isaac-sim/python.sh /home/solee/pez/step2_extract_raw.py
"""

import json
import os
import sys
import time
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
    """Patched forward that returns raw block outputs (no final LayerNorm)."""
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
            outs.append(x.clone())  # RAW output, no self.norm()

    if self.out_layers is not None:
        return outs

    if self.norm is not None:
        x = self.norm(x)
    return x


def load_model(model_size, device):
    """Load V-JEPA 2 model with patched forward for raw outputs."""
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

    state_dict = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=True)
    ckpt_key = cfg["checkpoint_key"]
    if ckpt_key in state_dict:
        state_dict = state_dict[ckpt_key]

    cleaned = {k.replace("module.", "").replace("backbone.", ""): v
               for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=True)

    # Patch forward to return raw block outputs
    model.__class__.forward = forward_raw

    model = model.to(device).eval()
    print(f"Loaded V-JEPA 2 {model_size} (RAW mode): embed_dim={cfg['embed_dim']}, "
          f"depth={n_layers}, checkpoint={os.path.basename(cfg['checkpoint'])}")
    return model, cfg


def get_transform():
    return transforms.Compose([
        transforms.Resize((VJEPA2_INPUT_SIZE, VJEPA2_INPUT_SIZE), antialias=True),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_video_frames(video_dir, transform):
    frames = []
    for i in range(N_FRAMES):
        path = os.path.join(video_dir, f"frame_{i:02d}.png")
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1)
        frames.append(transform(tensor))
    return torch.stack(frames)


def extract_features(model, video_dirs, transform, n_layers, embed_dim, batch_size, device):
    n_videos = len(video_dirs)
    all_features = [np.zeros((n_videos, embed_dim), dtype=np.float32) for _ in range(n_layers)]

    for batch_start in tqdm(range(0, n_videos, batch_size), desc="  Extracting"):
        batch_end = min(batch_start + batch_size, n_videos)
        batch_clips = []
        for vid_idx in range(batch_start, batch_end):
            frames = load_video_frames(video_dirs[vid_idx], transform)
            clip = frames.permute(1, 0, 2, 3)
            batch_clips.append(clip)

        batch = torch.stack(batch_clips).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            outputs = model(batch)

        if batch_start == 0:
            print(f"  Token shape: {outputs[0].shape}")

        for layer_idx, layer_out in enumerate(outputs):
            pooled = layer_out.mean(dim=1).float().cpu().numpy()
            all_features[layer_idx][batch_start:batch_end] = pooled

    return all_features


def main():
    print("=" * 60)
    print("PEZ Step 2: V-JEPA 2 Feature Extraction (RAW — no final LN)")
    print("=" * 60)

    device = "cuda:0"
    transform = get_transform()

    datasets = {}
    for dataset_name in ["velocity", "acceleration"]:
        video_base = os.path.join(DATA_ROOT, dataset_name, "videos")
        video_dirs = sorted([d for d in glob(os.path.join(video_base, "*")) if os.path.isdir(d)])
        datasets[dataset_name] = video_dirs
        print(f"{dataset_name}: {len(video_dirs)} videos")

    for model_size in ["large", "giant"]:
        cfg = MODEL_CONFIGS[model_size]
        model_name = cfg["name"]
        n_layers = cfg["depth"]
        embed_dim = cfg["embed_dim"]
        batch_size = cfg["batch_size"]

        print(f"\n{'='*40}")
        print(f"Model: {model_name} ({model_size}) — RAW mode")
        print(f"{'='*40}")

        model, _ = load_model(model_size, device)
        t0 = time.time()

        for dataset_name, video_dirs in datasets.items():
            n_videos = len(video_dirs)
            out_dir = os.path.join(FEATURES_ROOT, model_name, dataset_name)
            os.makedirs(out_dir, exist_ok=True)

            print(f"\n  Dataset: {dataset_name} ({n_videos} videos)")
            features = extract_features(
                model, video_dirs, transform, n_layers, embed_dim, batch_size, device
            )

            for l in range(n_layers):
                np.save(os.path.join(out_dir, f"layer_{l:02d}.npy"), features[l])

            # Quick sanity: norms across layers
            norms = [np.linalg.norm(features[l], axis=1).mean() for l in range(n_layers)]
            print(f"  Norm range: {min(norms):.1f} - {max(norms):.1f}")

        print(f"  {model_name} time: {time.time()-t0:.1f}s")
        del model
        torch.cuda.empty_cache()

    print("\nStep 2 (RAW) complete!")


if __name__ == "__main__":
    main()
