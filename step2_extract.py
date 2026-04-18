"""Step 2: Extract Figure 2(c) features from V-JEPA v2-L.

This rewrite keeps only the pieces needed for the paper-faithful Figure 2(c)
reproduction:

- V-JEPA v2-L only
- 16-frame 256x256 clips
- residual stream capture at every layer
- mean-pooling over space-time tokens
- two capture conventions for ablation:
  * resid_pre  : patch_embed output + pre-block residuals
  * resid_post : post-block residuals
- two preprocessing branches for ablation:
  * resize       : direct 256x256 resize
  * eval_preproc : Resize(293) -> CenterCrop(256)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, "/home/solee/pez")
from constants import (
    CHECKPOINT_DIR,
    DATA_ROOT,
    IMAGENET_MEAN,
    IMAGENET_STD,
    N_FRAMES,
    VJEPA2_INPUT_SIZE,
    VJEPA2_ROOT,
    VJEPA2_SRC,
)

sys.path.insert(0, VJEPA2_ROOT)
sys.path.insert(0, VJEPA2_SRC)


MODEL_NAME = "vjepa2_L"
EMBED_DIM = 1024
DEPTH = 24
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "vitl.pt")
TEMPORAL_TOKENS = N_FRAMES // 2
SPATIAL_GRID = VJEPA2_INPUT_SIZE // 16


def forward_resid_pre(self, x, masks=None):
    """Capture paper-style residual stream before each transformer block.

    For V-JEPA v2-L this yields 24 representations:
      layer_00 = patch_embed output (before block 0)
      layer_01 = post-block 0 = pre-block 1
      ...
      layer_23 = post-block 22 = pre-block 23

    This matches the paper's layer count (0..n-1) and keeps layer 0 as the
    representation before any attention block.
    """
    if masks is not None and not isinstance(masks, list):
        masks = [masks]

    if x.ndim == 4:
        _, _, height, width = x.shape
        tubelets = 1
    else:
        _, _, tubelets, height, width = x.shape
        tubelets = tubelets // self.tubelet_size

    h_patches = height // self.patch_size
    w_patches = width // self.patch_size
    if not self.handle_nonsquare_inputs:
        tubelets = h_patches = w_patches = None

    if not self.use_rope:
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x = self.patch_embed(x)
        x = x + pos_embed
    else:
        x = self.patch_embed(x)

    if masks is not None:
        from src.masks.utils import apply_masks

        x = apply_masks(x, masks)
        masks = torch.cat(masks, dim=0)

    outs = [x.clone()]
    for block_index, block in enumerate(self.blocks[:-1]):
        x = block(
            x,
            mask=masks,
            attn_mask=None,
            T=tubelets,
            H_patches=h_patches,
            W_patches=w_patches,
        )
        if block_index + 1 < DEPTH:
            outs.append(x.clone())

    return outs


def forward_resid_post(self, x, masks=None):
    """Capture residual stream after every transformer block."""
    if masks is not None and not isinstance(masks, list):
        masks = [masks]

    if x.ndim == 4:
        _, _, height, width = x.shape
        tubelets = 1
    else:
        _, _, tubelets, height, width = x.shape
        tubelets = tubelets // self.tubelet_size

    h_patches = height // self.patch_size
    w_patches = width // self.patch_size
    if not self.handle_nonsquare_inputs:
        tubelets = h_patches = w_patches = None

    if not self.use_rope:
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x = self.patch_embed(x)
        x = x + pos_embed
    else:
        x = self.patch_embed(x)

    if masks is not None:
        from src.masks.utils import apply_masks

        x = apply_masks(x, masks)
        masks = torch.cat(masks, dim=0)

    outs = []
    for block in self.blocks:
        x = block(
            x,
            mask=masks,
            attn_mask=None,
            T=tubelets,
            H_patches=h_patches,
            W_patches=w_patches,
        )
        outs.append(x.clone())

    return outs


def build_transform(transform_name: str):
    if transform_name == "resize":
        spatial = transforms.Resize((VJEPA2_INPUT_SIZE, VJEPA2_INPUT_SIZE), antialias=True)
    elif transform_name == "eval_preproc":
        spatial = transforms.Compose(
            [
                transforms.Resize(293, antialias=True),
                transforms.CenterCrop((VJEPA2_INPUT_SIZE, VJEPA2_INPUT_SIZE)),
            ]
        )
    else:
        raise ValueError(f"Unknown transform: {transform_name}")

    return transforms.Compose(
        [
            spatial,
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_model(device: str, capture: str):
    import models.vision_transformer as vit_module

    model = vit_module.vit_large(
        patch_size=16,
        img_size=(VJEPA2_INPUT_SIZE, VJEPA2_INPUT_SIZE),
        num_frames=64,
        tubelet_size=2,
        out_layers=list(range(DEPTH)),
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        use_rope=True,
    )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    state = checkpoint.get("target_encoder", checkpoint)
    cleaned = {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in state.items()
    }
    model.load_state_dict(cleaned, strict=True)

    if capture == "resid_pre":
        model.__class__.forward = forward_resid_pre
    elif capture == "resid_post":
        model.__class__.forward = forward_resid_post
    else:
        raise ValueError(f"Unknown capture: {capture}")

    return model.to(device).eval()


def list_video_dirs(dataset_name: str):
    base = os.path.join(DATA_ROOT, dataset_name, "videos")
    return sorted(path for path in glob(os.path.join(base, "*")) if os.path.isdir(path))


def load_clip(video_dir: str, transform):
    frames = []
    for frame_idx in range(N_FRAMES):
        path = os.path.join(video_dir, f"frame_{frame_idx:02d}.png")
        image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1)
        frames.append(transform(tensor))
    return torch.stack(frames).permute(1, 0, 2, 3)


def pool_tokens(layer_tokens: torch.Tensor, pooling: str):
    if pooling == "mean":
        return layer_tokens.mean(dim=1)

    grid = layer_tokens.view(
        layer_tokens.shape[0],
        TEMPORAL_TOKENS,
        SPATIAL_GRID,
        SPATIAL_GRID,
        layer_tokens.shape[-1],
    )
    if pooling == "temporal_last":
        return grid[:, -1].mean(dim=(1, 2))
    if pooling == "temporal_first":
        return grid[:, 0].mean(dim=(1, 2))
    if pooling == "temporal_diff":
        return grid[:, -1].mean(dim=(1, 2)) - grid[:, 0].mean(dim=(1, 2))
    if pooling == "temporal_last_patch":
        return grid[:, -1].reshape(layer_tokens.shape[0], SPATIAL_GRID * SPATIAL_GRID, layer_tokens.shape[-1])
    if pooling == "temporal_diff_patch":
        return (
            grid[:, -1].reshape(layer_tokens.shape[0], SPATIAL_GRID * SPATIAL_GRID, layer_tokens.shape[-1])
            - grid[:, 0].reshape(layer_tokens.shape[0], SPATIAL_GRID * SPATIAL_GRID, layer_tokens.shape[-1])
        )
    raise ValueError(f"Unknown pooling: {pooling}")


def extract_dataset(model, video_dirs, transform, batch_size: int, device: str, pooling: str):
    if pooling.endswith("_patch"):
        features = [
            np.zeros((len(video_dirs), SPATIAL_GRID * SPATIAL_GRID, EMBED_DIM), dtype=np.float32)
            for _ in range(DEPTH)
        ]
    else:
        features = [np.zeros((len(video_dirs), EMBED_DIM), dtype=np.float32) for _ in range(DEPTH)]

    for start in tqdm(range(0, len(video_dirs), batch_size), desc="extract"):
        end = min(start + batch_size, len(video_dirs))
        batch = torch.stack(
            [load_clip(video_dirs[index], transform) for index in range(start, end)]
        ).to(device)

        autocast = torch.amp.autocast("cuda") if device.startswith("cuda") else nullcontext()
        with torch.no_grad(), autocast:
            outputs = model(batch)

        for layer_index, layer_tokens in enumerate(outputs):
            pooled = pool_tokens(layer_tokens, pooling).float().cpu().numpy()
            features[layer_index][start:end] = pooled

    return features


def save_branch(output_root: str, dataset_name: str, features):
    dataset_root = Path(output_root) / MODEL_NAME / dataset_name
    dataset_root.mkdir(parents=True, exist_ok=True)
    for layer_index, array in enumerate(features):
        np.save(dataset_root / f"layer_{layer_index:02d}.npy", array)


def default_output_root(capture: str, transform_name: str, pooling: str):
    suffix = f"{capture}_{transform_name}" if pooling == "mean" else f"{capture}_{transform_name}_{pooling}"
    return os.path.join("/home/solee/pez/artifacts/features", suffix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", choices=["resid_pre", "resid_post"], default="resid_pre")
    parser.add_argument("--transform", choices=["resize", "eval_preproc"], default="resize")
    parser.add_argument(
        "--pooling",
        choices=[
            "mean",
            "temporal_last",
            "temporal_first",
            "temporal_diff",
            "temporal_last_patch",
            "temporal_diff_patch",
        ],
        default="mean",
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root or default_output_root(args.capture, args.transform, args.pooling)
    Path(output_root).mkdir(parents=True, exist_ok=True)

    metadata_path = Path(output_root) / "extraction_metadata.json"
    if args.reuse_existing and metadata_path.exists():
        print(f"Reusing existing extraction at {output_root}")
        return

    transform = build_transform(args.transform)
    model = load_model(args.device, args.capture)

    start_time = time.time()
    for dataset_name in ("velocity", "acceleration"):
        print(f"[{dataset_name}] capture={args.capture} transform={args.transform}")
        video_dirs = list_video_dirs(dataset_name)
        features = extract_dataset(
            model=model,
            video_dirs=video_dirs,
            transform=transform,
            batch_size=args.batch_size,
            device=args.device,
            pooling=args.pooling,
        )
        save_branch(output_root, dataset_name, features)

    metadata = {
        "model": MODEL_NAME,
        "capture": args.capture,
        "transform": args.transform,
        "pooling": args.pooling,
        "input_size": VJEPA2_INPUT_SIZE,
        "n_layers": DEPTH,
        "embed_dim": EMBED_DIM,
        "n_frames": N_FRAMES,
        "checkpoint": CHECKPOINT_PATH,
        "elapsed_sec": time.time() - start_time,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
