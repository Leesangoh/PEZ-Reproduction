"""Step 4: Appendix C.5-style local-to-global direction diagnostics.

This script focuses on the velocity dataset only and tests whether direction
is present locally before it becomes globally position-invariant.

Outputs:
- local patch-following direction probe vs layer
- train-on-one-region / test-on-unseen-region transfer vs layer
- patch-wise direction heatmaps at selected layers
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from constants import DATA_ROOT, N_FRAMES, RESOLUTION
from step2_extract import (
    DEPTH,
    EMBED_DIM,
    build_transform,
    list_video_dirs,
    load_clip,
    load_model,
)
from step3_probe import extract_spatial_sector_groups


TEMPORAL_TOKENS = N_FRAMES // 2
SPATIAL_GRID = RESOLUTION // 16
SELECTED_LAYERS = [0, 7, 8, 15]
RIDGE_ALPHA = 1.0
KFOLD = 5
SEED = 42


def standardize(train_x, test_x):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train_x - mean) / std, (test_x - mean) / std


def fit_ridge_r2(train_x, train_y, test_x, test_y, alpha=RIDGE_ALPHA):
    train_x_std, test_x_std = standardize(train_x, test_x)
    model = Ridge(alpha=alpha, random_state=SEED)
    model.fit(train_x_std, train_y)
    pred = model.predict(test_x_std)
    ss_res = np.square(pred - test_y).sum()
    ss_tot = np.square(test_y - test_y.mean()).sum()
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def load_velocity_metadata():
    df = pd.read_parquet(f"{DATA_ROOT}/velocity/gt_velocity.parquet")
    df = df.sort_values(["video_id", "frame_idx"]).reset_index(drop=True)

    grouped = []
    for video_id, video_df in df.groupby("video_id", sort=True):
        video_df = video_df.sort_values("frame_idx").reset_index(drop=True)
        patch_xy = []
        for token_index in range(TEMPORAL_TOKENS):
            start = token_index * 2
            stop = start + 2
            pair = video_df.iloc[start:stop]
            x_px = float(pair["pos_x_px"].mean())
            y_px = float(pair["pos_y_px"].mean())
            patch_x = int(np.clip(np.floor(x_px / 16.0), 0, SPATIAL_GRID - 1))
            patch_y = int(np.clip(np.floor(y_px / 16.0), 0, SPATIAL_GRID - 1))
            patch_xy.append((patch_x, patch_y))

        grouped.append(
            {
                "video_id": video_id,
                "direction_rad": float(video_df["direction_rad"].iloc[0]),
                "tubelet_patch_xy": patch_xy,
                "start_pos_x_px": float(video_df["pos_x_px"].iloc[0]),
                "start_pos_y_px": float(video_df["pos_y_px"].iloc[0]),
            }
        )

    meta = pd.DataFrame(grouped).sort_values("video_id").reset_index(drop=True)
    meta["spatial_sector"] = extract_spatial_sector_groups(
        meta["start_pos_x_px"].to_numpy(dtype=np.float32),
        meta["start_pos_y_px"].to_numpy(dtype=np.float32),
    )
    return meta


def extract_patch_features(device: str, batch_size: int):
    transform = build_transform("resize")
    model = load_model(device=device, capture="resid_pre")
    video_dirs = list_video_dirs("velocity")
    meta = load_velocity_metadata()
    expected_ids = [Path(path).name for path in video_dirs]
    if expected_ids != meta["video_id"].tolist():
        raise ValueError("Video directory order does not match GT metadata order")

    n_videos = len(video_dirs)
    local_features = [np.zeros((n_videos, EMBED_DIM), dtype=np.float32) for _ in range(DEPTH)]
    heatmap_features = {
        layer: np.zeros((n_videos, SPATIAL_GRID * SPATIAL_GRID, EMBED_DIM), dtype=np.float16)
        for layer in SELECTED_LAYERS
    }

    patch_xy = np.asarray(meta["tubelet_patch_xy"].tolist(), dtype=np.int64)
    patch_x = patch_xy[:, :, 0]
    patch_y = patch_xy[:, :, 1]

    for start in range(0, n_videos, batch_size):
        end = min(start + batch_size, n_videos)
        batch = torch.stack([load_clip(video_dirs[index], transform) for index in range(start, end)]).to(device)

        autocast = torch.amp.autocast("cuda") if device.startswith("cuda") else torch.no_grad()
        with torch.no_grad():
            if device.startswith("cuda"):
                with torch.amp.autocast("cuda"):
                    outputs = model(batch)
            else:
                outputs = model(batch)

        bx = torch.tensor(patch_x[start:end], device=device)
        by = torch.tensor(patch_y[start:end], device=device)
        time_index = torch.arange(TEMPORAL_TOKENS, device=device)[None, :]
        batch_index = torch.arange(end - start, device=device)[:, None]

        for layer_index, layer_tokens in enumerate(outputs):
            layer_grid = layer_tokens.view(end - start, TEMPORAL_TOKENS, SPATIAL_GRID, SPATIAL_GRID, EMBED_DIM)
            local_tokens = layer_grid[batch_index, time_index, by, bx]
            local_features[layer_index][start:end] = local_tokens.mean(dim=1).float().cpu().numpy()

            if layer_index in heatmap_features:
                spatial_mean = layer_grid.mean(dim=1).reshape(end - start, SPATIAL_GRID * SPATIAL_GRID, EMBED_DIM)
                heatmap_features[layer_index][start:end] = spatial_mean.float().cpu().numpy().astype(np.float16)

    return meta, local_features, heatmap_features


def kfold_r2_curve(local_features, targets):
    splitter = KFold(n_splits=KFOLD, shuffle=True, random_state=SEED)
    rows = []
    for layer in range(DEPTH):
        fold_scores = []
        x = local_features[layer]
        for train_idx, test_idx in splitter.split(x):
            fold_scores.append(
                fit_ridge_r2(
                    x[train_idx],
                    targets[train_idx],
                    x[test_idx],
                    targets[test_idx],
                )
            )
        rows.append(
            {
                "layer": layer,
                "r2_mean": float(np.mean(fold_scores)),
                "r2_std": float(np.std(fold_scores)),
            }
        )
    return pd.DataFrame(rows)


def cross_region_transfer_curve(local_features, targets, groups):
    unique_groups = sorted(np.unique(groups).astype(int).tolist())
    rows = []
    for layer in range(DEPTH):
        x = local_features[layer]
        pair_scores = []
        for source_group in unique_groups:
            train_idx = np.where(groups == source_group)[0]
            for target_group in unique_groups:
                if target_group == source_group:
                    continue
                test_idx = np.where(groups == target_group)[0]
                pair_scores.append(
                    fit_ridge_r2(
                        x[train_idx],
                        targets[train_idx],
                        x[test_idx],
                        targets[test_idx],
                    )
                )
        rows.append(
            {
                "layer": layer,
                "r2_mean": float(np.mean(pair_scores)),
                "r2_std": float(np.std(pair_scores)),
            }
        )
    return pd.DataFrame(rows)


def patch_heatmap_scores(features_by_layer, targets):
    splitter = KFold(n_splits=KFOLD, shuffle=True, random_state=SEED)
    heatmaps = {}
    for layer in SELECTED_LAYERS:
        patch_scores = np.zeros((SPATIAL_GRID * SPATIAL_GRID,), dtype=np.float32)
        x = features_by_layer[layer].astype(np.float32)
        for patch_index in range(SPATIAL_GRID * SPATIAL_GRID):
            fold_scores = []
            patch_x = x[:, patch_index, :]
            for train_idx, test_idx in splitter.split(patch_x):
                fold_scores.append(
                    fit_ridge_r2(
                        patch_x[train_idx],
                        targets[train_idx],
                        patch_x[test_idx],
                        targets[test_idx],
                    )
                )
            patch_scores[patch_index] = float(np.mean(fold_scores))
        heatmaps[layer] = patch_scores.reshape(SPATIAL_GRID, SPATIAL_GRID)
    return heatmaps


def save_outputs(output_root: str, local_curve, transfer_curve, heatmaps):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    local_csv = output_root / "step4_local_patch_curve.csv"
    transfer_csv = output_root / "step4_cross_region_transfer_curve.csv"
    heatmap_png = output_root / "step4_patch_heatmaps.png"
    curve_png = output_root / "step4_local_global_curves.png"
    summary_json = output_root / "step4_summary.json"

    local_curve.to_csv(local_csv, index=False)
    transfer_curve.to_csv(transfer_csv, index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(local_curve["layer"], local_curve["r2_mean"], label="Local trajectory-following patch", linewidth=2)
    ax.plot(transfer_curve["layer"], transfer_curve["r2_mean"], label="Cross-region transfer", linewidth=2)
    ax.axvline(8, color="gray", linestyle="--", linewidth=1.2, label="Layer 8")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Validation / transfer R$^2$")
    ax.set_title("Step 4: Local-to-Global Direction Diagnostics")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(curve_png, dpi=220)
    plt.close(fig)

    all_values = np.stack(list(heatmaps.values()))
    vmin = float(all_values.min())
    vmax = float(all_values.max())
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, layer in zip(axes.flat, SELECTED_LAYERS):
        image = ax.imshow(heatmaps[layer], cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"Layer {layer}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, label="Patch-wise R$^2$")
    fig.suptitle("Per-Patch Direction Decoding Heatmaps")
    fig.tight_layout()
    fig.savefig(heatmap_png, dpi=220)
    plt.close(fig)

    summary = {
        "local_curve_peak_layer": int(local_curve.iloc[local_curve["r2_mean"].idxmax()]["layer"]),
        "local_curve_layer_0": float(local_curve.iloc[0]["r2_mean"]),
        "local_curve_layer_7": float(local_curve[local_curve["layer"] == 7]["r2_mean"].iloc[0]),
        "local_curve_layer_8": float(local_curve[local_curve["layer"] == 8]["r2_mean"].iloc[0]),
        "transfer_peak_layer": int(transfer_curve.iloc[transfer_curve["r2_mean"].idxmax()]["layer"]),
        "transfer_layer_0": float(transfer_curve.iloc[0]["r2_mean"]),
        "transfer_layer_7": float(transfer_curve[transfer_curve["layer"] == 7]["r2_mean"].iloc[0]),
        "transfer_layer_8": float(transfer_curve[transfer_curve["layer"] == 8]["r2_mean"].iloc[0]),
        "selected_layers": SELECTED_LAYERS,
        "ridge_alpha": RIDGE_ALPHA,
    }
    summary_json.write_text(json.dumps(summary, indent=2))

    return {
        "local_csv": str(local_csv),
        "transfer_csv": str(transfer_csv),
        "heatmap_png": str(heatmap_png),
        "curve_png": str(curve_png),
        "summary_json": str(summary_json),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-root", default="/home/solee/pez/artifacts/results")
    args = parser.parse_args()

    meta, local_features, heatmap_features = extract_patch_features(
        device=args.device,
        batch_size=args.batch_size,
    )
    targets = meta["direction_rad"].to_numpy(dtype=np.float32)
    groups = meta["spatial_sector"].to_numpy(dtype=np.int64)

    local_curve = kfold_r2_curve(local_features, targets)
    transfer_curve = cross_region_transfer_curve(local_features, targets, groups)
    heatmaps = patch_heatmap_scores(heatmap_features, targets)
    outputs = save_outputs(args.output_root, local_curve, transfer_curve, heatmaps)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
