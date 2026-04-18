"""Step 3: Figure 2 probing for PEZ reproduction.

Primary support:

- V-JEPA v2-L only
- Figure 2(c) polar:
  - speed
  - direction
  - acceleration magnitude
- Figure 2(b) cartesian:
  - velocity_xy
  - acceleration_xy
- linear probe f(h) = Wh + b
- 20-config trainable sweep (lr x wd)
- 5-fold grouped cross-validation
- validation R^2 vs layer fraction

The four under-specified axes are exposed as ablations:
- CV group key
- direction target
- residual capture (handled via feature_root)
- preprocessing (handled via feature_root)
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

sys_path_added = False
if not sys_path_added:
    import sys

    sys.path.insert(0, "/home/solee/pez")
    sys_path_added = True

from constants import DATA_ROOT, RESULTS_ROOT


MODEL_NAME = "vjepa2_L"
DEPTH = 24
CV_SPLITS = 5
CV_RANDOM_SEED = 42
LR_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 5e-3]
WD_GRID = [0.01, 0.1, 0.4, 0.8]
MAX_EPOCHS = 400
PATIENCE = 40
ADAMW100_LR = 1e-3
ADAMW100_WD = 0.1
ADAMW100_EPOCHS = 100
ADAMW100_PATIENCE = 10

PROBE_SETS = {
    "fig2c": ("speed", "direction", "acceleration"),
    "fig2b_velocity_xy": ("velocity_xy",),
    "fig2b_acceleration_xy": ("acceleration_xy",),
    "fig2b_velocity_axes": ("velocity_x", "velocity_y"),
    "fig2b_acceleration_axes": ("acceleration_x", "acceleration_y"),
}

PLOT_LABELS = {
    "speed": "Speed",
    "direction": "Direction",
    "acceleration": "Acceleration",
    "velocity_xy": "Velocity XY",
    "acceleration_xy": "Acceleration XY",
    "velocity_x": "Velocity X",
    "velocity_y": "Velocity Y",
    "acceleration_x": "Acceleration X",
    "acceleration_y": "Acceleration Y",
}

PLOT_COLORS = {
    "speed": "#1f77b4",
    "direction": "#d62728",
    "acceleration": "#2ca02c",
    "velocity_xy": "#1f77b4",
    "acceleration_xy": "#2ca02c",
    "velocity_x": "#1f77b4",
    "velocity_y": "#ff7f0e",
    "acceleration_x": "#2ca02c",
    "acceleration_y": "#9467bd",
}


def load_targets(direction_target: str):
    velocity_df = pd.read_parquet(os.path.join(DATA_ROOT, "velocity", "gt_velocity.parquet"))
    acceleration_df = pd.read_parquet(
        os.path.join(DATA_ROOT, "acceleration", "gt_acceleration.parquet")
    )

    velocity_df = (
        velocity_df[velocity_df["frame_idx"] == 0].sort_values("video_id").reset_index(drop=True)
    )
    acceleration_df = (
        acceleration_df[acceleration_df["frame_idx"] == 0]
        .sort_values("video_id")
        .reset_index(drop=True)
    )

    if direction_target == "sincos":
        direction = np.stack(
            [
                np.sin(velocity_df["direction_rad"].to_numpy(dtype=np.float32)),
                np.cos(velocity_df["direction_rad"].to_numpy(dtype=np.float32)),
            ],
            axis=1,
        )
    elif direction_target == "angle":
        direction = velocity_df["direction_rad"].to_numpy(dtype=np.float32)
    elif direction_target == "vxy":
        direction = np.stack(
            [
                velocity_df["vx_px"].to_numpy(dtype=np.float32),
                velocity_df["vy_px"].to_numpy(dtype=np.float32),
            ],
            axis=1,
        )
    else:
        raise ValueError(f"Unknown direction_target: {direction_target}")

    return {
        "speed": {
            "dataset": "velocity",
            "target": velocity_df["speed"].to_numpy(dtype=np.float32),
            "video_ids": velocity_df["video_id"].tolist(),
            "output_dim": 1,
            "pos_x_px": velocity_df["pos_x_px"].to_numpy(dtype=np.float32),
            "pos_y_px": velocity_df["pos_y_px"].to_numpy(dtype=np.float32),
        },
        "direction": {
            "dataset": "velocity",
            "target": direction.astype(np.float32),
            "video_ids": velocity_df["video_id"].tolist(),
            "output_dim": 2 if direction_target in {"sincos", "vxy"} else 1,
            "pos_x_px": velocity_df["pos_x_px"].to_numpy(dtype=np.float32),
            "pos_y_px": velocity_df["pos_y_px"].to_numpy(dtype=np.float32),
        },
        "acceleration": {
            "dataset": "acceleration",
            "target": acceleration_df["accel_magnitude"].to_numpy(dtype=np.float32),
            "video_ids": acceleration_df["video_id"].tolist(),
            "output_dim": 1,
            "pos_x_px": acceleration_df["pos_x_px"].to_numpy(dtype=np.float32),
            "pos_y_px": acceleration_df["pos_y_px"].to_numpy(dtype=np.float32),
        },
        "velocity_xy": {
            "dataset": "velocity",
            "target": np.stack(
                [
                    velocity_df["vx_px"].to_numpy(dtype=np.float32),
                    velocity_df["vy_px"].to_numpy(dtype=np.float32),
                ],
                axis=1,
            ).astype(np.float32),
            "video_ids": velocity_df["video_id"].tolist(),
            "output_dim": 2,
            "pos_x_px": velocity_df["pos_x_px"].to_numpy(dtype=np.float32),
            "pos_y_px": velocity_df["pos_y_px"].to_numpy(dtype=np.float32),
        },
        "velocity_x": {
            "dataset": "velocity",
            "target": velocity_df["vx_px"].to_numpy(dtype=np.float32),
            "video_ids": velocity_df["video_id"].tolist(),
            "output_dim": 1,
            "pos_x_px": velocity_df["pos_x_px"].to_numpy(dtype=np.float32),
            "pos_y_px": velocity_df["pos_y_px"].to_numpy(dtype=np.float32),
        },
        "velocity_y": {
            "dataset": "velocity",
            "target": velocity_df["vy_px"].to_numpy(dtype=np.float32),
            "video_ids": velocity_df["video_id"].tolist(),
            "output_dim": 1,
            "pos_x_px": velocity_df["pos_x_px"].to_numpy(dtype=np.float32),
            "pos_y_px": velocity_df["pos_y_px"].to_numpy(dtype=np.float32),
        },
        "acceleration_xy": {
            "dataset": "acceleration",
            "target": np.stack(
                [
                    acceleration_df["ax_px"].to_numpy(dtype=np.float32),
                    acceleration_df["ay_px"].to_numpy(dtype=np.float32),
                ],
                axis=1,
            ).astype(np.float32),
            "video_ids": acceleration_df["video_id"].tolist(),
            "output_dim": 2,
            "pos_x_px": acceleration_df["pos_x_px"].to_numpy(dtype=np.float32),
            "pos_y_px": acceleration_df["pos_y_px"].to_numpy(dtype=np.float32),
        },
        "acceleration_x": {
            "dataset": "acceleration",
            "target": acceleration_df["ax_px"].to_numpy(dtype=np.float32),
            "video_ids": acceleration_df["video_id"].tolist(),
            "output_dim": 1,
            "pos_x_px": acceleration_df["pos_x_px"].to_numpy(dtype=np.float32),
            "pos_y_px": acceleration_df["pos_y_px"].to_numpy(dtype=np.float32),
        },
        "acceleration_y": {
            "dataset": "acceleration",
            "target": acceleration_df["ay_px"].to_numpy(dtype=np.float32),
            "video_ids": acceleration_df["video_id"].tolist(),
            "output_dim": 1,
            "pos_x_px": acceleration_df["pos_x_px"].to_numpy(dtype=np.float32),
            "pos_y_px": acceleration_df["pos_y_px"].to_numpy(dtype=np.float32),
        },
    }


def extract_position_groups(video_ids):
    groups = []
    for video_id in video_ids:
        match = re.search(r"_pos(\d+)$", video_id)
        if match is None:
            raise ValueError(f"Cannot parse position index from {video_id}")
        groups.append(int(match.group(1)))
    return np.asarray(groups, dtype=np.int64)


def extract_condition_groups(video_ids):
    groups = [re.sub(r"_pos\d+$", "", video_id) for video_id in video_ids]
    return pd.factorize(np.asarray(groups), sort=True)[0].astype(np.int64)


def extract_video_groups(video_ids):
    return np.arange(len(video_ids), dtype=np.int64)


def extract_direction_groups(video_ids):
    groups = []
    for video_id in video_ids:
        match = re.search(r"_dir(\d+)_", video_id)
        if match is None:
            raise ValueError(f"Cannot parse direction index from {video_id}")
        groups.append(int(match.group(1)))
    return np.asarray(groups, dtype=np.int64)


def extract_magnitude_groups(video_ids):
    groups = []
    for video_id in video_ids:
        if video_id.startswith("vel_"):
            match = re.search(r"_spd(\d+)_", video_id)
        elif video_id.startswith("acc_"):
            match = re.search(r"_acc(\d+)_", video_id)
        else:
            match = None
        if match is None:
            raise ValueError(f"Cannot parse magnitude index from {video_id}")
        groups.append(int(match.group(1)))
    return np.asarray(groups, dtype=np.int64)


def extract_pixel_region_groups(pos_x_px, pos_y_px):
    points = np.stack([pos_x_px, pos_y_px], axis=1).astype(np.float32)
    prototypes = np.asarray(
        [
            [64.0, 64.0],
            [192.0, 64.0],
            [64.0, 192.0],
            [192.0, 192.0],
            [128.0, 128.0],
        ],
        dtype=np.float32,
    )
    distances = np.square(points[:, None, :] - prototypes[None, :, :]).sum(axis=2)
    return distances.argmin(axis=1).astype(np.int64)


def extract_spatial_sector_groups(pos_x_px, pos_y_px):
    angles = np.arctan2(pos_y_px - 128.0, pos_x_px - 128.0)
    sectors = np.floor(((angles + np.pi) / (2.0 * np.pi)) * CV_SPLITS).astype(np.int64)
    return np.clip(sectors, 0, CV_SPLITS - 1)


def extract_spatial_cluster_groups(pos_x_px, pos_y_px):
    points = np.stack([pos_x_px, pos_y_px], axis=1).astype(np.float32)
    model = KMeans(n_clusters=CV_SPLITS, random_state=CV_RANDOM_SEED, n_init=20)
    labels = model.fit_predict(points)
    return labels.astype(np.int64)


def build_groups(video_ids, grouping: str, pos_x_px=None, pos_y_px=None):
    if grouping == "position":
        return extract_position_groups(video_ids)
    if grouping == "condition":
        return extract_condition_groups(video_ids)
    if grouping == "video":
        return extract_video_groups(video_ids)
    if grouping == "direction":
        return extract_direction_groups(video_ids)
    if grouping == "magnitude":
        return extract_magnitude_groups(video_ids)
    if grouping == "pixel_region":
        return extract_pixel_region_groups(pos_x_px, pos_y_px)
    if grouping == "spatial_sector":
        return extract_spatial_sector_groups(pos_x_px, pos_y_px)
    if grouping == "spatial_cluster":
        return extract_spatial_cluster_groups(pos_x_px, pos_y_px)
    if grouping == "direction_spatial_sector":
        direction = extract_direction_groups(video_ids)
        sector = extract_spatial_sector_groups(pos_x_px, pos_y_px)
        return pd.factorize(list(zip(direction.tolist(), sector.tolist())), sort=True)[0].astype(np.int64)
    if grouping == "magnitude_spatial_sector":
        magnitude = extract_magnitude_groups(video_ids)
        sector = extract_spatial_sector_groups(pos_x_px, pos_y_px)
        return pd.factorize(list(zip(magnitude.tolist(), sector.tolist())), sort=True)[0].astype(np.int64)
    raise ValueError(f"Unknown grouping: {grouping}")


def compute_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]

    ss_res = np.square(y_pred - y_true).sum()
    ss_tot = np.square(y_true - y_true.mean(axis=0, keepdims=True)).sum()
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def fit_probe(X_train, y_train, X_val, y_val, alpha):
    probe = Ridge(alpha=alpha, fit_intercept=True)
    probe.fit(X_train, y_train)
    pred_val = probe.predict(X_val)
    return compute_r2(y_val, pred_val)


def normalize_train_val(train, val, mode: str):
    mean = train.mean(axis=0, keepdims=True)
    if mode == "none":
        return train, val, np.zeros_like(mean), np.ones_like(mean)
    if mode == "center":
        return train - mean, val - mean, mean, np.ones_like(mean)
    if mode == "zscore":
        std = train.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        return (train - mean) / std, (val - mean) / std, mean, std
    raise ValueError(f"Unknown norm mode: {mode}")


def fit_trainable_probe_single(
    X_train,
    y_train,
    X_val,
    y_val,
    output_dim,
    lr,
    weight_decay,
    device,
    max_epochs,
    patience_limit,
    norm_mode="zscore",
):
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)

    if y_train.ndim == 1:
        y_train = y_train[:, None]
        y_val = y_val[:, None]

    X_train_std, X_val_std, _, _ = normalize_train_val(X_train, X_val, norm_mode)
    y_train_std, y_val_std, y_mean, y_std = normalize_train_val(y_train, y_val, norm_mode)

    X_tr = torch.tensor(X_train_std, dtype=torch.float32, device=device)
    X_va = torch.tensor(X_val_std, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train_std, dtype=torch.float32, device=device)
    y_va = torch.tensor(y_val_std, dtype=torch.float32, device=device)

    torch.manual_seed(CV_RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CV_RANDOM_SEED)

    probe = torch.nn.Linear(X_tr.shape[1], output_dim, bias=True).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    best_val_loss = float("inf")
    best_pred = None
    patience = 0

    for _ in range(max_epochs):
        probe.train()
        pred = probe(X_tr)
        loss = criterion(pred, y_tr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_pred = probe(X_va)
            val_loss = criterion(val_pred, y_va).item()

        if val_loss + 1e-8 < best_val_loss:
            best_val_loss = val_loss
            best_pred = val_pred.detach().cpu().numpy()
            patience = 0
        else:
            patience += 1
            if patience >= patience_limit:
                break

    if best_pred is None:
        with torch.no_grad():
            best_pred = probe(X_va).detach().cpu().numpy()

    pred_val = best_pred * y_std + y_mean
    return compute_r2(y_val, pred_val)


def fit_trainable_probe_batched(
    X_train,
    y_train,
    X_val,
    y_val,
    output_dim,
    lr_grid,
    wd_grid,
    device,
    max_epochs=MAX_EPOCHS,
    patience_limit=PATIENCE,
    norm_mode="zscore",
):
    configs = list(itertools.product(lr_grid, wd_grid))
    n_configs = len(configs)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)

    if y_train.ndim == 1:
        y_train = y_train[:, None]
        y_val = y_val[:, None]

    X_train_std, X_val_std, _, _ = normalize_train_val(X_train, X_val, norm_mode)
    y_train_std, y_val_std, y_mean, y_std = normalize_train_val(y_train, y_val, norm_mode)

    X_tr = torch.tensor(X_train_std, dtype=torch.float32, device=device)
    X_va = torch.tensor(X_val_std, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train_std, dtype=torch.float32, device=device)
    y_va = torch.tensor(y_val_std, dtype=torch.float32, device=device)

    input_dim = X_tr.shape[1]
    output_dim = int(output_dim)

    torch.manual_seed(CV_RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CV_RANDOM_SEED)
    template = torch.nn.Linear(input_dim, output_dim, bias=True)
    W = template.weight.data.T.contiguous().unsqueeze(0).expand(n_configs, input_dim, output_dim)
    b = template.bias.data.clone().unsqueeze(0).expand(n_configs, output_dim)
    W = W.contiguous().to(device)
    b = b.contiguous().to(device)
    W.requires_grad_(True)
    b.requires_grad_(True)

    m_W = torch.zeros_like(W)
    v_W = torch.zeros_like(W)
    m_b = torch.zeros_like(b)
    v_b = torch.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    lrs = torch.tensor([cfg[0] for cfg in configs], dtype=torch.float32, device=device)
    wds = torch.tensor([cfg[1] for cfg in configs], dtype=torch.float32, device=device)

    best_val_loss = torch.full((n_configs,), float("inf"), device=device)
    best_W = W.detach().clone()
    best_b = b.detach().clone()
    patience = torch.zeros(n_configs, dtype=torch.int32, device=device)
    active = torch.ones(n_configs, dtype=torch.bool, device=device)

    for step in range(1, max_epochs + 1):
        pred_tr = torch.einsum("nd,cdo->cno", X_tr, W) + b.unsqueeze(1)
        loss_per_cfg = ((pred_tr - y_tr.unsqueeze(0)) ** 2).mean(dim=(1, 2))
        total_loss = (loss_per_cfg * active.float()).sum()
        total_loss.backward()

        with torch.no_grad():
            W.grad.add_(W * wds[:, None, None])
            b.grad.add_(b * wds[:, None])

            m_W.mul_(beta1).add_(W.grad, alpha=1 - beta1)
            v_W.mul_(beta2).addcmul_(W.grad, W.grad, value=1 - beta2)
            m_b.mul_(beta1).add_(b.grad, alpha=1 - beta1)
            v_b.mul_(beta2).addcmul_(b.grad, b.grad, value=1 - beta2)

            bc1 = 1 - beta1**step
            bc2 = 1 - beta2**step
            m_W_hat = m_W / bc1
            v_W_hat = v_W / bc2
            m_b_hat = m_b / bc1
            v_b_hat = v_b / bc2

            active_mask_W = active.float()[:, None, None]
            active_mask_b = active.float()[:, None]

            W.data.sub_(active_mask_W * lrs[:, None, None] * m_W_hat / (v_W_hat.sqrt() + eps))
            b.data.sub_(active_mask_b * lrs[:, None] * m_b_hat / (v_b_hat.sqrt() + eps))
            W.grad.zero_()
            b.grad.zero_()

        with torch.no_grad():
            pred_va = torch.einsum("nd,cdo->cno", X_va, W) + b.unsqueeze(1)
            val_loss = ((pred_va - y_va.unsqueeze(0)) ** 2).mean(dim=(1, 2))
            improved = val_loss + 1e-8 < best_val_loss
            best_val_loss = torch.where(improved, val_loss, best_val_loss)
            best_W = torch.where(improved[:, None, None], W.detach(), best_W)
            best_b = torch.where(improved[:, None], b.detach(), best_b)
            patience = torch.where(improved, torch.zeros_like(patience), patience + 1)
            active = active & (patience < patience_limit)
            if not active.any():
                break

    with torch.no_grad():
        pred_va_best = torch.einsum("nd,cdo->cno", X_va, best_W) + best_b.unsqueeze(1)
    pred_va_unscaled = pred_va_best.cpu().numpy() * y_std[None, :, :] + y_mean[None, :, :]

    results = []
    for cfg_index, (lr, wd) in enumerate(configs):
        results.append((float(lr), float(wd), compute_r2(y_val, pred_va_unscaled[cfg_index])))
    return results


def evaluate_layer(features, targets, groups, output_dim, device, solver, norm_mode):
    splitter = GroupKFold(n_splits=min(CV_SPLITS, int(np.unique(groups).size)))
    fold_scores = []
    fold_best_lrs = []
    fold_best_wds = []

    for train_idx, val_idx in splitter.split(features, targets, groups):
        if solver == "trainable":
            cfg_results = fit_trainable_probe_batched(
                features[train_idx],
                targets[train_idx],
                features[val_idx],
                targets[val_idx],
                output_dim=output_dim,
                lr_grid=LR_GRID,
                wd_grid=WD_GRID,
                device=device,
                norm_mode=norm_mode,
            )
            best_lr, best_wd, best_r2 = max(cfg_results, key=lambda item: item[2])
            fold_scores.append(best_r2)
            fold_best_lrs.append(best_lr)
            fold_best_wds.append(best_wd)
        elif solver == "adamw100":
            score = fit_trainable_probe_single(
                features[train_idx],
                targets[train_idx],
                features[val_idx],
                targets[val_idx],
                output_dim=output_dim,
                lr=ADAMW100_LR,
                weight_decay=ADAMW100_WD,
                device=device,
                max_epochs=ADAMW100_EPOCHS,
                patience_limit=ADAMW100_PATIENCE,
                norm_mode=norm_mode,
            )
            fold_scores.append(score)
            fold_best_lrs.append(ADAMW100_LR)
            fold_best_wds.append(ADAMW100_WD)
        elif solver == "ridge":
            best_score = -np.inf
            best_wd = None
            for weight_decay in WD_GRID:
                score = fit_probe(
                    features[train_idx],
                    targets[train_idx],
                    features[val_idx],
                    targets[val_idx],
                    alpha=weight_decay,
                )
                if score > best_score:
                    best_score = score
                    best_wd = weight_decay
            fold_scores.append(float(best_score))
            fold_best_lrs.append(0.0)
            fold_best_wds.append(float(best_wd))
        else:
            raise ValueError(f"Unknown solver: {solver}")

    return {
        "r2_mean": float(np.mean(fold_scores)),
        "r2_std": float(np.std(fold_scores)),
        "best_lr_mode": float(pd.Series(fold_best_lrs).mode().iloc[0]),
        "best_wd_mode": float(pd.Series(fold_best_wds).mode().iloc[0]),
    }


def load_layer(feature_root: str, dataset_name: str, layer_index: int):
    path = os.path.join(feature_root, MODEL_NAME, dataset_name, f"layer_{layer_index:02d}.npy")
    return np.load(path)


def run_single_config(args):
    output_csv = os.path.join(RESULTS_ROOT, f"results_{args.run_name}.csv")
    output_json = os.path.join(RESULTS_ROOT, f"config_{args.run_name}.json")
    output_png = os.path.join(RESULTS_ROOT, f"figure_{args.run_name}.png")

    targets = load_targets(args.direction_target)
    rows = []
    variable_names = PROBE_SETS[args.probe_set]

    for variable_name in variable_names:
        spec = targets[variable_name]
        groups = build_groups(
            spec["video_ids"],
            args.grouping,
            pos_x_px=spec["pos_x_px"],
            pos_y_px=spec["pos_y_px"],
        )
        for layer in range(DEPTH):
            features = load_layer(args.feature_root, spec["dataset"], layer)
            result = evaluate_layer(
                features=features,
                targets=spec["target"],
                groups=groups,
                output_dim=spec["output_dim"],
                device=args.device,
                solver=args.solver,
                norm_mode=args.norm_mode,
            )
            row = {
                "run_name": args.run_name,
                "variable": variable_name,
                "layer": layer,
                "layer_fraction": layer / (DEPTH - 1),
                "r2_mean": result["r2_mean"],
                "r2_std": result["r2_std"],
                "best_lr_mode": result["best_lr_mode"],
                "best_wd_mode": result["best_wd_mode"],
                "grouping": args.grouping,
                "direction_target": args.direction_target,
                "residual_capture": args.residual_capture,
                "preprocessing": args.preprocessing,
                "solver": args.solver,
                "probe_set": args.probe_set,
                "norm_mode": args.norm_mode,
                "feature_root": args.feature_root,
            }
            rows.append(row)
            if layer in (0, 7, 8, 12, 16, 23):
                print(
                    f"{args.run_name} | {variable_name:12s} | L{layer:02d} "
                    f"R^2={result['r2_mean']:.4f} ± {result['r2_std']:.4f}"
                )

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    config = {
        "run_name": args.run_name,
        "feature_root": args.feature_root,
        "grouping": args.grouping,
        "direction_target": args.direction_target,
        "residual_capture": args.residual_capture,
        "preprocessing": args.preprocessing,
        "solver": args.solver,
        "probe_set": args.probe_set,
        "norm_mode": args.norm_mode,
        "cv_splits": CV_SPLITS,
        "model": MODEL_NAME,
        "n_layers": DEPTH,
        "lr_grid": LR_GRID,
        "wd_grid": WD_GRID,
    }
    Path(output_json).write_text(json.dumps(config, indent=2))

    plot_run(df, output_png, title=args.run_name)
    return df


def plot_run(df: pd.DataFrame, output_png: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for variable_name in df["variable"].unique():
        subset = df[df["variable"] == variable_name].sort_values("layer")
        x = subset["layer_fraction"].to_numpy()
        mean = subset["r2_mean"].to_numpy()
        std = subset["r2_std"].to_numpy()
        ax.plot(
            x,
            mean,
            color=PLOT_COLORS[variable_name],
            linewidth=2,
            label=PLOT_LABELS[variable_name],
        )
        ax.fill_between(x, mean - std, mean + std, color=PLOT_COLORS[variable_name], alpha=0.15)

    ax.axvline(8 / (DEPTH - 1), color="gray", linestyle="--", linewidth=1.2, label="Layer 8")
    ax.set_xlabel("Layer Fraction")
    ax.set_ylabel("Validation R$^2$")
    probe_set = df["probe_set"].iloc[0] if "probe_set" in df.columns else "fig2c"
    title_prefix = "V-JEPA v2-L: Polar" if probe_set == "fig2c" else "V-JEPA v2-L: Cartesian"
    ax.set_title(f"{title_prefix} | {title}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(min(-0.1, float(df["r2_mean"].min()) - 0.05), 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def first_ge_threshold(values, threshold):
    for index, value in enumerate(values):
        if value >= threshold:
            return index
    return math.inf


def summarize_runs(args):
    csv_paths = sorted(Path(RESULTS_ROOT).glob("results_*.csv"))
    csv_paths = [
        path
        for path in csv_paths
        if path.name not in {"ablation_summary.csv"} and not path.name.startswith("results_overlay")
    ]
    if not csv_paths:
        raise FileNotFoundError("No run CSVs found under artifacts/results")

    run_frames = [pd.read_csv(path) for path in csv_paths]
    all_df = pd.concat(run_frames, ignore_index=True)

    summary_rows = []
    for run_name, run_df in all_df.groupby("run_name"):
        if "direction" not in set(run_df["variable"].unique()):
            continue
        direction_df = run_df[run_df["variable"] == "direction"].sort_values("layer")
        direction_curve = direction_df["r2_mean"].to_numpy()
        peak_idx = int(direction_df.iloc[np.argmax(direction_curve)]["layer"])
        peak_val = float(np.max(direction_curve))
        first_08 = first_ge_threshold(direction_curve, 0.8)
        summary_rows.append(
            {
                "run_name": run_name,
                "grouping": direction_df["grouping"].iloc[0],
                "direction_target": direction_df["direction_target"].iloc[0],
                "residual_capture": direction_df["residual_capture"].iloc[0],
                "preprocessing": direction_df["preprocessing"].iloc[0],
                "solver": direction_df["solver"].iloc[0] if "solver" in direction_df.columns else "trainable",
                "probe_set": direction_df["probe_set"].iloc[0] if "probe_set" in direction_df.columns else "fig2c",
                "l0_direction_r2": float(direction_curve[0]),
                "l1_l6_direction_mean": float(direction_curve[1:7].mean()),
                "direction_first_ge_0p8_layer": first_08,
                "peak_direction_r2": peak_val,
                "peak_layer_index": peak_idx,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df["paper_distance"] = (
        (summary_df["direction_first_ge_0p8_layer"].replace(math.inf, 999) - 8).abs()
        + (summary_df["peak_layer_index"] - 8).abs()
        + summary_df["l0_direction_r2"].abs() * 5.0
        + summary_df["l1_l6_direction_mean"].abs()
    )
    summary_df = summary_df.sort_values(
        ["paper_distance", "peak_direction_r2"], ascending=[True, False]
    ).reset_index(drop=True)
    summary_df["paper_like_rank"] = np.arange(1, len(summary_df) + 1)
    summary_df.to_csv(os.path.join(RESULTS_ROOT, "ablation_summary.csv"), index=False)

    plot_overlay(all_df, os.path.join(RESULTS_ROOT, "figure2c_ablation_overlay.png"))

    best_row = summary_df.iloc[0].to_dict()
    Path(os.path.join(RESULTS_ROOT, "best_config.json")).write_text(json.dumps(best_row, indent=2))
    print(json.dumps(best_row, indent=2))


def summarize_fig2b_runs(args):
    csv_paths = sorted(Path(RESULTS_ROOT).glob("results_fig2b_*.csv"))
    if not csv_paths:
        raise FileNotFoundError("No fig2b CSVs found under artifacts/results")

    run_frames = [pd.read_csv(path) for path in csv_paths]
    all_df = pd.concat(run_frames, ignore_index=True)
    rows = []

    for run_name, run_df in all_df.groupby("run_name"):
        variable = run_df["variable"].iloc[0]
        curve_df = run_df.sort_values("layer").reset_index(drop=True)
        curve = curve_df["r2_mean"].to_numpy()
        first_08 = first_ge_threshold(curve, 0.8)
        peak_idx = int(curve_df.iloc[np.argmax(curve)]["layer"])
        peak_val = float(np.max(curve))
        rows.append(
            {
                "run_name": run_name,
                "variable": variable,
                "grouping": curve_df["grouping"].iloc[0],
                "residual_capture": curve_df["residual_capture"].iloc[0],
                "preprocessing": curve_df["preprocessing"].iloc[0],
                "solver": curve_df["solver"].iloc[0] if "solver" in curve_df.columns else "trainable",
                "l0_r2": float(curve_df[curve_df["layer"] == 0]["r2_mean"].iloc[0]),
                "l8_r2": float(curve_df[curve_df["layer"] == 8]["r2_mean"].iloc[0]),
                "peak_r2": peak_val,
                "peak_layer": peak_idx,
                "first_ge_0p8_layer": first_08,
                "late_layer_r2": float(curve_df[curve_df["layer"] == 23]["r2_mean"].iloc[0]),
            }
        )

    summary_df = pd.DataFrame(rows).sort_values(["variable", "grouping"]).reset_index(drop=True)
    summary_path = os.path.join(RESULTS_ROOT, "fig2b_ablation_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    plot_fig2b_overlay(all_df, os.path.join(RESULTS_ROOT, "fig2b_overlay.png"))
    print(summary_df.to_string(index=False))


def plot_fig2b_overlay(all_df: pd.DataFrame, output_png: str):
    grouping_to_color = {
        "pixel_region": "#1f77b4",
        "spatial_sector": "#d62728",
        "spatial_cluster": "#2ca02c",
    }
    variable_order = ["velocity_xy", "acceleration_xy"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True, sharey=True)

    for ax, variable_name in zip(axes, variable_order):
        subset_var = all_df[all_df["variable"] == variable_name]
        for grouping in ["pixel_region", "spatial_sector", "spatial_cluster"]:
            subset = subset_var[subset_var["grouping"] == grouping].sort_values("layer")
            ax.plot(
                subset["layer_fraction"],
                subset["r2_mean"],
                linewidth=2,
                color=grouping_to_color[grouping],
                label=grouping,
            )
            ax.fill_between(
                subset["layer_fraction"],
                subset["r2_mean"] - subset["r2_std"],
                subset["r2_mean"] + subset["r2_std"],
                color=grouping_to_color[grouping],
                alpha=0.12,
            )
        ax.axvline(8 / (DEPTH - 1), color="gray", linestyle="--", linewidth=1.2, label="Layer 8")
        ax.set_title(PLOT_LABELS[variable_name])
        ax.set_xlabel("Layer Fraction")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Validation R$^2$")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    fig.suptitle("Figure 2(b) Cartesian Ablations", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(all_df: pd.DataFrame, output_png: str):
    colors = plt.cm.tab10(np.linspace(0, 1, all_df["run_name"].nunique()))
    run_to_color = {
        run_name: colors[index]
        for index, run_name in enumerate(sorted(all_df["run_name"].unique()))
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), sharex=True, sharey=True)
    for ax, variable_name in zip(axes, ("speed", "direction", "acceleration")):
        for run_name in sorted(all_df["run_name"].unique()):
            subset = all_df[
                (all_df["run_name"] == run_name) & (all_df["variable"] == variable_name)
            ].sort_values("layer")
            ax.plot(
                subset["layer_fraction"],
                subset["r2_mean"],
                linewidth=2,
                color=run_to_color[run_name],
                alpha=0.95,
                label=run_name,
            )
        ax.axvline(8 / (DEPTH - 1), color="gray", linestyle="--", linewidth=1.2)
        ax.set_title(variable_name.capitalize())
        ax.set_xlabel("Layer Fraction")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Validation R$^2$")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    fig.suptitle("Figure 2(c) Ablations Overlay", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--run-name", required=True)
    run_parser.add_argument("--feature-root", required=True)
    run_parser.add_argument(
        "--probe-set",
        choices=[
            "fig2c",
            "fig2b_velocity_xy",
            "fig2b_acceleration_xy",
            "fig2b_velocity_axes",
            "fig2b_acceleration_axes",
        ],
        default="fig2c",
    )
    run_parser.add_argument("--solver", choices=["trainable", "adamw100", "ridge"], default="trainable")
    run_parser.add_argument("--norm-mode", choices=["zscore", "center", "none"], default="zscore")
    run_parser.add_argument(
        "--grouping",
        choices=[
            "position",
            "condition",
            "video",
            "direction",
            "magnitude",
            "pixel_region",
            "spatial_sector",
            "spatial_cluster",
            "direction_spatial_sector",
            "magnitude_spatial_sector",
        ],
        required=True,
    )
    run_parser.add_argument("--direction-target", choices=["sincos", "angle", "vxy"], required=True)
    run_parser.add_argument("--residual-capture", choices=["resid_pre", "resid_post"], required=True)
    run_parser.add_argument("--preprocessing", choices=["resize", "eval_preproc"], required=True)
    run_parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--device", default="cpu")
    subparsers.add_parser("summarize_fig2b")

    args = parser.parse_args()
    Path(RESULTS_ROOT).mkdir(parents=True, exist_ok=True)

    if args.command == "run":
        run_single_config(args)
    elif args.command == "summarize":
        summarize_runs(args)
    elif args.command == "summarize_fig2b":
        summarize_fig2b_runs(args)


if __name__ == "__main__":
    main()
