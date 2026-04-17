"""Step 3: Linear probing + PEZ Figure 2c generation.

Trains layer-wise linear probes on V-JEPA 2 features to predict physics variables
(speed, direction, acceleration), then generates the PEZ reproduction figures.

Usage:
    /isaac-sim/python.sh /home/solee/pez/step3_probe.py
"""

import json
import os
import sys
import time
import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold

sys.path.insert(0, "/home/solee/pez")
from constants import DATA_ROOT, FEATURES_ROOT, RESULTS_ROOT

# Paper Appendix B probe protocol.
# Appendix B specifies a grouped 5-fold CV protocol with hyperparameter sweeps over
# learning rate and weight decay, but does not specify the optimizer or epoch count.
# We therefore use exact ridge regression for the linear probes while preserving the
# grouped CV and regularization sweep, avoiding the underfit seen with the earlier
# C.11-style fixed Adam implementation.
CV_N_SPLITS = 5
CV_RANDOM_SEED = 42
APPENDIX_B_LR_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 5e-3]
APPENDIX_B_WD_GRID = [0.01, 0.1, 0.4, 0.8]
TRAINABLE_MAX_EPOCHS = 400
TRAINABLE_PATIENCE = 40

# Phase 2 "weak probe" grids — closer to paper's unspecified minimal linear probe.
RIDGE_WEAK_ALPHA_GRID = [1.0, 10.0, 100.0]
ADAMW100_LR = 1e-3
ADAMW100_WD = 0.1
ADAMW100_MAX_EPOCHS = 100
ADAMW100_PATIENCE = 10

PROBE_SETS = {
    "polar": [
        ("speed", 1, "velocity", "speed"),
        ("direction", 2, "velocity", "direction_sincos"),
        ("acceleration", 1, "acceleration", "accel_magnitude"),
    ],
    "cartesian": [
        ("velocity_xy", 2, "velocity", "velocity_xy"),
        ("acceleration_xy", 2, "acceleration", "accel_xy"),
    ],
}

MODEL_CONFIGS = {
    "vjepa2_L": {"depth": 24, "embed_dim": 1024, "display": "V-JEPA 2-L"},
    "vjepa2_G": {"depth": 40, "embed_dim": 1408, "display": "V-JEPA 2-G"},
}


def extract_position_groups(video_ids):
    """Group videos by their sampled start-position index."""
    groups = []
    for video_id in video_ids:
        match = re.search(r"_pos(\d+)$", video_id)
        if match is None:
            groups.append(len(groups))
        else:
            groups.append(int(match.group(1)))
    return np.asarray(groups, dtype=np.int64)


def extract_condition_groups(video_ids):
    """Group videos by motion condition, ignoring start-position variation."""
    groups = [re.sub(r"_pos\d+$", "", video_id) for video_id in video_ids]
    return pd.factorize(np.asarray(groups), sort=True)[0].astype(np.int64)


def extract_direction_groups(video_ids):
    """Group videos by direction index only.

    Used for leave-one-direction-out CV: with 8 directions, 8-fold GroupKFold
    puts every video of one angle entirely into the val fold so the probe
    cannot memorize the angles it has seen during training.
    """
    groups = []
    for video_id in video_ids:
        match = re.search(r"_dir(\d+)_", video_id)
        if match is None:
            raise ValueError(f"Cannot extract direction from video_id: {video_id}")
        groups.append(int(match.group(1)))
    return np.asarray(groups, dtype=np.int64)


def extract_video_groups(video_ids):
    """Treat each video as its own group.

    This is the minimum-assumption 5-fold setting when the paper specifies
    grouped CV but does not reveal the grouping key.
    """
    return np.arange(len(video_ids), dtype=np.int64)


def extract_groups(video_ids, grouping):
    if grouping == "position":
        return extract_position_groups(video_ids)
    if grouping == "condition":
        return extract_condition_groups(video_ids)
    if grouping == "direction":
        return extract_direction_groups(video_ids)
    if grouping == "video":
        return extract_video_groups(video_ids)
    if grouping == "video_shuffled":
        return extract_video_groups(video_ids)
    raise ValueError(f"Unknown grouping: {grouping}")


def result_csv_path(model_name, probe_set, suffix=""):
    sfx = f"_{suffix}" if suffix else ""
    if probe_set == "polar":
        return os.path.join(RESULTS_ROOT, f"results_{model_name}{sfx}.csv")
    return os.path.join(RESULTS_ROOT, f"results_{probe_set}_{model_name}{sfx}.csv")


def probing_config_path(probe_set):
    if probe_set == "polar":
        return os.path.join(RESULTS_ROOT, "probing_config.json")
    return os.path.join(RESULTS_ROOT, f"probing_config_{probe_set}.json")


def load_targets():
    """Load per-video ground truth targets."""
    # Velocity dataset
    vel_df = pd.read_parquet(os.path.join(DATA_ROOT, "velocity", "gt_velocity.parquet"))
    vel_per_video = vel_df[vel_df["frame_idx"] == 0].sort_values("video_id").reset_index(drop=True)

    speed_targets = vel_per_video["speed"].values.astype(np.float32)
    direction_rad = vel_per_video["direction_rad"].values.astype(np.float32)
    direction_sincos = np.stack([
        np.sin(direction_rad), np.cos(direction_rad)
    ], axis=1).astype(np.float32)
    velocity_xy = np.stack([
        vel_per_video["vx_px"].values.astype(np.float32),
        vel_per_video["vy_px"].values.astype(np.float32),
    ], axis=1).astype(np.float32)

    # Acceleration dataset
    acc_df = pd.read_parquet(os.path.join(DATA_ROOT, "acceleration", "gt_acceleration.parquet"))
    acc_per_video = acc_df[acc_df["frame_idx"] == 0].sort_values("video_id").reset_index(drop=True)
    accel_targets = acc_per_video["accel_magnitude"].values.astype(np.float32)
    accel_xy = np.stack([
        acc_per_video["ax_px"].values.astype(np.float32),
        acc_per_video["ay_px"].values.astype(np.float32),
    ], axis=1).astype(np.float32)

    targets = {
        "velocity": {
            "speed": speed_targets,
            "direction_sincos": direction_sincos,
            "velocity_xy": velocity_xy,
            "video_ids": vel_per_video["video_id"].tolist(),
        },
        "acceleration": {
            "accel_magnitude": accel_targets,
            "accel_xy": accel_xy,
            "video_ids": acc_per_video["video_id"].tolist(),
        },
    }

    print(f"Targets loaded: speed ({len(speed_targets)}), "
          f"direction ({direction_sincos.shape}), "
          f"velocity_xy ({velocity_xy.shape}), "
          f"acceleration ({len(accel_targets)})")
    return targets


def compute_r2(y_true, y_pred):
    """Compute variance-weighted multi-output R^2."""
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
    return 1.0 - ss_res / ss_tot


def fit_probe(X_train, y_train, X_val, y_val, alpha):
    """Fit a single linear ridge probe and return validation R^2."""
    probe = Ridge(alpha=alpha, fit_intercept=True)
    probe.fit(X_train, y_train)
    pred_val = probe.predict(X_val)
    return compute_r2(y_val, pred_val)


def fit_trainable_probe(
    X_train, y_train, X_val, y_val, output_dim, lr, weight_decay, device,
    max_epochs=None, patience_limit=None,
):
    """Fit a trainable linear probe with standardized inputs/targets."""
    if max_epochs is None:
        max_epochs = TRAINABLE_MAX_EPOCHS
    if patience_limit is None:
        patience_limit = TRAINABLE_PATIENCE
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)

    if y_train.ndim == 1:
        y_train = y_train[:, None]
        y_val = y_val[:, None]

    x_mean = X_train.mean(axis=0, keepdims=True)
    x_std = X_train.std(axis=0, keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    X_train_std = (X_train - x_mean) / x_std
    X_val_std = (X_val - x_mean) / x_std

    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True)
    y_std[y_std < 1e-6] = 1.0
    y_train_std = (y_train - y_mean) / y_std
    y_val_std = (y_val - y_mean) / y_std

    X_tr = torch.tensor(X_train_std, dtype=torch.float32, device=device)
    X_va = torch.tensor(X_val_std, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train_std, dtype=torch.float32, device=device)
    y_va = torch.tensor(y_val_std, dtype=torch.float32, device=device)

    torch.manual_seed(CV_RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CV_RANDOM_SEED)

    probe = torch.nn.Linear(X_tr.shape[1], output_dim, bias=True).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    best_state = None
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
            best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
            best_pred = val_pred.detach().cpu().numpy()
            patience = 0
        else:
            patience += 1
            if patience >= patience_limit:
                break

    if best_state is None:
        with torch.no_grad():
            best_pred = probe(X_va).detach().cpu().numpy()
    else:
        probe.load_state_dict(best_state)

    pred_val = best_pred * y_std + y_mean
    return compute_r2(y_val, pred_val)


def fit_trainable_probe_batched(
    X_train, y_train, X_val, y_val, output_dim,
    lr_grid, wd_grid, device,
    max_epochs=TRAINABLE_MAX_EPOCHS, patience_limit=TRAINABLE_PATIENCE,
):
    """Fit ALL (lr, wd) configs simultaneously as a batched linear layer.

    Mathematically equivalent to calling fit_trainable_probe() once per config,
    but ~20x faster because a single forward/backward pass updates all configs
    in parallel (each config has its own weight slice + its own Adam state +
    its own LR/WD).

    Returns:
        List[Tuple[float, float, float]]: (lr, wd, val_r2) for each config
        in the same order as itertools.product(lr_grid, wd_grid).
    """
    import itertools

    configs = list(itertools.product(lr_grid, wd_grid))
    n_configs = len(configs)

    # --- Preprocess (same standardization as single version) ---
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)

    if y_train.ndim == 1:
        y_train = y_train[:, None]
        y_val = y_val[:, None]

    x_mean = X_train.mean(axis=0, keepdims=True)
    x_std = X_train.std(axis=0, keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    X_train_std = (X_train - x_mean) / x_std
    X_val_std = (X_val - x_mean) / x_std

    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True)
    y_std[y_std < 1e-6] = 1.0
    y_train_std = (y_train - y_mean) / y_std
    y_val_std = (y_val - y_mean) / y_std

    X_tr = torch.tensor(X_train_std, dtype=torch.float32, device=device)
    X_va = torch.tensor(X_val_std, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train_std, dtype=torch.float32, device=device)
    y_va = torch.tensor(y_val_std, dtype=torch.float32, device=device)

    D = X_tr.shape[1]
    O = output_dim

    # --- Init all configs with identical weights (match single version) ---
    # Single fit_trainable_probe calls torch.manual_seed(CV_RANDOM_SEED) then
    # creates nn.Linear(D, O), so every config starts from the same (W, b).
    torch.manual_seed(CV_RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CV_RANDOM_SEED)
    template = torch.nn.Linear(D, O, bias=True)
    # nn.Linear weight shape: (O, D). We want W of shape (n_configs, D, O)
    W_init = template.weight.data.T.contiguous()  # (D, O)
    b_init = template.bias.data.clone()           # (O,)

    W = W_init.unsqueeze(0).expand(n_configs, D, O).contiguous().to(device)
    b = b_init.unsqueeze(0).expand(n_configs, O).contiguous().to(device)
    W.requires_grad_(True)
    b.requires_grad_(True)

    # --- Per-config Adam state ---
    m_W = torch.zeros_like(W)
    v_W = torch.zeros_like(W)
    m_b = torch.zeros_like(b)
    v_b = torch.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    lrs = torch.tensor([c[0] for c in configs], dtype=torch.float32, device=device)  # (C,)
    wds = torch.tensor([c[1] for c in configs], dtype=torch.float32, device=device)  # (C,)

    # --- Per-config early-stopping state ---
    best_val_loss = torch.full((n_configs,), float("inf"), device=device)
    best_W = W.detach().clone()
    best_b = b.detach().clone()
    patience = torch.zeros(n_configs, dtype=torch.int32, device=device)
    active = torch.ones(n_configs, dtype=torch.bool, device=device)

    for step in range(1, max_epochs + 1):
        # Forward (all configs in parallel): (N, D) x (C, D, O) -> (C, N, O)
        pred_tr = torch.einsum("nd,cdo->cno", X_tr, W) + b.unsqueeze(1)
        # Per-config MSE
        loss_per_cfg = ((pred_tr - y_tr.unsqueeze(0)) ** 2).mean(dim=(1, 2))  # (C,)

        # Sum ONLY active configs (inactive get zero grad contribution)
        total_loss = (loss_per_cfg * active.float()).sum()
        total_loss.backward()

        with torch.no_grad():
            # Adam update with per-config LR + coupled L2 weight decay on W AND b.
            # Matches torch.optim.Adam(weight_decay=wd) which applies WD to every
            # parameter (weight_decay * param added to gradient before Adam step).
            W.grad.add_(W * wds[:, None, None])
            b.grad.add_(b * wds[:, None])

            # Adam step (per-config)
            m_W.mul_(beta1).add_(W.grad, alpha=1 - beta1)
            v_W.mul_(beta2).addcmul_(W.grad, W.grad, value=1 - beta2)
            m_b.mul_(beta1).add_(b.grad, alpha=1 - beta1)
            v_b.mul_(beta2).addcmul_(b.grad, b.grad, value=1 - beta2)

            bc1 = 1 - beta1 ** step
            bc2 = 1 - beta2 ** step
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

        # Val loss per config
        with torch.no_grad():
            pred_va = torch.einsum("nd,cdo->cno", X_va, W) + b.unsqueeze(1)
            val_loss_per_cfg = ((pred_va - y_va.unsqueeze(0)) ** 2).mean(dim=(1, 2))

            improved = val_loss_per_cfg + 1e-8 < best_val_loss
            best_val_loss = torch.where(improved, val_loss_per_cfg, best_val_loss)
            best_W = torch.where(
                improved[:, None, None].expand_as(W), W.detach(), best_W
            )
            best_b = torch.where(
                improved[:, None].expand_as(b), b.detach(), best_b
            )

            patience = torch.where(
                improved, torch.zeros_like(patience), patience + 1
            )
            active = active & (patience < patience_limit)
            if not active.any():
                break

    # --- Final val R² per config using best_W, best_b ---
    with torch.no_grad():
        pred_va_best = torch.einsum("nd,cdo->cno", X_va, best_W) + best_b.unsqueeze(1)
    pred_va_best_np = pred_va_best.cpu().numpy()  # (C, N_val, O)

    # Un-standardize predictions back to raw target space
    pred_va_unscaled = pred_va_best_np * y_std[None, :, :] + y_mean[None, :, :]
    y_val_raw = y_val  # (N_val, O) original scale

    results = []
    ss_tot = np.square(y_val_raw - y_val_raw.mean(axis=0, keepdims=True)).sum()
    for i, (lr_val, wd_val) in enumerate(configs):
        ss_res = np.square(pred_va_unscaled[i] - y_val_raw).sum()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        results.append((float(lr_val), float(wd_val), float(r2)))
    return results


def evaluate_layer(features, targets, groups, output_dim, solver, device,
                   cv_n_splits=None, grouping=None):
    """Evaluate one layer using Appendix B grouped k-fold cross-validation."""

    n_splits = cv_n_splits if cv_n_splits is not None else CV_N_SPLITS
    if grouping == "video_shuffled":
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=CV_RANDOM_SEED)
        split_iter = cv.split(features, targets)
    else:
        n_unique = int(np.unique(groups).size)
        if n_splits > n_unique:
            # GroupKFold requires n_splits <= #groups. Clamp silently for LOO.
            n_splits = n_unique
        cv = GroupKFold(n_splits=n_splits)
        split_iter = cv.split(features, targets, groups)
    fold_scores = []
    fold_best_lrs = []
    fold_best_wds = []

    for train_idx, val_idx in split_iter:
        if solver == "trainable":
            # Batched: all 20 HP configs in one go (~20x faster)
            cfg_results = fit_trainable_probe_batched(
                features[train_idx], targets[train_idx],
                features[val_idx], targets[val_idx],
                output_dim=output_dim,
                lr_grid=APPENDIX_B_LR_GRID,
                wd_grid=APPENDIX_B_WD_GRID,
                device=device,
            )
            # Pick best config by val R²
            best_lr, best_wd, best_score = max(cfg_results, key=lambda r: r[2])
            fold_scores.append(best_score)
            fold_best_lrs.append(best_lr)
            fold_best_wds.append(best_wd)
            continue

        if solver == "ridge_weak":
            # Phase 2-A: closed-form ridge with a small alpha grid.
            best_score = -np.inf
            best_alpha = None
            for alpha in RIDGE_WEAK_ALPHA_GRID:
                score = fit_probe(
                    features[train_idx], targets[train_idx],
                    features[val_idx], targets[val_idx],
                    alpha=alpha,
                )
                if score > best_score:
                    best_score = score
                    best_alpha = alpha
            fold_scores.append(best_score)
            fold_best_lrs.append(0.0)          # no LR for closed-form
            fold_best_wds.append(best_alpha)
            continue

        if solver == "adamw100":
            # Phase 2-B: single-HP AdamW (lr=1e-3, wd=0.1), 100 epochs, patience 10.
            # Intentionally paper-minimal: no HP sweep, short training.
            score = fit_trainable_probe(
                features[train_idx], targets[train_idx],
                features[val_idx], targets[val_idx],
                output_dim=output_dim,
                lr=ADAMW100_LR,
                weight_decay=ADAMW100_WD,
                device=device,
                max_epochs=ADAMW100_MAX_EPOCHS,
                patience_limit=ADAMW100_PATIENCE,
            )
            fold_scores.append(score)
            fold_best_lrs.append(ADAMW100_LR)
            fold_best_wds.append(ADAMW100_WD)
            continue

        # --- Legacy path (ridge, trainable_unbatched) ---
        best_score = -np.inf
        best_lr = None
        best_wd = None

        for learning_rate in APPENDIX_B_LR_GRID:
            for weight_decay in APPENDIX_B_WD_GRID:
                if solver == "ridge":
                    score = fit_probe(
                        features[train_idx], targets[train_idx],
                        features[val_idx], targets[val_idx],
                        alpha=weight_decay,
                    )
                elif solver == "trainable_unbatched":
                    score = fit_trainable_probe(
                        features[train_idx], targets[train_idx],
                        features[val_idx], targets[val_idx],
                        output_dim=output_dim,
                        lr=learning_rate,
                        weight_decay=weight_decay,
                        device=device,
                    )
                else:
                    raise ValueError(f"Unknown solver: {solver}")

                if score > best_score:
                    best_score = score
                    best_lr = learning_rate
                    best_wd = weight_decay

        fold_scores.append(best_score)
        fold_best_lrs.append(best_lr)
        fold_best_wds.append(best_wd)

    return float(np.mean(fold_scores)), float(np.std(fold_scores)), fold_best_lrs, fold_best_wds


def run_probing(model_name, targets, probe_set, solver, grouping, device,
                features_root=None, output_suffix="", cv_n_splits=None):
    """Run all probes for a model across all layers."""
    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["depth"]
    embed_dim = cfg["embed_dim"]
    results = []
    probe_configs = PROBE_SETS[probe_set]
    feats_root = features_root if features_root is not None else FEATURES_ROOT

    for probe_name, output_dim, dataset, target_key in probe_configs:
        print(f"\n  Probe: {probe_name} (output_dim={output_dim}, dataset={dataset})")
        target_data = targets[dataset][target_key]
        groups = extract_groups(targets[dataset]["video_ids"], grouping)

        for layer in range(n_layers):
            feat_path = os.path.join(feats_root, model_name, dataset, f"layer_{layer:02d}.npy")
            features = np.load(feat_path)
            assert features.shape[1] == embed_dim, \
                f"Feature dim mismatch at layer {layer}: {features.shape[1]} != {embed_dim}"

            mean_r2, std_r2, best_lrs, best_wds = evaluate_layer(
                features, target_data, groups, output_dim, solver, device,
                cv_n_splits=cv_n_splits,
                grouping=grouping,
            )
            results.append({
                "layer": layer,
                "probe": probe_name,
                "mean_r2": mean_r2,
                "std_r2": std_r2,
                "best_lr": float(pd.Series(best_lrs).mode().iloc[0]),
                "best_wd": float(pd.Series(best_wds).mode().iloc[0]),
            })

            if layer % 5 == 0 or layer == n_layers - 1:
                print(f"    Layer {layer:2d}: R²={mean_r2:.4f} ± {std_r2:.4f} "
                      f"(best lr mode={results[-1]['best_lr']}, "
                      f"best wd mode={results[-1]['best_wd']})")

    df = pd.DataFrame(results)
    csv_path = result_csv_path(model_name, probe_set, suffix=output_suffix)
    df.to_csv(csv_path, index=False)
    print(f"\n  Results saved to {csv_path}")
    return df


def generate_figure_2c():
    """Generate PEZ Figure 2c: Speed vs Direction R^2 by layer."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, (model_name, cfg) in zip(axes, MODEL_CONFIGS.items()):
        csv_path = result_csv_path(model_name, "polar")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        n_layers = cfg["depth"]
        layers = np.arange(n_layers)

        # PEZ zone shading
        ax.axvspan(n_layers * 0.3, n_layers * 0.5, alpha=0.1, color="orange", label="PEZ zone")

        for probe_name, color, marker in [("speed", "blue", "o"), ("direction", "red", "s")]:
            probe_df = df[df["probe"] == probe_name].sort_values("layer")
            mean = probe_df["mean_r2"].values
            std = probe_df["std_r2"].values
            ax.plot(layers, mean, color=color, marker=marker, markersize=3,
                    label=f"{probe_name.capitalize()} R$^2$", linewidth=1.5)
            ax.fill_between(layers, mean - std, mean + std, alpha=0.2, color=color)

        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("R$^2$", fontsize=12)
        ax.set_title(f"{cfg['display']} ({n_layers} layers)", fontsize=13)
        ax.set_ylim(-0.1, 1.05)
        ax.set_xlim(-0.5, n_layers - 0.5)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle("PEZ Reproduction: Speed vs Direction R$^2$ by Layer", fontsize=14, y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(RESULTS_ROOT, f"pez_reproduction_fig2c.{ext}")
        plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure 2c saved to {RESULTS_ROOT}/pez_reproduction_fig2c.{{png,pdf}}")


def generate_acceleration_figure():
    """Generate acceleration R^2 curve."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, (model_name, cfg) in zip(axes, MODEL_CONFIGS.items()):
        csv_path = result_csv_path(model_name, "polar")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        n_layers = cfg["depth"]
        layers = np.arange(n_layers)

        ax.axvspan(n_layers * 0.3, n_layers * 0.5, alpha=0.1, color="orange", label="PEZ zone")

        probe_df = df[df["probe"] == "acceleration"].sort_values("layer")
        mean = probe_df["mean_r2"].values
        std = probe_df["std_r2"].values
        ax.plot(layers, mean, color="green", marker="^", markersize=3,
                label="Acceleration R$^2$", linewidth=1.5)
        ax.fill_between(layers, mean - std, mean + std, alpha=0.2, color="green")

        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("R$^2$", fontsize=12)
        ax.set_title(f"{cfg['display']} ({n_layers} layers)", fontsize=13)
        ax.set_ylim(-0.1, 1.05)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle("PEZ Reproduction: Acceleration R$^2$ by Layer", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_ROOT, "pez_reproduction_acceleration.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Acceleration figure saved.")


def generate_all_probes_figure():
    """Generate combined figure with all three probes overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    probe_styles = {
        "speed": ("blue", "o", "Speed"),
        "direction": ("red", "s", "Direction"),
        "acceleration": ("green", "^", "Acceleration"),
    }

    for ax, (model_name, cfg) in zip(axes, MODEL_CONFIGS.items()):
        csv_path = result_csv_path(model_name, "polar")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        n_layers = cfg["depth"]
        layers = np.arange(n_layers)

        ax.axvspan(n_layers * 0.3, n_layers * 0.5, alpha=0.1, color="orange", label="PEZ zone")

        for probe_name, (color, marker, label) in probe_styles.items():
            probe_df = df[df["probe"] == probe_name].sort_values("layer")
            if len(probe_df) == 0:
                continue
            mean = probe_df["mean_r2"].values
            std = probe_df["std_r2"].values
            ax.plot(layers[:len(mean)], mean, color=color, marker=marker, markersize=3,
                    label=f"{label} R$^2$", linewidth=1.5)
            ax.fill_between(layers[:len(mean)], mean - std, mean + std, alpha=0.15, color=color)

        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("R$^2$", fontsize=12)
        ax.set_title(f"{cfg['display']} ({n_layers} layers)", fontsize=13)
        ax.set_ylim(-0.1, 1.05)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle("PEZ Reproduction: All Probes R$^2$ by Layer", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_ROOT, "pez_reproduction_all_probes.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"All-probes figure saved.")


def generate_cartesian_figure():
    """Generate the Cartesian velocity/acceleration figure (paper Fig. 2b-style)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, (model_name, cfg) in zip(axes, MODEL_CONFIGS.items()):
        csv_path = result_csv_path(model_name, "cartesian")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        n_layers = cfg["depth"]
        layers = np.arange(n_layers)

        ax.axvspan(n_layers * 0.3, n_layers * 0.5, alpha=0.1, color="orange", label="PEZ zone")

        for probe_name, color, marker, label in [
            ("velocity_xy", "blue", "o", "Velocity (vx, vy)"),
            ("acceleration_xy", "green", "^", "Acceleration (ax, ay)"),
        ]:
            probe_df = df[df["probe"] == probe_name].sort_values("layer")
            mean = probe_df["mean_r2"].values
            std = probe_df["std_r2"].values
            ax.plot(layers[:len(mean)], mean, color=color, marker=marker, markersize=3,
                    label=label, linewidth=1.5)
            ax.fill_between(layers[:len(mean)], mean - std, mean + std, alpha=0.2, color=color)

        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("R$^2$", fontsize=12)
        ax.set_title(f"{cfg['display']} ({n_layers} layers)", fontsize=13)
        ax.set_ylim(-0.1, 1.05)
        ax.set_xlim(-0.5, n_layers - 0.5)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle("PEZ Reproduction: Cartesian Velocity vs Acceleration R$^2$ by Layer", fontsize=14, y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(RESULTS_ROOT, f"pez_reproduction_fig2b_cartesian.{ext}")
        plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Cartesian figure saved to {RESULTS_ROOT}/pez_reproduction_fig2b_cartesian.{{png,pdf}}")


def check_pez_pattern():
    """Check whether results match expected PEZ pattern."""
    print("\n" + "=" * 60)
    print("PEZ PATTERN CHECK")
    print("=" * 60)

    checks = []

    for model_name, cfg in MODEL_CONFIGS.items():
        csv_path = result_csv_path(model_name, "polar")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        n_layers = cfg["depth"]
        display = cfg["display"]

        speed_df = df[df["probe"] == "speed"].sort_values("layer")
        dir_df = df[df["probe"] == "direction"].sort_values("layer")
        acc_df = df[df["probe"] == "acceleration"].sort_values("layer")

        speed_r2 = speed_df["mean_r2"].values
        dir_r2 = dir_df["mean_r2"].values

        # Check 1: Speed R^2 high from early layers
        early_speed = speed_r2[:3].mean()
        checks.append(("Speed early R^2 > 0.5", display, early_speed, early_speed > 0.5))

        # Check 2: Speed relatively flat
        speed_range = speed_r2.max() - speed_r2.min()
        checks.append(("Speed R^2 range < 0.4", display, speed_range, speed_range < 0.4))

        # Check 3: Direction R^2 near 0 before PEZ
        pre_pez = int(n_layers * 0.25)
        dir_pre_pez = dir_r2[:pre_pez].mean()
        checks.append(("Direction pre-PEZ R^2 < 0.3", display, dir_pre_pez, dir_pre_pez < 0.3))

        # Check 4: Direction R^2 peak
        dir_peak = dir_r2.max()
        checks.append(("Direction peak R^2 > 0.5", display, dir_peak, dir_peak > 0.5))

        # Check 5: Direction transition (sharp increase)
        if len(dir_r2) > 5:
            max_jump = max(dir_r2[i + 1] - dir_r2[i] for i in range(len(dir_r2) - 1))
            transition_layer = np.argmax([dir_r2[i + 1] - dir_r2[i]
                                          for i in range(len(dir_r2) - 1)])
            checks.append((f"Direction transition at ~layer {transition_layer}",
                           display, max_jump, max_jump > 0.1))

        # Check 6: Acceleration similar to speed
        if len(acc_df) > 0:
            acc_r2 = acc_df["mean_r2"].values
            early_acc = acc_r2[:3].mean()
            checks.append(("Accel early R^2 > 0.3", display, early_acc, early_acc > 0.3))

    print(f"\n{'Check':<40} {'Model':<15} {'Value':>8} {'Pass':>6}")
    print("-" * 75)
    for check_name, model, value, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"{check_name:<40} {model:<15} {value:>8.4f} {status:>6}")

    n_pass = sum(1 for _, _, _, p in checks if p)
    n_total = len(checks)
    print(f"\n{n_pass}/{n_total} checks passed")


def main():
    print("=" * 60)
    print("PEZ Step 3: Linear Probing + Figure Generation")
    print("=" * 60)

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODEL_CONFIGS.keys()), default=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--probe-set", choices=list(PROBE_SETS.keys()), default="polar")
    parser.add_argument(
        "--solver",
        choices=["ridge", "ridge_weak", "trainable", "trainable_unbatched", "adamw100"],
        default="trainable",
        help=(
            "ridge: closed-form L2 regression over Appendix B WD grid. "
            "ridge_weak: closed-form L2 with alpha in {1, 10, 100} (Phase 2-A weak probe). "
            "trainable: Adam-trained linear probe, all 20 HP configs batched (default). "
            "trainable_unbatched: legacy per-config loop (slow, for verification). "
            "adamw100: single-HP AdamW (lr=1e-3, wd=0.1) for 100 epochs "
            "with patience 10 (Phase 2-B weak probe)."
        ),
    )
    parser.add_argument(
        "--grouping",
        choices=["position", "condition", "direction", "video", "video_shuffled"],
        default="condition",
        help=(
            "GroupKFold key. position/condition share direction angles across "
            "train and val folds. direction puts every video of one angle into "
            "a single fold (leave-one-direction-out when combined with "
            "--cv-splits 8). video treats every sample as its own group under "
            "GroupKFold. video_shuffled uses shuffled KFold, closest to "
            "ordinary 5-fold when grouping metadata is unknown."
        ),
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=None,
        help=(
            "Override GroupKFold n_splits. Defaults to CV_N_SPLITS=5. For LOO "
            "direction use --grouping direction --cv-splits 8."
        ),
    )
    parser.add_argument(
        "--features-root",
        default=None,
        help="Override FEATURES_ROOT. Use to point at features_preblock/ for paper-convention extraction.",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffix appended to result CSV filename (e.g. 'preblock_ridge' -> results_vjepa2_L_preblock_ridge.csv).",
    )
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    targets = load_targets()

    start_time = time.time()

    for model_name in args.models:
        print(f"\n{'='*40}")
        print(f"Model: {model_name}")
        print(f"{'='*40}")
        run_probing(
            model_name, targets, args.probe_set, args.solver, args.grouping, device,
            features_root=args.features_root, output_suffix=args.output_suffix,
            cv_n_splits=args.cv_splits,
        )

    elapsed = time.time() - start_time
    print(f"\nTotal probing time: {elapsed:.1f}s")

    # Save config
    config = {
        "probe_set": args.probe_set,
        "paper_appendix_b_learning_rates": APPENDIX_B_LR_GRID,
        "paper_appendix_b_weight_decays": APPENDIX_B_WD_GRID,
        "cv_splits": CV_N_SPLITS,
        "cv_random_seed": CV_RANDOM_SEED,
        "grouping": args.grouping,
        "probes": [
            {"name": n, "output_dim": od, "dataset": d, "target": t}
            for n, od, d, t in PROBE_SETS[args.probe_set]
        ],
        "optimizer": "Adam" if args.solver == "trainable" else "ridge_closed_form",
        "loss": "MSE",
        "batch_mode": "full_batch",
        "protocol": "appendix_b_grouped_5fold_cv",
        "note": (
            "Appendix B specifies grouped 5-fold CV plus lr/wd sweeps, but does not "
            "specify optimizer or epoch count for the main layerwise probes."
        ),
        "solver": args.solver,
        "device": device,
        "trainable_max_epochs": TRAINABLE_MAX_EPOCHS if args.solver == "trainable" else None,
        "trainable_patience": TRAINABLE_PATIENCE if args.solver == "trainable" else None,
    }
    cfg_sfx = f"_{args.output_suffix}" if args.output_suffix else ""
    cfg_path = probing_config_path(args.probe_set).replace(
        ".json", f"{cfg_sfx}.json"
    )
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)

    # Generate figures (skip when using a custom output suffix — the default
    # figure pipeline reads the canonical filenames).
    if args.output_suffix:
        print(f"\nSkipping figure generation (output-suffix={args.output_suffix!r}).")
    else:
        print("\nGenerating figures...")
        if args.probe_set == "polar":
            generate_figure_2c()
            generate_acceleration_figure()
            generate_all_probes_figure()
            check_pez_pattern()
        elif args.probe_set == "cartesian":
            generate_cartesian_figure()

    print(f"\nStep 3 complete!")
    print(f"Output: {RESULTS_ROOT}/")


if __name__ == "__main__":
    main()
