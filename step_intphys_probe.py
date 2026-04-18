"""Reproduce the paper's possible/impossible linear probe on IntPhys dev.

This is an approximate reproduction path using the public IntPhys dev split:

- data: IntPhys dev split (public, labeled)
- model: V-JEPA 2 Large / Huge / Giant
- features: spatiotemporally mean-pooled residual stream
- task: binary possible/impossible classification
- split: 5-fold GroupKFold grouped by matched scene quadruplet
- metric: validation accuracy (%) and AUC

The paper does not fully specify the exact split / preprocessing for this task,
so this script is a best-effort reconstruction rather than a guaranteed
paper-identical implementation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from step2_extract import build_transform, load_model, resolve_model_spec


INTPHYS_ROOT = Path("/home/solee/pez/artifacts/intphys/dev")
INTPHYS_STARTING_KIT = Path("/home/solee/pez/artifacts/intphys/starting_kit.zip")
FEATURE_ROOT_BASE = Path("/home/solee/pez/artifacts/features")
RESULTS_ROOT = Path("/home/solee/pez/artifacts/results")

DEFAULT_N_FRAMES_SAMPLE = 16
CV_SPLITS = 5
CV_RANDOM_SEED = 42
LR_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 5e-3]
WD_GRID = [0.01, 0.1, 0.4, 0.8]
MAX_EPOCHS = 400
PATIENCE = 40


def load_dev_reference(max_scenes_per_block: int | None = None) -> pd.DataFrame:
    rows = []
    for block_dir in sorted(INTPHYS_ROOT.glob("O*")):
        if not block_dir.is_dir():
            continue
        block = block_dir.name
        scene_dirs = [p for p in sorted(block_dir.glob("*")) if p.is_dir()]
        if max_scenes_per_block is not None:
            scene_dirs = scene_dirs[:max_scenes_per_block]
        for scene_dir in scene_dirs:
            if not scene_dir.is_dir():
                continue
            scene_id = scene_dir.name
            scene_group = f"{block}/{scene_id}"
            for movie_dir in sorted(scene_dir.glob("*")):
                if not movie_dir.is_dir():
                    continue
                movie_id = movie_dir.name
                status_path = movie_dir / "status.json"
                if not status_path.exists():
                    continue
                status = json.loads(status_path.read_text())
                label = int(bool(status["header"]["is_possible"]))
                rows.append(
                    {
                        "block": block,
                        "scene_id": scene_id,
                        "movie_id": movie_id,
                        "scene_group": scene_group,
                        "label": label,
                        "rel_path": f"{block}/{scene_id}/{movie_id}",
                        "scene_dir": str(movie_dir / "scene"),
                    }
                )

    df = pd.DataFrame(rows).sort_values(["block", "scene_id", "movie_id"]).reset_index(drop=True)
    return df


def list_frames(scene_dir: str):
    return sorted(Path(scene_dir).glob("scene_*.png"))


def load_clip(scene_dir: str, transform, n_frames_sample: int) -> torch.Tensor:
    frames = list_frames(scene_dir)
    if len(frames) == 0:
        raise FileNotFoundError(f"No frames found in {scene_dir}")

    sample_indices = np.linspace(0, len(frames) - 1, n_frames_sample).round().astype(int)
    sample_paths = [frames[idx] for idx in sample_indices]

    tensor_frames = []
    for path in sample_paths:
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1)
        tensor_frames.append(transform(tensor))
    return torch.stack(tensor_frames).permute(1, 0, 2, 3)


def extract_features(
    device: str,
    batch_size: int,
    capture: str,
    transform_name: str,
    feature_root: Path,
    max_scenes_per_block: int | None,
    n_frames_sample: int,
    model_name: str,
):
    feature_root.mkdir(parents=True, exist_ok=True)
    metadata_path = feature_root / "metadata.json"
    manifest_path = feature_root / "manifest.csv"

    df = load_dev_reference(max_scenes_per_block=max_scenes_per_block)
    transform = build_transform(transform_name)
    model, spec = load_model(device=device, capture=capture, model_name=model_name)

    features = [np.zeros((len(df), spec["embed_dim"]), dtype=np.float32) for _ in range(spec["depth"])]

    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        print(f"extract batch {start}:{end} / {len(df)}")
        batch = torch.stack(
            [
                load_clip(df.iloc[idx]["scene_dir"], transform, n_frames_sample=n_frames_sample)
                for idx in range(start, end)
            ]
        ).to(device)

        with torch.no_grad():
            outputs = model(batch)

        for layer_index, layer_tokens in enumerate(outputs):
            pooled = layer_tokens.mean(dim=1).float().cpu().numpy()
            features[layer_index][start:end] = pooled

    for layer_index, array in enumerate(features):
        np.save(feature_root / f"layer_{layer_index:02d}.npy", array)

    df.to_csv(manifest_path, index=False)
    metadata = {
        "model": spec["repo_name"],
        "model_key": model_name,
        "capture": capture,
        "transform": transform_name,
        "n_layers": spec["depth"],
        "embed_dim": spec["embed_dim"],
        "n_clips": len(df),
        "n_frames_sample": n_frames_sample,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return df


def compute_accuracy(y_true: np.ndarray, prob: np.ndarray) -> float:
    pred = (prob >= 0.5).astype(np.int64)
    return float((pred == y_true.astype(np.int64)).mean())


def compute_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return 0.5
    return float(roc_auc_score(y_true, prob))


def compute_relative_accuracy(scene_groups, movie_ids, y_true, prob) -> float:
    frame = pd.DataFrame(
        {
            "scene_group": np.asarray(scene_groups),
            "movie_id": np.asarray(movie_ids),
            "label": np.asarray(y_true).astype(np.int64),
            "prob": np.asarray(prob).astype(np.float64),
        }
    )

    correct = 0
    total = 0
    for _, sub in frame.groupby("scene_group", sort=True):
        pos = float(sub.loc[sub["label"] == 1, "prob"].sum())
        imp = float(sub.loc[sub["label"] == 0, "prob"].sum())
        correct += int(pos > imp)
        total += 1
    if total == 0:
        return 0.0
    return float(correct / total)


def fit_binary_probe_batched(
    X_train,
    y_train,
    X_val,
    y_val,
    val_scene_groups,
    val_movie_ids,
    device: str,
):
    configs = list(itertools.product(LR_GRID, WD_GRID))
    n_configs = len(configs)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
    y_val = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)

    x_mean = X_train.mean(axis=0, keepdims=True)
    x_std = X_train.std(axis=0, keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    X_train_std = (X_train - x_mean) / x_std
    X_val_std = (X_val - x_mean) / x_std

    X_tr = torch.tensor(X_train_std, dtype=torch.float32, device=device)
    X_va = torch.tensor(X_val_std, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.float32, device=device)

    input_dim = X_tr.shape[1]
    output_dim = 1

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

    best_val_acc = torch.full((n_configs,), -1.0, device=device)
    best_W = W.detach().clone()
    best_b = b.detach().clone()
    patience = torch.zeros(n_configs, dtype=torch.int32, device=device)
    active = torch.ones(n_configs, dtype=torch.bool, device=device)

    for step in range(1, MAX_EPOCHS + 1):
        logits_tr = torch.einsum("nd,cdo->cno", X_tr, W) + b.unsqueeze(1)
        loss_per_cfg = torch.nn.functional.binary_cross_entropy_with_logits(
            logits_tr, y_tr.unsqueeze(0).expand(n_configs, -1, -1), reduction="none"
        ).mean(dim=(1, 2))
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

            logits_va = torch.einsum("nd,cdo->cno", X_va, W) + b.unsqueeze(1)
            prob_va = torch.sigmoid(logits_va).squeeze(-1)
            acc_va = (prob_va.ge(0.5) == torch.tensor(y_val[:, 0], device=device).bool()).float().mean(dim=1)

            improved = acc_va > best_val_acc + 1e-8
            best_val_acc = torch.where(improved, acc_va, best_val_acc)
            best_W = torch.where(improved[:, None, None], W.detach(), best_W)
            best_b = torch.where(improved[:, None], b.detach(), best_b)
            patience = torch.where(improved, torch.zeros_like(patience), patience + 1)
            active = active & (patience < PATIENCE)
            if not active.any():
                break

    with torch.no_grad():
        logits_best = torch.einsum("nd,cdo->cno", X_va, best_W) + best_b.unsqueeze(1)
        prob_best = torch.sigmoid(logits_best).squeeze(-1).cpu().numpy()

    results = []
    y_true = y_val[:, 0]
    for cfg_index, (lr, wd) in enumerate(configs):
        prob = prob_best[cfg_index]
        results.append(
            {
                "lr": float(lr),
                "wd": float(wd),
                "acc": compute_accuracy(y_true, prob),
                "auc": compute_auc(y_true, prob),
                "relative_acc": compute_relative_accuracy(
                    val_scene_groups, val_movie_ids, y_true, prob
                ),
                "prob": prob,
            }
        )
    return results


def evaluate_layers(feature_root: Path, df: pd.DataFrame, device: str, selection_metric: str, depth: int):
    groups = df["scene_group"].to_numpy()
    labels = df["label"].to_numpy(dtype=np.int64)
    movie_ids = df["movie_id"].to_numpy()
    splitter = GroupKFold(n_splits=min(CV_SPLITS, int(np.unique(groups).size)))

    rows = []
    for layer in range(depth):
        X = np.load(feature_root / f"layer_{layer:02d}.npy")
        fold_accs = []
        fold_aucs = []
        fold_rel_accs = []
        fold_best_lrs = []
        fold_best_wds = []

        for train_idx, val_idx in splitter.split(X, labels, groups):
            cfg_results = fit_binary_probe_batched(
                X[train_idx],
                labels[train_idx],
                X[val_idx],
                labels[val_idx],
                groups[val_idx],
                movie_ids[val_idx],
                device=device,
            )
            if selection_metric == "accuracy":
                best = max(cfg_results, key=lambda item: (item["acc"], item["auc"]))
            elif selection_metric == "relative_accuracy":
                best = max(cfg_results, key=lambda item: (item["relative_acc"], item["acc"]))
            else:
                raise ValueError(f"Unknown selection_metric: {selection_metric}")
            fold_accs.append(best["acc"])
            fold_aucs.append(best["auc"])
            fold_rel_accs.append(best["relative_acc"])
            fold_best_lrs.append(best["lr"])
            fold_best_wds.append(best["wd"])

        rows.append(
            {
                "layer": layer,
                "accuracy_mean": float(np.mean(fold_accs)),
                "accuracy_std": float(np.std(fold_accs)),
                "auc_mean": float(np.mean(fold_aucs)),
                "auc_std": float(np.std(fold_aucs)),
                "relative_accuracy_mean": float(np.mean(fold_rel_accs)),
                "relative_accuracy_std": float(np.std(fold_rel_accs)),
                "best_lr_mode": float(pd.Series(fold_best_lrs).mode().iloc[0]),
                "best_wd_mode": float(pd.Series(fold_best_wds).mode().iloc[0]),
            }
        )

    return pd.DataFrame(rows)


def plot_curve(df: pd.DataFrame, output_png: Path, depth: int):
    x = df["layer"].to_numpy() / (depth - 1)
    y = df["accuracy_mean"].to_numpy() * 100.0
    yerr = df["accuracy_std"].to_numpy() * 100.0

    plt.figure(figsize=(7, 4.5))
    plt.plot(x, y, color="#1f77b4", linewidth=2, label="IntPhys linear probe")
    plt.fill_between(x, y - yerr, y + yerr, color="#1f77b4", alpha=0.2)
    pez_left = 8 / depth
    pez_right = min(9, depth - 1) / (depth - 1)
    plt.axvspan(pez_left, pez_right, color="gray", alpha=0.2, label="Layer 8 PEZ marker")
    plt.xlabel("Layer Fraction")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("IntPhys Possible/Impossible Probe")
    plt.ylim(40, 100)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


def summarize(df: pd.DataFrame, depth: int) -> dict:
    acc = df["accuracy_mean"].to_numpy()
    peak_layer = int(df.loc[df["accuracy_mean"].idxmax(), "layer"])
    ge85 = np.where(acc >= 0.85)[0]
    l8_index = min(8, depth - 1)
    return {
        "l0_acc": float(acc[0]),
        "l8_acc": float(acc[l8_index]),
        "peak_acc": float(acc.max()),
        "peak_layer": peak_layer,
        "first_ge_85_layer": None if len(ge85) == 0 else int(ge85[0]),
        "late_acc": float(acc[-1]),
        "l0_relative_acc": float(df["relative_accuracy_mean"].to_numpy()[0]),
        "l8_relative_acc": float(df["relative_accuracy_mean"].to_numpy()[l8_index]),
        "peak_relative_acc": float(df["relative_accuracy_mean"].to_numpy().max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model", choices=["large", "huge", "giant"], default="large")
    parser.add_argument("--capture", choices=["resid_pre", "resid_post"], default="resid_pre")
    parser.add_argument("--transform", choices=["resize", "eval_preproc"], default="resize")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--feature-root", default=None)
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument("--max-scenes-per-block", type=int, default=None)
    parser.add_argument("--run-name", default="intphys_possible_impossible")
    parser.add_argument("--n-frames-sample", type=int, default=DEFAULT_N_FRAMES_SAMPLE)
    parser.add_argument("--selection-metric", choices=["accuracy", "relative_accuracy"], default="accuracy")
    args = parser.parse_args()

    spec = resolve_model_spec(args.model)
    if args.feature_root is None:
        feature_root = FEATURE_ROOT_BASE / f"intphys_{spec['repo_name']}_{args.capture}_{args.transform}"
    else:
        feature_root = Path(args.feature_root)
    results_csv = RESULTS_ROOT / f"results_{args.run_name}.csv"
    results_png = RESULTS_ROOT / f"figure_{args.run_name}.png"
    summary_json = RESULTS_ROOT / f"summary_{args.run_name}.json"

    if args.reuse_features and (feature_root / "manifest.csv").exists():
        df = pd.read_csv(feature_root / "manifest.csv")
    else:
        df = extract_features(
            device=args.device,
            batch_size=args.batch_size,
            capture=args.capture,
            transform_name=args.transform,
            feature_root=feature_root,
            max_scenes_per_block=args.max_scenes_per_block,
            n_frames_sample=args.n_frames_sample,
            model_name=args.model,
        )

    result_df = evaluate_layers(
        feature_root=feature_root,
        df=df,
        device=args.device,
        selection_metric=args.selection_metric,
        depth=spec["depth"],
    )
    result_df.to_csv(results_csv, index=False)
    plot_curve(result_df, results_png, depth=spec["depth"])
    summary = summarize(result_df, depth=spec["depth"])
    summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
