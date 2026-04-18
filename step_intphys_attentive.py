"""Approximate Figure 8 reproduction with attentive probes on IntPhys dev.

This script builds a patch-preserving attentive-MLP probe on top of frozen
V-JEPA 2 layer features. It is intended for overall-only reproduction of the
possible/impossible IntPhys task across model sizes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset

from step2_extract import build_transform, load_model, pool_tokens, resolve_model_spec

import sys

sys.path.insert(0, "/home/solee/vjepa2")
sys.path.insert(0, "/home/solee/vjepa2/src")
from src.models.attentive_pooler import AttentiveClassifier


INTPHYS_ROOT = Path("/home/solee/pez/artifacts/intphys/dev")
FEATURE_ROOT_BASE = Path("/home/solee/pez/artifacts/features")
RESULTS_ROOT = Path("/home/solee/pez/artifacts/results")

CV_SPLITS = 5
RANDOM_SEED = 42


def load_dev_reference(max_scenes_per_block: int | None = None) -> pd.DataFrame:
    rows = []
    for block_dir in sorted(INTPHYS_ROOT.glob("O*")):
        if not block_dir.is_dir():
            continue
        scene_dirs = [p for p in sorted(block_dir.glob("*")) if p.is_dir()]
        if max_scenes_per_block is not None:
            scene_dirs = scene_dirs[:max_scenes_per_block]
        for scene_dir in scene_dirs:
            scene_group = f"{block_dir.name}/{scene_dir.name}"
            for movie_dir in sorted(scene_dir.glob("*")):
                if not movie_dir.is_dir():
                    continue
                status_path = movie_dir / "status.json"
                if not status_path.exists():
                    continue
                status = json.loads(status_path.read_text())
                rows.append(
                    {
                        "block": block_dir.name,
                        "scene_id": scene_dir.name,
                        "movie_id": movie_dir.name,
                        "scene_group": scene_group,
                        "label": int(bool(status["header"]["is_possible"])),
                        "scene_dir": str(movie_dir / "scene"),
                    }
                )
    return pd.DataFrame(rows).sort_values(["block", "scene_id", "movie_id"]).reset_index(drop=True)


def list_frames(scene_dir: str):
    return sorted(Path(scene_dir).glob("scene_*.png"))


def load_clip(scene_dir: str, transform, n_frames_sample: int) -> torch.Tensor:
    frames = list_frames(scene_dir)
    sample_indices = np.linspace(0, len(frames) - 1, n_frames_sample).round().astype(int)
    tensors = []
    for idx in sample_indices:
        image_bgr = cv2.imread(str(frames[idx]), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1)
        tensors.append(transform(tensor))
    return torch.stack(tensors).permute(1, 0, 2, 3)


def default_feature_root(model: str, capture: str, transform_name: str, patch_pool: str, n_frames: int) -> Path:
    spec = resolve_model_spec(model)
    return FEATURE_ROOT_BASE / f"intphys_{spec['repo_name']}_{capture}_{transform_name}_{patch_pool}_{n_frames}f"


def extract_patch_features(
    device: str,
    batch_size: int,
    capture: str,
    transform_name: str,
    feature_root: Path,
    n_frames_sample: int,
    model_name: str,
    patch_pool: str,
):
    metadata_path = feature_root / "metadata.json"
    manifest_path = feature_root / "manifest.csv"
    if metadata_path.exists() and manifest_path.exists():
        return pd.read_csv(manifest_path), json.loads(metadata_path.read_text())

    feature_root.mkdir(parents=True, exist_ok=True)
    df = load_dev_reference()
    transform = build_transform(transform_name)
    model, spec = load_model(device=device, capture=capture, model_name=model_name)

    layer_arrays = [
        open_memmap(
            feature_root / f"layer_{layer_index:02d}.npy",
            mode="w+",
            dtype=np.float16,
            shape=(len(df), 256, spec["embed_dim"]),
        )
        for layer_index in range(spec["depth"])
    ]

    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        print(f"extract batch {start}:{end} / {len(df)}")
        batch = torch.stack(
            [load_clip(df.iloc[idx]["scene_dir"], transform, n_frames_sample=n_frames_sample) for idx in range(start, end)]
        ).to(device)

        with torch.no_grad():
            outputs = model(batch)

        for layer_index, layer_tokens in enumerate(outputs):
            patch_tokens = pool_tokens(layer_tokens, patch_pool).float().cpu().numpy().astype(np.float16)
            layer_arrays[layer_index][start:end] = patch_tokens

    del layer_arrays
    df.to_csv(manifest_path, index=False)
    metadata = {
        "model": spec["repo_name"],
        "model_key": model_name,
        "capture": capture,
        "transform": transform_name,
        "patch_pool": patch_pool,
        "n_layers": spec["depth"],
        "embed_dim": spec["embed_dim"],
        "n_clips": len(df),
        "n_frames_sample": n_frames_sample,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return df, metadata


def compute_accuracy(y_true: np.ndarray, prob: np.ndarray) -> float:
    pred = (prob >= 0.5).astype(np.int64)
    return float((pred == y_true.astype(np.int64)).mean())


def compute_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return 0.5
    return float(roc_auc_score(y_true, prob))


def compute_relative_accuracy(scene_groups, y_true, prob) -> float:
    frame = pd.DataFrame(
        {
            "scene_group": np.asarray(scene_groups),
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
    return float(correct / total) if total else 0.0


def fit_attentive_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_scene_groups: np.ndarray,
    device: str,
    num_probe_blocks: int,
    num_heads: int,
    lr: float,
    wd: float,
    num_epochs: int,
    batch_size: int,
):
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)

    feat_mean = X_train.mean(axis=(0, 1), keepdims=True)
    feat_std = X_train.std(axis=(0, 1), keepdims=True)
    feat_std[feat_std < 1e-6] = 1.0
    X_train = (X_train - feat_mean) / feat_std
    X_val = (X_val - feat_mean) / feat_std

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train[:, None])),
        batch_size=batch_size,
        shuffle=True,
    )
    val_tensor = torch.from_numpy(X_val).to(device)

    model = AttentiveClassifier(
        embed_dim=X_train.shape[-1],
        num_heads=num_heads,
        depth=num_probe_blocks,
        num_classes=1,
        use_activation_checkpointing=False,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best_relative = -1.0
    best_prob = None
    best_state = None
    patience = 0

    for epoch in range(num_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch_y.squeeze(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(val_tensor).squeeze(-1)).cpu().numpy()
        rel = compute_relative_accuracy(val_scene_groups, y_val, prob)
        if rel > best_relative:
            best_relative = rel
            best_prob = prob.copy()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                break

    if best_prob is None:
        raise RuntimeError("Attentive probe failed to produce validation predictions")
    model.load_state_dict(best_state)
    return best_prob


def evaluate_layers(
    feature_root: Path,
    manifest_df: pd.DataFrame,
    n_layers: int,
    device: str,
    num_probe_blocks: int,
    num_heads: int,
    lr: float,
    wd: float,
    num_epochs: int,
    batch_size: int,
):
    groups = manifest_df["scene_group"].to_numpy()
    y = manifest_df["label"].to_numpy(dtype=np.int64)
    splitter = GroupKFold(n_splits=CV_SPLITS)
    rows = []

    for layer in range(n_layers):
        print(f"evaluate layer {layer}/{n_layers - 1}")
        X = np.load(feature_root / f"layer_{layer:02d}.npy", mmap_mode="r")
        fold_acc = []
        fold_auc = []
        fold_rel = []
        for train_idx, val_idx in splitter.split(np.zeros(len(y)), y, groups):
            prob = fit_attentive_probe(
                X[train_idx],
                y[train_idx],
                X[val_idx],
                y[val_idx],
                groups[val_idx],
                device=device,
                num_probe_blocks=num_probe_blocks,
                num_heads=num_heads,
                lr=lr,
                wd=wd,
                num_epochs=num_epochs,
                batch_size=batch_size,
            )
            fold_acc.append(compute_accuracy(y[val_idx], prob))
            fold_auc.append(compute_auc(y[val_idx], prob))
            fold_rel.append(compute_relative_accuracy(groups[val_idx], y[val_idx], prob))

        rows.append(
            {
                "layer": layer,
                "clip_accuracy_mean": float(np.mean(fold_acc)),
                "clip_accuracy_std": float(np.std(fold_acc)),
                "auc_mean": float(np.mean(fold_auc)),
                "auc_std": float(np.std(fold_auc)),
                "relative_accuracy_mean": float(np.mean(fold_rel)),
                "relative_accuracy_std": float(np.std(fold_rel)),
            }
        )
    return pd.DataFrame(rows)


def summarize(result_df: pd.DataFrame):
    best_idx = int(result_df["relative_accuracy_mean"].idxmax())
    peak = result_df.iloc[best_idx]
    layer8 = result_df.loc[result_df["layer"] == 8].iloc[0]
    last = result_df.iloc[-1]
    first_above = result_df.loc[result_df["relative_accuracy_mean"] >= 0.8, "layer"]
    return {
        "l0_relative_acc": float(result_df.iloc[0]["relative_accuracy_mean"]),
        "l8_relative_acc": float(layer8["relative_accuracy_mean"]),
        "peak_relative_acc": float(peak["relative_accuracy_mean"]),
        "peak_relative_layer": int(peak["layer"]),
        "late_relative_acc": float(last["relative_accuracy_mean"]),
        "first_relative_ge_0.8": None if len(first_above) == 0 else int(first_above.iloc[0]),
        "l0_clip_acc": float(result_df.iloc[0]["clip_accuracy_mean"]),
        "l8_clip_acc": float(layer8["clip_accuracy_mean"]),
        "peak_clip_acc": float(result_df.iloc[result_df["clip_accuracy_mean"].idxmax()]["clip_accuracy_mean"]),
    }


def plot_curve(result_df: pd.DataFrame, png_path: Path, title: str):
    layers = result_df["layer"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layers, result_df["clip_accuracy_mean"] * 100.0, label="Clip Accuracy", linewidth=2)
    ax.plot(layers, result_df["relative_accuracy_mean"] * 100.0, label="Relative Accuracy", linewidth=2)
    ax.axvline(8, color="gray", linestyle="--", alpha=0.7, label="Layer 8")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(45, 101)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["large", "huge", "giant"], default="large")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--capture", choices=["resid_pre", "resid_post"], default="resid_pre")
    parser.add_argument("--transform", choices=["resize", "eval_preproc"], default="resize")
    parser.add_argument("--patch-pool", choices=["temporal_last_patch", "temporal_diff_patch"], default="temporal_last_patch")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--probe-batch-size", type=int, default=16)
    parser.add_argument("--n-frames-sample", type=int, default=16)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--feature-root", type=Path, default=None)
    parser.add_argument("--num-probe-blocks", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--num-epochs", type=int, default=20)
    args = parser.parse_args()

    feature_root = (
        args.feature_root
        if args.feature_root is not None
        else default_feature_root(args.model, args.capture, args.transform, args.patch_pool, args.n_frames_sample)
    )

    manifest_df, metadata = extract_patch_features(
        device=args.device,
        batch_size=args.batch_size,
        capture=args.capture,
        transform_name=args.transform,
        feature_root=feature_root,
        n_frames_sample=args.n_frames_sample,
        model_name=args.model,
        patch_pool=args.patch_pool,
    )

    results = evaluate_layers(
        feature_root=feature_root,
        manifest_df=manifest_df,
        n_layers=metadata["n_layers"],
        device=args.device,
        num_probe_blocks=args.num_probe_blocks,
        num_heads=args.num_heads,
        lr=args.lr,
        wd=args.wd,
        num_epochs=args.num_epochs,
        batch_size=args.probe_batch_size,
    )

    run_name = args.run_name
    csv_path = RESULTS_ROOT / f"results_{run_name}.csv"
    png_path = RESULTS_ROOT / f"figure_{run_name}.png"
    summary_path = RESULTS_ROOT / f"summary_{run_name}.json"
    results.to_csv(csv_path, index=False)
    plot_curve(results, png_path, title=run_name)
    summary_path.write_text(json.dumps(summarize(results), indent=2))
    print(csv_path)
    print(png_path)
    print(summary_path)


if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    main()
