#!/usr/bin/env python3

from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOM_KEY_TO_NAME = {
    "classroom": "classroom",
    "meetingroom": "meeting_room",
    "emptyroom": "empty_room",
}

DEFAULT_RUNS = ["source_only", "strong_aug", "consistency", "cond_coral", "dann", "full"]


def clean_version(value: str) -> str:
    value = value.strip().lower()
    if not value:
        return "v5"
    return value if value.startswith("v") else f"v{value}"


def default_output_dir(version: str) -> str:
    return f"runs_router_csi_{version}"


def load_training_module(code_dir: Path) -> ModuleType:
    train_path = code_dir / "train_router_csi_dg_v5.py"
    if not train_path.exists():
        raise FileNotFoundError(f"Could not find training script: {train_path}")
    module_name = "train_router_csi_dg_v5"
    spec = importlib.util.spec_from_file_location(module_name, train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import training script: {train_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def checkpoint_path(output_dir: Path, version: str, run_name: str, band: str, fold: str) -> Path:
    return output_dir / f"tiny_csi_dg_{version}_{run_name}_{band}_{fold}.pt"


def build_loader(
    train_mod: ModuleType,
    data_dir: Path,
    band: str,
    fold: str,
    split_name: str,
    batch_size: int,
    workers: int,
    n_doppler_bins: int,
    feature_norm: str,
    drop_dc: bool,
):
    splits = json.loads((data_dir / f"loeo_splits_{band}.json").read_text())
    indices = splits[fold][split_name]
    dataset = train_mod.CSIFoldDataset(
        data_dir / f"X_csi_50Hz_2x2_{band}.npy",
        data_dir / f"y_labels_{band}.npy",
        data_dir / f"env_labels_{band}.npy",
        indices,
        n_doppler_bins=n_doppler_bins,
        feature_norm=feature_norm,
        drop_dc=drop_dc,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)


def load_model(train_mod: ModuleType, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args: dict[str, Any] = ckpt.get("args", {})
    width = int(ckpt_args.get("width_effective", ckpt_args.get("width", 16)))
    model = train_mod.TinyCSIDomainGeneralizerV5(
        width=width,
        dropout=float(ckpt_args.get("dropout", 0.30)),
        domain_classes=2,
        use_attention=bool(ckpt_args.get("use_attention", False)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def collect_predictions(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    probs = []
    labels = []
    losses = []
    with torch.no_grad():
        for x, y, _domains in loader:
            x = x.to(device)
            y = y.to(device)
            logits, _domain_logits, _features = model(x, grl_lambda=0.0)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            losses.append(float(loss.item()) * x.shape[0])
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y.cpu().numpy())
    y_true = np.concatenate(labels).astype(np.int64)
    y_prob = np.concatenate(probs).astype(np.float64)
    return y_true, y_prob, sum(losses) / max(1, len(y_true))


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    pos_ranks = ranks[y_true == 1].sum()
    return float((pos_ranks - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, loss: float, threshold: float) -> dict[str, float | int]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    accuracy = (tp + tn) / max(1, len(y_true))
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "loss": loss,
        "threshold": threshold,
        "accuracy": accuracy,
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "auc": roc_auc(y_true, y_prob),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n": int(len(y_true)),
    }


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    candidates = np.unique(np.concatenate([np.linspace(0.05, 0.95, 181), y_prob]))
    best_threshold = 0.5
    best_score = -math.inf
    for threshold in candidates:
        score = metrics_at_threshold(y_true, y_prob, loss=0.0, threshold=float(threshold))["balanced_accuracy"]
        if float(score) > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def evaluate_checkpoint(
    train_mod: ModuleType,
    args: argparse.Namespace,
    run_name: str,
    band: str,
    fold: str,
    device: torch.device,
) -> dict[str, Any] | None:
    ckpt_path = checkpoint_path(Path(args.output_dir), args.version, run_name, band, fold)
    if not ckpt_path.exists():
        message = f"Missing checkpoint: {ckpt_path}"
        if args.strict:
            raise FileNotFoundError(message)
        print(f"Skipping. {message}")
        return None

    model, saved = load_model(train_mod, ckpt_path, device)
    ckpt_args = saved.get("args", {})
    n_doppler_bins = int(ckpt_args.get("n_doppler_bins_effective", ckpt_args.get("n_doppler_bins", 32)))
    feature_norm = str(ckpt_args.get("feature_norm_effective", ckpt_args.get("feature_norm", "instance")))
    drop_dc = bool(ckpt_args.get("drop_dc_effective", not bool(ckpt_args.get("keep_dc", False))))

    val_loader = build_loader(
        train_mod,
        Path(args.data_dir),
        band,
        fold,
        "val",
        args.batch_size,
        args.workers,
        n_doppler_bins,
        feature_norm,
        drop_dc,
    )
    test_loader = build_loader(
        train_mod,
        Path(args.data_dir),
        band,
        fold,
        "test",
        args.batch_size,
        args.workers,
        n_doppler_bins,
        feature_norm,
        drop_dc,
    )

    val_y, val_prob, val_loss = collect_predictions(model, val_loader, device)
    threshold = choose_threshold(val_y, val_prob) if args.tune_threshold else 0.5
    test_y, test_prob, test_loss = collect_predictions(model, test_loader, device)
    val_metrics = metrics_at_threshold(val_y, val_prob, val_loss, threshold)
    test_metrics = metrics_at_threshold(test_y, test_prob, test_loss, threshold)

    params = int(saved.get("params", count_parameters(model)))
    row: dict[str, Any] = {
        "version": args.version,
        "run_name": run_name,
        "band": band,
        "fold": fold,
        "test_room": ROOM_KEY_TO_NAME[fold],
        "checkpoint": str(ckpt_path),
        "epochs_requested": int(ckpt_args.get("epochs", -1)),
        "width": int(ckpt_args.get("width_effective", ckpt_args.get("width", -1))),
        "n_doppler_bins": n_doppler_bins,
        "feature_norm": feature_norm,
        "dropout": float(ckpt_args.get("dropout", float("nan"))),
        "weight_decay": float(ckpt_args.get("weight_decay", float("nan"))),
        "augment_strength": ckpt_args.get("augment_strength", ""),
        "use_attention": int(bool(ckpt_args.get("use_attention", False))),
        "lambda_domain": float(ckpt_args.get("lambda_domain", 0.0)),
        "lambda_coral": float(ckpt_args.get("lambda_coral", 0.0)),
        "lambda_consistency": float(ckpt_args.get("lambda_consistency", 0.0)),
        "label_smoothing": float(ckpt_args.get("label_smoothing", 0.0)),
        "selected_threshold": threshold,
        "params": params,
        "float32_mb": params * 4 / 1_000_000,
        "int8_weight_mb_est": params / 1_000_000,
    }
    for key, value in val_metrics.items():
        row[f"val_{key}"] = value
    for key, value in test_metrics.items():
        row[f"test_{key}"] = value

    if args.predictions:
        write_predictions(Path(args.output_dir), args.version, run_name, band, fold, test_y, test_prob, threshold)

    print(
        f"{run_name:16s} {band:6s} {fold:11s} "
        f"test_acc={test_metrics['accuracy']:.4f} "
        f"test_bal={test_metrics['balanced_accuracy']:.4f} "
        f"test_f1={test_metrics['f1']:.4f} "
        f"thr={threshold:.3f}"
    )
    return row


def write_predictions(
    out_dir: Path,
    version: str,
    run_name: str,
    band: str,
    fold: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> None:
    path = out_dir / f"{version}_predictions_{run_name}_{band}_{fold}.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_order", "y_true", "y_prob", "y_pred", "threshold"])
        writer.writeheader()
        for i, (label, prob) in enumerate(zip(y_true, y_prob)):
            writer.writerow({
                "sample_order": i,
                "y_true": int(label),
                "y_prob": float(prob),
                "y_pred": int(prob >= threshold),
                "threshold": threshold,
            })


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_keys = [
        "test_accuracy",
        "test_balanced_accuracy",
        "test_f1",
        "test_auc",
        "test_precision",
        "test_recall",
        "test_specificity",
        "val_accuracy",
        "val_balanced_accuracy",
        "params",
        "float32_mb",
        "int8_weight_mb_est",
    ]
    summary = []
    for run_name in sorted(set(str(r["run_name"]) for r in rows)):
        run_rows = [r for r in rows if str(r["run_name"]) == run_name]
        for band in sorted(set(str(r["band"]) for r in run_rows)) + ["overall"]:
            subset = run_rows if band == "overall" else [r for r in run_rows if str(r["band"]) == band]
            if not subset:
                continue
            row = {"version": subset[0]["version"], "run_name": run_name, "band": band, "n_folds": len(subset)}
            for key in metric_keys:
                row[key] = float(np.mean([float(r[key]) for r in subset]))
            summary.append(row)

    baseline = next((r for r in summary if r["run_name"] == "source_only" and r["band"] == "overall"), None)
    deltas = []
    if baseline is not None:
        for row in summary:
            if row["band"] != "overall" or row["run_name"] == "source_only":
                continue
            deltas.append({
                "version": row["version"],
                "run_name": row["run_name"],
                "vs_run": "source_only",
                "delta_test_accuracy": row["test_accuracy"] - baseline["test_accuracy"],
                "delta_test_balanced_accuracy": row["test_balanced_accuracy"] - baseline["test_balanced_accuracy"],
                "delta_test_f1": row["test_f1"] - baseline["test_f1"],
                "delta_test_auc": row["test_auc"] - baseline["test_auc"],
                "delta_params": row["params"] - baseline["params"],
                "delta_int8_weight_mb_est": row["int8_weight_mb_est"] - baseline["int8_weight_mb_est"],
            })
    return summary, deltas


def resolve_run_names(args: argparse.Namespace) -> list[str]:
    if args.run_names:
        return [name.strip() for name in args.run_names.split(",") if name.strip()]
    return DEFAULT_RUNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v5")
    parser.add_argument("--code-dir", default=".")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--run-names", default="", help="Comma-separated run names. Defaults to standard v5 preset names.")
    parser.add_argument("--band", choices=["2.4ghz", "5ghz", "both"], default="both")
    parser.add_argument("--fold", choices=["classroom", "meetingroom", "emptyroom", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--tune-threshold", action="store_true")
    parser.add_argument("--predictions", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    args.version = clean_version(args.version)
    if not args.output_dir:
        args.output_dir = default_output_dir(args.version)
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_mod = load_training_module(Path(args.code_dir))
    device = choose_device(args.device)
    print(f"Using device: {device}")
    print(f"Writing all test files to: {output_dir}")

    run_names = resolve_run_names(args)
    bands = ["2.4ghz", "5ghz"] if args.band == "both" else [args.band]
    folds = ["classroom", "meetingroom", "emptyroom"] if args.fold == "all" else [args.fold]

    rows = []
    for run_name in run_names:
        for band in bands:
            for fold in folds:
                row = evaluate_checkpoint(train_mod, args, run_name, band, fold, device)
                if row is not None:
                    rows.append(row)

    if not rows:
        raise SystemExit("No checkpoints were evaluated.")

    summary, deltas = summarize(rows)
    write_csv(output_dir / f"{args.version}_test_per_fold.csv", rows)
    write_csv(output_dir / f"{args.version}_test_summary.csv", summary)
    write_csv(output_dir / f"{args.version}_test_delta_vs_source_only.csv", deltas)

    print(f"\nSaved test per-fold results: {output_dir / f'{args.version}_test_per_fold.csv'}")
    print(f"Saved test summary: {output_dir / f'{args.version}_test_summary.csv'}")
    if deltas:
        print(f"Saved deltas: {output_dir / f'{args.version}_test_delta_vs_source_only.csv'}")

    print("\nOverall test summary:")
    for row in summary:
        if row["band"] != "overall":
            continue
        print(
            f"{row['run_name']}: "
            f"test_acc={row['test_accuracy']:.4f}, "
            f"test_bal_acc={row['test_balanced_accuracy']:.4f}, "
            f"test_f1={row['test_f1']:.4f}, "
            f"params={int(row['params'])}"
        )


if __name__ == "__main__":
    main()
