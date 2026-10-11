#!/usr/bin/env python3

from __future__ import annotations
import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
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

PRESETS: dict[str, dict[str, Any]] = {
    "source_only": {
        "augment_strength": "none",
        "lambda_domain": 0.0,
        "lambda_coral": 0.0,
        "lambda_consistency": 0.0,
        "use_attention": False,
    },
    "strong_aug": {
        "augment_strength": "strong",
        "lambda_domain": 0.0,
        "lambda_coral": 0.0,
        "lambda_consistency": 0.0,
        "use_attention": False,
    },
    "consistency": {
        "augment_strength": "strong",
        "lambda_domain": 0.0,
        "lambda_coral": 0.0,
        "lambda_consistency": 0.30,
        "use_attention": False,
    },
    "cond_coral": {
        "augment_strength": "strong",
        "lambda_domain": 0.0,
        "lambda_coral": 0.05,
        "lambda_consistency": 0.30,
        "use_attention": False,
    },
    "dann": {
        "augment_strength": "strong",
        "lambda_domain": 0.15,
        "lambda_coral": 0.0,
        "lambda_consistency": 0.30,
        "use_attention": False,
    },
    "full": {
        "augment_strength": "strong",
        "lambda_domain": 0.15,
        "lambda_coral": 0.05,
        "lambda_consistency": 0.30,
        "use_attention": True,
    },
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def clean_version(value: str) -> str:
    value = value.strip().lower()
    if not value:
        return "v5"
    return value if value.startswith("v") else f"v{value}"


def default_output_dir(version: str) -> str:
    return f"runs_router_csi_{version}"


def effective_width(args: argparse.Namespace, band: str) -> int:
    if args.width > 0:
        return args.width
    return args.width_5ghz if band == "5ghz" else args.width_24ghz


def effective_doppler_bins(args: argparse.Namespace, band: str) -> int:
    if args.n_doppler_bins > 0:
        return args.n_doppler_bins
    return args.n_doppler_bins_5ghz if band == "5ghz" else args.n_doppler_bins_24ghz


def effective_feature_norm(args: argparse.Namespace, band: str) -> str:
    if band == "2.4ghz" and args.use_v3_24ghz_features:
        return "peak"
    return args.feature_norm


def effective_drop_dc(args: argparse.Namespace, band: str) -> bool:
    if band == "2.4ghz" and args.use_v3_24ghz_features:
        return False
    return not args.keep_dc


def amplitude_doppler_spectrogram(
    x: torch.Tensor,
    n_doppler_bins: int,
    feature_norm: str = "instance",
    drop_dc: bool = True,
) -> torch.Tensor:
    """Convert [T,2,2,30] raw amplitude CSI into [F,2,2,30] Doppler features."""
    t = x.shape[0]
    hann = torch.hann_window(t, device=x.device, dtype=x.dtype).view(t, 1, 1, 1)
    centered = x - x.mean(dim=0, keepdim=True)
    n_fft = max(t, (n_doppler_bins + int(drop_dc)) * 2)
    spec = torch.fft.rfft(centered * hann, n=n_fft, dim=0).abs().pow(2)

    start = 1 if drop_dc else 0
    spec = spec[start : start + n_doppler_bins]
    spec_db = 10.0 * torch.log10(spec + 1e-8)

    if feature_norm == "peak":
        spec_db = spec_db - spec_db.amax(dim=0, keepdim=True)
    elif feature_norm == "instance":
        spec_db = (spec_db - spec_db.mean()) / (spec_db.std(unbiased=False) + 1e-5)
    elif feature_norm == "channel_instance":
        spec_db = (spec_db - spec_db.mean(dim=0, keepdim=True)) / (
            spec_db.std(dim=0, keepdim=True, unbiased=False) + 1e-5
        )
    elif feature_norm != "none":
        raise ValueError(f"Unknown feature_norm: {feature_norm}")

    return spec_db


class CSIFoldDataset(Dataset):
    def __init__(
        self,
        x_path: Path,
        y_path: Path,
        env_path: Path,
        indices: list[int],
        n_doppler_bins: int,
        feature_norm: str,
        drop_dc: bool,
    ) -> None:
        self.X = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        self.env = np.load(env_path, allow_pickle=True)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.n_doppler_bins = n_doppler_bins
        self.feature_norm = feature_norm
        self.drop_dc = drop_dc

        source_rooms = sorted(set(str(v) for v in self.env[self.indices]))
        self.room_to_id = {room: i for i, room in enumerate(source_rooms)}

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int):
        idx = int(self.indices[item])
        x = torch.from_numpy(np.asarray(self.X[idx], dtype=np.float32))
        x = amplitude_doppler_spectrogram(
            x,
            n_doppler_bins=self.n_doppler_bins,
            feature_norm=self.feature_norm,
            drop_dc=self.drop_dc,
        )
        y = torch.tensor(float(self.y[idx]), dtype=torch.float32)
        domain = torch.tensor(self.room_to_id.get(str(self.env[idx]), -1), dtype=torch.long)
        return x, y, domain


def augment_csi_batch(x: torch.Tensor, strength: str) -> torch.Tensor:
    """Augment Doppler features. x shape: [B,F,2,2,30]."""
    if strength == "none":
        return x

    out = x.clone()
    b, f, rx, tx, sc = out.shape
    strong = strength == "strong"

    if torch.rand((), device=out.device) < 0.85:
        gain = torch.empty(b, 1, 1, 1, 1, device=out.device).uniform_(0.80 if strong else 0.92, 1.25 if strong else 1.08)
        out = out * gain

    if torch.rand((), device=out.device) < 0.80:
        sigma = 0.08 if strong else 0.03
        out = out + torch.randn_like(out) * sigma

    if torch.rand((), device=out.device) < 0.55:
        max_bins = max(2, int(round(f * (0.18 if strong else 0.08))))
        for i in range(b):
            width = int(torch.randint(1, max_bins + 1, (1,), device=out.device).item())
            start = int(torch.randint(0, max(1, f - width + 1), (1,), device=out.device).item())
            out[i, start : start + width] = 0.0

    if torch.rand((), device=out.device) < 0.65:
        max_sc = 8 if strong else 3
        for i in range(b):
            width = int(torch.randint(1, max_sc + 1, (1,), device=out.device).item())
            start = int(torch.randint(0, sc - width + 1, (1,), device=out.device).item())
            out[i, ..., start : start + width] = 0.0

    if torch.rand((), device=out.device) < 0.40:
        for i in range(b):
            r = int(torch.randint(0, rx, (1,), device=out.device).item())
            c = int(torch.randint(0, tx, (1,), device=out.device).item())
            out[i, :, r, c, :] = 0.0

    if torch.rand((), device=out.device) < 0.45:
        max_shift = max(1, int(round(f * (0.06 if strong else 0.03))))
        for i in range(b):
            shift = int(torch.randint(-max_shift, max_shift + 1, (1,), device=out.device).item())
            out[i] = torch.roll(out[i], shifts=shift, dims=0)

    return out


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


class SubcarrierAttention(nn.Module):
    def __init__(self, subcarriers: int = 30, hidden: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(subcarriers, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, subcarriers),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.abs().mean(dim=(1, 2, 3))
        weights = self.net(pooled).view(x.shape[0], 1, 1, 1, x.shape[-1])
        return x * weights


class IdentityAttention(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class DepthwiseTemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = ((kernel_size - 1) // 2) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation, groups=channels, bias=False),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            group_norm(channels),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TinyCSIDomainGeneralizerV5(nn.Module):
    def __init__(self, width: int = 16, dropout: float = 0.30, domain_classes: int = 2, use_attention: bool = False) -> None:
        super().__init__()
        self.width = width
        self.use_attention = use_attention
        self.attn = SubcarrierAttention() if use_attention else IdentityAttention()
        self.stem = nn.Sequential(
            nn.Conv1d(120, width, kernel_size=1, bias=False),
            group_norm(width),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            DepthwiseTemporalBlock(width, 3, 1, dropout),
            DepthwiseTemporalBlock(width, 5, 2, dropout),
            DepthwiseTemporalBlock(width, 7, 4, dropout),
        )
        self.classifier = nn.Sequential(nn.LayerNorm(width), nn.Dropout(dropout), nn.Linear(width, 1))
        self.domain_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(width, domain_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = x.permute(0, 2, 3, 4, 1).reshape(x.shape[0], 120, x.shape[1])
        x = self.stem(x)
        x = self.blocks(x)
        return x.mean(dim=-1)

    def forward(self, x: torch.Tensor, grl_lambda: float = 0.0):
        features = self.encode(x)
        label_logits = self.classifier(features).squeeze(-1)
        domain_logits = self.domain_head(grad_reverse(features, grl_lambda))
        return label_logits, domain_logits, features


TinyCSIDomainGeneralizer = TinyCSIDomainGeneralizerV5


def covariance(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    return x.T @ x / max(1, x.shape[0] - 1)


def coral_loss(features: torch.Tensor, domains: torch.Tensor) -> torch.Tensor:
    unique = torch.unique(domains[domains >= 0])
    if unique.numel() < 2:
        return features.new_tensor(0.0)
    a = features[domains == unique[0]]
    b = features[domains == unique[1]]
    if a.shape[0] < 2 or b.shape[0] < 2:
        return features.new_tensor(0.0)
    return (covariance(a) - covariance(b)).pow(2).mean()


def conditional_coral_loss(features: torch.Tensor, domains: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    losses = []
    hard_labels = (labels >= 0.5).long()
    for cls in [0, 1]:
        mask = hard_labels == cls
        if int(mask.sum()) < 4:
            continue
        loss = coral_loss(features[mask], domains[mask])
        if torch.isfinite(loss):
            losses.append(loss)
    return torch.stack(losses).mean() if losses else features.new_tensor(0.0)


def consistency_loss(clean_features: torch.Tensor, aug_features: torch.Tensor) -> torch.Tensor:
    clean = F.normalize(clean_features, dim=1)
    aug = F.normalize(aug_features, dim=1)
    return 1.0 - (clean * aug).sum(dim=1).mean()


@dataclass
class Metrics:
    loss: float
    accuracy: float
    balanced_accuracy: float
    f1: float
    auc: float


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


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, loss: float) -> Metrics:
    y_pred = (y_prob >= 0.5).astype(np.int64)
    y_true = y_true.astype(np.int64)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / max(1, len(y_true))
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return Metrics(loss=loss, accuracy=acc, balanced_accuracy=(recall + specificity) / 2, f1=f1, auc=roc_auc(y_true, y_prob))


def evaluate_val(model: nn.Module, loader: DataLoader, device: torch.device) -> Metrics:
    model.eval()
    losses = []
    probs = []
    labels = []
    with torch.no_grad():
        for x, y, _domains in loader:
            x = x.to(device)
            y = y.to(device)
            logits, _domain_logits, _features = model(x, grl_lambda=0.0)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            losses.append(float(loss.item()) * x.shape[0])
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y.cpu().numpy())
    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probs)
    return binary_metrics(y_true, y_prob, loss=sum(losses) / max(1, len(y_true)))


def make_pos_weight(y_path: Path, indices: Iterable[int], device: torch.device) -> torch.Tensor:
    y = np.load(y_path, mmap_mode="r")[list(indices)]
    positives = float((y == 1).sum())
    negatives = float((y == 0).sum())
    return torch.tensor([negatives / max(1.0, positives)], dtype=torch.float32, device=device)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def export_int8_weights(model: nn.Module, path: Path) -> Path:
    payload = {}
    for name, tensor in model.state_dict().items():
        arr = tensor.detach().cpu().numpy()
        if not np.issubdtype(arr.dtype, np.floating):
            payload[name] = arr
            continue
        max_abs = float(np.max(np.abs(arr)))
        scale = max(max_abs / 127.0, 1e-12)
        payload[name] = np.round(arr / scale).clip(-127, 127).astype(np.int8)
        payload[f"{name}__scale"] = np.asarray(scale, dtype=np.float32)
    np.savez_compressed(path, **payload)
    return path


def grad_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().item())
    return math.sqrt(total)


def grl_value(epoch: int, epochs: int, mode: str, max_lambda: float) -> float:
    if max_lambda <= 0:
        return 0.0
    if mode == "constant":
        return max_lambda
    progress = epoch / max(1, epochs)
    return max_lambda * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


def train_one_fold(args: argparse.Namespace, band: str, fold_key: str, device: torch.device) -> dict[str, Any]:
    band_offset = 0 if band == "2.4ghz" else 100
    fold_offset = {"classroom": 0, "meetingroom": 10, "emptyroom": 20}[fold_key]
    run_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(args.run_name)) % 1000
    seed_everything(args.seed + band_offset + fold_offset + run_offset)

    data_dir = Path(args.data_dir)
    splits = json.loads((data_dir / f"loeo_splits_{band}.json").read_text())
    fold = splits[fold_key]
    n_doppler_bins = effective_doppler_bins(args, band)
    width = effective_width(args, band)
    feature_norm = effective_feature_norm(args, band)
    drop_dc = effective_drop_dc(args, band)

    x_path = data_dir / f"X_csi_50Hz_2x2_{band}.npy"
    y_path = data_dir / f"y_labels_{band}.npy"
    env_path = data_dir / f"env_labels_{band}.npy"

    train_ds = CSIFoldDataset(x_path, y_path, env_path, fold["train"], n_doppler_bins, feature_norm, drop_dc)
    val_ds = CSIFoldDataset(x_path, y_path, env_path, fold["val"], n_doppler_bins, feature_norm, drop_dc)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = TinyCSIDomainGeneralizerV5(width=width, dropout=args.dropout, domain_classes=2, use_attention=args.use_attention).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    pos_weight = make_pos_weight(y_path, fold["train"], device)

    best_state = None
    best_val = -math.inf
    stale = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {k: 0.0 for k in ["loss", "cls_loss", "domain_loss", "align_loss", "consistency_loss", "encoder_grad_norm", "classifier_grad_norm", "domain_grad_norm"]}
        n_train = 0
        grl_lambda = grl_value(epoch, args.epochs, args.grl_schedule, args.lambda_domain)

        for x, y, domains in train_loader:
            x = x.to(device)
            y = y.to(device)
            domains = domains.to(device)
            x_aug = augment_csi_batch(x, args.augment_strength)
            y_train = y * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing

            optimizer.zero_grad(set_to_none=True)
            logits, domain_logits, features = model(x_aug, grl_lambda=grl_lambda)
            cls_loss = F.binary_cross_entropy_with_logits(logits, y_train, pos_weight=pos_weight)
            domain_loss = F.cross_entropy(domain_logits, domains) if args.lambda_domain > 0 else features.new_tensor(0.0)
            align_loss = conditional_coral_loss(features, domains, y) if args.lambda_coral > 0 else features.new_tensor(0.0)

            if args.lambda_consistency > 0 and args.augment_strength != "none":
                with torch.no_grad():
                    clean_logits, _clean_domain, clean_features = model(x, grl_lambda=0.0)
                feat_cons = consistency_loss(clean_features, features)
                prob_cons = F.mse_loss(torch.sigmoid(logits), torch.sigmoid(clean_logits))
                cons_loss = feat_cons + 0.25 * prob_cons
            else:
                cons_loss = features.new_tensor(0.0)

            loss = cls_loss + domain_loss + args.lambda_coral * align_loss + args.lambda_consistency * cons_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            enc_params = list(model.stem.parameters()) + list(model.blocks.parameters())
            if args.use_attention:
                enc_params += list(model.attn.parameters())
            batch_n = x.shape[0]
            sums["encoder_grad_norm"] += grad_norm(enc_params) * batch_n
            sums["classifier_grad_norm"] += grad_norm(model.classifier.parameters()) * batch_n
            sums["domain_grad_norm"] += grad_norm(model.domain_head.parameters()) * batch_n

            optimizer.step()

            n_train += batch_n
            sums["loss"] += float(loss.item()) * batch_n
            sums["cls_loss"] += float(cls_loss.item()) * batch_n
            sums["domain_loss"] += float(domain_loss.item()) * batch_n
            sums["align_loss"] += float(align_loss.item()) * batch_n
            sums["consistency_loss"] += float(cons_loss.item()) * batch_n

        scheduler.step()
        val = evaluate_val(model, val_loader, device)
        score = val.balanced_accuracy
        row = {
            "version": args.version,
            "run_name": args.run_name,
            "band": band,
            "fold": fold_key,
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "grl_lambda": grl_lambda,
            "n_doppler_bins": n_doppler_bins,
            "width": width,
            "feature_norm": feature_norm,
            "drop_dc": int(drop_dc),
            "use_v3_24ghz_features": int(args.use_v3_24ghz_features),
            "augment_strength": args.augment_strength,
            "lambda_domain": args.lambda_domain,
            "lambda_coral": args.lambda_coral,
            "lambda_consistency": args.lambda_consistency,
            "label_smoothing": args.label_smoothing,
            "train_loss": sums["loss"] / max(1, n_train),
            "train_cls_loss": sums["cls_loss"] / max(1, n_train),
            "train_domain_loss": sums["domain_loss"] / max(1, n_train),
            "train_align_loss": sums["align_loss"] / max(1, n_train),
            "train_consistency_loss": sums["consistency_loss"] / max(1, n_train),
            "encoder_grad_norm": sums["encoder_grad_norm"] / max(1, n_train),
            "classifier_grad_norm": sums["classifier_grad_norm"] / max(1, n_train),
            "domain_grad_norm": sums["domain_grad_norm"] / max(1, n_train),
            "val_loss": val.loss,
            "val_accuracy": val.accuracy,
            "val_balanced_accuracy": val.balanced_accuracy,
            "val_f1": val.f1,
            "val_auc": val.auc,
        }
        history.append(row)

        if score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if args.verbose:
            print(
                f"{args.version} {args.run_name} {band} {fold_key} epoch {epoch:03d} "
                f"val_bal={val.balanced_accuracy:.4f} cls={row['train_cls_loss']:.4f} "
                f"dom={row['train_domain_loss']:.4f} coral={row['train_align_loss']:.6f}"
            )

        if stale >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val = evaluate_val(model, val_loader, device)
    params = count_parameters(model)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"tiny_csi_dg_{args.version}_{args.run_name}_{band}_{fold_key}.pt"
    history_path = out_dir / f"history_{args.version}_{args.run_name}_{band}_{fold_key}.csv"

    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args) | {
                "width_effective": width,
                "n_doppler_bins_effective": n_doppler_bins,
                "feature_norm_effective": feature_norm,
                "drop_dc_effective": drop_dc,
            },
            "band": band,
            "fold": fold_key,
            "heldout_room": ROOM_KEY_TO_NAME[fold_key],
            "run_name": args.run_name,
            "version": args.version,
            "params": params,
            "model_class": "TinyCSIDomainGeneralizerV5",
        },
        ckpt_path,
    )
    write_csv(history_path, history)

    int8_path = ""
    if args.export_int8_weights:
        int8_path = str(export_int8_weights(model, out_dir / f"tiny_csi_dg_{args.version}_{args.run_name}_{band}_{fold_key}_int8_weights.npz"))

    return {
        "version": args.version,
        "run_name": args.run_name,
        "band": band,
        "fold": fold_key,
        "heldout_room": ROOM_KEY_TO_NAME[fold_key],
        "epochs_requested": args.epochs,
        "epochs_trained": len(history),
        "n_doppler_bins": n_doppler_bins,
        "feature_norm": feature_norm,
        "drop_dc": int(drop_dc),
        "use_v3_24ghz_features": int(args.use_v3_24ghz_features),
        "width": width,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "augment_strength": args.augment_strength,
        "use_attention": int(args.use_attention),
        "lambda_domain": args.lambda_domain,
        "lambda_coral": args.lambda_coral,
        "lambda_consistency": args.lambda_consistency,
        "label_smoothing": args.label_smoothing,
        "params": params,
        "float32_mb": params * 4 / 1_000_000,
        "int8_weight_mb_est": params / 1_000_000,
        "val_accuracy": val.accuracy,
        "val_balanced_accuracy": val.balanced_accuracy,
        "val_f1": val.f1,
        "val_auc": val.auc,
        "checkpoint": str(ckpt_path),
        "history_csv": str(history_path),
        "int8_weights": int8_path,
        "_history_rows": history,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_keys = ["val_accuracy", "val_balanced_accuracy", "val_f1", "val_auc", "params", "float32_mb", "int8_weight_mb_est"]
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
    return summary


def write_results(rows: list[dict[str, Any]], output_dir: Path, version: str) -> None:
    public_rows = []
    all_history = []
    for row in rows:
        public_row = dict(row)
        all_history.extend(public_row.pop("_history_rows", []))
        public_rows.append(public_row)

    write_csv(output_dir / f"{version}_train_val_per_fold.csv", public_rows)
    write_csv(output_dir / f"{version}_train_val_summary.csv", summarize(public_rows))
    write_csv(output_dir / f"{version}_train_history_all.csv", all_history)
    print(f"\nSaved train/val per-fold results: {output_dir / f'{version}_train_val_per_fold.csv'}")
    print(f"Saved train/val summary: {output_dir / f'{version}_train_val_summary.csv'}")
    print(f"Saved combined training history: {output_dir / f'{version}_train_history_all.csv'}")


def apply_preset(args: argparse.Namespace, preset: str) -> argparse.Namespace:
    run_args = argparse.Namespace(**vars(args))
    if preset != "custom":
        for key, value in PRESETS[preset].items():
            setattr(run_args, key, value)
    if not run_args.run_name:
        run_args.run_name = preset
    return run_args


def grid_runs(args: argparse.Namespace) -> list[argparse.Namespace]:
    runs = []
    for lambda_domain in [0.05, 0.15, 0.30]:
        for lambda_coral in [0.0, 0.05, 0.10]:
            run_args = argparse.Namespace(**vars(args))
            run_args.augment_strength = "strong"
            run_args.lambda_domain = lambda_domain
            run_args.lambda_coral = lambda_coral
            run_args.lambda_consistency = args.lambda_consistency
            run_args.use_attention = False
            run_args.run_name = f"grid_d{lambda_domain:g}_c{lambda_coral:g}_cons{args.lambda_consistency:g}".replace(".", "p")
            runs.append(run_args)
    return runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v5")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--band", choices=["2.4ghz", "5ghz", "both"], default="both")
    parser.add_argument("--fold", choices=["classroom", "meetingroom", "emptyroom", "all"], default="all")
    parser.add_argument("--preset", choices=list(PRESETS) + ["all", "grid", "custom"], default="all")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=0, help="Override both band widths. 0 uses band-aware defaults.")
    parser.add_argument("--width-24ghz", type=int, default=16)
    parser.add_argument("--width-5ghz", type=int, default=24)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--n-doppler-bins", type=int, default=0, help="Override both band Doppler bins. 0 uses band-aware defaults.")
    parser.add_argument("--n-doppler-bins-24ghz", type=int, default=32)
    parser.add_argument("--n-doppler-bins-5ghz", type=int, default=64)
    parser.add_argument("--feature-norm", choices=["instance", "channel_instance", "peak", "none"], default="instance")
    parser.add_argument("--keep-dc", action="store_true")
    parser.add_argument("--use-v3-24ghz-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment-strength", choices=["none", "mild", "strong"], default="strong")
    parser.add_argument("--use-attention", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lambda-domain", type=float, default=0.15)
    parser.add_argument("--lambda-coral", type=float, default=0.05)
    parser.add_argument("--lambda-consistency", type=float, default=0.30)
    parser.add_argument("--label-smoothing", type=float, default=0.10)
    parser.add_argument("--grl-schedule", choices=["constant", "ramp"], default="ramp")
    parser.add_argument("--grad-clip", type=float, default=3.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--export-int8-weights", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    args.version = clean_version(args.version)
    if not args.output_dir:
        args.output_dir = default_output_dir(args.version)
    return args


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    print(f"Using device: {device}")
    print(f"Writing all training files to: {args.output_dir}")

    bands = ["2.4ghz", "5ghz"] if args.band == "both" else [args.band]
    folds = ["classroom", "meetingroom", "emptyroom"] if args.fold == "all" else [args.fold]

    if args.preset == "all":
        run_args_list = [apply_preset(args, preset) for preset in PRESETS]
    elif args.preset == "grid":
        run_args_list = grid_runs(args)
    else:
        run_args_list = [apply_preset(args, args.preset)]

    rows = []
    for run_args in run_args_list:
        for band in bands:
            for fold in folds:
                print(f"\n=== Training {run_args.version} run={run_args.run_name}, band={band}, heldout_room={ROOM_KEY_TO_NAME[fold]} ===")
                rows.append(train_one_fold(run_args, band, fold, device))

    write_results(rows, Path(args.output_dir), args.version)


if __name__ == "__main__":
    main()
