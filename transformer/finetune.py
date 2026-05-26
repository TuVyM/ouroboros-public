#!/usr/bin/env python3
"""
Supervised fine-tuning of MultiScaleEncoder on BUY/SELL/HOLD labels.

Usage:
    cd /home/ouroboroz-tech/trading
    .venv/bin/python transformer/finetune.py --symbol BTCUSDT
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformer.encoder import MultiScaleEncoder

PRETRAINED_PATH = Path(__file__).parent / "pretrained.pt"
FINETUNED_PATH = Path(__file__).parent / "finetuned.pt"
WINDOW = 89   # same as orderflow/features.py WINDOW — ohlcv index for output row i is WINDOW+i


class FinetuneDataset(Dataset):
    def __init__(
        self,
        ohlcv: np.ndarray,
        bar_indices: np.ndarray,
        labels: np.ndarray,
    ):
        self.ohlcv = ohlcv.astype(np.float32)
        self.bar_indices = bar_indices
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.bar_indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.bar_indices[i])
        lo = max(0, idx - 719)
        win = self.ohlcv[lo : idx + 1]
        if len(win) < 720:
            pad = np.zeros((720 - len(win), 5), dtype=np.float32)
            win = np.concatenate([pad, win], axis=0)
        return torch.from_numpy(win), torch.tensor(self.labels[i], dtype=torch.long)


def build_regime_labels(
    trend_labels: np.ndarray,
    range_labels: np.ndarray,
    regime_mask: np.ndarray,
) -> np.ndarray:
    """
    Route per-bar labels based on regime.
    regime_mask: bool array, True = ranging bar.
    Returns int64 array: ranging bars get range_labels, others get trend_labels.
    """
    combined = trend_labels.copy().astype(np.int64)
    combined[regime_mask] = range_labels[regime_mask].astype(np.int64)
    return combined


def freeze_bottom_layers(model: MultiScaleEncoder, n_freeze: int = 2) -> None:
    """Freeze input_proj, pos_embed, and bottom n_freeze transformer layers of each encoder."""
    for enc in (model.enc_1h, model.enc_4h, model.enc_1d):
        for p in enc.input_proj.parameters():
            p.requires_grad = False
        enc.pos_embed.requires_grad = False
        for layer in enc.transformer.layers[:n_freeze]:
            for p in layer.parameters():
                p.requires_grad = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if not PRETRAINED_PATH.exists():
        raise SystemExit(
            f"pretrained.pt not found at {PRETRAINED_PATH}.\n"
            "Run: .venv/bin/python transformer/pretrain.py --symbol BTCUSDT"
        )

    # ---- Load OHLCV ----
    from backtest_feed import _find_cache_file, DATA_CACHE_DIR
    cache = _find_cache_file(args.symbol, DATA_CACHE_DIR, interval="1h")
    if cache is None:
        raise SystemExit(f"No 1h cache for {args.symbol}.")
    ts_path = cache.replace(".npy", "_ts.npy")
    ohlcv = np.load(cache).astype(np.float32)
    timestamps = np.load(ts_path)
    print(f"Loaded {len(ohlcv)} bars")

    # ---- Build feature matrix (24-feature) and aligned OHLCV indices ----
    from orderflow.features import build_features
    X, out_ts = build_features(args.symbol, ohlcv, timestamps)
    M = len(X)
    # Output row i corresponds to ohlcv bar at index WINDOW + i
    bar_indices = np.arange(WINDOW, WINDOW + M)

    # ---- Compute regime mask (classify_batch — no hysteresis) ----
    from lgbm.regime_detector import RegimeDetector
    adx = RegimeDetector.compute_adx(ohlcv)                # (N-1,)
    regime_batch = RegimeDetector.classify_batch(adx)       # (N-1,) object array
    # Pad to N (first bar inherits from second)
    regime_full = np.empty(len(ohlcv), dtype=object)
    regime_full[0] = regime_batch[0]
    regime_full[1:] = regime_batch
    regime_mask = np.array([regime_full[i] == "ranging" for i in bar_indices])
    print(f"Ranging bars: {regime_mask.sum()} / {M}")

    # ---- Compute labels ----
    from lgbm.labels import triple_barrier_labels
    from lgbm.labels_range import value_area_labels
    from orderflow.range_features import build_range_features

    closes = ohlcv[bar_indices, 3].astype(np.float64)
    highs  = ohlcv[bar_indices, 1].astype(np.float64)
    lows   = ohlcv[bar_indices, 2].astype(np.float64)

    trend_y = triple_barrier_labels(closes, highs, lows, tp=0.010, sl=0.005, horizon=8)

    X_range, _, va_levels = build_range_features(args.symbol, ohlcv, timestamps)
    tr = np.zeros(len(closes))
    tr[1:] = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])),
    )
    atr14 = np.convolve(tr, np.ones(14) / 14, mode="same")
    range_y = value_area_labels(closes, highs, lows, atr14, va_levels, regime_mask, X_range, horizon=6)

    M_labels = min(len(trend_y), len(range_y))
    y = build_regime_labels(trend_y[:M_labels], range_y[:M_labels], regime_mask[:M_labels])
    bar_indices = bar_indices[:M_labels]
    print(f"Labels: sell={int((y==0).sum())} hold={int((y==1).sum())} buy={int((y==2).sum())}")

    # ---- Time-series split ----
    GAP = 8
    split = int(M_labels * 0.8)
    tr_idx, tr_y = bar_indices[:split - GAP], y[:split - GAP]
    va_idx, va_y = bar_indices[split:], y[split:]

    train_ds = FinetuneDataset(ohlcv, tr_idx, tr_y)
    val_ds = FinetuneDataset(ohlcv, va_idx, va_y)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # ---- Class weights ----
    counts = np.bincount(tr_y, minlength=3).clip(1).astype(np.float32)
    weights = torch.tensor(len(tr_y) / (3 * counts)).to(args.device)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # ---- Model ----
    model = MultiScaleEncoder().to(args.device)
    state = torch.load(PRETRAINED_PATH, map_location=args.device, weights_only=True)
    model.load_state_dict(state)
    freeze_bottom_layers(model, n_freeze=2)

    head = nn.Linear(64, 3).to(args.device)
    trainable = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    total_steps = args.epochs * len(train_dl)
    warmup_steps = args.warmup_epochs * len(train_dl)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val = float("inf")
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        model.train(); head.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(args.device), yb.to(args.device)
            emb = model(xb)
            loss = F.cross_entropy(head(emb), yb, weight=weights)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        model.eval(); head.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(args.device), yb.to(args.device)
                val_loss += F.cross_entropy(head(model(xb)), yb, weight=weights).item()
        train_loss /= len(train_dl)
        val_loss /= len(val_dl)
        print(f"Epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_count = 0
            torch.save(model.state_dict(), FINETUNED_PATH)
            print(f"  Saved → {FINETUNED_PATH}")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nFine-tuning complete. Best val loss: {best_val:.4f}")
    print(f"Checkpoint: {FINETUNED_PATH}")


if __name__ == "__main__":
    main()
