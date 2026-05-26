#!/usr/bin/env python3
"""
Self-supervised masked bar reconstruction pretraining for MultiScaleEncoder.

Usage:
    cd /home/ouroboroz-tech/trading
    .venv/bin/python transformer/pretrain.py --symbol BTCUSDT
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
DATA_DIR = Path(__file__).parent.parent / "data_cache"
WIN = 720  # 1h bars per sample


class OHLCVDataset(Dataset):
    def __init__(self, ohlcv: np.ndarray, indices: np.ndarray | None = None):
        self.ohlcv = ohlcv.astype(np.float32)
        # Default: all positions where we have a full WIN-bar history
        self.indices = indices if indices is not None else np.arange(WIN - 1, len(ohlcv))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> torch.Tensor:
        end = int(self.indices[i])
        return torch.from_numpy(self.ohlcv[end - WIN + 1 : end + 1].copy())


def apply_mask(
    x: torch.Tensor, mask_ratio: float = 0.20
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero out mask_ratio of bars. Returns (masked_x, bool_mask) where True = masked."""
    B, T, _ = x.shape
    mask = torch.rand(B, T, device=x.device) < mask_ratio
    x_masked = x.clone()
    x_masked[mask] = 0.0
    return x_masked, mask


def make_train_val_split(
    ohlcv: np.ndarray,
    val_frac: float = 0.10,
    gap: int = 168,
) -> tuple[OHLCVDataset, OHLCVDataset]:
    """Time-series split with a gap to prevent sliding-window leakage.

    Val set starts at val_frac from the end. Train set ends gap positions
    before the val set (measured in sample positions, which equals bar positions
    since all_idx has stride 1). This guarantees the bar-index gap between the
    last training window and first validation window exceeds the window overlap.
    """
    all_idx = np.arange(WIN - 1, len(ohlcv))
    val_start = int(len(all_idx) * (1 - val_frac))
    train_idx = all_idx[:val_start - gap]
    val_idx = all_idx[val_start:]
    return OHLCVDataset(ohlcv, train_idx), OHLCVDataset(ohlcv, val_idx)


def _pretrain_epoch(
    model: MultiScaleEncoder,
    decoders: dict[str, nn.Linear],
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    mask_ratio: float = 0.20,
) -> float:
    model.train()
    for dec in decoders.values():
        dec.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)                        # (B, 720, 5)
        x1h = batch[:, -168:, :]                       # (B, 168, 5)
        x4h = MultiScaleEncoder._resample(batch[:, -168:, :], 4)   # (B, 42, 5)
        x1d = MultiScaleEncoder._resample(batch, 24)[:, -30:, :]   # (B, 30, 5)

        x1h_n = MultiScaleEncoder._znorm(x1h)
        x4h_n = MultiScaleEncoder._znorm(x4h)
        x1d_n = MultiScaleEncoder._znorm(x1d)

        x1h_m, m1h = apply_mask(x1h_n, mask_ratio)
        x4h_m, m4h = apply_mask(x4h_n, mask_ratio)
        x1d_m, m1d = apply_mask(x1d_n, mask_ratio)

        seq1h = model.enc_1h(x1h_m, return_sequence=True)   # (B, 168, 64)
        seq4h = model.enc_4h(x4h_m, return_sequence=True)   # (B, 42, 64)
        seq1d = model.enc_1d(x1d_m, return_sequence=True)   # (B, 30, 64)

        loss_1h = F.mse_loss(decoders["1h"](seq1h)[m1h], x1h_n[m1h])
        loss_4h = F.mse_loss(decoders["4h"](seq4h)[m4h], x4h_n[m4h])
        loss_1d = F.mse_loss(decoders["1d"](seq1d)[m1d], x1d_n[m1d])
        loss = loss_1h + loss_4h + loss_1d

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(model.parameters()) + [p for d in decoders.values() for p in d.parameters()],
            max_norm=1.0,
        )
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def _val_loss(
    model: MultiScaleEncoder,
    decoders: dict[str, nn.Linear],
    loader: DataLoader,
    device: str,
    mask_ratio: float = 0.20,
) -> float:
    model.eval()
    for dec in decoders.values():
        dec.eval()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        x1h = MultiScaleEncoder._znorm(batch[:, -168:, :])
        x4h = MultiScaleEncoder._znorm(MultiScaleEncoder._resample(batch[:, -168:, :], 4))
        x1d = MultiScaleEncoder._znorm(MultiScaleEncoder._resample(batch, 24)[:, -30:, :])
        x1h_m, m1h = apply_mask(x1h, mask_ratio)
        x4h_m, m4h = apply_mask(x4h, mask_ratio)
        x1d_m, m1d = apply_mask(x1d, mask_ratio)
        loss = (
            F.mse_loss(decoders["1h"](model.enc_1h(x1h_m, return_sequence=True))[m1h], x1h[m1h])
            + F.mse_loss(decoders["4h"](model.enc_4h(x4h_m, return_sequence=True))[m4h], x4h[m4h])
            + F.mse_loss(decoders["1d"](model.enc_1d(x1d_m, return_sequence=True))[m1d], x1d[m1d])
        )
        total += loss.item()
    return total / len(loader)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    from backtest_feed import _find_cache_file, DATA_CACHE_DIR
    cache = _find_cache_file(args.symbol, DATA_CACHE_DIR, interval="1h")
    if cache is None:
        raise SystemExit(f"No 1h cache for {args.symbol}.")
    ohlcv = np.load(cache).astype(np.float32)
    print(f"Loaded {len(ohlcv)} bars from {cache}")

    train_ds, val_ds = make_train_val_split(ohlcv, val_frac=0.10, gap=168)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = MultiScaleEncoder().to(args.device)
    decoders = {
        "1h": nn.Linear(64, 5).to(args.device),
        "4h": nn.Linear(64, 5).to(args.device),
        "1d": nn.Linear(64, 5).to(args.device),
    }
    all_params = list(model.parameters()) + [p for d in decoders.values() for p in d.parameters()]
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(train_dl)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    best_val = float("inf")
    patience_count = 0
    for epoch in range(1, args.epochs + 1):
        tr_loss = _pretrain_epoch(model, decoders, train_dl, optimizer, args.device)
        scheduler.step()
        val_loss = _val_loss(model, decoders, val_dl, args.device)
        print(f"Epoch {epoch:3d}  train={tr_loss:.4f}  val={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            patience_count = 0
            torch.save(model.state_dict(), PRETRAINED_PATH)
            print(f"  Saved → {PRETRAINED_PATH}  (val={val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nPretraining complete. Best val MSE: {best_val:.4f}")
    print(f"Checkpoint: {PRETRAINED_PATH}")


if __name__ == "__main__":
    main()
