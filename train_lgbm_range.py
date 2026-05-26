#!/usr/bin/env python3
"""
Train LightGBM range specialist on BTC value-area features.

Usage:
    python train_lgbm_range.py                    # uses longest available 1h cache
    python train_lgbm_range.py --symbol BTCUSDT
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np

from backtest_feed import _find_cache_file, DATA_CACHE_DIR
from lgbm.regime_detector import RegimeDetector
from lgbm.labels_range import value_area_labels
from orderflow.range_features import build_range_features

MODEL_OUT = Path("lgbm/btc_range_model.lgbm")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    args = p.parse_args()

    cache_path = _find_cache_file(args.symbol, DATA_CACHE_DIR, interval="1h")
    if cache_path is None:
        raise SystemExit(
            f"No 1h cache for {args.symbol}. "
            "Run: python pretrain.py --symbols BTCUSDT --days 365 --interval 1h --cache-only"
        )
    ts_path    = cache_path.replace(".npy", "_ts.npy")
    ohlcv      = np.load(cache_path)
    timestamps = np.load(ts_path)
    print(f"OHLCV loaded: {len(ohlcv)} bars from {cache_path}")

    # ---- Step 1: Compute ADX regime labels (batch mode — stateless, reproducible) ----
    print("Computing ADX regime labels (batch mode)…")
    adx          = RegimeDetector.compute_adx(ohlcv)            # (N-1,)
    regime_batch = RegimeDetector.classify_batch(adx)           # (N-1,) object array
    # Pad to length N (first bar gets regime of second)
    regime_full  = np.empty(len(ohlcv), dtype=object)
    regime_full[0]  = regime_batch[0]
    regime_full[1:] = regime_batch

    # ---- Step 2: Build range feature matrix (with encoder embeddings if finetuned.pt exists) ----
    from transformer.embed import load_encoder
    encoder = load_encoder()
    if encoder is not None:
        from orderflow.range_features import build_range_features_with_embedding
        print("Building range feature matrix (93 features — with encoder embeddings)…")
        X, out_ts, va_levels = build_range_features_with_embedding(
            args.symbol, ohlcv, timestamps, encoder
        )
    else:
        print("Building range feature matrix (29 features — no finetuned.pt found)…")
        X, out_ts, va_levels = build_range_features(args.symbol, ohlcv, timestamps)
    print(f"Range feature matrix: {X.shape}")

    # ---- Step 3: Align regime mask to output timestamps ----
    ts_to_idx   = {int(t): i for i, t in enumerate(timestamps)}
    bar_indices = np.array([ts_to_idx[int(t)] for t in out_ts])
    regime_mask = np.array([regime_full[i] == "ranging" for i in bar_indices])
    print(f"Ranging bars: {regime_mask.sum()} / {len(regime_mask)}")

    # ---- Step 4: Align OHLCV for labels ----
    closes = ohlcv[bar_indices, 3].astype(np.float64)
    highs  = ohlcv[bar_indices, 1].astype(np.float64)
    lows   = ohlcv[bar_indices, 2].astype(np.float64)

    # ATR(14) on bar-aligned closes
    tr = np.zeros(len(closes))
    tr[1:] = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])),
    )
    atr14 = np.convolve(tr, np.ones(14) / 14, mode="same")

    # ---- Step 5: Compute value-area labels ----
    HORIZON = 6
    print("Computing value-area labels…")
    y = value_area_labels(
        closes, highs, lows, atr14, va_levels, regime_mask, X,
        horizon=HORIZON,
    )
    X = X[:len(y)]
    print(f"Labels: sell={int((y==0).sum())} hold={int((y==1).sum())} buy={int((y==2).sum())}")

    if (y == 0).sum() < 10 or (y == 2).sum() < 10:
        raise SystemExit("Too few BUY/SELL labels — extend dataset or lower entry_pct in labels_range.py")

    # ---- Step 6: Time-series train/test split ----
    GAP   = HORIZON
    split = int(len(X) * 0.8)
    X_tr, y_tr = X[:split - GAP], y[:split - GAP]
    X_te, y_te = X[split:],       y[split:]
    print(f"Train: {len(X_tr)}  Test: {len(X_te)}")

    counts = np.bincount(y_tr, minlength=3).clip(1)
    w_tr   = (len(y_tr) / (3 * counts))[y_tr]

    train_ds = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
    test_ds  = lgb.Dataset(X_te, label=y_te, reference=train_ds)

    params = {
        "objective":        "multiclass",
        "num_class":        3,
        "metric":           "multi_logloss",
        "learning_rate":    0.03,
        "num_leaves":       31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "lambda_l1":        0.1,
        "verbose":          -1,
    }

    model = lgb.train(
        params, train_ds,
        num_boost_round=1000,
        valid_sets=[test_ds],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    MODEL_OUT.parent.mkdir(exist_ok=True)
    model.save_model(str(MODEL_OUT))
    print(f"\nModel saved → {MODEL_OUT}")
    print("Next: run fast_backtest.py --lgbm to validate pf_ranging before going live")


if __name__ == "__main__":
    main()
