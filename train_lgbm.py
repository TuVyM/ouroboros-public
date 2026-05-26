#!/usr/bin/env python3
"""
Train LightGBM signal classifier on BTC orderflow + price features.

Usage:
    python train_lgbm.py                    # uses longest available 1h cache
    python train_lgbm.py --symbol BTCUSDT
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import shap

from backtest_feed import _find_cache_file, DATA_CACHE_DIR
from lgbm.labels import triple_barrier_labels
from orderflow.features import build_features, FEATURE_NAMES

MODEL_OUT = Path("lgbm/btc_trend_model.lgbm")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--no-embed", action="store_true", help="Skip encoder embeddings, use 24 raw features only")
    args = p.parse_args()

    # ---- Load OHLCV from longest available 1h cache ----
    cache_path = _find_cache_file(args.symbol, DATA_CACHE_DIR, interval="1h")
    if cache_path is None:
        raise SystemExit(
            f"No 1h cache for {args.symbol}. "
            "Run: python pretrain.py --symbols BTCUSDT --days 365 --interval 1h --cache-only"
        )
    ts_path = cache_path.replace(".npy", "_ts.npy")
    ohlcv      = np.load(cache_path)          # (N, 5) float32
    timestamps = np.load(ts_path)             # (N,) int64 unix seconds

    print(f"OHLCV loaded: {len(ohlcv)} bars from {cache_path}")

    # ---- Build feature matrix (with encoder embeddings if finetuned.pt exists) ----
    encoder = None
    if not args.no_embed:
        from transformer.embed import load_encoder
        encoder = load_encoder()
    if encoder is not None:
        from orderflow.features import build_features_with_embedding
        print("Building feature matrix (88 features — with encoder embeddings)…")
        X, out_ts = build_features_with_embedding(args.symbol, ohlcv, timestamps, encoder)
        feat_names = FEATURE_NAMES + [f"emb_{i}" for i in range(64)]
    else:
        print("Building feature matrix (24 features — no embeddings)…")
        X, out_ts = build_features(args.symbol, ohlcv, timestamps)
        feat_names = FEATURE_NAMES
    print(f"Feature matrix: {X.shape}")

    # ---- Align OHLCV to out_ts for label construction ----
    # Map unix-second timestamps back to ohlcv row indices
    ts_to_idx = {t: i for i, t in enumerate(timestamps)}
    idx = np.array([ts_to_idx[t] for t in out_ts])
    closes = ohlcv[idx, 3].astype(np.float64)
    highs  = ohlcv[idx, 1].astype(np.float64)
    lows   = ohlcv[idx, 2].astype(np.float64)

    # ---- Compute triple barrier labels ----
    HORIZON = 8
    print("Computing triple barrier labels…")
    y = triple_barrier_labels(closes, highs, lows, tp=0.010, sl=0.005, horizon=HORIZON)
    X = X[:len(y)]   # drop last HORIZON rows (no future data)
    print(f"Labels: sell={int((y==0).sum())} hold={int((y==1).sum())} buy={int((y==2).sum())}")

    # ---- Time-series train/test split (no shuffling) ----
    GAP    = HORIZON
    split  = int(len(X) * 0.8)
    X_tr,  y_tr  = X[:split - GAP],  y[:split - GAP]
    X_te,  y_te  = X[split:],         y[split:]

    counts = np.bincount(y_tr, minlength=3).clip(1)
    w_tr   = (len(y_tr) / (3 * counts))[y_tr]   # inverse-frequency class weights

    print(f"Train: {len(X_tr)}  Test: {len(X_te)}")

    train_ds = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=feat_names)
    test_ds  = lgb.Dataset(X_te, label=y_te, reference=train_ds)

    params = {
        "objective":         "multiclass",
        "num_class":         3,
        "metric":            "multi_logloss",
        "learning_rate":     0.01,
        "num_leaves":        63,
        "min_data_in_leaf":  100,
        "feature_fraction":  0.8,
        "bagging_fraction":  0.8,
        "bagging_freq":      5,
        "lambda_l1":         0.1,
        "verbose":           -1,
    }

    model = lgb.train(
        params, train_ds,
        num_boost_round=2000,
        valid_sets=[test_ds],
        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(100)],
    )

    # ---- SHAP feature importance ----
    print("\nComputing SHAP importance (500 test samples)…")
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_te[:500])
    # sv is list of 3 arrays (one per class) — average absolute across classes
    mean_abs = np.abs(np.array(sv)).mean(axis=(0, 1))
    print("\nTop-10 SHAP feature importances:")
    for name, val in sorted(zip(feat_names, mean_abs), key=lambda x: -x[1])[:10]:
        print(f"  {name:25s}: {val:.5f}")

    MODEL_OUT.parent.mkdir(exist_ok=True)
    model.save_model(str(MODEL_OUT))
    print(f"\nModel saved → {MODEL_OUT}")


if __name__ == "__main__":
    main()
