#!/usr/bin/env python3
"""
Train 1m scalp LightGBM model (3-class multiclass: sell=0, hold=1, buy=2).

Usage:
    python train_lgbm_scalp.py                    # uses longest available 1m cache
    python train_lgbm_scalp.py --symbol BTCUSDT
"""
import argparse
from collections import deque
from pathlib import Path

import lightgbm as lgb
import numpy as np

from backtest_feed import _find_cache_file, DATA_CACHE_DIR
from lgbm.labels import triple_barrier_labels
from lgbm.predictor import LGBMPredictor
from lgbm.regime_detector import RegimeDetector
from orderflow.scalp_features import build_scalp_features, SCALP_FEATURE_NAMES

MODEL_OUT = Path("lgbm/btc_scalp_model.lgbm")

LGBM_PARAMS = {
    "objective":        "multiclass",
    "num_class":        3,
    "metric":           "multi_logloss",
    "learning_rate":    0.05,
    "num_leaves":       31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "lambda_l1":        0.1,
    "verbose":          -1,
}

# ATR on 1m BTC bars is ~0.06–0.10%. Use TP >= 3× ATR to clear noise floor.
GRID_TP  = [0.20, 0.30, 0.40, 0.60, 0.80]   # percent
GRID_SL  = [0.10, 0.15, 0.20, 0.30, 0.40]   # percent
GRID_HOR = [10, 15, 20, 30]                   # bars


def _build_htf_context(ohlcv_1m: np.ndarray, ts_1m: np.ndarray,
                        ohlcv_1h: np.ndarray, ts_1h: np.ndarray) -> np.ndarray:
    """
    Reconstruct (N_1m, 3) HTF context columns: [htf_signal, htf_conf, htf_regime].

    Strategy:
      1. Run LGBMPredictor on each 1h bar (stateless batch — classify_batch for regime).
      2. Record (signal_enc, conf, regime_enc) per 1h timestamp.
      3. Forward-fill to 1m timestamps using searchsorted.

    Ranging bars use the trend model path (known distribution shift, acceptable for training).
    """
    predictor = LGBMPredictor()
    adx       = RegimeDetector.compute_adx(ohlcv_1h)
    regimes   = RegimeDetector.classify_batch(adx)          # (N_1h - 1,)
    regime_full = np.empty(len(ohlcv_1h), dtype=object)
    regime_full[0]  = regimes[0]
    regime_full[1:] = regimes

    _SIGNAL_ENC = {"sell": 0, "hold": 1, "buy": 2}
    _REGIME_ENC = {"ranging": 0, "trending": 1, "volatile": 2}

    from orderflow.features import build_features
    X_1h, ts_out_1h = build_features("BTCUSDT", ohlcv_1h, ts_1h)
    ts_to_idx = {int(t): i for i, t in enumerate(ts_1h)}

    htf_rows = []
    for i, t in enumerate(ts_out_1h):
        idx    = ts_to_idx.get(int(t))
        if idx is None:
            continue
        regime = regime_full[idx]
        f24    = X_1h[i]
        va     = {}
        try:
            # Force trending path for all regimes — avoids needing range features
            eff_regime = "trending" if str(regime) == "ranging" else str(regime)
            res = predictor.predict(f24, None, eff_regime, va)
        except Exception:
            res = {"signal": "hold", "confidence": 0.0, "regime": str(regime)}
        htf_rows.append((
            int(t),
            _SIGNAL_ENC.get(res["signal"], 1),
            float(res["confidence"]),
            _REGIME_ENC.get(str(regime), 0),
        ))

    if not htf_rows:
        return np.zeros((len(ts_1m), 3), dtype=np.float32)

    htf_ts  = np.array([r[0] for r in htf_rows], dtype=np.int64)
    htf_sig = np.array([r[1] for r in htf_rows], dtype=np.float32)
    htf_con = np.array([r[2] for r in htf_rows], dtype=np.float32)
    htf_reg = np.array([r[3] for r in htf_rows], dtype=np.float32)

    idxs = np.searchsorted(htf_ts, ts_1m, side="right") - 1
    idxs = np.clip(idxs, 0, len(htf_ts) - 1)

    out = np.column_stack([htf_sig[idxs], htf_con[idxs], htf_reg[idxs]])
    return out.astype(np.float32)


def _build_feature_matrix(ohlcv_1m: np.ndarray, htf_ctx: np.ndarray) -> tuple:
    """Build (M, 15) feature matrix. Returns (X, bar_offsets)."""
    buf     = deque(maxlen=30)
    rows    = []
    offsets = []
    for i, row in enumerate(ohlcv_1m):
        ohlcv = row.astype(np.float32)
        buf.append((ohlcv, 0.0, 0.0))   # ob_imb=0, liq_vol=0 (unavailable historically)
        feat = build_scalp_features(
            buf,
            htf_signal=int(htf_ctx[i, 0]),
            htf_conf=float(htf_ctx[i, 1]),
            htf_regime=int(htf_ctx[i, 2]),
            training=True,
        )
        if feat is not None:
            rows.append(feat)
            offsets.append(i)

    return np.array(rows, dtype=np.float32), np.array(offsets, dtype=np.int64)


def _grid_search(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> tuple:
    """
    Search GRID_TP × GRID_SL × GRID_HOR combos.

    Selection: maximise min(buy%, sell%) subject to non-HOLD >= 16% of all bars.
    Tiebreaker: prefer smallest TP.

    Returns: (best_tp, best_sl, best_horizon, top5_candidates)
    Each candidate: (min_frac, tp, sl, horizon, buy_pct, sell_pct)
    """
    candidates = []
    for tp in GRID_TP:
        for sl in GRID_SL:
            for hor in GRID_HOR:
                y = triple_barrier_labels(
                    closes, highs, lows,
                    tp=tp / 100.0, sl=sl / 100.0, horizon=hor,
                )
                n        = len(y)
                buy_pct  = float((y == 2).sum() / n * 100)
                sel_pct  = float((y == 0).sum() / n * 100)
                non_hold = buy_pct + sel_pct
                if non_hold < 16.0:
                    continue
                min_frac = min(buy_pct, sel_pct)
                candidates.append((min_frac, tp, sl, hor, buy_pct, sel_pct))

    if not candidates:
        raise SystemExit(
            "No (TP, SL, horizon) combo satisfies the >=16% non-HOLD constraint. "
            "Try running with a longer cache or wider grid."
        )

    candidates.sort(key=lambda x: (-x[0], x[1]))
    best = candidates[0]
    return best[1], best[2], best[3], candidates[:5]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    args = p.parse_args()

    # ── Load caches ──────────────────────────────────────────────────────────
    cache_1m = _find_cache_file(args.symbol, DATA_CACHE_DIR, interval="1m")
    if cache_1m is None:
        raise SystemExit(
            f"No 1m cache for {args.symbol}. Run:\n"
            "  python pretrain.py --symbols BTCUSDT --days 90 --interval 1m --cache-only"
        )
    ohlcv_1m = np.load(cache_1m)
    ts_1m    = np.load(cache_1m.replace(".npy", "_ts.npy"))
    print(f"1m OHLCV loaded: {len(ohlcv_1m)} bars from {cache_1m}")

    cache_1h = _find_cache_file(args.symbol, DATA_CACHE_DIR, interval="1h")
    if cache_1h is None:
        raise SystemExit("No 1h cache found.")
    ohlcv_1h = np.load(cache_1h)
    ts_1h    = np.load(cache_1h.replace(".npy", "_ts.npy"))
    print(f"1h OHLCV loaded: {len(ohlcv_1h)} bars from {cache_1h}")

    # ── HTF context ──────────────────────────────────────────────────────────
    print("Reconstructing HTF context (1h signal forward-filled to 1m)...")
    htf_ctx = _build_htf_context(ohlcv_1m, ts_1m, ohlcv_1h, ts_1h)
    print(f"HTF context shape: {htf_ctx.shape}")

    # ── Feature matrix ───────────────────────────────────────────────────────
    print("Building scalp feature matrix (15 features)...")
    X, bar_offsets = _build_feature_matrix(ohlcv_1m, htf_ctx)
    print(f"Feature matrix: {X.shape}")

    closes = ohlcv_1m[bar_offsets, 3].astype(np.float64)
    highs  = ohlcv_1m[bar_offsets, 1].astype(np.float64)
    lows   = ohlcv_1m[bar_offsets, 2].astype(np.float64)

    # ── Grid search ──────────────────────────────────────────────────────────
    n_combos = len(GRID_TP) * len(GRID_SL) * len(GRID_HOR)
    print(f"Grid search: {len(GRID_TP)}x{len(GRID_SL)}x{len(GRID_HOR)} = {n_combos} combos...")
    best_tp, best_sl, best_hor, top5 = _grid_search(closes, highs, lows)

    print(f"\nSelected: TP={best_tp}% SL={best_sl}% horizon={best_hor} bars")
    print("Top-5 candidates (min_frac, tp%, sl%, horizon, buy%, sell%):")
    for i, (mf, tp, sl, hor, bp, sp) in enumerate(top5):
        mark = " <- selected" if i == 0 else ""
        print(f"  #{i+1}: min={mf:.2f}% tp={tp}% sl={sl}% hor={hor} buy={bp:.2f}% sell={sp:.2f}%{mark}")

    # ── ATR env-var equivalents ──────────────────────────────────────────────
    mean_atr_pct = float(X[:, 7].mean()) * 100.0
    if mean_atr_pct > 1e-6:
        print(f"\nMean ATR%: {mean_atr_pct:.4f}%")
        print("Env-var equivalents for live SL/TP:")
        print(f"  SCALP_SL_PCT={best_sl/100:.4f}   (or SCALP_SL_ATR={best_sl/mean_atr_pct:.2f})")
        print(f"  SCALP_TP_PCT={best_tp/100:.4f}   (or SCALP_TP_ATR={best_tp/mean_atr_pct:.2f})")

    # ── Labels ───────────────────────────────────────────────────────────────
    print(f"\nBuilding labels: TP={best_tp}% SL={best_sl}% horizon={best_hor}...")
    y = triple_barrier_labels(closes, highs, lows,
                               tp=best_tp/100.0, sl=best_sl/100.0, horizon=best_hor)
    X = X[:len(y)]
    print(f"Labels: sell={int((y==0).sum())} hold={int((y==1).sum())} buy={int((y==2).sum())}")

    # ── Train/test split ─────────────────────────────────────────────────────
    GAP   = best_hor
    split = int(len(X) * 0.8)
    X_tr, y_tr = X[:split - GAP], y[:split - GAP]
    X_te, y_te = X[split:],        y[split:]

    counts = np.bincount(y_tr.astype(np.int64), minlength=3).clip(1)
    w_tr   = (len(y_tr) / (3 * counts))[y_tr]

    print(f"Train: {len(X_tr)}  Test: {len(X_te)}")

    train_ds = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=SCALP_FEATURE_NAMES)
    test_ds  = lgb.Dataset(X_te, label=y_te, reference=train_ds)

    model = lgb.train(
        LGBM_PARAMS, train_ds,
        num_boost_round=1000,
        valid_sets=[test_ds],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    # ── SHAP importance ──────────────────────────────────────────────────────
    try:
        import shap
        print("\nComputing SHAP importance (500 test samples)...")
        explainer = shap.TreeExplainer(model)
        sv        = explainer.shap_values(X_te[:500])
        mean_abs  = np.abs(np.array(sv)).mean(axis=(0, 1))
        print("\nTop-10 SHAP feature importances:")
        for name, val in sorted(zip(SCALP_FEATURE_NAMES, mean_abs), key=lambda x: -x[1])[:10]:
            print(f"  {name:25s}: {val:.5f}")
    except ImportError:
        print("\nshap not installed — skipping (pip install shap to enable)")

    # ── Validation gate ──────────────────────────────────────────────────────
    logloss  = model.best_score["valid_0"]["multi_logloss"]
    baseline = 1.099   # ln(3) — random 3-class baseline
    gate_ok  = "OK" if logloss < baseline else "FAIL"
    print(f"\nTest multi_logloss: {logloss:.4f}  (random baseline={baseline:.3f}: {gate_ok})")
    if logloss > 1.0:
        print("  NOTE: logloss > 1.0 — model is barely better than random. "
              "Consider retraining with more data or tuned features.")

    # ── Save ─────────────────────────────────────────────────────────────────
    MODEL_OUT.parent.mkdir(exist_ok=True)
    model.save_model(str(MODEL_OUT))
    print(f"\nModel saved -> {MODEL_OUT}")
    print(f"Next: run live_trader.py --lgbm --scalp --dry-run --symbol BTCUSDT")


if __name__ == "__main__":
    main()
