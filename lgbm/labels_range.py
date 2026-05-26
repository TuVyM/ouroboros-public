# lgbm/labels_range.py
import numpy as np

SELL = np.int8(0)
HOLD = np.int8(1)
BUY  = np.int8(2)

_ENTRY_PCT        = 0.005   # 0.5% from VAL/VAH to qualify as candidate
_SL_ATR_MULT      = 0.5     # SL = 0.5 × ATR below VAL / above VAH
_BREAK_TP_MULT    = 2.0     # TP for breakdown/breakout retests = 2 × SL distance
_MIN_CONFLUENCE   = 2       # confluence required for mean-reversion cases
_BREAK_CONFLUENCE = 1       # confluence required for breakdown/breakout retest cases


def _confluence_count(bar_features: np.ndarray, va: dict, price: float) -> int:
    """Count how many confluence conditions hold (0–3)."""
    count = 0
    # (a) price within 0.3% of VAL or VAH
    if abs(price - va["val"]) / (price + 1e-8) < 0.003 or \
       abs(price - va["vah"]) / (price + 1e-8) < 0.003:
        count += 1
    # (b) any OB-Fib feature (cols 10–14) within 0.5%
    if np.any(np.abs(bar_features[10:15]) < 0.005):
        count += 1
    # (c) Fib channel position (cols 15–17) near 0.236, 0.500, or 0.764 ±0.05
    key = np.array([0.236, 0.500, 0.764])
    if np.any(np.abs(bar_features[15:18, None] - key[None, :]) < 0.05):
        count += 1
    return count


def value_area_labels(
    closes:      np.ndarray,   # (N,) float64
    highs:       np.ndarray,   # (N,) float64
    lows:        np.ndarray,   # (N,) float64
    atr14:       np.ndarray,   # (N,) float64 — ATR(14) in price units
    va_levels:   list,         # list of N dicts {"val", "poc", "vah"}
    regime_mask: np.ndarray,   # (N,) bool — True where regime == "ranging"
    X_base:      np.ndarray,   # (N, 24) float32 — base features for confluence
    horizon:     int = 6,
) -> np.ndarray:
    """
    Value-area triple barrier labels for ranging bars only.

    Mean-reversion cases (confluence ≥ 2):
      at_val_above  price within 0.5% above VAL → BUY toward POC, SL below VAL
      at_vah_below  price within 0.5% below VAH → SELL toward POC, SL above VAH

    Breakdown/breakout retest cases (confluence ≥ 1):
      at_val_below  price within 0.5% below VAL → SELL (old VAL = new resistance),
                    SL above VAL, TP = VAL − 2×SL_dist
      at_vah_above  price within 0.5% above VAH → BUY (old VAH = new support),
                    SL below VAH, TP = VAH + 2×SL_dist

    All other bars: HOLD.
    """
    n      = len(closes)
    labels = np.full(n, HOLD, dtype=np.int8)

    for t in range(n - horizon):
        if not regime_mask[t]:
            continue
        va    = va_levels[t]
        price = float(closes[t])
        atr   = float(atr14[t])
        val, poc, vah = va["val"], va["poc"], va["vah"]
        sl_dist = _SL_ATR_MULT * atr

        near_val = abs(price - val) / (price + 1e-8) <= _ENTRY_PCT
        near_vah = abs(price - vah) / (price + 1e-8) <= _ENTRY_PCT
        if not near_val and not near_vah:
            continue

        confluence = _confluence_count(X_base[t], va, price)

        if near_val and price >= val:
            # Mean-reversion BUY: approaching VAL from above
            if confluence < _MIN_CONFLUENCE:
                continue
            for i in range(1, horizon + 1):
                if highs[t + i] >= poc:
                    labels[t] = BUY;  break
                if lows[t + i]  <= val - sl_dist:
                    labels[t] = SELL; break

        elif near_val and price < val:
            # Breakdown retest SELL: old VAL is now resistance
            if confluence < _BREAK_CONFLUENCE:
                continue
            sl_price = val + sl_dist
            tp_price = val - _BREAK_TP_MULT * sl_dist
            for i in range(1, horizon + 1):
                if lows[t + i]  <= tp_price:
                    labels[t] = SELL; break
                if highs[t + i] >= sl_price:
                    labels[t] = BUY;  break   # fake breakdown, price reclaimed VAL

        elif near_vah and price <= vah:
            # Mean-reversion SELL: approaching VAH from below
            if confluence < _MIN_CONFLUENCE:
                continue
            for i in range(1, horizon + 1):
                if lows[t + i]  <= poc:
                    labels[t] = SELL; break
                if highs[t + i] >= vah + sl_dist:
                    labels[t] = BUY;  break

        else:  # near_vah and price > vah
            # Breakout retest BUY: old VAH is now support
            if confluence < _BREAK_CONFLUENCE:
                continue
            sl_price = vah - sl_dist
            tp_price = vah + _BREAK_TP_MULT * sl_dist
            for i in range(1, horizon + 1):
                if highs[t + i] >= tp_price:
                    labels[t] = BUY;  break
                if lows[t + i]  <= sl_price:
                    labels[t] = SELL; break   # fake breakout, price fell back below VAH

    return labels
