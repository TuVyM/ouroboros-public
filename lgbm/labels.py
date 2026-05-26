import numpy as np

SELL = np.int8(0)
HOLD = np.int8(1)
BUY  = np.int8(2)


def triple_barrier_labels(
    closes:  np.ndarray,
    highs:   np.ndarray,
    lows:    np.ndarray,
    tp:      float = 0.00618,
    sl:      float = 0.00382,
    horizon: int   = 8,
) -> np.ndarray:
    """
    Compute triple barrier labels for each bar t in range [0, len(closes)-horizon).

    tp:      take-profit as a fraction (0.010 = 1.0%)
    sl:      stop-loss   as a fraction (0.005 = 0.5%)
    horizon: max bars to look forward

    Returns int8 array of length len(closes) - horizon.
    Encoding: SELL=0, HOLD=1, BUY=2
    """
    n = len(closes)
    labels = np.full(n - horizon, HOLD, dtype=np.int8)

    for t in range(n - horizon):
        entry    = closes[t]
        tp_price = entry * (1.0 + tp)
        sl_price = entry * (1.0 - sl)

        for i in range(1, horizon + 1):
            hit_tp = highs[t + i] >= tp_price
            hit_sl = lows[t + i]  <= sl_price

            if hit_tp and hit_sl:
                # Both breach on same bar — label by which price close is nearer
                if abs(closes[t + i] - tp_price) < abs(closes[t + i] - sl_price):
                    labels[t] = BUY
                else:
                    labels[t] = SELL
                break
            elif hit_tp:
                labels[t] = BUY
                break
            elif hit_sl:
                labels[t] = SELL
                break
        # else: remains HOLD

    return labels
