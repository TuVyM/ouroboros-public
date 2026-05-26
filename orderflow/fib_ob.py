# orderflow/fib_ob.py
"""
Market-structure feature functions for LGBM.

_fib_time_ob_features_batch, _fib_channel_features_batch, _ict_features_batch
validated against BTCUSDT 1h historical data (2023–2026).
_tpo_features_batch is pure OHLCV — no external data required.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from typing import Optional

_FIB_LOOKBACKS = (1, 5, 15, 60, 89)


def _fib_time_ob_features_batch(windows: np.ndarray) -> np.ndarray:
    """Fully vectorised (B, T, 5) → (B, 5). No Python loop over B.

    Algorithm mirrors _fib_time_ob_features / _fib_time_ob_anchors but uses
    sliding_window_view for swing detection and numpy fancy-indexing for the
    OB search, eliminating all per-window Python overhead.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    B, T, _ = windows.shape
    o = windows[:, :, 0]  # (B, T)
    h = windows[:, :, 1]
    l = windows[:, :, 2]
    c = windows[:, :, 3]
    current = c[:, -1]    # (B,)

    _n   = 3   # swing look-left / look-right
    _tol = 2   # OB search tolerance around fib_t
    _min_seg = 2 * _n + 1

    out = np.zeros((B, len(_FIB_LOOKBACKS)), dtype=np.float32)
    bi   = np.arange(B)   # batch index — reused for fancy indexing

    for k, lb in enumerate(_FIB_LOOKBACKS):
        seg_h = h[:, -lb:]   # (B, lb)
        seg_l = l[:, -lb:]
        seg_o = o[:, -lb:]
        seg_c = c[:, -lb:]

        ob_high = seg_h.max(axis=1)   # fallback = segment high  (B,)
        ob_low  = seg_l.min(axis=1)   # fallback = segment low

        if lb >= _min_seg:
            # ---- swing detection (vectorised across B) -------------------------
            nb_h = sliding_window_view(seg_h, 2 * _n + 1, axis=1)  # (B, lb-2n, 2n+1)
            nb_l = sliding_window_view(seg_l, 2 * _n + 1, axis=1)

            is_sh = seg_h[:, _n:-_n] >= nb_h.max(axis=2)  # (B, lb-2n)
            is_sl = seg_l[:, _n:-_n] <= nb_l.min(axis=2)

            valid = is_sh.any(axis=1) & is_sl.any(axis=1)  # (B,)

            if valid.any():
                # Best swing = highest SH vs lowest SL → maximum price range
                sh_vals = np.where(is_sh, seg_h[:, _n:-_n], -np.inf)  # (B, lb-2n)
                sl_vals = np.where(is_sl, seg_l[:, _n:-_n],  np.inf)

                sh_idx = sh_vals.argmax(axis=1) + _n   # position in seg (B,)
                sl_idx = sl_vals.argmin(axis=1) + _n

                sw_start = np.minimum(sh_idx, sl_idx)
                sw_end   = np.maximum(sh_idx, sl_idx)
                dur      = (sw_end - sw_start).astype(np.float32)

                valid_dur = valid & (dur >= 2)

                fib_t = np.clip(
                    (sw_start + np.round(0.618 * dur)).astype(np.int32),
                    1, lb - 1
                )  # (B,) — clipped so prev=fib_t-1 is always ≥0

                # ---- OB detection around fib_t (5-offset unroll) ---------------
                bodies     = np.abs(seg_c - seg_o)
                ranges     = seg_h - seg_l + 1e-8
                is_impulse = (bodies / ranges) >= 0.618
                is_bull    = seg_c > seg_o
                is_bear    = seg_c < seg_o

                found_high = np.zeros(B, dtype=bool)
                found_low  = np.zeros(B, dtype=bool)

                for delta in range(-_tol, _tol + 1):
                    idx  = np.clip(fib_t + delta, 1, lb - 1)
                    prev = idx - 1

                    # Bearish OB candle before a bearish impulse → resistance
                    cond_h = (valid_dur & ~found_high
                              & is_impulse[bi, idx]
                              & is_bear[bi, idx]
                              & is_bull[bi, prev])
                    ob_high = np.where(cond_h, seg_h[bi, prev], ob_high)
                    found_high |= cond_h

                    # Bullish OB candle before a bullish impulse → support
                    cond_l = (valid_dur & ~found_low
                              & is_impulse[bi, idx]
                              & is_bull[bi, idx]
                              & is_bear[bi, prev])
                    ob_low = np.where(cond_l, seg_l[bi, prev], ob_low)
                    found_low |= cond_l

        rng      = ob_high - ob_low                        # (B,)
        safe_rng = np.where(rng < 1e-8, 1.0, rng)
        fib_618  = ob_low + 0.618 * safe_rng
        dist     = np.clip((current - fib_618) / safe_rng, -2.0, 2.0)
        out[:, k] = np.where(rng < 1e-8, 0.0, dist).astype(np.float32)

    return out


def _ict_features_batch(windows: np.ndarray,
                        timestamps_ms: Optional[np.ndarray] = None) -> np.ndarray:
    """ICT Smart Money features. (B, T, 5) → (B, 3).

    Feature 0 — Fair Value Gap (FVG): signed imbalance between candles T-3 and T-1.
        Bullish FVG: low[-1] > high[-3]  (gap up, unmitigated)
        Bearish FVG: low[-3] > high[-1]  (gap down, unmitigated)
        Normalised by close price.

    Feature 1 — Liquidity Sweep: ±1.0 when the last 5 bars swept a prior swing
        high/low and then closed back through it (stop-hunt reversal signal).
        +1 = swept low (bullish reversal), -1 = swept high (bearish reversal).

    Feature 2 — Kill Zone: session timing scaled to [0, 1].
        London open (07–10 UTC) = 0.5, NY open (13–16 UTC) = 1.0, else = 0.0.
    """
    B, T, _ = windows.shape
    h = windows[:, :, 1]   # (B, T)
    l = windows[:, :, 2]
    c = windows[:, :, 3]

    # --- FVG (requires ≥3 bars) ---
    if T >= 3:
        fvg_bull = np.maximum(0.0, l[:, -1] - h[:, -3])   # (B,)
        fvg_bear = np.maximum(0.0, l[:, -3] - h[:, -1])
    else:
        fvg_bull = np.zeros(B, dtype=np.float32)
        fvg_bear = np.zeros(B, dtype=np.float32)
    fvg = ((fvg_bull - fvg_bear) / (c[:, -1] + 1e-8)).astype(np.float32)

    # --- Liquidity sweep (requires >5 bars) ---
    if T > 5:
        prev_high = h[:, :-5].max(axis=1)                  # (B,)
        prev_low  = l[:, :-5].min(axis=1)
        recent_high = h[:, -5:].max(axis=1)
        recent_low  = l[:, -5:].min(axis=1)
        # Swept high: recent pierced prev_high but closed below it → bearish
        swept_hi = ((recent_high > prev_high) & (c[:, -1] < prev_high))
        # Swept low:  recent pierced prev_low  but closed above it → bullish
        swept_lo = ((recent_low  < prev_low)  & (c[:, -1] > prev_low))
        sweep = (swept_lo.astype(np.float32) - swept_hi.astype(np.float32))
    else:
        sweep = np.zeros(B, dtype=np.float32)

    # --- Kill zone ---
    if timestamps_ms is not None:
        hours = (np.asarray(timestamps_ms) // 3_600_000) % 24  # (B,)
        kill = np.where((hours >= 7) & (hours < 10), 0.5,
               np.where((hours >= 13) & (hours < 16), 1.0, 0.0)).astype(np.float32)
    else:
        kill = np.zeros(B, dtype=np.float32)

    return np.stack([fvg, sweep, kill], axis=1)  # (B, 3)


def _fib_channel_features_batch(windows: np.ndarray) -> np.ndarray:
    """Fibonacci channel features. (B, T, 5) → (B, 3).

    Measures where current close sits within the N-bar channel expressed as
    signed distances from symmetric Fibonacci channel levels (0.236, 0.500, 0.764).

    Unlike OB-Fib (trend-anchored from impulse order blocks), the channel uses raw
    window high/low — making it the primary spatial context for mean_reversion in
    ranging regime where no directional impulse may be present.

      dist > 0 at level 0.764 → price near channel top → mean_reversion sell setup
      dist < 0 at level 0.236 → price near channel bottom → mean_reversion buy setup

    Distances normalised by channel range → values in ~[-1, 1].
    """
    h   = windows[:, :, 1].max(axis=1)   # (B,) window high
    l   = windows[:, :, 2].min(axis=1)   # (B,) window low
    c   = windows[:, -1, 3]              # (B,) current close
    rng = h - l + 1e-8
    levels = np.stack([
        l + 0.236 * rng,
        l + 0.500 * rng,
        l + 0.764 * rng,
    ], axis=1)                            # (B, 3)
    distances = (c[:, np.newaxis] - levels) / rng[:, np.newaxis]  # (B, 3)
    return distances.astype(np.float32)


def _tpo_features_batch(windows: np.ndarray, n_bins: int = 21) -> np.ndarray:
    """TPO (Time Price Opportunity) / Market Profile. (B, T, 5) → (B, 3).

    Builds a time-at-price histogram for each window:
      tpo_count[b, k] = number of bars in window b whose [low, high] spans bin k.
    POC = bin with highest count.
    Value Area = bins (sorted by count desc) until 70% of T is covered.
    VAH/VAL = highest/lowest bin midpoint inside the value area.

    Features (all divided by window range to normalise):
      0: (close - POC) / range
      1: (close - VAH) / range
      2: (close - VAL) / range
    """
    B, T, _ = windows.shape
    h = windows[:, :, 1].astype(np.float64)
    l = windows[:, :, 2].astype(np.float64)
    c = windows[:, -1, 3].astype(np.float64)

    win_h = h.max(axis=1)
    win_l = l.min(axis=1)
    rng   = win_h - win_l + 1e-8

    # Bin edges: (B, n_bins+1)
    edges   = win_l[:, None] + np.outer(rng, np.linspace(0.0, 1.0, n_bins + 1))
    bin_mid = (edges[:, :-1] + edges[:, 1:]) * 0.5  # (B, n_bins)

    # Count bars touching each bin (vectorised)
    l_e = l[:, None, :]          # (B, 1, T)
    h_e = h[:, None, :]          # (B, 1, T)
    e_lo = edges[:, :-1, None]   # (B, n_bins, 1)
    e_hi = edges[:, 1:,  None]   # (B, n_bins, 1)
    tpo_count = ((l_e <= e_hi) & (h_e >= e_lo)).sum(axis=2).astype(np.float32)  # (B, n_bins)

    # POC
    poc_idx  = tpo_count.argmax(axis=1)                  # (B,)
    poc      = bin_mid[np.arange(B), poc_idx]            # (B,)

    # Value Area: take bins by count (desc) until ≥70% of total count covered
    order        = np.argsort(-tpo_count, axis=1)        # (B, n_bins)
    sorted_cnt   = tpo_count[np.arange(B)[:, None], order]
    cum          = np.cumsum(sorted_cnt, axis=1)         # (B, n_bins)
    target       = 0.70 * tpo_count.sum(axis=1, keepdims=True)
    prev_cum     = np.concatenate([np.zeros((B, 1), dtype=np.float32), cum[:, :-1]], axis=1)
    in_va        = prev_cum < target                     # (B, n_bins) in sorted order

    # Scatter back to original bin ordering
    va_mask = np.zeros((B, n_bins), dtype=bool)
    np.put_along_axis(va_mask, order, in_va, axis=1)

    vah = np.where(va_mask, bin_mid, -np.inf).max(axis=1)
    val = np.where(va_mask, bin_mid,  np.inf).min(axis=1)

    out = np.stack([
        (c - poc) / rng,
        (c - vah) / rng,
        (c - val) / rng,
    ], axis=1).astype(np.float32)

    return np.clip(out, -3.0, 3.0)


def build_structural_features(
    ohlcv:      np.ndarray,   # (N, 5) float32 — open high low close volume
    timestamps_s: np.ndarray, # (N,) int64 — unix seconds UTC
    window:     int = 89,
) -> np.ndarray:
    """
    Build (N - window, 14) structural features via a single sliding-window pass.

    Alignment: feature row k = OHLCV bar (k + window). Matches the [WINDOW:] drop
    in orderflow/features.py — concatenate directly with the existing feature matrix.

    Output columns:
      [0:5]   ob_fib_dist_{1,5,15,60,89}
      [5:8]   fib_ch_{236,500,764}
      [8:11]  ict_{fvg, sweep, killzone}
      [11:14] tpo_{poc, vah, val}
    """
    N = len(ohlcv)
    assert N > window, f"Need at least {window+1} bars, got {N}"

    # sliding_window_view(ohlcv, window, axis=0) → (N-window+1, 5, window)
    # Transpose → (N-window+1, window, 5) then drop first window (alignment)
    raw   = sliding_window_view(ohlcv.astype(np.float32), window, axis=0)  # (N-w+1, 5, w)
    wins  = raw.transpose(0, 2, 1)[1:]  # (N-window, window, 5)  ← drop index 0

    # Timestamps for ICT kill-zone (milliseconds, one per window)
    ts_ms = (timestamps_s[window:] * 1000).astype(np.int64)   # (N-window,)

    ob_fib = _fib_time_ob_features_batch(wins)         # (N-window, 5)
    fib_ch = _fib_channel_features_batch(wins)         # (N-window, 3)
    ict    = _ict_features_batch(wins, ts_ms)           # (N-window, 3)
    tpo    = _tpo_features_batch(wins)                  # (N-window, 3)

    return np.concatenate([ob_fib, fib_ch, ict, tpo], axis=1)  # (N-window, 14)
