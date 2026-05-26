# lgbm/regime_detector.py
import numpy as np


class RegimeDetector:
    """ADX-based market regime classifier.

    classify(htf_1h)           — stateful, 3-bar confirmation, live/backtest
    classify_batch(adx_series) — stateless, pure thresholds, training labels
    compute_adx(htf_1h)        — shared static ADX(14) computation
    """

    TREND_THRESH = 25.0
    RANGE_THRESH = 20.0
    CONFIRM_BARS = 3

    def __init__(self):
        self._current_regime     = "ranging"
        self._pending_regime     = None
        self._confirmation_count = 0

    # ------------------------------------------------------------------
    # Live / backtest mode (stateful)
    # ------------------------------------------------------------------

    def classify(self, htf_1h: np.ndarray) -> str:
        """Stateful classify — call once per bar in arrival order."""
        adx = self.compute_adx(htf_1h)
        range_pct, median_range_pct = self._atr_stats(htf_1h)
        label = self._threshold(float(adx[-1]), range_pct, median_range_pct)

        if label is None:
            # Hold-zone bars (ADX 20-25, no ATR spike) neither advance nor reset the
            # confirmation counter — spec requirement. A series like [trending, hold, trending]
            # still counts as 2 trending bars toward the 3-bar confirmation window.
            return self._current_regime
        if label == self._current_regime:
            self._pending_regime     = None
            self._confirmation_count = 0
            return self._current_regime

        if label != self._pending_regime:
            self._pending_regime     = label
            self._confirmation_count = 1
        else:
            self._confirmation_count += 1

        if self._confirmation_count >= self.CONFIRM_BARS:
            self._current_regime     = label
            self._pending_regime     = None
            self._confirmation_count = 0

        return self._current_regime

    # ------------------------------------------------------------------
    # Batch mode (stateless) — for training data labeling
    # ------------------------------------------------------------------

    @classmethod
    def classify_batch(cls, adx_series: np.ndarray) -> np.ndarray:
        """Pure ADX thresholds, no hysteresis. Returns object array of regime strings."""
        out = np.empty(len(adx_series), dtype=object)
        for i, adx in enumerate(adx_series):
            if adx > cls.TREND_THRESH:
                out[i] = "trending"
            elif adx < cls.RANGE_THRESH:
                out[i] = "ranging"
            else:
                out[i] = "volatile"
        return out

    # ------------------------------------------------------------------
    # Shared ADX computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_adx(htf_1h: np.ndarray) -> np.ndarray:
        """Wilder-smoothed ADX(14). Input: (N,5) OHLCV. Returns (N-1,) float64."""
        hi = htf_1h[:, 1].astype(np.float64)
        lo = htf_1h[:, 2].astype(np.float64)
        cl = htf_1h[:, 3].astype(np.float64)
        n  = len(cl)

        tr = np.zeros(n)
        tr[1:] = np.maximum(
            hi[1:] - lo[1:],
            np.maximum(np.abs(hi[1:] - cl[:-1]), np.abs(lo[1:] - cl[:-1])),
        )

        up   = hi[1:] - hi[:-1]
        dn   = lo[:-1] - lo[1:]
        pdm  = np.where((up > dn) & (up > 0), up, 0.0)
        ndm  = np.where((dn > up) & (dn > 0), dn, 0.0)

        def _wilder(arr: np.ndarray, p: int) -> np.ndarray:
            s = np.zeros(len(arr))
            if len(arr) < p:
                return s
            s[p - 1] = arr[:p].mean()
            for i in range(p, len(arr)):
                s[i] = (s[i - 1] * (p - 1) + arr[i]) / p
            return s

        p     = 14
        atr14 = _wilder(tr[1:], p)
        pdi   = 100.0 * _wilder(pdm, p) / (atr14 + 1e-8)
        ndi   = 100.0 * _wilder(ndm, p) / (atr14 + 1e-8)
        dx    = 100.0 * np.abs(pdi - ndi) / (pdi + ndi + 1e-8)
        adx   = _wilder(dx, p)
        return adx   # shape (N-1,)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _atr_stats(htf_1h: np.ndarray):
        # Uses H-L range (not true range) for simplicity — this is used only for
        # tie-breaking in the ADX 20-25 band, where an approximate relative spike
        # is sufficient. True range would require aligning prev-close across the window.
        window = min(30, len(htf_1h))
        hl          = (htf_1h[-window:, 1] - htf_1h[-window:, 2]).astype(np.float64)
        cl          = float(htf_1h[-1, 3])
        range_pct        = float(hl[-1] / (cl + 1e-8))
        median_range_pct = float(np.median(hl) / (cl + 1e-8))
        return range_pct, median_range_pct

    def _threshold(self, adx: float, range_pct: float, median_range_pct: float):
        if adx > self.TREND_THRESH:
            return "trending"
        if adx < self.RANGE_THRESH:
            return "ranging"
        if range_pct > 2.0 * median_range_pct:
            return "volatile"
        return None   # hold zone
