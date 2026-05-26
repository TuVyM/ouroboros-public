# lgbm/scalp_layer.py
import logging
import os
import threading
from collections import deque
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np

from orderflow.scalp_features import build_scalp_features

log = logging.getLogger(__name__)

SCALP_MODEL_PATH     = Path(__file__).parent / "btc_scalp_model.lgbm"
MIN_SCALP_CONFIDENCE = 0.60
FEE_PCT              = 0.0004    # one-way taker fee

_SIGNAL_ENC = {"sell": 0, "hold": 1, "buy": 2}
_REGIME_ENC = {"ranging": 0, "trending": 1, "volatile": 2}


class ScalpLayer:
    """1m scalp inference + position management on a partitioned balance slice."""

    def __init__(
        self,
        balance_slice: float,
        total_balance: float,
        dry_run: bool = True,
        model_path: Optional[Path] = SCALP_MODEL_PATH,
    ):
        self._balance       = float(balance_slice)
        self._initial_slice = float(balance_slice)
        self._total_balance = float(total_balance)
        self._dry_run       = dry_run

        if model_path is None:
            self._model = None  # set externally (used in tests)
        else:
            if not Path(model_path).exists():
                raise FileNotFoundError(
                    f"Scalp model not found: {model_path}\n"
                    "Run: python train_lgbm_scalp.py --symbol BTCUSDT"
                )
            self._model = lgb.Booster(model_file=str(model_path))

        self._buf: deque = deque(maxlen=30)   # (ohlcv, ob_imbalance, liq_vol) tuples

        # HTF context — initialised to HOLD/ranging until first update
        self._htf_signal: int      = 1    # HOLD
        self._htf_conf:   float    = 0.0
        self._htf_regime: int      = 0    # ranging
        self._htf_initialized: bool = False

        self._position: Optional[dict] = None   # keys: direction, entry_price, size_usdc, stop_loss, take_profit
        self._warned_low_balance: bool = False
        self._lock = threading.Lock()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def update_htf_context(self, signal: str, conf: float, regime: str) -> None:
        with self._lock:
            self._htf_signal      = _SIGNAL_ENC.get(signal.lower(), 1)
            self._htf_conf        = float(conf)
            self._htf_regime      = _REGIME_ENC.get(regime.lower(), 0)
            self._htf_initialized = True

    def on_1m_bar(
        self,
        ohlcv: np.ndarray,
        ob_imbalance: float,
        liq_vol: float,
        timestamp_ms: int,
    ) -> dict:
        with self._lock:
            self._buf.append((ohlcv, float(ob_imbalance), float(liq_vol)))

            base = {"signal": "hold", "conf": 0.0, "trade_pnl": None,
                    "balance": self._balance, "position": self._position_str()}

            if not self._htf_initialized:
                return base

            feat = build_scalp_features(
                self._buf, self._htf_signal, self._htf_conf, self._htf_regime
            )
            if feat is None:
                return base

            probs    = self._model.predict(feat.reshape(1, -1))[0]  # (3,) [sell, hold, buy]
            pred_cls = int(probs.argmax())
            conf     = float(probs[pred_cls])
            sig      = {0: "sell", 1: "hold", 2: "buy"}[pred_cls]
            close    = float(ohlcv[3])
            atr_ratio = float(feat[7])   # index 7 = atr_ratio_1m

            trade_pnl = None
            if self._position is not None:
                trade_pnl = self._check_exit(close)

            if self._position is None and sig in ("buy", "sell") and conf >= MIN_SCALP_CONFIDENCE:
                self._open_position(sig, close, atr_ratio, conf)

            if self._balance < 0.60 * self._initial_slice:
                if not self._warned_low_balance:
                    log.warning(
                        "ScalpLayer: balance %.2f is below 60%% of initial slice %.2f — consider restart",
                        self._balance, self._initial_slice,
                    )
                    self._warned_low_balance = True
            elif self._warned_low_balance:
                self._warned_low_balance = False

            return {
                "signal":    sig,
                "conf":      conf,
                "trade_pnl": trade_pnl,
                "balance":   self._balance,
                "position":  self._position_str(),
            }

    @property
    def balance(self) -> float:
        return self._balance

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _position_str(self) -> Optional[str]:
        return self._position["direction"] if self._position else None

    def _sl_tp_dist(self, close: float, atr_ratio: float) -> tuple:
        """Return (sl_dist, tp_dist) in price units.

        Prefers absolute-percent env vars (SCALP_SL_PCT / SCALP_TP_PCT) over ATR multiples.
        ATR multiples sourced from SCALP_SL_ATR / SCALP_TP_ATR env vars (default 1.0 / 2.0).
        """
        sl_pct_env = os.environ.get("SCALP_SL_PCT", "")
        tp_pct_env = os.environ.get("SCALP_TP_PCT", "")
        if sl_pct_env and tp_pct_env:
            return float(sl_pct_env) * close, float(tp_pct_env) * close
        atr_price = atr_ratio * close
        sl_atr    = float(os.environ.get("SCALP_SL_ATR", "1.0"))
        tp_atr    = float(os.environ.get("SCALP_TP_ATR", "2.0"))
        return sl_atr * atr_price, tp_atr * atr_price

    def _open_position(self, sig: str, close: float, atr_ratio: float, conf: float) -> None:
        sl_dist, tp_dist = self._sl_tp_dist(close, atr_ratio)
        if sl_dist < 1e-8:
            return

        tp_ratio = tp_dist / sl_dist   # sl_dist > 1e-8 guaranteed by guard above
        kelly_f  = max(0.0, (conf * tp_ratio - (1 - conf)) / tp_ratio)
        cap      = 0.01 * self._total_balance
        size     = min(kelly_f * 0.25 * self._balance, cap)
        if size < 1.0:
            return

        if sig == "buy":
            sl, tp, direction = close - sl_dist, close + tp_dist, "long"
        else:
            sl, tp, direction = close + sl_dist, close - tp_dist, "short"

        self._position = {
            "direction":   direction,
            "entry_price": close,
            "size_usdc":   size,
            "stop_loss":   sl,
            "take_profit": tp,
        }
        log.info(
            "[ScalpLayer%s] %s @ %.2f size=%.2f sl=%.2f tp=%.2f",
            " DRY" if self._dry_run else "", direction, close, size, sl, tp,
        )

    def _check_exit(self, close: float) -> Optional[float]:
        pos    = self._position
        is_lng = pos["direction"] == "long"
        sl_hit = (is_lng and close <= pos["stop_loss"]) or \
                 (not is_lng and close >= pos["stop_loss"])
        tp_hit = (is_lng and close >= pos["take_profit"]) or \
                 (not is_lng and close <= pos["take_profit"])
        if not (sl_hit or tp_hit):
            return None

        raw_pnl = (close - pos["entry_price"]) / pos["entry_price"]
        pnl_pct = raw_pnl if is_lng else -raw_pnl
        fee     = pos["size_usdc"] * FEE_PCT * 2
        pnl     = pos["size_usdc"] * pnl_pct - fee
        self._balance += pnl
        reason = "take_profit" if tp_hit else "stop_loss"
        log.info(
            "[ScalpLayer] %s closed %s pnl=%.2f balance=%.2f",
            reason, pos["direction"], pnl, self._balance,
        )
        self._position = None
        return pnl
