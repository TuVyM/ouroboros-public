# lgbm/shadow_trainer.py
"""
LGBM shadow trainer — continuous self-improvement for LGBMPredictor.

Every MIN_TRADE_BUFFER closed trades, incrementally retrains a shadow copy of
the LGBM model by adding new trees (lgb.train with init_model). When the shadow
model's profit factor on held-out trades exceeds the live model's PF by
SWAP_PF_MARGIN, it atomically replaces the live model file (btc_trend_model.lgbm
or btc_range_model.lgbm) and hot-reloads the live LGBMPredictor.

Signal encoding: sell=0, hold=1, buy=2.
"""
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import lightgbm as lgb
import numpy as np

log = logging.getLogger(__name__)

_PHI             = (1 + 5 ** 0.5) / 2
MIN_TRADE_BUFFER = 55                           # Fibonacci — trades before first retrain
VAL_FRACTION     = 0.20                         # newest 20% held out for PF eval
SWAP_COOLDOWN_S  = 3600                         # seconds between swaps
SWAP_PF_MARGIN   = 1.0 + 1.0 / (_PHI * 100)   # ≈ 1.00618 — φ-scaled margin
INCREMENTAL_ROUNDS = 21                         # Fibonacci — new trees per retrain cycle
SHADOW_BACKUP_DIR  = Path(__file__).resolve().parent.parent / "lgbm" / "shadow_backups"

# (features: np.ndarray (24,), action: int, pnl: float)
_TradeRecord = Tuple[np.ndarray, int, float]


class LGBMTradeBuffer:
    """Thread-safe ring buffer of closed trade records for LGBM shadow training."""

    def __init__(self, maxlen: int = 1597):     # Fibonacci
        self._buf: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, features: np.ndarray, action: int, pnl: float) -> None:
        """Store one closed trade. features: (24,) float32."""
        with self._lock:
            self._buf.append((features.copy(), int(action), float(pnl)))

    def train_val_split(self) -> Tuple[List[_TradeRecord], List[_TradeRecord]]:
        """Returns (train_records, val_records). Val is the newest VAL_FRACTION."""
        with self._lock:
            buf = list(self._buf)
        if len(buf) < 2:
            return buf, []
        n_val = max(1, min(int(len(buf) * VAL_FRACTION), len(buf) - 1))
        return buf[:-n_val], buf[-n_val:]

    @staticmethod
    def records_to_arrays(
        records: List[_TradeRecord],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unpack records into (X, y, pnl) numpy arrays."""
        X   = np.stack([r[0] for r in records]).astype(np.float32)
        y   = np.array([r[1] for r in records], dtype=np.int32)
        pnl = np.array([r[2] for r in records], dtype=np.float32)
        return X, y, pnl

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


def _compute_lgbm_pf(
    records,
    shadow_model: lgb.Booster,
) -> float:
    """
    Evaluate shadow model PF on held-out records.

    For each record: if shadow predicts same action as actual → keep PnL sign.
    If prediction differs → flip PnL sign (wrong direction = opposite outcome).
    PF = sum(positive outcomes) / sum(|negative outcomes|).
    Signal encoding: sell=0, hold=1, buy=2.
    """
    if not records:
        return 0.0
    X = np.stack([r[0] for r in records]).astype(np.float32)
    probs = shadow_model.predict(X)                  # (N, 3)
    preds = probs.argmax(axis=1).astype(int)         # argmax → predicted action

    wins = losses = 0.0
    for (_, action, pnl), pred in zip(records, preds):
        effective_pnl = pnl if pred == action else -pnl
        if effective_pnl > 0:
            wins += effective_pnl
        else:
            losses += abs(effective_pnl)
    return wins / (losses + 1e-8)


class LGBMShadowTrainer:
    """
    Incremental shadow trainer for LGBMPredictor.

    Accumulates closed trade records, retrains a shadow copy every
    MIN_TRADE_BUFFER new trades, and atomically swaps the model file
    + hot-reloads LGBMPredictor when shadow PF > live PF × SWAP_PF_MARGIN.

    Usage (fast_backtest --mode lgbm):
        shadow = LGBMShadowTrainer(live_predictor=_lgbm_predictor)
        # after each closed trade:
        shadow.push_trade(features, action, pnl)
        shadow.run_sync_cycle()   # no-op until MIN_TRADE_BUFFER reached

    Usage (live trading daemon):
        shadow = LGBMShadowTrainer(live_predictor=predictor, async_mode=True)
        shadow.start()
        # push_trade() from main thread; daemon retrains in background
    """

    def __init__(
        self,
        live_predictor,                           # LGBMPredictor instance
        async_mode: bool = False,
        backup_dir: Path = SHADOW_BACKUP_DIR,
    ):
        self._predictor     = live_predictor
        self._async_mode    = async_mode
        self._backup_dir    = Path(backup_dir)
        self._lock          = threading.Lock()
        self._last_swap_t   = 0.0
        self._swap_count    = 0
        self._trades_since_last_train = 0

        self.trade_buf = LGBMTradeBuffer()

        # Shadow model starts as a copy of the live model (load fresh from disk)
        self._shadow_model: Optional[lgb.Booster] = None
        self._load_shadow()

        if async_mode:
            self._thread = threading.Thread(
                target=self._training_loop, daemon=True, name="lgbm-shadow"
            )

    def _load_shadow(self) -> None:
        model_path = self._predictor._model_path
        if model_path.exists():
            self._shadow_model = lgb.Booster(model_file=str(model_path))
        else:
            log.warning("LGBMShadowTrainer: model file not found at %s", model_path)

    def start(self) -> None:
        """Launch daemon thread (async_mode only)."""
        if self._async_mode:
            self._thread.start()
            log.info("LGBMShadowTrainer daemon started")

    def push_trade(self, features: np.ndarray, action: int, pnl: float) -> None:
        """Record a closed trade outcome."""
        self.trade_buf.push(features, action, pnl)
        with self._lock:
            self._trades_since_last_train += 1

    def run_sync_cycle(self) -> dict:
        """
        Check if retraining is due. Retrain + maybe swap if so.
        Returns metrics dict. No-op (returns skipped=True) if buffer too small.
        """
        n = len(self.trade_buf)
        if n < MIN_TRADE_BUFFER:
            return {"skipped": True, "n_trades": n}

        with self._lock:
            since = self._trades_since_last_train
            if since < MIN_TRADE_BUFFER:
                return {"skipped": True, "n_trades": n, "since_last": since}
            self._trades_since_last_train = 0

        metrics = {"skipped": False, "n_trades": n}
        shadow_loss = self._retrain_incremental()
        metrics["shadow_loss"] = shadow_loss

        swapped, shadow_pf, live_pf = self._maybe_swap()
        metrics.update({"swapped": swapped, "shadow_pf": shadow_pf, "live_pf": live_pf})
        return metrics

    def _retrain_incremental(self) -> Optional[float]:
        """Add INCREMENTAL_ROUNDS new trees to the shadow model on recent trade data."""
        train_records, _ = self.trade_buf.train_val_split()
        if len(train_records) < 10:
            return None

        X, y, pnl = LGBMTradeBuffer.records_to_arrays(train_records)
        counts = np.bincount(y, minlength=3).clip(1)
        w = (len(y) / (3 * counts))[y]

        train_ds = lgb.Dataset(X, label=y, weight=w, free_raw_data=True)
        params = {
            "objective":        "multiclass",
            "num_class":        3,
            "metric":           "multi_logloss",
            "learning_rate":    0.01,           # conservative for incremental
            "num_leaves":       31,
            "min_data_in_leaf": 5,
            "verbose":          -1,
        }
        try:
            self._shadow_model = lgb.train(
                params, train_ds,
                num_boost_round=INCREMENTAL_ROUNDS,
                init_model=self._shadow_model,
            )
            return float(self._shadow_model.best_score.get("training", {}).get("multi_logloss", -1))
        except Exception as e:
            log.warning("Shadow retrain failed: %s", e)
            return None

    def _maybe_swap(self) -> Tuple[bool, float, float]:
        """Evaluate shadow vs live PF. Atomically swap model file + reload if shadow wins."""
        now = time.time()
        if now - self._last_swap_t < SWAP_COOLDOWN_S:
            return False, 0.0, 0.0
        if self._shadow_model is None:
            return False, 0.0, 0.0

        _, val = self.trade_buf.train_val_split()
        if not val:
            return False, 0.0, 0.0

        shadow_pf = _compute_lgbm_pf(val, self._shadow_model)
        live_pf   = _compute_lgbm_pf(val, self._predictor.model)

        log.info(
            "Shadow eval: shadow_pf=%.3f  live_pf=%.3f  (need %.3f to swap)",
            shadow_pf, live_pf, live_pf * SWAP_PF_MARGIN,
        )

        if shadow_pf <= 1.0 or shadow_pf < live_pf * SWAP_PF_MARGIN:
            return False, shadow_pf, live_pf

        # Back up live model before overwriting
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup_dir = self._backup_dir / ts
        model_path = self._predictor._model_path
        try:
            os.makedirs(backup_dir, exist_ok=True)
            import shutil
            shutil.copy2(model_path, backup_dir / "btc_model_pre_swap.lgbm")
            log.info("Shadow pre-swap backup → %s", backup_dir)
        except Exception as e:
            log.warning("Pre-swap backup failed: %s — aborting swap", e)
            return False, shadow_pf, live_pf

        # Atomic: write shadow model to tmp, rename over live file
        tmp_path = model_path.with_suffix(".lgbm.tmp")
        try:
            self._shadow_model.save_model(str(tmp_path))
            os.replace(tmp_path, model_path)         # atomic on Linux
            self._predictor.reload()                 # hot-reload live predictor
        except Exception as e:
            log.warning("Shadow swap write failed: %s", e)
            tmp_path.unlink(missing_ok=True)
            return False, shadow_pf, live_pf

        self._last_swap_t = now
        self._swap_count += 1
        log.info(
            "Shadow swap #%d: live_pf=%.3f → shadow_pf=%.3f (backup at %s)",
            self._swap_count, live_pf, shadow_pf, backup_dir,
        )
        return True, shadow_pf, live_pf

    def _training_loop(self) -> None:
        """Daemon loop for async mode."""
        while True:
            time.sleep(SWAP_COOLDOWN_S / 10)   # check every 6 minutes
            try:
                metrics = self.run_sync_cycle()
                if not metrics.get("skipped"):
                    log.info("Shadow cycle: %s", metrics)
            except Exception as e:
                log.error("Shadow training loop error: %s", e)
