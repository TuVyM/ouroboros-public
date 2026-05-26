"""
Production live trading loop.

Consumes closed candles from BinanceFeed WebSocket.
Runs WM inference + actor decision each bar.
Pushes experiences to ShadowTrainer daemon for continuous learning.

Usage:
    python live_trader.py --symbol BTCUSDT --device cuda --dry-run

Environment variables (same as fast_backtest):
    WORLD_MODEL_CHECKPOINT   — path to WM checkpoint dir
    ACTOR_CRITIC_CHECKPOINT  — path to AC checkpoint dir
"""

import argparse
import logging
import os
from pathlib import Path
import signal
import sys

_HERE = Path(__file__).resolve().parent
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

log = logging.getLogger(__name__)

_PHI = 1.6180339887498949

# Candle window expected by the WM encoder (same as fast_backtest)
WORLD_MODEL_WINDOW   = int(os.environ.get("WORLD_MODEL_WINDOW",   64))
CHRONOS_INPUT_WINDOW = int(os.environ.get("CHRONOS_INPUT_WINDOW", 64))

WORLD_MODEL_CHECKPOINT   = os.environ.get("WORLD_MODEL_CHECKPOINT",   str(_HERE / "checkpoints" / "v4_base"))
ACTOR_CRITIC_CHECKPOINT  = os.environ.get("ACTOR_CRITIC_CHECKPOINT",  str(_HERE / "checkpoints" / "v4_actor"))

MIN_CONFIDENCE = float(os.environ.get("MIN_CONFIDENCE", "0.55"))
MAX_POSITION_PCT = float(os.environ.get("MAX_POSITION_PCT", "0.02"))
PAPER_PORTFOLIO_USDC = float(os.environ.get("PAPER_PORTFOLIO_USDC", "300.0"))
FEE_PCT = 0.0004

STOP_LOSS_PCT = 0.00382
TP_MULT       = _PHI


# Module-level references so unit-test patches (patch("live_trader.X")) resolve correctly.
# The real imports happen lazily inside __init__ when use_lgbm=True.
try:
    from lgbm.predictor import LGBMPredictor
except Exception:
    LGBMPredictor = None  # type: ignore[assignment,misc]

try:
    from lgbm.regime import RegimeDetector
except Exception:
    RegimeDetector = None  # type: ignore[assignment,misc]

try:
    from backtest_feed import _find_cache_file, DATA_CACHE_DIR
except ImportError:
    _find_cache_file = None  # type: ignore
    DATA_CACHE_DIR = ""


@dataclass
class _LivePosition:
    direction:           str
    entry_price:         float
    size_usdc:           float
    stop_loss:           float
    take_profit:         float
    entry_tick:          int
    regime_idx:          int           = -1
    regime:              str           = "trending"
    val_price:           float | None  = None
    poc_price:           float | None  = None
    vah_price:           float | None  = None
    partial_exit_1_done: bool          = False
    partial_exit_2_done: bool          = False
    peak_price:          float | None  = None


def _compute_ob_imbalance(order_book) -> float:
    """Compute (bid_vol - ask_vol) / (bid_vol + ask_vol) from a depth5 OrderBook.

    Returns 0.0 if order_book is None or either side is empty.
    """
    if order_book is None:
        return 0.0
    bids = order_book.bids or []
    asks = order_book.asks or []
    if not bids or not asks:
        return 0.0
    bid_vol = sum(q for _, q in bids[:5])
    ask_vol = sum(q for _, q in asks[:5])
    return float((bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8))


def _build_pos_state(
    position,
    price: float,
    tick: int,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    regime_idx: int,
    bvr: float,
    bph: int,
) -> torch.Tensor:
    """Build pos_state (1, 10) matching train_actor_critic feature order exactly.

    Features:
      0  open_flag        — 1.0 if position is open
      1  pnl_pct          — unrealised P&L as fraction (0 if flat)
      2  ticks_held/100   — bars held / 100
      3  mom_8h           — 8h log-return (bph-normalised)
      4  ret_1h           — 1h log-return
      5  ret_24h          — 24h log-return
      6  close_vs_sma24   — deviation from 24h SMA
      7  atr_ratio        — cur bar H-L / mean 24h H-L  - 1
      8  regime           — regime_idx - 1  →  {-1, 0, 1}
      9  buy_vol_ratio    — taker-buy fraction [0,1]; 0.5 if unknown (tick_delta absent)

    Lookbacks that exceed the available buffer fall back to 0.0 gracefully.
    """
    n  = len(closes)
    hl = highs - lows

    ret_1h = float(np.clip(
        np.log(closes[-1] / (closes[-1 - bph] + 1e-8)), -0.05, 0.05
    )) if n > bph else 0.0

    ret_24h = float(np.clip(
        np.log(closes[-1] / (closes[-1 - 24 * bph] + 1e-8)), -0.20, 0.20
    )) if n > 24 * bph else 0.0

    sma24 = float(closes[-24 * bph:].mean()) if n >= 24 * bph else float(closes.mean())
    cvs   = float(np.clip((closes[-1] - sma24) / (sma24 + 1e-8), -0.10, 0.10))

    mom_8h = float(np.clip(
        np.log(closes[-1] / (closes[-1 - 8 * bph] + 1e-8)), -0.10, 0.10
    )) if n > 8 * bph else 0.0

    mean_hl   = float(hl[-24 * bph:].mean()) if n >= 24 * bph else (float(hl.mean()) or 1e-8)
    atr_ratio = float(np.clip(float(hl[-1]) / (mean_hl + 1e-8) - 1.0, -1.0, 2.0))

    ps = torch.zeros(1, 10)
    ps[0, 3] = mom_8h
    ps[0, 4] = ret_1h
    ps[0, 5] = ret_24h
    ps[0, 6] = cvs
    ps[0, 7] = atr_ratio
    ps[0, 8] = float(regime_idx - 1)
    ps[0, 9] = float(np.clip(bvr, 0.0, 1.0))

    if position is not None:
        is_buy  = position.direction == 'buy'
        raw_pnl = (price - position.entry_price) / position.entry_price
        pnl_pct = raw_pnl if is_buy else -raw_pnl
        ps[0, 0] = 1.0
        ps[0, 1] = float(pnl_pct)
        ps[0, 2] = min(float(tick - position.entry_tick) / 100.0, 5.0)

    return ps


class LiveTrader:
    """Bar-by-bar live trading engine.

    Initialise once, then call on_candle() for each closed candle.
    Manages position state, invokes WM + actor, pushes experiences to shadow.
    """

    def __init__(self, symbol: str, device: str = 'cuda',
                 dry_run: bool = True, shadow_trainer=None, bph: int = 60,
                 use_swarm: bool = False, use_lgbm: bool = False,
                 scalp_layer=None):
        self.symbol        = symbol
        self.device        = device
        self.dry_run       = dry_run
        self.shadow        = shadow_trainer
        self._bph          = bph          # bars per hour: 60 for 1m, 1 for 1h
        self._buf_cap      = max(WORLD_MODEL_WINDOW * 2, 25 * bph)  # enough for all bph lookbacks
        self.balance       = PAPER_PORTFOLIO_USDC
        self.position: Optional[_LivePosition] = None
        self.tick          = 0
        self._ohlcv_buf    = []   # rolling window of closed candles
        self._use_swarm    = use_swarm
        self._use_lgbm     = use_lgbm
        self._warmup: bool = False

        # Scalp layer state
        self._scalp:       "ScalpLayer | None" = scalp_layer
        self._cached_htf:  "dict | None"       = None
        self._tick_1m:     int                 = 0
        self._last_bar_ts: int                 = 0

        if use_lgbm:
            from lgbm.predictor import LGBMPredictor
            from transformer.embed import load_encoder as _load_enc
            self._lgbm = LGBMPredictor()
            self._encoder = _load_enc()
            self._trend_uses_emb = self._encoder is not None and self._lgbm._trend_model.num_feature() > 24
            self._range_uses_emb = self._encoder is not None and self._lgbm._range_model.num_feature() > 29
            log.info(
                "MultiScaleEncoder loaded — trend=%d feat, range=%d feat",
                self._lgbm._trend_model.num_feature(),
                self._lgbm._range_model.num_feature(),
            ) if self._encoder is not None else log.info("No finetuned.pt — bare 24/29-feature matrices")
            self.wm = None
            self.ac = None
            log.info("LiveTrader initialised (LGBM mode): symbol=%s dry_run=%s", symbol, dry_run)
        elif use_swarm:
            from llm_swarm import LLMSwarm, LLMSwarmError
            self._swarm = LLMSwarm()
            self._LLMSwarmError = LLMSwarmError
            self.wm = None
            self.ac = None
            log.info("LiveTrader initialised (LLM swarm mode): symbol=%s dry_run=%s", symbol, dry_run)
        else:
            from world_model import WorldModelWrapper, ActorCriticWrapper
            # Load live models
            self.wm = WorldModelWrapper(device=device, checkpoint_path=WORLD_MODEL_CHECKPOINT)
            self.wm.load()
            self.ac = ActorCriticWrapper(
                self.wm.model, device=device, checkpoint_path=ACTOR_CRITIC_CHECKPOINT)
            self.ac.load()
            log.info("LiveTrader initialised: symbol=%s device=%s dry_run=%s", symbol, device, dry_run)

    def on_candle(self, ohlcv: np.ndarray, bvr: float = 0.5,
                  htf_1h: Optional[np.ndarray] = None,
                  timestamp_ms: int = 0) -> dict:
        """Process one closed candle. ohlcv: (5,) array [open,high,low,close,vol].

        bvr: taker-buy volume fraction [0,1]. Pass (1+state.tick_delta)/2 from BinanceFeed;
             defaults to 0.5 (neutral) when unavailable.
        htf_1h: (N, 5) array of 1h candles — required when use_lgbm=True.
        timestamp_ms: close timestamp of the current 1m bar in milliseconds.

        Returns action dict: {'signal': str, 'confidence': float, 'regime': int}
        """
        self._ohlcv_buf.append(ohlcv)
        if len(self._ohlcv_buf) > self._buf_cap:
            self._ohlcv_buf.pop(0)

        price  = float(ohlcv[3])   # close

        # ── LGBM path ───────────────────────────────────────────────────────
        # Does not use _ohlcv_buf — skip the WM warmup guard.
        if self._use_lgbm:
            price = float(ohlcv[3])
            if htf_1h is None or len(htf_1h) < 90:
                self.tick += 1
                self._tick_1m += 1
                return {'signal': 'hold', 'confidence': 0.0, 'regime': -1}

            htf    = htf_1h.astype(np.float32)
            now_s  = (timestamp_ms // 1000) if timestamp_ms else int(time.time())
            hour_s = now_s // 3600 * 3600
            ts     = np.array(
                [hour_s - (len(htf) - 1 - i) * 3600 for i in range(len(htf))],
                dtype=np.int64,
            )

            # Feature rebuild: every 5 ticks when scalp active; every tick otherwise
            do_rebuild = (
                self._scalp is None
                or self._cached_htf is None
                or self._tick_1m % 5 == 0
            )

            if do_rebuild:
                regime = self._lgbm.classify_regime(htf)

                if regime == "ranging":
                    if self._range_uses_emb:
                        from orderflow.range_features import build_range_features_with_embedding
                        X_r, _, va_levels = build_range_features_with_embedding(
                            self.symbol, htf, ts, self._encoder
                        )
                    else:
                        from orderflow.range_features import build_range_features
                        X_r, _, va_levels = build_range_features(self.symbol, htf, ts)
                    if len(X_r) == 0:
                        self.tick += 1
                        self._tick_1m += 1
                        return {'signal': 'hold', 'confidence': 0.0, 'regime': -1}
                    features_24    = None
                    features_range = X_r[-1]
                    va_levels_last = va_levels[-1]
                else:
                    if self._trend_uses_emb:
                        from orderflow.features import build_features_with_embedding
                        X, _ = build_features_with_embedding(self.symbol, htf, ts, self._encoder)
                    else:
                        from orderflow.features import build_features
                        X, _ = build_features(self.symbol, htf, ts)
                    if len(X) == 0:
                        self.tick += 1
                        self._tick_1m += 1
                        return {'signal': 'hold', 'confidence': 0.0, 'regime': -1}
                    features_24    = X[-1]
                    features_range = None
                    va_levels_last = {}

                hl24    = htf[-24:, 1] - htf[-24:, 2]
                mean_hl = float(hl24.mean()) if len(hl24) >= 1 else 1e-8
                atr_sl  = float(np.clip(1.0 * mean_hl / (price + 1e-8), 0.003, 0.010))
                atr_tp  = float(np.clip(5.0 * mean_hl / (price + 1e-8), 0.007, 0.050))

                self._cached_htf = dict(
                    features_24=features_24,
                    features_range=features_range,
                    regime=regime,
                    va_levels_last=va_levels_last,
                    atr_sl=atr_sl,
                    atr_tp=atr_tp,
                )

            c      = self._cached_htf
            result = self._lgbm.predict(
                c["features_24"], c["features_range"], c["regime"], c["va_levels_last"]
            )
            sig  = result['signal']
            conf = result['confidence']

            # Update scalp HTF context every 1m bar
            if self._scalp is not None:
                self._scalp.update_htf_context(sig, conf, result['regime'])

            # 1h position management: every 5 ticks when scalp active; every tick otherwise
            if self._scalp is None or self._tick_1m % 5 == 0:
                self._manage_position_lgbm(sig, conf, price, c["atr_sl"], c["atr_tp"], result)

            self.tick     += 1
            self._tick_1m += 1
            return {'signal': sig, 'confidence': conf, 'regime': -1}

        # WM / Swarm paths need _ohlcv_buf warmup; LGBM does not.
        if len(self._ohlcv_buf) < WORLD_MODEL_WINDOW:
            return {'signal': 'hold', 'confidence': 0.0, 'regime': -1}

        window = np.stack(self._ohlcv_buf[-WORLD_MODEL_WINDOW:])   # (W, 5)

        # ── LLM Swarm path ──────────────────────────────────────────────────
        if self._use_swarm:
            pos_dict = {
                "open_flag":   int(self.position is not None),
                "pnl_pct":     float(
                    ((price - self.position.entry_price) / self.position.entry_price
                     * (1 if self.position.direction == 'buy' else -1))
                    if self.position is not None else 0.0
                ),
                "ticks_held":  int(self.tick - self.position.entry_tick)
                               if self.position is not None else 0,
            }
            try:
                result = self._swarm.decide(window, pos_dict)
            except self._LLMSwarmError as e:
                log.error("LLMSwarmError: %s — shutting down", e)
                os.kill(os.getpid(), signal.SIGTERM)
                return {"signal": "hold", "confidence": 0.0, "regime": -1}
            sig  = result['signal']
            conf = result['confidence']
            regime_idx = result.get('regime', -1)
            # No shadow trainer support in swarm mode (no h_s/z_s)
            self._manage_position(sig, conf, price, None, None, None, regime_idx)
            self.tick += 1
            return {'signal': sig, 'confidence': conf, 'regime': regime_idx}

        # ── WM / Actor-Critic path (default) ────────────────────────────────
        with torch.inference_mode():
            ohlcv_t, book_t = self.wm.preprocess(window, None)
            embs = self.wm.model.encoder(ohlcv_t.to(self.device), book_t.to(self.device))
            h_s, z_s = self.wm._rssm_states.get(self.symbol, (None, None))
            _, new_h, new_z = self.wm.model.forward_from_embeddings(embs, h=h_s, z=z_s)
            self.wm._rssm_states[self.symbol] = (new_h.detach(), new_z.detach())
            h_s, z_s = new_h.detach(), new_z.detach()

            regime_idx = int(
                self.wm.model.regime_head(
                    torch.cat([h_s, z_s], dim=-1)
                ).argmax(dim=-1).item()
            )

        # Build features from the full available buffer (bph-normalised lookbacks)
        buf_arr = np.stack(self._ohlcv_buf)
        closes  = buf_arr[:, 3].astype(np.float64)
        highs   = buf_arr[:, 1].astype(np.float64)
        lows    = buf_arr[:, 2].astype(np.float64)

        ps = _build_pos_state(
            position=self.position, price=price, tick=self.tick,
            closes=closes, highs=highs, lows=lows,
            regime_idx=regime_idx, bvr=bvr, bph=self._bph,
        ).to(self.device)

        ac_result = self.ac.decide(h_s, z_s, pos_state=ps)
        sig  = ac_result['signal']
        conf = ac_result['confidence']

        # Push bar to shadow trainer
        if self.shadow is not None:
            self.shadow.push_bar(window, h_s, z_s)
            self.shadow.set_position_open(self.position is not None)

        # Position management
        self._manage_position(sig, conf, price, h_s, z_s, ps, regime_idx)
        self.tick += 1
        return {'signal': sig, 'confidence': conf, 'regime': regime_idx}

    def _manage_position_lgbm(
        self, sig: str, conf: float, price: float,
        atr_sl: float, atr_tp: float, predict_result: dict,
    ) -> None:
        """Position management for LGBM path.

        Trending regime: single ATR-based TP/SL.
        Ranging regime:  3-level partial ladder — 50% at POC, 25% at 75%-level, 25% trailing.
        """
        if self._warmup:
            return

        regime = predict_result.get("regime", "trending")

        # ── Manage existing position ──────────────────────────────────────────
        if self.position is not None:
            is_buy = self.position.direction == 'buy'
            flip   = ((is_buy and sig == 'sell') or (not is_buy and sig == 'buy')) \
                     and conf >= 0.40

            if self.position.regime == "ranging":
                self._check_ranging_exit(price, is_buy, flip, price * atr_sl)
            else:
                tp_hit = (is_buy  and price >= self.position.take_profit) or \
                         (not is_buy and price <= self.position.take_profit)
                sl_hit = (is_buy  and price <= self.position.stop_loss) or \
                         (not is_buy and price >= self.position.stop_loss)
                closed = tp_hit or sl_hit or flip
                if closed:
                    raw_pnl = (price - self.position.entry_price) / self.position.entry_price
                    pnl_pct = raw_pnl if is_buy else -raw_pnl
                    fee     = self.position.size_usdc * FEE_PCT * 2
                    pnl     = self.position.size_usdc * pnl_pct - fee
                    self.balance += pnl
                    reason = 'take_profit' if tp_hit else ('stop_loss' if sl_hit else 'signal')
                    log.info("Trade closed: %s dir=%s pnl=%.2f balance=%.2f",
                             reason, self.position.direction, pnl, self.balance)
                    self.position = None

        # ── Open new position ─────────────────────────────────────────────────
        if self.position is None and sig in ('buy', 'sell'):
            if regime == "ranging":
                self._open_ranging_position(sig, conf, price, predict_result)
            else:
                kelly_f  = max(0.0, (conf * (atr_tp / (atr_sl + 1e-8)) - (1 - conf))
                               / (atr_tp / (atr_sl + 1e-8)))
                size_pct = min(kelly_f * 0.25, MAX_POSITION_PCT)
                size     = self.balance * size_pct
                if size < 1.0:
                    return
                sl_dist = price * atr_sl
                tp_dist = price * atr_tp
                sl = (price - sl_dist) if sig == 'buy' else (price + sl_dist)
                tp = (price + tp_dist) if sig == 'buy' else (price - tp_dist)
                self.position = _LivePosition(
                    direction=sig, entry_price=price, size_usdc=size,
                    stop_loss=sl, take_profit=tp, entry_tick=self.tick,
                    regime="trending",
                )
                log.info("[LGBM%s] Trend trade opened: %s price=%.2f size=%.2f",
                         " DRY" if self.dry_run else "", sig, price, size)

    def _open_ranging_position(
        self, sig: str, conf: float, price: float, predict_result: dict,
    ) -> None:
        val = predict_result.get("val_price") or price
        poc = predict_result.get("poc_price") or price
        vah = predict_result.get("vah_price") or price
        atr_abs = abs(price - val) * 0.5
        sl = (val - atr_abs) if sig == 'buy' else (vah + atr_abs)

        kelly_f  = max(0.0, (conf * abs(poc - price) / (abs(sl - price) + 1e-8) - (1 - conf))
                       / (abs(poc - price) / (abs(sl - price) + 1e-8) + 1e-8))
        size_pct = min(kelly_f * 0.25, MAX_POSITION_PCT)
        size     = self.balance * size_pct
        if size < 1.0:
            return

        self.position = _LivePosition(
            direction=sig, entry_price=price, size_usdc=size,
            stop_loss=sl, take_profit=poc, entry_tick=self.tick,
            regime="ranging",
            val_price=val, poc_price=poc, vah_price=vah,
        )
        log.info("[LGBM%s] Range trade opened: %s price=%.2f val=%.2f poc=%.2f vah=%.2f",
                 " DRY" if self.dry_run else "", sig, price, val, poc, vah)

    def _check_ranging_exit(
        self, price: float, is_buy: bool, flip: bool, atr_abs: float,
    ) -> bool:
        """3-level ladder exit for ranging trades. Returns True if position fully closed."""
        pos = self.position

        # Initial SL check
        sl_hit = (is_buy  and price <= pos.stop_loss) or \
                 (not is_buy and price >= pos.stop_loss)
        if sl_hit or flip:
            raw_pnl = (price - pos.entry_price) / pos.entry_price
            pnl_pct = raw_pnl if is_buy else -raw_pnl
            pnl     = pos.size_usdc * pnl_pct - pos.size_usdc * FEE_PCT * 2
            self.balance += pnl
            reason = 'stop_loss' if sl_hit else 'signal_flip'
            log.info("Range trade closed (%s): dir=%s pnl=%.2f balance=%.2f",
                     reason, pos.direction, pnl, self.balance)
            self.position = None
            return True

        poc = pos.poc_price or pos.entry_price
        val = pos.val_price or pos.entry_price
        vah = pos.vah_price or pos.entry_price

        # Exit 1: 50% at POC
        if not pos.partial_exit_1_done:
            poc_hit = (is_buy and price >= poc) or (not is_buy and price <= poc)
            if poc_hit:
                partial_size = pos.size_usdc * 0.50
                raw_pnl = (price - pos.entry_price) / pos.entry_price
                pnl_pct = raw_pnl if is_buy else -raw_pnl
                pnl     = partial_size * pnl_pct - partial_size * FEE_PCT * 2
                self.balance     += pnl
                pos.size_usdc    -= partial_size
                pos.stop_loss     = pos.entry_price
                pos.partial_exit_1_done = True
                log.info("Range partial exit 1: dir=%s size=50.0%% pnl=%.2f remaining_usdc=%.2f balance=%.2f",
                         pos.direction, pnl, pos.size_usdc, self.balance)

        # Exit 2: 25% at 75% of range
        elif not pos.partial_exit_2_done:
            level_75 = val + 0.75 * (vah - val) if is_buy else vah - 0.75 * (vah - val)
            hit_75   = (is_buy and price >= level_75) or (not is_buy and price <= level_75)
            if hit_75:
                partial_size = pos.size_usdc * 0.50
                raw_pnl = (price - pos.entry_price) / pos.entry_price
                pnl_pct = raw_pnl if is_buy else -raw_pnl
                pnl     = partial_size * pnl_pct - partial_size * FEE_PCT * 2
                self.balance     += pnl
                pos.size_usdc    -= partial_size
                pos.partial_exit_2_done = True
                pos.peak_price   = price
                log.info("Range partial exit 2: dir=%s size=25.0%% pnl=%.2f remaining_usdc=%.2f balance=%.2f",
                         pos.direction, pnl, pos.size_usdc, self.balance)

        # Exit 3: trailing stop or hard exit at VAH/VAL
        else:
            if pos.peak_price is not None:
                pos.peak_price = max(pos.peak_price, price) if is_buy else min(pos.peak_price, price)
            trail_hit = pos.peak_price is not None and (
                (is_buy  and price < pos.peak_price - atr_abs * 0.3) or
                (not is_buy and price > pos.peak_price + atr_abs * 0.3)
            )
            hard_hit  = (is_buy and price >= vah) or (not is_buy and price <= val)
            if trail_hit or hard_hit:
                raw_pnl = (price - pos.entry_price) / pos.entry_price
                pnl_pct = raw_pnl if is_buy else -raw_pnl
                pnl     = pos.size_usdc * pnl_pct - pos.size_usdc * FEE_PCT * 2
                self.balance += pnl
                reason   = 'trailing_stop' if trail_hit else 'vah_exit'
                log.info("Range partial exit 3 (%s): dir=%s pnl=%.2f trailing_stop=%.5f balance=%.2f",
                         reason, pos.direction, pnl,
                         pos.peak_price or price, self.balance)
                self.position = None
                return True

        return False

    def _manage_position(self, sig, conf, price, h_s, z_s, ps, regime_idx):
        action_idx = {'buy': 0, 'sell': 1, 'hold': 2}.get(sig, 2)

        if self.position is not None:
            is_buy  = self.position.direction == 'buy'
            tp_hit  = (is_buy and price >= self.position.take_profit) or \
                      (not is_buy and price <= self.position.take_profit)
            sl_hit  = (is_buy and price <= self.position.stop_loss) or \
                      (not is_buy and price >= self.position.stop_loss)

            if tp_hit or sl_hit:
                raw_pnl = (price - self.position.entry_price) / self.position.entry_price
                pnl_pct = raw_pnl if is_buy else -raw_pnl
                fee     = self.position.size_usdc * FEE_PCT * 2
                pnl     = self.position.size_usdc * pnl_pct - fee
                self.balance += pnl
                reason = 'take_profit' if tp_hit else 'stop_loss'
                log.info("Trade closed: %s dir=%s pnl=%.2f balance=%.2f",
                         reason, self.position.direction, pnl, self.balance)
                if self.shadow is not None and h_s is not None:
                    self.shadow.push_trade(h_s, z_s, ps, action_idx, pnl)
                self.position = None

        if self.position is None and sig in ('buy', 'sell') and conf >= MIN_CONFIDENCE:
            kelly_f  = max(0.0, (conf * TP_MULT - (1 - conf)) / TP_MULT)
            size_pct = min(kelly_f * 0.25, MAX_POSITION_PCT)
            size     = self.balance * size_pct
            if size < 1.0:
                return
            sl_dist = price * STOP_LOSS_PCT
            tp_dist = price * STOP_LOSS_PCT * TP_MULT
            if sig == 'buy':
                sl = price - sl_dist
                tp = price + tp_dist
            else:
                sl = price + sl_dist
                tp = price - tp_dist
            self.position = _LivePosition(
                direction=sig, entry_price=price, size_usdc=size,
                stop_loss=sl, take_profit=tp, entry_tick=self.tick,
                regime_idx=regime_idx,
            )
            if not self.dry_run:
                log.info("Trade opened: %s price=%.2f size=%.2f sl=%.2f tp=%.2f",
                         sig, price, size, sl, tp)
            else:
                log.debug("[DRY RUN] Trade opened: %s price=%.2f size=%.2f",
                          sig, price, size)


def _run_warmup(trader: "LiveTrader", symbol: str, n_bars: int) -> None:
    """Replay the last `n_bars` 1m bars from cache through the trader (no trades).

    Sets trader._warmup=True for the duration so _manage_position_lgbm is a no-op.
    Calls ScalpLayer.on_1m_bar for each bar if scalp is active, priming _buf.
    Clears _warmup and logs completion when done.
    """
    if _find_cache_file is None:
        log.warning("_run_warmup: backtest_feed not available — skipping warmup")
        trader._warmup = False
        return

    cache_1m = _find_cache_file(symbol, DATA_CACHE_DIR, interval="1m")
    if cache_1m is None:
        log.warning("_run_warmup: no 1m cache for %s — skipping warmup", symbol)
        trader._warmup = False
        return

    cache_1h = _find_cache_file(symbol, DATA_CACHE_DIR, interval="1h")
    if cache_1h is None:
        log.warning("_run_warmup: no 1h cache for %s — skipping warmup", symbol)
        trader._warmup = False
        return

    ohlcv_1m  = np.load(cache_1m)
    # 1m _ts.npy is (N, 6) with ms timestamp in col 0; 1h _ts.npy is 1D seconds
    ts_raw_1m = np.load(cache_1m.replace(".npy", "_ts.npy"))
    ts_1m_ms  = ts_raw_1m[:, 0].astype(np.int64) if ts_raw_1m.ndim == 2 else ts_raw_1m.astype(np.int64)
    ohlcv_1h  = np.load(cache_1h)
    ts_1h_raw = np.load(cache_1h.replace(".npy", "_ts.npy"))
    ts_1h_s   = (ts_1h_raw[:, 0].astype(np.int64) // 1000) if ts_1h_raw.ndim == 2 else ts_1h_raw.astype(np.int64)

    start_idx = max(0, len(ohlcv_1m) - n_bars)
    ohlcv_1m  = ohlcv_1m[start_idx:]
    ts_1m_ms  = ts_1m_ms[start_idx:]

    trader._warmup = True
    t0 = time.monotonic()
    log.info("_run_warmup: replaying %d 1m bars for %s...", len(ohlcv_1m), symbol)

    for row, ts_ms in zip(ohlcv_1m, ts_1m_ms):
        ts_s   = ts_ms // 1000
        idx_1h = int(np.searchsorted(ts_1h_s, ts_s, side="right")) - 1
        idx_1h = max(0, min(idx_1h, len(ohlcv_1h) - 1))
        htf_slice = ohlcv_1h[max(0, idx_1h - 511): idx_1h + 1].astype(np.float32)

        ohlcv_row = row.astype(np.float32)
        trader.on_candle(
            ohlcv_row,
            bvr=0.5,
            htf_1h=htf_slice,
            timestamp_ms=int(ts_ms),
        )

        if trader._scalp is not None:
            trader._scalp.on_1m_bar(
                ohlcv_row,
                ob_imbalance=0.0,
                liq_vol=0.0,
                timestamp_ms=int(ts_ms),
            )

    elapsed = time.monotonic() - t0
    trader._warmup = False
    log.info(
        "_run_warmup: complete — %d bars in %.1fs (%.0f bars/s)",
        len(ohlcv_1m), elapsed, len(ohlcv_1m) / max(elapsed, 1e-9),
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Ouroboros live trader")
    parser.add_argument("--symbol",  default="BTCUSDT")
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Paper trading only (default). Set DRY_RUN=false in env to disable.")
    parser.add_argument("--no-shadow", action="store_true",
                        help="Disable shadow trainer daemon.")
    parser.add_argument("--swap-cooldown", type=int, default=3600,
                        help="Minimum seconds between weight swaps (default 3600).")
    parser.add_argument("--interval", choices=['1m', '5m', '1h'], default='1m',
                        help="Candle interval (default 1m). Sets bars-per-hour for feature lookbacks.")
    parser.add_argument("--swarm", action="store_true",
                        help="Use LLM swarm instead of world model for decisions.")
    parser.add_argument("--lgbm", action="store_true",
                        help="Use LightGBM predictor instead of world model. "
                             "Works with --interval 1m/5m/1h; features always computed from htf_1h.")
    parser.add_argument("--scalp", action="store_true",
                        help="Enable 1m scalp layer (requires --lgbm)")
    parser.add_argument(
        "--warmup-bars", type=int, default=500, metavar="N",
        help="Number of historical 1m bars to replay before going live (default 500 ≈ 8h). "
             "Set 0 to skip.",
    )
    args = parser.parse_args()

    if args.scalp and not args.lgbm:
        raise SystemExit("--scalp requires --lgbm")

    dry_run = args.dry_run or os.environ.get("DRY_RUN", "true").lower() != "false"
    bph = {"1m": 60, "5m": 12, "1h": 1}.get(args.interval, 60)
    log.info("Starting LiveTrader: symbol=%s device=%s dry_run=%s interval=%s",
             args.symbol, args.device, dry_run, args.interval)

    trader = LiveTrader(args.symbol, device=args.device, dry_run=dry_run, bph=bph,
                        use_swarm=args.swarm, use_lgbm=args.lgbm)

    if args.scalp:
        from lgbm.scalp_layer import ScalpLayer
        total_bal  = trader.balance
        scalp_frac = float(os.environ.get("SCALPER_BALANCE_FRACTION", "0.33"))
        scalp_bal  = total_bal * scalp_frac
        trader._scalp = ScalpLayer(
            balance_slice=scalp_bal,
            total_balance=total_bal,
            dry_run=dry_run,
        )
        log.info("ScalpLayer initialised: slice=%.2f total=%.2f", scalp_bal, total_bal)

    if args.warmup_bars > 0:
        _run_warmup(trader, args.symbol, n_bars=args.warmup_bars)

    shadow = None
    if not args.no_shadow and not args.swarm and not args.lgbm:
        try:
            from .shadow_trainer import ShadowTrainer, SWAP_COOLDOWN_S
            from . import shadow_trainer as _st
        except ImportError:
            from shadow_trainer import ShadowTrainer, SWAP_COOLDOWN_S
            import shadow_trainer as _st
        _st.SWAP_COOLDOWN_S = args.swap_cooldown
        shadow = ShadowTrainer(trader.wm, trader.ac, device=args.device, async_mode=True)
        trader.shadow = shadow
        shadow.start()

    # Graceful shutdown
    def _shutdown(sig, frame):
        log.info("Shutting down...")
        if shadow is not None:
            shadow.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Live data feed (WebSocket event-driven)
    try:
        from .data_feed import BinanceFeed
    except ImportError:
        from data_feed import BinanceFeed
    feed = BinanceFeed(symbols=[args.symbol])
    feed.start()

    log.info("Waiting for market data (WebSocket event-driven, interval=%s)...", args.interval)
    _bar_count   = 0
    _bar_stride = 1 if args.scalp else {"1m": 1, "5m": 5, "1h": 60}.get(args.interval, 1)
    try:
        while True:
            if not feed.wait_for_candle(timeout=90.0):
                log.warning("No closed candle in 90s — feed may have stalled")
                continue
            _bar_count += 1
            if _bar_count % _bar_stride != 0:
                continue
            state = feed.get_state(args.symbol)
            if state is None or state.ohlcv_window is None or len(state.ohlcv_window) == 0:
                continue
            ohlcv = state.ohlcv_window[-1].astype(np.float32)   # most recent closed bar
            bvr   = float(np.clip((1.0 + state.tick_delta) / 2.0, 0.0, 1.0))
            # get_state() doesn't populate htf_1h — fetch directly from HTF fetcher
            htf_1h, _ = feed._htf.get(args.symbol.upper())
            result = trader.on_candle(
                ohlcv, bvr=bvr,
                htf_1h=htf_1h,
                timestamp_ms=state.timestamp,
            )
            scalp_result = None
            if args.scalp and trader._scalp is not None:
                ob_imb  = _compute_ob_imbalance(state.order_book)
                liq_vol = feed.get_liq_since(args.symbol, trader._last_bar_ts)
                scalp_result = trader._scalp.on_1m_bar(
                    ohlcv, ob_imb, liq_vol, state.timestamp
                )
                trader._last_bar_ts = state.timestamp
            if scalp_result is not None:
                log.info(
                    "Bar: signal=%s conf=%.3f regime=%d "
                    "scalp=%s scalp_conf=%.3f balance=%.2f scalp_balance=%.2f bvr=%.2f",
                    result['signal'], result['confidence'], result['regime'],
                    scalp_result['signal'], scalp_result['conf'],
                    trader.balance, trader._scalp.balance, bvr,
                )
            else:
                log.info("Bar: signal=%s conf=%.3f regime=%d balance=%.2f bvr=%.2f",
                         result['signal'], result['confidence'],
                         result['regime'], trader.balance, bvr)
    except KeyboardInterrupt:
        _shutdown(None, None)


if __name__ == '__main__':
    main()
