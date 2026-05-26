# Ouroboros — LGBM + MultiScaleEncoder Model Card

**As of:** 2026-05-24  
**Status:** Live (dry-run) — `live_trader.py --lgbm --interval 5m --dry-run --symbol BTCUSDT`  
**Symbol:** BTCUSDT (Binance Futures perpetual)  
**Bar interval:** 5m live tick, 1h feature cadence

---

## 1. Architecture

The encoder and regime detector run **in parallel on the same raw OHLCV**. They are independent — the encoder does not feed the regime detector.

```
1h OHLCV (raw)
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
MultiScaleEncoder               RegimeDetector (ADX-14)
  ├── 1h Transformer  168 bars         ├── ADX > 25   → "trending"
  ├── 4h Transformer   42 bars         ├── ADX < 20   → "ranging"
  └── 1d Transformer   30 bars         └── ADX 20-25  → "transition zone"
  concat → Linear(192→64)                   (ATR spike override → "volatile")
  64-dim latent z                              ↓
    │                            trending / volatile → Trend LGBM
    │ (z used only for           ranging             → Range LGBM
    │  Range LGBM; discarded
    │  for Trend LGBM)
    ▼
  ┌───────────────────────────────────┐
  │ Regime routing                    │
  │                                   │
  ▼                                   ▼
Trend LGBM                      Range LGBM
24 raw features                 24 raw + 5 VA + 64-dim z = 93 features
  │                                   │
  └──────────────┬────────────────────┘
                 ▼
         BUY / SELL / HOLD
         (conf ≥ 0.55 trend, ≥ 0.50 range)
                 │
         Position manager
         (Kelly-capped, ATR SL/TP)
```

**Why split?** Encoder embeddings consistently hurt the trend LGBM (top SHAP feature `delta_ratio_8h` SHAP = 0.07 raw vs 0.028 with embeddings — the 64-dim vector dilutes the orderflow signal). The same embeddings consistently help the range LGBM, where value-area geometry maps naturally to the encoder's multi-scale representation.

---

## 2. MultiScaleEncoder

| Property | Value |
|---|---|
| File | `transformer/encoder.py` |
| Weights | `transformer/finetuned.pt` (fine-tuned) / `transformer/pretrained.pt` (pretrain only) |
| Architecture | 3× independent TransformerEncoder (PyTorch), fused by linear projection |
| Windows | 1h: 168 bars · 4h: 42 bars · 1d: 30 bars |
| Output dim | 64 |
| Trainable params | 628,800 (~629K) |
| Pretraining | Masked bar reconstruction (MAE-style, 20% masking), 50 epochs, 1h OHLCV |
| Fine-tuning | Supervised BUY/SELL/HOLD on 1h bars with regime-routed labels, lr=1e-4 |
| Val loss (fine-tune) | ~1.03–1.04 (random baseline = ln(3) = 1.099) — weak label signal |
| Used for | Range LGBM embedding only (trend LGBM discards it) |

---

## 3. Trend LGBM

| Property | Value |
|---|---|
| File | `lgbm/btc_trend_model.lgbm` |
| Script | `train_lgbm.py --no-embed` |
| Features | 24 raw orderflow features (see §5.1) |
| Labels | Triple-barrier: TP=1.0%, SL=0.5%, horizon=8 bars |
| Training data | 1,095 days 1h OHLCV, BTCUSDT (~2023-05-24 → 2026-05-24) |
| Train/test split | 80/20 time-series (no shuffle), GAP=8 bars |
| LightGBM params | lr=0.01, num_leaves=63, min_data_in_leaf=100, rounds=2000, early_stop=150 |
| Class weights | Inverse-frequency |
| Best iteration | ~678 (varies by run) |
| Test logloss | ~0.993 |
| Confidence gate | 0.55 (MIN_CONFIDENCE) |

**Backtest (30d, 1h bars):** PF trend = 1.362, 19 trades

---

## 4. Range LGBM

| Property | Value |
|---|---|
| File | `lgbm/btc_range_model.lgbm` |
| Script | `train_lgbm_range.py` |
| Features | 29 range features + 64-dim MSE embedding = 93 total |
| Labels | Value-area labels: 4 cases (see §4.1) |
| Training data | 1,095 days 1h OHLCV, BTCUSDT — ranging bars only (ADX < 20) |
| Train/test split | 80/20 time-series, GAP=6 bars |
| LightGBM params | lr=0.03, num_leaves=31, min_data_in_leaf=20, rounds=1000, early_stop=50 |
| Class weights | Inverse-frequency |
| Best iteration | 699 |
| Test logloss | 0.2007 |
| Confidence gate | 0.50 (MIN_RANGE_CONFIDENCE) |

**Backtest (30d, 1h bars):** PF range = 3.917, 18 trades

### 4.1 Value-Area Label Cases

| Case | Entry condition | Signal | TP | SL | Confluence |
|---|---|---|---|---|---|
| at_val_above | price ≥ VAL and within 0.5% | BUY → POC | POC | VAL − 0.5×ATR | ≥ 2 |
| at_val_below | price < VAL and within 0.5% | SELL — old VAL now resistance | VAL − 2×ATR | VAL + 0.5×ATR | ≥ 1 |
| at_vah_below | price ≤ VAH and within 0.5% | SELL → POC | POC | VAH + 0.5×ATR | ≥ 2 |
| at_vah_above | price > VAH and within 0.5% | BUY — old VAH now support | VAH + 2×ATR | VAH − 0.5×ATR | ≥ 1 |

The breakdown/retest cases (at_val_below, at_vah_above) use lower confluence (1 vs 2) because the level break is itself the primary signal. Value area is computed per ranging episode using a 20-bin volume-profile histogram.

---

## 5. Feature Spaces

### 5.1 Trend Features (24, raw orderflow)

| Index | Name | Description |
|---|---|---|
| 0 | buy_ratio_1h | Taker buy fraction this bar [0,1] |
| 1 | delta_ratio_8h | 8h cumulative delta / total vol [-1,1] |
| 2 | vwap_deviation | (close − VWAP) / VWAP |
| 3 | funding_rate | 8h perp funding rate |
| 4 | ret_1h | 1-bar return |
| 5 | ret_8h | 8-bar return |
| 6 | ret_24h | 24-bar return |
| 7 | atr_ratio | ATR(14) / close |
| 8 | sma24_dist | (close − SMA24) / SMA24 |
| 9 | range_position | (close − window_low) / (window_high − window_low) |
| 10–14 | ob_fib_dist_* | Order-block anchored Fibonacci distances (1, 5, 15, 60, 89 bars) |
| 15–17 | fib_ch_* | Fibonacci channel position (0.236, 0.500, 0.764 levels) |
| 18 | ict_fvg | ICT fair value gap |
| 19 | ict_sweep | ICT liquidity sweep |
| 20 | ict_killzone | ICT kill zone flag |
| 21–23 | tpo_poc/vah/val | TPO market profile levels |

### 5.2 Range Features (29 raw + 64 embedding = 93)

Indices 0–23: same as trend features above.

| Index | Name | Description |
|---|---|---|
| 24 | rsi_bull_div | 1 if price makes lower low AND RSI makes higher low (4-bar) |
| 25 | rsi_bear_div | 1 if price makes higher high AND RSI makes lower high (4-bar) |
| 26 | val_dist | (price − VAL) / price — positive = above VAL |
| 27 | vah_dist | (VAH − price) / price — positive = below VAH |
| 28 | poc_dist | (price − POC) / price |
| 29–92 | emb_0 … emb_63 | 64-dim MultiScaleEncoder latent z |

---

## 6. Regime Detection

| Property | Value |
|---|---|
| File | `lgbm/regime_detector.py` |
| Indicator | Wilder-smoothed ADX(14) on 1h OHLCV |
| Trending threshold | ADX > 25 → routes to Trend LGBM |
| Ranging threshold | ADX < 20 → routes to Range LGBM |
| Volatile | ADX 20–25 AND current H-L > 2× 30-bar median → routes to Trend LGBM |
| Transition zone | ADX 20–25 with no ATR spike → regime unchanged (hysteresis) |
| Confirmation | 3 consecutive qualifying bars required to switch regime (stateful) |
| Cold start | Initialises as "ranging"; takes up to ~45 min after restart to confirm trending (3 × ~15-min HTF refresh cycles) |

---

## 7. Position Management

| Parameter | Value |
|---|---|
| Max position size | Kelly fraction, capped at 2% of balance (MAX_POSITION_PCT) |
| Stop-loss (trend) | 1.0× mean H-L of last 24 1h bars / price, clipped [0.3%, 1.0%] |
| Take-profit (trend) | 5.0× mean H-L of last 24 1h bars / price, clipped [0.7%, 5.0%] |
| Stop-loss (range, mean-reversion) | VAL − 0.5×ATR (buy) / VAH + 0.5×ATR (sell) |
| Take-profit (range, mean-reversion) | POC |
| Stop-loss (range, breakdown) | VAL + 0.5×ATR (short) / VAH − 0.5×ATR (long) |
| Take-profit (range, breakdown) | 2× SL distance from level |
| Min confidence | 0.55 (trend), 0.50 (range) |
| DRY_RUN default | True — no real orders until `.env` sets `DRY_RUN=false` |

---

## 8. Backtest Performance (2026-05-24, 30-day 1h window)

| Metric | Value | Gate | Notes |
|---|---|---|---|
| Overall PF | 1.846 | — | |
| Trending PF | 1.362 | ≥ 1.39 ❌ | Slightly below gate |
| Ranging PF | 3.917 | ≥ 1.75 ✅ | |
| Total trades | 37 (19 trend + 18 range) | — | Low count; statistics are indicative |
| Win rate | 59.5% | > 52% ✅ | |
| Max drawdown | 0.0% | — | Measured on closed-trade P&L only; intra-trade drawdown not tracked |
| Sharpe | 19.74 | — | Annualised on per-trade returns (√8760 factor); unreliable at n=37 |
| Exit breakdown | SL=8 trail=5 TP=18 sig=6 | — | |

**Note on V1 gate:** Consensus rate 5.9% fails the legacy gate (10–25%). That gate was calibrated for a higher-frequency world-model. At 1h cadence with specialist routing, ~1.2 trades/day is the intended conservative behaviour.

---

## 9. Training Pipeline

Steps 1–2 (encoder) are needed only for the range LGBM. The trend LGBM uses `--no-embed` and never loads the encoder. Do **not** omit `--no-embed` from step 3 — without it the trend model trains on 88 features and breaks the predictor's runtime routing.

```bash
# 1. Pretrain encoder (needed for range model only)
.venv/bin/python -m transformer.pretrain --symbol BTCUSDT

# 2. Fine-tune encoder (needed for range model only)
.venv/bin/python -m transformer.finetune --symbol BTCUSDT

# 3. Train trend LGBM — raw features, no encoder
.venv/bin/python train_lgbm.py --no-embed --symbol BTCUSDT

# 4. Train range LGBM — uses finetuned.pt embeddings
.venv/bin/python train_lgbm_range.py --symbol BTCUSDT

# 5. Validate (must use --interval 1h; LGBM features are 1h-aligned)
.venv/bin/python fast_backtest.py --mode lgbm --interval 1h

# 6. Train scalp LGBM (optional — enables 1m scalp layer)
.venv/bin/python train_lgbm_scalp.py --symbol BTCUSDT

# 7. Run live (dry-run), with scalp layer and 500-bar warm-start
.venv/bin/python live_trader.py --lgbm --scalp --interval 1m --dry-run --symbol BTCUSDT --warmup-bars 500
```

**Minimum lookback:** Feature builders require ≥ 89 bars (WINDOW=89). The live trader bootstraps 511 historical 1m candles at startup; the 1h HTF cache must have ≥ 90 bars.

**Warm-start pipeline (`--warmup-bars N`, default 500 ≈ 8h):** Before the WebSocket connects, `_run_warmup` replays the last N bars of the 1m cache through `on_candle` (HTF forward-filled via searchsorted) and `ScalpLayer.on_1m_bar`, with `_warmup=True` suppressing all position management. Primes `RegimeDetector` (needs ~45 bars / 3 cycles to confirm trending) and fills `ScalpLayer._buf` (needs 30 bars). Set `--warmup-bars 0` to skip.

---

## 10. Key Files

| File | Purpose |
|---|---|
| `transformer/encoder.py` | MultiScaleEncoder architecture |
| `transformer/pretrain.py` | Self-supervised masked bar reconstruction pretraining |
| `transformer/finetune.py` | Supervised fine-tuning on BUY/SELL/HOLD labels |
| `transformer/embed.py` | `load_encoder()` — loads finetuned.pt if present, else pretrained.pt |
| `lgbm/predictor.py` | LGBMPredictor — loads both models, routes by regime |
| `lgbm/regime_detector.py` | ADX-based RegimeDetector (stateful live / stateless batch) |
| `lgbm/labels.py` | Triple-barrier labels (trend) |
| `lgbm/labels_range.py` | Value-area labels with breakdown/retest cases (range) |
| `orderflow/features.py` | 24-feature trend matrix builder |
| `orderflow/range_features.py` | 29/93-feature range matrix builder |
| `train_lgbm.py` | Trend model training script |
| `train_lgbm_range.py` | Range model training script |
| `fast_backtest.py` | Backtester (use `--mode lgbm --interval 1h`) |
| `live_trader.py` | Live trading loop — includes `_run_warmup` warm-start pipeline |
| `orderflow/scalp_features.py` | 15-feature 1m scalp feature builder |
| `lgbm/scalp_layer.py` | ScalpLayer — 1m inference + Kelly position management on partitioned balance slice |
| `train_lgbm_scalp.py` | 1m scalp model training (multiclass, grid-searched TP/SL/horizon) |
| `lgbm/btc_scalp_model.lgbm` | Trained scalp model — TP=0.2% SL=0.2% horizon=30, logloss=0.9076 |

---

## 11. Known Limitations

- **Trending PF slightly below gate (1.362 vs 1.39):** Triple-barrier labels use a fixed TP=1.0%. In sustained trends the model exits early at TP while the move continues. ATR-scaled or trailing TP would likely improve this.
- **Fine-tuning signal is weak:** Val loss hovers near random (~1.03 vs 1.099 baseline). The encoder learns multi-scale structure in pretraining but BUY/SELL/HOLD labels are too noisy to produce strongly directional embeddings. Useful for range geometry; not useful for trend direction.
- **Cold-start regime bias:** Mitigated by `--warmup-bars 500` (default). Without warmup, detector initialises as "ranging" and takes ~45 min to confirm trending. With warmup, 500 1m bars (≈8h) replay in seconds and pre-prime the detector.
- **Single symbol:** Trained and validated on BTCUSDT only. Generalisation to other instruments is untested.
- **Backtest statistics unreliable at n=37:** The 30-day window produces too few trades for stable PF or Sharpe estimates. Treat as directional signal only; a longer OOS window is needed before live capital deployment.
