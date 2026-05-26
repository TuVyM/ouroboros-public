# Ouroboros — Regime-Adaptive LGBM Trading System

A live algorithmic trading system for BTCUSDT perpetual futures on Binance. Uses LightGBM with regime-dependent routing, a MultiScale Transformer encoder, and a 1m scalp layer on top of a 1h HTF signal.

**Status:** Live dry-run validated. See [MODEL_CARD.md](docs/MODEL_CARD.md) for full architecture, backtest results, and known limitations.

---

## Architecture

```
1h OHLCV (live WebSocket)
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
MultiScaleEncoder               RegimeDetector (ADX-14)
  (1h/4h/1d Transformer)          trending / ranging / volatile
  64-dim latent z                        │
    │                                    ▼
    │                        Trend LGBM / Range LGBM
    └──────────────────────────────────►│
                                        ▼
                                  BUY / SELL / HOLD
                                        │
                                 Position Manager
                                 (Kelly-capped, ATR SL/TP)
                                        │
                              1m Scalp Layer (optional)
                              (separate balance slice)
```

- **Trend LGBM** — 24 raw orderflow features, no encoder
- **Range LGBM** — 29 range features + 64-dim encoder embedding
- **ScalpLayer** — 15-feature 1m inference, Kelly position sizing on partitioned balance

Full architecture details in [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

---

## Quickstart

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Fetch historical data

```bash
python pretrain.py --symbols BTCUSDT --days 90 --interval 1h --cache-only
python pretrain.py --symbols BTCUSDT --days 90 --interval 1m --cache-only
```

### 3. Train models

```bash
# Pretrain + fine-tune encoder (needed for range model)
python -m transformer.pretrain --symbol BTCUSDT
python -m transformer.finetune --symbol BTCUSDT

# Train trend LGBM
python train_lgbm.py --no-embed --symbol BTCUSDT

# Train range LGBM
python train_lgbm_range.py --symbol BTCUSDT

# Train scalp LGBM (optional)
python train_lgbm_scalp.py --symbol BTCUSDT
```

### 4. Backtest

```bash
python fast_backtest.py --mode lgbm --interval 1h
```

### 5. Run live (dry-run)

```bash
# HTF only
python live_trader.py --lgbm --interval 1m --dry-run --symbol BTCUSDT

# With 1m scalp layer and historical warm-start
python live_trader.py --lgbm --scalp --dry-run --symbol BTCUSDT --warmup-bars 500
```

---

## Key Files

| File | Purpose |
|---|---|
| `live_trader.py` | Live trading loop, WebSocket event-driven |
| `train_lgbm.py` | Trend LGBM training |
| `train_lgbm_range.py` | Range LGBM training |
| `train_lgbm_scalp.py` | 1m scalp LGBM training |
| `fast_backtest.py` | Backtester |
| `lgbm/predictor.py` | Regime-routing inference |
| `lgbm/regime_detector.py` | ADX-based regime detection |
| `lgbm/scalp_layer.py` | 1m scalp inference + position management |
| `orderflow/features.py` | 24-feature trend matrix builder |
| `orderflow/scalp_features.py` | 15-feature 1m scalp builder |
| `transformer/encoder.py` | MultiScaleEncoder (1h/4h/1d Transformer) |

---

## Configuration

Copy `.env.example` to `.env` and set values:

```bash
# Trading
DRY_RUN=true                    # set false for live trading
SCALPER_BALANCE_FRACTION=0.33   # fraction of balance for scalp layer

# Scalp SL/TP (override ATR-based defaults)
SCALP_SL_PCT=0.0020
SCALP_TP_PCT=0.0020
```

---

## Backtest Results (30-day, 1h bars)

| Metric | Value |
|---|---|
| Overall PF | 1.846 |
| Trending PF | 1.362 |
| Ranging PF | 3.917 |
| Win rate | 59.5% |
| Total trades | 37 |

*See [MODEL_CARD.md](docs/MODEL_CARD.md) for full results and caveats.*

---

## Tests

```bash
pytest tests/ -q
```

282 tests, ~8s.

---

## Disclaimer

This is experimental software. It trades real money if `DRY_RUN=false`. Past backtest performance does not guarantee future results. Use at your own risk.
