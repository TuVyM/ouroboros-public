# tests/test_lgbm_shadow.py
import numpy as np
import pytest

from lgbm.shadow_trainer import LGBMTradeBuffer, MIN_TRADE_BUFFER


def _feat(seed=0):
    rng = np.random.default_rng(seed)
    return rng.random(24).astype(np.float32)


def test_buffer_starts_empty():
    buf = LGBMTradeBuffer()
    assert len(buf) == 0


def test_push_increases_length():
    buf = LGBMTradeBuffer()
    buf.push(_feat(), action=2, pnl=0.01)
    assert len(buf) == 1


def test_train_val_split_proportions():
    buf = LGBMTradeBuffer()
    for i in range(100):
        buf.push(_feat(i), action=i % 3, pnl=0.01 * (1 if i % 2 == 0 else -1))
    train, val = buf.train_val_split()
    assert len(val) == 20   # VAL_FRACTION=0.20 → 20 val records
    assert len(train) == 80


def test_train_val_split_val_is_newest():
    buf = LGBMTradeBuffer()
    for i in range(10):
        buf.push(_feat(i), action=i % 3, pnl=float(i))
    _, val = buf.train_val_split()
    # The val records should have the largest pnl values (newest pushes)
    val_pnls = [r[2] for r in val]
    assert max(val_pnls) >= 7.0   # last 20% of 10 = 2 records, pnl=8 and 9


def test_get_arrays_returns_correct_shapes():
    buf = LGBMTradeBuffer()
    for i in range(10):
        buf.push(_feat(i), action=i % 3, pnl=float(i))
    records = buf.train_val_split()[0]
    X, y, pnl = LGBMTradeBuffer.records_to_arrays(records)
    assert X.shape == (8, 24)
    assert y.shape == (8,)
    assert pnl.shape == (8,)


def test_min_trade_buffer_is_fibonacci():
    import math
    fibs = set()
    a, b = 0, 1
    while b <= 10_000:
        fibs.add(b)
        a, b = b, a + b
    assert MIN_TRADE_BUFFER in fibs


def test_maxlen_enforced():
    buf = LGBMTradeBuffer(maxlen=5)
    for i in range(10):
        buf.push(_feat(i), action=0, pnl=0.0)
    assert len(buf) == 5


from unittest.mock import MagicMock, patch
from lgbm.shadow_trainer import LGBMShadowTrainer, _compute_lgbm_pf, SWAP_PF_MARGIN


def _make_predictor_mock(signal="buy", confidence=0.8):
    mock = MagicMock()
    mock.predict.return_value = {"signal": signal, "confidence": confidence}
    mock._model_path = Path("lgbm/btc_model.lgbm")
    return mock


def _fill_buffer(n=60, win_rate=0.6):
    buf = LGBMTradeBuffer()
    for i in range(n):
        pnl = 0.01 if i / n < win_rate else -0.005
        buf.push(_feat(i), action=2, pnl=pnl)   # all buy signals
    return buf


def test_compute_lgbm_pf_all_wins():
    records = [(_feat(i), 2, 0.01) for i in range(10)]
    mock_model = MagicMock()
    # Model always predicts buy (action=2) — matches all records
    mock_model.predict.return_value = np.array([[0.1, 0.1, 0.8]] * 10)
    pf = _compute_lgbm_pf(records, mock_model)
    assert pf > 1.0


def test_compute_lgbm_pf_all_wrong():
    # Records say buy (action=2), model predicts sell (action=0) → flip all pnls
    records = [(_feat(i), 2, 0.01) for i in range(10)]
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.8, 0.1, 0.1]] * 10)
    pf = _compute_lgbm_pf(records, mock_model)
    assert pf == pytest.approx(0.0, abs=1e-6)   # all losses → wins=0 → PF=0


def test_compute_lgbm_pf_empty_returns_zero():
    mock_model = MagicMock()
    pf = _compute_lgbm_pf([], mock_model)
    assert pf == pytest.approx(0.0)


def test_shadow_trainer_run_sync_skips_when_too_few_trades():
    mock_pred = _make_predictor_mock()
    mock_pred._model_path = MagicMock()
    mock_pred._model_path.exists.return_value = False
    trainer = LGBMShadowTrainer(live_predictor=mock_pred)
    result = trainer.run_sync_cycle()
    assert result["skipped"] is True
    assert result["n_trades"] < MIN_TRADE_BUFFER


def test_shadow_trainer_push_trade_increases_buffer():
    mock_pred = _make_predictor_mock()
    mock_pred._model_path = MagicMock()
    mock_pred._model_path.exists.return_value = False
    trainer = LGBMShadowTrainer(live_predictor=mock_pred)
    trainer.push_trade(_feat(), action=2, pnl=0.01)
    assert len(trainer.trade_buf) == 1


def test_swap_pf_margin_is_phi_based():
    import math
    phi = (1 + math.sqrt(5)) / 2
    expected = 1.0 + 1.0 / (phi * 100)
    assert abs(SWAP_PF_MARGIN - expected) < 1e-9
