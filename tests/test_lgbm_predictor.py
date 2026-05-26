# tests/test_lgbm_predictor.py
import numpy as np
import pytest
from unittest.mock import MagicMock

from lgbm.predictor import LGBMPredictor, MIN_CONFIDENCE, MIN_RANGE_CONFIDENCE
from orderflow.features import N_FEATURES
from orderflow.range_features import N_RANGE_FEATURES


def _make_predictor(trend_probs=(0.1, 0.1, 0.8), range_probs=(0.1, 0.1, 0.8)):
    mock_trend = MagicMock()
    mock_trend.predict.return_value = np.array([trend_probs])
    mock_range = MagicMock()
    mock_range.predict.return_value = np.array([range_probs])
    mock_det   = MagicMock()
    pred = LGBMPredictor.__new__(LGBMPredictor)
    pred._trend_model = mock_trend
    pred._range_model = mock_range
    pred._detector    = mock_det
    return pred


def test_routes_to_trend_model_when_trending():
    pred = _make_predictor(trend_probs=(0.1, 0.1, 0.8))
    result = pred.predict(
        features_24    = np.zeros(N_FEATURES, dtype=np.float32),
        features_range = None,
        regime         = "trending",
        va_levels_last = {},
    )
    assert result["signal"] == "buy"
    assert result["confidence"] == pytest.approx(0.8)
    pred._trend_model.predict.assert_called_once()
    pred._range_model.predict.assert_not_called()


def test_routes_to_trend_model_when_volatile():
    pred = _make_predictor(trend_probs=(0.8, 0.1, 0.1))
    result = pred.predict(
        features_24    = np.zeros(N_FEATURES, dtype=np.float32),
        features_range = None,
        regime         = "volatile",
        va_levels_last = {},
    )
    assert result["regime"] == "volatile"
    pred._trend_model.predict.assert_called_once()


def test_routes_to_range_model_when_ranging():
    pred = _make_predictor(range_probs=(0.8, 0.1, 0.1))
    result = pred.predict(
        features_24    = None,
        features_range = np.zeros(N_RANGE_FEATURES, dtype=np.float32),
        regime         = "ranging",
        va_levels_last = {"val": 49800.0, "poc": 50000.0, "vah": 50200.0},
    )
    assert result["signal"] == "sell"
    assert result["regime"] == "ranging"
    assert result["val_price"] == pytest.approx(49800.0)
    assert result["poc_price"] == pytest.approx(50000.0)
    assert result["vah_price"] == pytest.approx(50200.0)
    pred._range_model.predict.assert_called_once()
    pred._trend_model.predict.assert_not_called()


def test_trend_result_has_none_va_prices():
    pred = _make_predictor()
    result = pred.predict(
        features_24    = np.zeros(N_FEATURES, dtype=np.float32),
        features_range = None,
        regime         = "trending",
        va_levels_last = {},
    )
    assert result["val_price"] is None
    assert result["poc_price"] is None
    assert result["vah_price"] is None


def test_hold_when_below_min_confidence_trending():
    pred = _make_predictor(trend_probs=(0.4, 0.3, 0.3))  # max directional = 0.4 < 0.55
    result = pred.predict(
        features_24    = np.zeros(N_FEATURES, dtype=np.float32),
        features_range = None,
        regime         = "trending",
        va_levels_last = {},
    )
    assert result["signal"] == "hold"


def test_hold_when_below_min_range_confidence():
    pred = _make_predictor(range_probs=(0.3, 0.5, 0.2))  # max directional = 0.3 < 0.50
    result = pred.predict(
        features_24    = None,
        features_range = np.zeros(N_RANGE_FEATURES, dtype=np.float32),
        regime         = "ranging",
        va_levels_last = {"val": 0.0, "poc": 0.0, "vah": 0.0},
    )
    assert result["signal"] == "hold"


def test_classify_regime_delegates_to_detector():
    pred = _make_predictor()
    pred._detector.classify.return_value = "ranging"
    ohlcv = np.zeros((60, 5), dtype=np.float32)
    regime = pred.classify_regime(ohlcv)
    assert regime == "ranging"
    pred._detector.classify.assert_called_once_with(ohlcv)


def test_file_not_found_raises_for_missing_trend_model():
    with pytest.raises(FileNotFoundError, match="train_lgbm.py"):
        LGBMPredictor(
            trend_model_path=Path("/nonexistent/trend.lgbm"),
            range_model_path=Path("/nonexistent/range.lgbm"),
        )


def test_reload_method_exists():
    pred = LGBMPredictor.__new__(LGBMPredictor)
    pred._trend_model = MagicMock()
    pred._range_model = MagicMock()
    assert callable(getattr(pred, "reload", None))
