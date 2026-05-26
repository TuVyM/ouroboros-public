# lgbm/predictor.py
from pathlib import Path

import lightgbm as lgb
import numpy as np

from lgbm.regime_detector import RegimeDetector

TREND_MODEL_PATH = Path(__file__).parent / "btc_trend_model.lgbm"
RANGE_MODEL_PATH = Path(__file__).parent / "btc_range_model.lgbm"

MIN_CONFIDENCE       = 0.55   # trend + volatile models
MIN_RANGE_CONFIDENCE = 0.50   # range model (pre-filtered by VAL/VAH window)


class LGBMPredictor:
    """Load trend and range LGBM models and route based on regime."""

    def __init__(
        self,
        trend_model_path: Path = TREND_MODEL_PATH,
        range_model_path: Path = RANGE_MODEL_PATH,
    ):
        trend_model_path = Path(trend_model_path)
        range_model_path = Path(range_model_path)

        if not trend_model_path.exists():
            raise FileNotFoundError(
                f"Trend model not found: {trend_model_path}\n"
                "Run: python train_lgbm.py --symbol BTCUSDT"
            )
        if not range_model_path.exists():
            raise FileNotFoundError(
                f"Range model not found: {range_model_path}\n"
                "Run: python train_lgbm_range.py --symbol BTCUSDT"
            )
        self._trend_model_path = trend_model_path
        self._range_model_path = range_model_path
        self._trend_model = lgb.Booster(model_file=str(trend_model_path))
        self._range_model = lgb.Booster(model_file=str(range_model_path))
        self._detector    = RegimeDetector()

    def classify_regime(self, htf_1h: np.ndarray) -> str:
        """Classify regime from (N, 5) OHLCV. Stateful — call once per bar."""
        return self._detector.classify(htf_1h)

    def predict(
        self,
        features_24:    np.ndarray | None,
        features_range: np.ndarray | None,
        regime:         str,
        va_levels_last: dict,
    ) -> dict:
        """
        Route to the correct specialist model.

        Args:
            features_24:    (24,) float32 — used when regime is trending/volatile
            features_range: (29,) float32 — used when regime is ranging
            regime:         pre-classified regime string from classify_regime()
            va_levels_last: {"val", "poc", "vah"} from build_range_features() — ranging only

        Returns dict with keys: signal, confidence, regime, val_price, poc_price, vah_price
        """
        if regime == "ranging":
            probs  = self._range_model.predict(features_range.reshape(1, -1))[0]
            result = self._interpret_probs(probs, MIN_RANGE_CONFIDENCE)
            result["val_price"] = float(va_levels_last.get("val", 0.0))
            result["poc_price"] = float(va_levels_last.get("poc", 0.0))
            result["vah_price"] = float(va_levels_last.get("vah", 0.0))
        else:
            probs  = self._trend_model.predict(features_24.reshape(1, -1))[0]
            result = self._interpret_probs(probs, MIN_CONFIDENCE)
            result["val_price"] = result["poc_price"] = result["vah_price"] = None
        result["regime"] = regime
        return result

    def reload(self) -> None:
        """Hot-reload both models from disk (called after shadow swap)."""
        self._trend_model = lgb.Booster(model_file=str(self._trend_model_path))
        self._range_model = lgb.Booster(model_file=str(self._range_model_path))

    @staticmethod
    def _interpret_probs(probs: np.ndarray, min_conf: float) -> dict:
        sell_conf = float(probs[0])
        hold_conf = float(probs[1])
        buy_conf  = float(probs[2])
        best = max(buy_conf, sell_conf)
        if best < min_conf:
            return {"signal": "hold", "confidence": hold_conf}
        if buy_conf >= sell_conf:
            return {"signal": "buy",  "confidence": buy_conf}
        return {"signal": "sell", "confidence": sell_conf}
