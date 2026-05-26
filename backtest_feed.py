"""
BacktestFeed — replays cached .npy OHLCV candles through the same interface as BinanceFeed.

Usage (via main.py):
    python3 main.py --backtest --backtest-speed 0   # max speed (120d in ~minutes)
    python3 main.py --backtest --backtest-speed 1.0 # real-time (1 candle/min)
    python3 main.py --backtest --backtest-speed 60  # 60× faster than real-time

Data format expected in data_cache/:
    {SYMBOL}_{N}d.npy  — shape (T, 5) float32, columns: open high low close volume
"""
import logging
import os
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from .config import BUFFER_SIZE, SYMBOLS
    from .data_feed import Candle, MarketState, OrderBook, SymbolBuffer
except ImportError:
    from config import BUFFER_SIZE, SYMBOLS
    from data_feed import Candle, MarketState, OrderBook, SymbolBuffer

log = logging.getLogger(__name__)

DATA_CACHE_DIR = str(Path(__file__).resolve().parent / "data_cache")


def _find_cache_file(symbol: str, cache_dir: str, interval: str = None) -> Optional[str]:
    """Find the largest available cache file for a symbol.

    Handles both old-style (BTCUSDT_90d.npy) and new-style (BTCUSDT_1700d_1h.npy) names.
    If interval is given, only files matching that interval are considered.
    """
    import re
    best = None
    best_days = 0
    pattern = re.compile(r'^(.+?)_(\d+)d(?:_([^.]+))?\.npy$')
    for fname in os.listdir(cache_dir):
        if fname.endswith('_ts.npy'):
            continue
        m = pattern.match(fname)
        if not m:
            continue
        sym, days_str, file_interval = m.group(1), m.group(2), m.group(3)
        if sym.upper() != symbol.upper():
            continue
        if interval and file_interval != interval:
            continue
        try:
            days = int(days_str)
            if days > best_days:
                best_days = days
                best = os.path.join(cache_dir, fname)
        except ValueError:
            pass
    return best


class BacktestFeed:
    """
    Replays cached 1m OHLCV candles, exposing the same interface as BinanceFeed.

    speed_multiplier:
        0   = as fast as possible (no sleep between candles)
        1.0 = real-time (sleep 60s between candles)
        60  = 60× speed (sleep 1s between candles)
    """

    def __init__(
        self,
        symbols: List[str] = SYMBOLS,
        cache_dir: str = DATA_CACHE_DIR,
        speed_multiplier: float = 0.0,
        start_offset: int = 0,        # skip first N candles (default: start from beginning)
    ):
        self.symbols = [s.upper() for s in symbols]
        self.cache_dir = cache_dir
        self.speed_multiplier = speed_multiplier
        self.start_offset = start_offset

        self.buffers: Dict[str, SymbolBuffer] = {
            s: SymbolBuffer(s, maxlen=BUFFER_SIZE) for s in self.symbols
        }

        # maxsize=1 blocks the producer until the consumer processes each candle
        self._candle_queue: queue.Queue = queue.Queue(maxsize=1)
        self._done = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Load data
        self._data: Dict[str, np.ndarray] = {}
        for sym in self.symbols:
            path = _find_cache_file(sym, cache_dir)
            if path is None:
                raise FileNotFoundError(
                    f"No cache file for {sym} in {cache_dir}. "
                    f"Run: python3 pretrain.py --symbols {sym} --days 120 --cache-only"
                )
            arr = np.load(path)
            self._data[sym] = arr[start_offset:]
            log.info("BacktestFeed: loaded %s — %d candles from %s", sym, len(arr) - start_offset, path)

        total = min(len(v) for v in self._data.values())
        log.info(
            "BacktestFeed: %d symbols × %d candles = %d ticks total (speed=%.0fx)",
            len(self.symbols), total, total * len(self.symbols),
            speed_multiplier if speed_multiplier > 0 else float("inf"),
        )

    # ------------------------------------------------------------------
    # Public API — matches BinanceFeed exactly
    # ------------------------------------------------------------------

    def start(self):
        self._thread = threading.Thread(target=self._replay, daemon=True)
        self._thread.start()
        log.info("BacktestFeed started (%d symbols, speed=%.0fx)", len(self.symbols), self.speed_multiplier)

    def stop(self):
        self._done.set()

    def wait_for_candle(self, timeout: float = 90.0) -> bool:
        try:
            self._candle_queue.get(timeout=timeout)
            return True
        except queue.Empty:
            return False

    def wait_for_tick(self, timeout: float = 5.0) -> bool:
        return self.wait_for_candle(timeout=timeout)

    def get_state(self, symbol: str) -> Optional[MarketState]:
        return self.buffers[symbol.upper()].get_state()

    def get_all_states(self) -> Dict[str, MarketState]:
        states = {}
        for sym, buf in self.buffers.items():
            s = buf.get_state()
            if s is not None:
                states[sym] = s
        return states

    def is_ready(self) -> bool:
        return all(len(b) >= 2 for b in self.buffers.values())

    def is_tick_ready(self) -> bool:
        return self.is_ready()

    def is_done(self) -> bool:
        """True when all historical candles have been replayed AND consumed."""
        return self._done.is_set() and self._candle_queue.empty()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _replay(self):
        """Push candles into buffers in chronological order."""
        arrays = {sym: self._data[sym] for sym in self.symbols}
        n_candles = min(len(v) for v in arrays.values())

        # Fake timestamps: start from a fixed epoch, 1 min apart
        base_ts = 1_600_000_000_000  # ms — arbitrary historical start

        sleep_per_candle = 0.0
        if self.speed_multiplier > 0:
            sleep_per_candle = 60.0 / self.speed_multiplier

        for i in range(n_candles):
            if self._done.is_set():
                break

            ts = base_ts + i * 60_000  # ms

            for sym in self.symbols:
                row = arrays[sym][i]
                o, h, l, c, v = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
                candle = Candle(
                    timestamp=ts,
                    open=o, high=h, low=l, close=c, volume=v,
                    closed=True,
                )
                self.buffers[sym].update_candle(candle)

            self._candle_queue.put(1)  # blocks until main loop consumes previous candle

            if sleep_per_candle > 0:
                time.sleep(sleep_per_candle)

        log.info("BacktestFeed: replay complete (%d candles)", n_candles)
        self._done.set()
        # Unblock any waiter so main loop can detect completion
        self._candle_queue.put(1, block=False) if self._candle_queue.empty() else None
