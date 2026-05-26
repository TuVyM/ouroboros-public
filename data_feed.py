"""
Binance WebSocket data feed.

Streams:
  @aggTrade        — aggregated trades (sub-second, ~ms latency)
                     Used to build 1s candles for high-frequency swarm ticks.
  @kline_1m        — 1-minute closed candles (for world model training)
  @depth5@100ms    — order book top-5 snapshots

Tick hierarchy:
  tick_window  — last TICK_BUFFER_SIZE × 1s candles (built from aggTrades)
  ohlcv_window — last BUFFER_SIZE × 1m candles
"""
import asyncio
import json
import logging
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import websockets

try:
    from .config import SYMBOLS, BUFFER_SIZE, BINANCE_WS_URL
except ImportError:
    from config import SYMBOLS, BUFFER_SIZE, BINANCE_WS_URL

log = logging.getLogger(__name__)

TICK_BUFFER_SIZE = 60   # seconds of 1s candles kept in tick_window


@dataclass
class Candle:
    timestamp: int       # ms
    open: float
    high: float
    low: float
    close: float
    volume: float
    closed: bool         # True = final candle for that minute


@dataclass
class OrderBook:
    bids: List[Tuple[float, float]] = field(default_factory=list)  # (price, qty)
    asks: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class MarketState:
    symbol: str
    last_candle: Optional[Candle]
    ohlcv_window: np.ndarray      # (BUFFER_SIZE, 5) — 1m candles
    order_book: OrderBook
    timestamp: int                 # ms
    tick_window: Optional[np.ndarray] = None  # (TICK_BUFFER_SIZE, 5) — 1s candles, may be None
    funding_rate: float = 0.0     # current 8h funding rate (e.g. 0.0001 = 0.01%)
    oi_usd: float = 0.0           # open interest in USD (mark_price × contracts)
    tick_delta: float = 0.0       # (buy_vol - sell_vol) / total_vol over tick_window [-1, 1]
    peer_corr: float = 0.0        # rolling return correlation with peer symbols [-1, 1]
    liq_usd_5m: float = 0.0       # USD liquidated in last 5 min (all symbols on Binance Futures)
    liq_direction: float = 0.0    # (short_liq - long_liq) / total: positive = short squeeze (bullish)
    htf_1h: Optional[np.ndarray] = None   # (N, 5) — last 168 1h candles
    htf_1d: Optional[np.ndarray] = None   # (N, 5) — last 365 1d candles
    fear_greed: int = 50           # Alternative.me Fear & Greed 0-100
    dxy: float = 100.0             # FRED Broad Dollar Index (proxy for USD strength)
    mayer_multiple: float = 1.0    # price / 200d SMA — Mayer Multiple (computed from htf_1d)


class _SecondCandle:
    """Accumulates aggTrades within the current second into a 1s OHLCV bar."""

    __slots__ = ("second_ts", "open", "high", "low", "close", "volume", "buy_vol", "sell_vol")

    def __init__(self, ts_sec: int, price: float, qty: float, is_buyer_maker: bool = False):
        self.second_ts = ts_sec
        self.open = price
        self.high = price
        self.low = price
        self.close = price
        self.volume = qty
        self.buy_vol = 0.0 if is_buyer_maker else qty   # buyer is taker → buy aggressor
        self.sell_vol = qty if is_buyer_maker else 0.0  # buyer is maker → sell aggressor

    def update(self, price: float, qty: float, is_buyer_maker: bool = False):
        if price > self.high:
            self.high = price
        if price < self.low:
            self.low = price
        self.close = price
        self.volume += qty
        if is_buyer_maker:
            self.sell_vol += qty
        else:
            self.buy_vol += qty

    def to_ohlcv(self) -> Tuple[float, float, float, float, float]:
        return self.open, self.high, self.low, self.close, self.volume

    @property
    def delta(self) -> float:
        """(buy_vol - sell_vol) / total_vol — positive = buy pressure."""
        total = self.buy_vol + self.sell_vol + 1e-8
        return (self.buy_vol - self.sell_vol) / total


class SymbolBuffer:
    def __init__(self, symbol: str, maxlen: int = BUFFER_SIZE):
        self.symbol = symbol
        self.maxlen = maxlen
        self._candles: deque = deque(maxlen=maxlen)        # 1m candles
        self._tick_candles: deque = deque(maxlen=TICK_BUFFER_SIZE)  # 1s candles
        self._vol_deltas: deque = deque(maxlen=TICK_BUFFER_SIZE)   # per-second tick deltas
        self._current_second: Optional[_SecondCandle] = None
        self._order_book = OrderBook()
        self._lock = threading.Lock()
        self._liq_events: deque = deque()  # (ts_ms, usd_value, is_short_liq) — pruned to 5 min

    # ------------------------------------------------------------------
    # 1m candle (from @kline_1m)
    # ------------------------------------------------------------------

    def update_candle(self, candle: Candle):
        with self._lock:
            if candle.closed:
                self._candles.append(candle)

    # ------------------------------------------------------------------
    # Sub-second trades (from @aggTrade) → 1s candles
    # ------------------------------------------------------------------

    def update_trade(self, price: float, qty: float, trade_ts_ms: int,
                     is_buyer_maker: bool = False) -> bool:
        """
        Accumulate a trade into the current 1s candle.
        Returns True when a new 1s candle is completed (second boundary crossed).
        is_buyer_maker=True means the buyer is the passive side → sell aggressor.
        """
        ts_sec = trade_ts_ms // 1000
        with self._lock:
            if self._current_second is None:
                self._current_second = _SecondCandle(ts_sec, price, qty, is_buyer_maker)
                return False

            if ts_sec == self._current_second.second_ts:
                self._current_second.update(price, qty, is_buyer_maker)
                return False

            # Second boundary — flush current candle
            self._tick_candles.append(self._current_second.to_ohlcv())
            self._vol_deltas.append(self._current_second.delta)
            self._current_second = _SecondCandle(ts_sec, price, qty, is_buyer_maker)
            return True  # new 1s candle ready

    # ------------------------------------------------------------------
    # Order book
    # ------------------------------------------------------------------

    def update_order_book(self, bids: list, asks: list):
        with self._lock:
            self._order_book = OrderBook(
                bids=[(float(p), float(q)) for p, q in bids[:5]],
                asks=[(float(p), float(q)) for p, q in asks[:5]],
            )

    def update_liquidation(self, ts_ms: int, usd_value: float, is_short_liq: bool):
        """Accumulate a futures liquidation event. is_short_liq=True means shorts squeezed (bullish)."""
        with self._lock:
            self._liq_events.append((ts_ms, usd_value, is_short_liq))

    def get_liq_since(self, since_ms: int) -> float:
        """Return signed liq notional for events at or after since_ms.

        Positive = short liquidations (bullish short squeeze).
        Negative = long liquidations (bearish cascade).
        """
        with self._lock:
            total = 0.0
            for ts, usd, is_short in self._liq_events:
                if ts >= since_ms:
                    total += usd if is_short else -usd
            return total

    # ------------------------------------------------------------------
    # State snapshot
    # ------------------------------------------------------------------

    def get_state(self) -> Optional[MarketState]:
        with self._lock:
            if len(self._candles) < 2:
                return None
            arr_1m = np.array(
                [[c.open, c.high, c.low, c.close, c.volume] for c in self._candles],
                dtype=np.float32,
            )
            last = self._candles[-1]

            tick_arr = None
            if len(self._tick_candles) >= 2:
                tick_arr = np.array(list(self._tick_candles), dtype=np.float32)  # (N, 5)

            tick_delta = float(np.mean(list(self._vol_deltas))) if self._vol_deltas else 0.0

            # Liquidation stats — prune to last 5 min, compute directional pressure
            now_ms = int(time.time() * 1000)
            cutoff_ms = now_ms - 300_000
            while self._liq_events and self._liq_events[0][0] < cutoff_ms:
                self._liq_events.popleft()
            liq_total = liq_short = liq_long = 0.0
            for _ts, usd, is_short in self._liq_events:
                liq_total += usd
                if is_short:
                    liq_short += usd
                else:
                    liq_long += usd
            liq_direction = (liq_short - liq_long) / (liq_total + 1e-8)

            return MarketState(
                symbol=self.symbol,
                last_candle=last,
                ohlcv_window=arr_1m,
                order_book=OrderBook(
                    bids=list(self._order_book.bids),
                    asks=list(self._order_book.asks),
                ),
                timestamp=last.timestamp,
                tick_window=tick_arr,
                tick_delta=tick_delta,
                liq_usd_5m=liq_total,
                liq_direction=liq_direction,
            )

    def __len__(self):
        return len(self._candles)


class FundingFetcher:
    """
    Polls Binance FAPI every 5 minutes for funding rate + open interest.
    Runs in a daemon thread — safe to ignore if FAPI is unavailable (perpetuals only).
    """
    _FAPI = "https://fapi.binance.com/fapi/v1"

    def __init__(self, symbols: List[str]):
        self._symbols = [s.upper() for s in symbols]
        self._data: Dict[str, Tuple[float, float]] = {s: (0.0, 0.0) for s in self._symbols}
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def get(self, symbol: str) -> Tuple[float, float]:
        """Returns (funding_rate, oi_usd). Thread-safe."""
        with self._lock:
            return self._data.get(symbol.upper(), (0.0, 0.0))

    def _fetch(self, url: str) -> dict:
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read())

    def _run(self):
        while True:
            for sym in self._symbols:
                try:
                    d = self._fetch(f"{self._FAPI}/premiumIndex?symbol={sym}")
                    fr = float(d.get("lastFundingRate", 0.0))
                    mark = float(d.get("markPrice", 0.0))
                    d2 = self._fetch(f"{self._FAPI}/openInterest?symbol={sym}")
                    oi_usd = float(d2.get("openInterest", 0.0)) * mark
                    with self._lock:
                        self._data[sym] = (fr, oi_usd)
                except Exception as exc:
                    log.debug("FundingFetcher %s: %s", sym, exc)
            time.sleep(300)


class HTFCandleFetcher:
    """
    Polls Binance REST every 15 min for higher-timeframe OHLCV:
      1h candles: last 168 bars (1 week of context)
      1d candles: last 365 bars (1 year of context)
    Gives _range_position_signal() macro range awareness.
    """
    _BASE = "https://api.binance.com/api/v3/klines"
    _1H_BARS = 721   # yields 720 closed 1h bars — needed for 1d encoder (30 × 24 = 720)
    _1D_BARS = 366

    def __init__(self, symbols: List[str]):
        self._symbols = [s.upper() for s in symbols]
        self._data: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {
            s: (None, None) for s in self._symbols
        }
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._fetch_all()
        self._thread.start()

    def get(self, symbol: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._lock:
            return self._data.get(symbol.upper(), (None, None))

    def _fetch_candles(self, symbol: str, interval: str, limit: int) -> Optional[np.ndarray]:
        url = f"{self._BASE}?symbol={symbol}&interval={interval}&limit={limit}"
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                rows = json.loads(r.read())
            arr = np.array([[float(row[1]), float(row[2]), float(row[3]),
                              float(row[4]), float(row[5])] for row in rows[:-1]],
                            dtype=np.float32)
            return arr
        except Exception as exc:
            log.debug("HTFCandleFetcher %s %s: %s", symbol, interval, exc)
            return None

    def _fetch_all(self):
        for sym in self._symbols:
            h1 = self._fetch_candles(sym, "1h", self._1H_BARS)
            d1 = self._fetch_candles(sym, "1d", self._1D_BARS)
            with self._lock:
                self._data[sym] = (h1, d1)
            time.sleep(0.25)
        log.info("HTFCandleFetcher: refreshed 1h+1d candles for %d symbols", len(self._symbols))

    def _run(self):
        while True:
            time.sleep(900)
            self._fetch_all()


class MacroFetcher:
    """
    Polls macro indicators hourly:
      - Alternative.me Fear & Greed Index (no key, updates daily)
      - FRED Broad Dollar Index CSV (no key, daily)
    Mayer Multiple (MVRV proxy) is computed from HTFCandleFetcher 1d candles — no API needed.
    """
    _FNG_URL = "https://api.alternative.me/fng/?limit=1"
    _DXY_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXBGS"

    def __init__(self):
        self.fear_greed: int = 50
        self.dxy: float = 100.0
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._fetch_all()
        self._thread.start()

    def _fetch(self, url: str) -> bytes:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.read()

    def _update_fear_greed(self):
        try:
            data = json.loads(self._fetch(self._FNG_URL))
            with self._lock:
                self.fear_greed = int(data["data"][0]["value"])
        except Exception as exc:
            log.debug("MacroFetcher F&G: %s", exc)

    def _update_dxy(self):
        try:
            text = self._fetch(self._DXY_URL).decode()
            for line in reversed(text.strip().split("\n")):
                if line and not line.startswith("DATE"):
                    parts = line.split(",")
                    if len(parts) >= 2 and parts[1] and parts[1] != ".":
                        with self._lock:
                            self.dxy = float(parts[1])
                        break
        except Exception as exc:
            log.debug("MacroFetcher DXY: %s", exc)

    def _fetch_all(self):
        self._update_fear_greed()
        self._update_dxy()
        log.info("MacroFetcher: fear_greed=%d dxy=%.2f", self.fear_greed, self.dxy)

    def get(self) -> Tuple[int, float]:
        with self._lock:
            return self.fear_greed, self.dxy

    def _run(self):
        while True:
            time.sleep(3600)
            self._fetch_all()


class BinanceFeed:
    """
    Subscribes to:
      {symbol}@aggTrade        — sub-second trade stream
      {symbol}@kline_1m        — 1-minute candles
      {symbol}@depth5@100ms    — order book

    Fires two events:
      _new_tick    — set whenever a 1s candle completes (sub-second trading)
      _new_candle  — set whenever a 1m candle closes (training / slow path)
    """

    def __init__(self, symbols: List[str] = SYMBOLS):
        self.symbols = [s.lower() for s in symbols]
        self.buffers: Dict[str, SymbolBuffer] = {
            s.upper(): SymbolBuffer(s.upper()) for s in self.symbols
        }
        self._funding = FundingFetcher(symbols)
        self._htf = HTFCandleFetcher(symbols)
        self._macro = MacroFetcher()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._new_candle = threading.Event()   # 1m candle closed
        self._new_tick = threading.Event()     # 1s candle completed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        self._funding.start()
        self._htf.start()
        self._macro.start()
        self._bootstrap_history()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log.info("BinanceFeed started for %s (aggTrade + kline_1m + depth5)", self.symbols)

    def _bootstrap_history(self):
        """Pre-fill 1m candle buffers via REST so the window is immediately warm."""
        base = "https://api.binance.com/api/v3/klines"
        for sym in self.symbols:
            try:
                url = f"{base}?symbol={sym.upper()}&interval=1m&limit={BUFFER_SIZE}"
                with urllib.request.urlopen(url, timeout=10) as r:
                    rows = json.loads(r.read())
                buf = self.buffers[sym.upper()]
                for row in rows[:-1]:   # skip the current (open) candle
                    c = Candle(
                        timestamp=int(row[0]),
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=float(row[5]),
                        closed=True,
                    )
                    buf._candles.append(c)
                log.info("Bootstrapped %d historical 1m candles for %s", len(buf._candles), sym)
            except Exception as exc:
                log.warning("History bootstrap failed for %s: %s", sym, exc)

    def stop(self):
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

    def wait_for_candle(self, timeout: float = 90.0) -> bool:
        """Block until a new 1m closed candle arrives. Returns True if arrived."""
        fired = self._new_candle.wait(timeout=timeout)
        self._new_candle.clear()
        return fired

    def wait_for_tick(self, timeout: float = 5.0) -> bool:
        """Block until a new 1s candle completes. Returns True if arrived."""
        fired = self._new_tick.wait(timeout=timeout)
        self._new_tick.clear()
        return fired

    def get_state(self, symbol: str) -> Optional[MarketState]:
        return self.buffers[symbol.upper()].get_state()

    def get_liq_since(self, symbol: str, since_ms: int) -> float:
        """Signed liq notional for symbol since since_ms. Returns 0.0 if symbol unknown."""
        buf = self.buffers.get(symbol.upper())
        return buf.get_liq_since(since_ms) if buf else 0.0

    def get_all_states(self) -> Dict[str, MarketState]:
        raw: Dict[str, MarketState] = {}
        for sym, buf in self.buffers.items():
            s = buf.get_state()
            if s is not None:
                raw[sym] = s

        # Per-symbol log returns for cross-asset correlation
        returns: Dict[str, np.ndarray] = {}
        for sym, state in raw.items():
            c = state.ohlcv_window[:, 3]
            if len(c) > 5:
                returns[sym] = np.diff(np.log(c + 1e-8))

        for sym, state in raw.items():
            # Funding rate + OI from background fetcher
            state.funding_rate, state.oi_usd = self._funding.get(sym)

            # Peer correlation: average pairwise with all other symbols
            if sym in returns:
                corrs = []
                for other, other_ret in returns.items():
                    if other == sym:
                        continue
                    n = min(len(returns[sym]), len(other_ret))
                    if n > 5:
                        r1, r2 = returns[sym][-n:], other_ret[-n:]
                        c = np.corrcoef(r1, r2)
                        if c.shape == (2, 2):
                            corrs.append(float(np.clip(c[0, 1], -1.0, 1.0)))
                state.peer_corr = float(np.mean(corrs)) if corrs else 0.0

        # Populate HTF candles and macro indicators
        fear_greed, dxy = self._macro.get()
        for sym, state in raw.items():
            htf_1h, htf_1d = self._htf.get(sym)
            state.htf_1h = htf_1h
            state.htf_1d = htf_1d
            state.fear_greed = fear_greed
            state.dxy = dxy
            # Mayer Multiple: price / 200d SMA — proxy for BTC cycle position
            if htf_1d is not None and len(htf_1d) >= 200:
                sma_200 = float(htf_1d[-200:, 3].mean())
                state.mayer_multiple = float(htf_1d[-1, 3]) / (sma_200 + 1e-8)
            else:
                state.mayer_multiple = 1.0

        return raw

    def is_ready(self) -> bool:
        """Ready when all symbols have ≥2 1m candles."""
        return all(len(b) >= 2 for b in self.buffers.values())

    def is_tick_ready(self) -> bool:
        """Ready when all symbols have ≥2 1s candles (available within ~2s of startup)."""
        for buf in self.buffers.values():
            with buf._lock:
                if len(buf._tick_candles) < 2:
                    return False
        return True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(asyncio.gather(
            self._connect(),
            self._connect_liquidations(),
        ))

    async def _connect(self):
        streams = []
        for s in self.symbols:
            streams.append(f"{s}@aggTrade")
            streams.append(f"{s}@kline_1m")
            streams.append(f"{s}@depth5@100ms")
        url = f"{BINANCE_WS_URL}?streams=" + "/".join(streams)

        while True:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    log.info("WebSocket connected: %d streams", len(streams))
                    self._running.set()
                    async for raw in ws:
                        self._handle(json.loads(raw))
            except Exception as exc:
                log.warning("WebSocket disconnected (%s), reconnecting in 5s…", exc)
                await asyncio.sleep(5)

    async def _connect_liquidations(self):
        """Subscribe to Binance Futures all-liquidation stream (public, no auth)."""
        url = "wss://fstream.binance.com/ws/!forceOrder@arr"
        while True:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    log.info("Futures liquidation stream connected")
                    async for raw in ws:
                        self._handle_liquidation(json.loads(raw))
            except Exception as exc:
                log.debug("Liquidation stream disconnected (%s), reconnecting in 10s", exc)
                await asyncio.sleep(10)

    def _handle_liquidation(self, msg: dict):
        """
        Process a @forceOrder message.
        S="BUY"  → a short position was liquidated (forced buy) → bullish short squeeze
        S="SELL" → a long position was liquidated (forced sell) → bearish cascade
        """
        if msg.get("e") != "forceOrder":
            return
        o = msg.get("o", {})
        sym = o.get("s", "").upper()
        if sym not in self.buffers:
            return
        qty   = float(o.get("q", 0) or 0)
        price = float(o.get("ap", 0) or o.get("p", 0) or 0)
        usd_value = qty * price
        if usd_value <= 0:
            return
        is_short_liq = o.get("S") == "BUY"  # BUY order = short being liquidated
        ts_ms = int(msg.get("E", time.time() * 1000))
        self.buffers[sym].update_liquidation(ts_ms, usd_value, is_short_liq)

    def _handle(self, msg: dict):
        data = msg.get("data", {})
        stream = msg.get("stream", "")

        if "@aggTrade" in stream:
            symbol = data.get("s", "")
            if symbol in self.buffers:
                price = float(data["p"])
                qty = float(data["q"])
                ts_ms = int(data["T"])
                is_buyer_maker = bool(data.get("m", False))
                new_second = self.buffers[symbol].update_trade(price, qty, ts_ms, is_buyer_maker)
                if new_second:
                    self._new_tick.set()

        elif "@kline" in stream:
            k = data.get("k", {})
            symbol = k.get("s", "")
            candle = Candle(
                timestamp=k["t"],
                open=float(k["o"]),
                high=float(k["h"]),
                low=float(k["l"]),
                close=float(k["c"]),
                volume=float(k["v"]),
                closed=k["x"],
            )
            if symbol in self.buffers:
                self.buffers[symbol].update_candle(candle)
                if candle.closed:
                    self._new_candle.set()

        elif "@depth" in stream:
            symbol = stream.split("@")[0].upper()
            if symbol in self.buffers:
                self.buffers[symbol].update_order_book(
                    data.get("bids", []),
                    data.get("asks", []),
                )
