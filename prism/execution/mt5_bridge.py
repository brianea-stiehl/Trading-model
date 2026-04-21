"""
PRISM MT5 Execution Bridge — Exness integration.
Connects to MetaTrader5 terminal via Python API.
Supports CONFIRM mode (Slack approval gated elsewhere), AUTO mode, NOTIFY mode.

Requirements:
- MetaTrader5 package: pip install MetaTrader5
- MT5 terminal running on same machine (Windows or Wine on Linux)
- Exness account credentials in environment variables
"""
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, List

try:
    import pandas as pd
except Exception:  # pragma: no cover — pandas is a hard runtime dep
    pd = None  # type: ignore

logger = logging.getLogger(__name__)

# ---- Constants ----
MAGIC_NUMBER = 20260420  # PRISM identifier in MT5
RISK_PCT = float(os.environ.get("PRISM_RISK_PCT", "0.01"))  # 1% per trade
MAX_CONCURRENT = int(os.environ.get("PRISM_MAX_CONCURRENT", "3"))

# Pip size per instrument
PIP_SIZE = {
    "XAUUSD": 0.01,    # Gold: 1 pip = $0.01 per oz
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
}

# APPROXIMATE retail pip values per 1.0 standard lot. These are fallbacks when
# the MT5 symbol info isn't available (offline, mock, or misconfigured). In
# production we read symbol_info.trade_tick_value / trade_tick_size from MT5.
APPROX_PIP_VALUE_PER_LOT = {
    "XAUUSD": 1.0,   # Approx; real value depends on broker contract size (usually 100 oz)
    "USDJPY": 7.0,   # Approx; varies with USD/JPY price
    "__DEFAULT__": 10.0,  # Approx for EURUSD/GBPUSD major pairs
}

# Common Exness symbol suffixes. Resolved at connect() time against
# mt5.symbols_get() so we don't hard-code broker-specific naming.
EXNESS_SUFFIX_CANDIDATES = ("", "m", "z", ".", ".m", "-m", "_m")

# Bar period in minutes — used for the freshness guard. If the latest bar
# pulled from MT5 is more than ~1.5× its own period old, the feed is stale.
BAR_PERIOD_MINUTES = {
    "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440,
}

# MT5 timeframe constant lookup. Set lazily at connect() time because the
# MetaTrader5 module is Windows/Wine-only and we can't import it from tests.
# Keys are PRISM canonical labels; values are the string attribute names on
# the mt5 module that we resolve via getattr.
_MT5_TF_ATTR = {
    "M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15", "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4", "D1": "TIMEFRAME_D1",
}


@dataclass
class SignalPacket:
    """Complete signal ready for MT5 execution."""
    instrument: str
    direction: str        # "LONG" or "SHORT"
    entry: float
    sl: float
    tp1: float
    tp2: float
    rr_ratio: float
    confidence: float
    confidence_level: str
    magnitude_pips: float
    regime: str
    news_bias: str
    fvg_zone: Optional[dict]
    signal_time: str
    model_version: str = "prism_v2.0"
    # UUID stamped at construction time so every packet — whether built by the
    # SignalGenerator, a retry, a reconciliation replay, or a test fixture —
    # carries a stable audit ID without needing a post-hoc assignment.
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ExecutionResult:
    success: bool
    ticket: Optional[int]      # MT5 order ticket
    error: Optional[str]
    actual_entry: Optional[float]
    actual_sl: Optional[float]
    actual_tp: Optional[float]
    executed_at: Optional[str]
    status: str = "EXECUTED"   # EXECUTED | PENDING_APPROVAL | NOTIFY | REJECTED


class MT5Bridge:
    """
    MetaTrader5 execution bridge for PRISM.
    Handles connection, position sizing, order placement, and trade management.
    """

    # --- Reconnect defaults (overridable via env / constructor) -------------
    # Heartbeat cadence is set by the runner's scan_interval — the bridge
    # itself doesn't poll; ensure_connected() is called from the scan loop.
    _DEFAULT_RECONNECT_BASE_COOLDOWN_SEC = 10
    _DEFAULT_RECONNECT_MAX_COOLDOWN_SEC = 300     # cap at 5min so bad creds
                                                  # don't hammer the broker
    _DEFAULT_DISCONNECT_ALERT_THRESHOLD_SEC = 120 # post Slack alert after 2min
                                                  # of sustained disconnect

    def __init__(
        self,
        mode: str = "CONFIRM",
        reconnect_base_cooldown_sec: Optional[int] = None,
        reconnect_max_cooldown_sec: Optional[int] = None,
        disconnect_alert_threshold_sec: Optional[int] = None,
    ):
        """
        mode: "CONFIRM" — returns PENDING_APPROVAL result; an external approval
                          surface (e.g. OpenClaw → Slack) must call submit_order()
                          to actually send the trade.
              "AUTO"    — executes immediately on signal.
              "NOTIFY"  — sends alert only, no execution.
        """
        self.mode = mode
        self._mt5 = None
        self._connected = False
        # instrument -> resolved broker symbol name (may differ from instrument)
        self._symbol_cache: dict = {}

        # --- Reconnect state ------------------------------------------------
        # connect() stashes the kwargs it used so ensure_connected() can
        # replay them on reinit without re-reading env.
        self._last_connect_kwargs: dict = {}
        self._disconnect_at: Optional[datetime] = None
        self._reconnect_attempts: int = 0
        self._next_reconnect_at: Optional[datetime] = None
        self._disconnect_alert_sent: bool = False
        self._reconnect_just_happened: bool = False  # flag cleared after caller reads it
        self.reconnect_base_cooldown_sec = (
            reconnect_base_cooldown_sec
            or int(os.environ.get("PRISM_MT5_RECONNECT_BASE_SEC",
                                  self._DEFAULT_RECONNECT_BASE_COOLDOWN_SEC))
        )
        self.reconnect_max_cooldown_sec = (
            reconnect_max_cooldown_sec
            or int(os.environ.get("PRISM_MT5_RECONNECT_MAX_SEC",
                                  self._DEFAULT_RECONNECT_MAX_COOLDOWN_SEC))
        )
        self.disconnect_alert_threshold_sec = (
            disconnect_alert_threshold_sec
            or int(os.environ.get("PRISM_MT5_DISCONNECT_ALERT_SEC",
                                  self._DEFAULT_DISCONNECT_ALERT_THRESHOLD_SEC))
        )

    def connect(
        self,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        path: Optional[str] = None,
    ) -> bool:
        """Connect to MT5 terminal. Credentials from args or environment."""
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5

            login = login or int(os.environ.get("MT5_LOGIN", "0"))
            password = password or os.environ.get("MT5_PASSWORD", "")
            server = server or os.environ.get("MT5_SERVER", "")
            path = path or os.environ.get("MT5_PATH", "")

            kwargs = {"login": login, "password": password, "server": server}
            if path:
                kwargs["path"] = path

            # Stash for ensure_connected() reinit — we don't want to
            # re-read env mid-outage in case the operator rotated creds
            # after launch (they'd get picked up on the next full restart).
            self._last_connect_kwargs = dict(kwargs)

            return self._initialize_mt5(kwargs, initial=True)

        except ImportError:
            logger.error("MetaTrader5 package not installed. Run: pip install MetaTrader5")
            return False
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False

    def _initialize_mt5(self, kwargs: dict, initial: bool = False) -> bool:
        """
        Run mt5.initialize() + account_info() sanity check. Factored out so
        ensure_connected() can replay it on reconnect without duplicating
        the error handling.

        Returns True on success. On failure, logs and returns False — caller
        is responsible for backoff.
        """
        if self._mt5 is None:
            return False
        try:
            if not self._mt5.initialize(**kwargs):
                if initial:
                    logger.error("MT5 init failed: %s", self._mt5.last_error())
                return False
            info = self._mt5.account_info()
            if info is None:
                if initial:
                    logger.error("MT5 account info unavailable after connect")
                return False
            if initial:
                logger.info(
                    "MT5 connected: account=%s balance=%s server=%s",
                    info.login, info.balance, info.server,
                )
            self._connected = True
            return True
        except Exception as e:
            logger.warning("MT5 init exception: %s", e)
            return False

    def disconnect(self):
        if self._mt5 and self._connected:
            self._mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected")

    # ------------------------------------------------------------------
    # Reconnect / heartbeat (Phase 4)
    # ------------------------------------------------------------------
    # Connections drop. Broker maintenance windows, network flaps, laptop
    # sleep, VPN churn — real deployments disconnect. Without recovery
    # logic, PRISM goes silent (no errors, just no signals). Worse: if
    # get_account_balance / deals_since_utc_midnight / symbol_info start
    # returning None, risk sizing silently falls back to approximations
    # and the drawdown guard loses its canonical PnL source.
    #
    # Contract used by the runner:
    #   ensure_connected(now) -> bool
    #     True  => link is alive (either already or just reconnected)
    #     False => link is down AND cooldown says don't retry yet
    # When False, the runner SKIPS the scan rather than falling through
    # to the demo cache path — stale data + live intent = bad trades.

    def _heartbeat_ok(self) -> bool:
        """Cheapest probe: terminal_info(). Returns False on None / raise."""
        if self._mt5 is None:
            return False
        try:
            info = self._mt5.terminal_info()
            if info is None:
                return False
            # MT5 reports "connected" = terminal->broker link. A False
            # here means the terminal is running but can't reach the
            # broker — treat as disconnected from PRISM's perspective.
            return bool(getattr(info, "connected", True))
        except Exception:
            return False

    def ensure_connected(self, now: Optional[datetime] = None) -> bool:
        """
        Verify the MT5 link is alive; attempt one reinit if not. Safe to
        call every scan (the heartbeat is cheap). On sustained outage,
        flips ``_connected = False`` so ``supports_live_bars()`` starts
        reporting False — which in turn prevents the runner from falling
        through to the demo parquet cache on a live deployment.
        """
        now = now or datetime.now(timezone.utc)

        if self._heartbeat_ok():
            # Transition from disconnected -> connected: clear state +
            # log recovery time so the audit trail shows the outage.
            if self._disconnect_at is not None:
                dur = (now - self._disconnect_at).total_seconds()
                logger.info(
                    "MT5 reconnected after %.0fs (%d attempts)",
                    dur, self._reconnect_attempts,
                )
                self._disconnect_at = None
                self._reconnect_attempts = 0
                self._next_reconnect_at = None
                self._disconnect_alert_sent = False
                self._reconnect_just_happened = True
            self._connected = True
            return True

        # --- Disconnect detected ---
        if self._disconnect_at is None:
            self._disconnect_at = now
            logger.warning("MT5 heartbeat failed — entering reconnect loop")
        # Mark disconnected so downstream capability checks (supports_live_bars,
        # get_bars, deals_since_utc_midnight) report the outage correctly.
        self._connected = False

        # Cooldown gate — don't hammer the broker.
        if self._next_reconnect_at is not None and now < self._next_reconnect_at:
            return False

        self._reconnect_attempts += 1
        logger.info(
            "MT5 reconnect attempt %d (cooldown ends %s)",
            self._reconnect_attempts, self._next_reconnect_at,
        )
        # Best-effort shutdown to clear any half-open state before reinit.
        try:
            if self._mt5 is not None:
                self._mt5.shutdown()
        except Exception:
            pass

        ok = self._initialize_mt5(self._last_connect_kwargs or {}, initial=False)
        if ok and self._heartbeat_ok():
            dur = (now - self._disconnect_at).total_seconds()
            logger.info(
                "MT5 reconnected after %.0fs (%d attempts)",
                dur, self._reconnect_attempts,
            )
            self._disconnect_at = None
            self._reconnect_attempts = 0
            self._next_reconnect_at = None
            self._disconnect_alert_sent = False
            self._reconnect_just_happened = True
            return True

        # Still dead — schedule next attempt with exponential backoff,
        # capped so bad creds don't keep doubling forever.
        cooldown = min(
            self.reconnect_base_cooldown_sec * (2 ** (self._reconnect_attempts - 1)),
            self.reconnect_max_cooldown_sec,
        )
        self._next_reconnect_at = now + timedelta(seconds=cooldown)
        return False

    @property
    def disconnected_duration_sec(self) -> Optional[float]:
        """Seconds since the current outage started, or None if connected."""
        if self._disconnect_at is None:
            return None
        return (datetime.now(timezone.utc) - self._disconnect_at).total_seconds()

    def should_alert_disconnect(self, now: Optional[datetime] = None) -> bool:
        """
        True when the outage has lasted long enough to warrant a Slack
        alert AND we haven't already sent one this outage. One-shot
        semantics reset on reconnect.
        """
        if self._disconnect_alert_sent or self._disconnect_at is None:
            return False
        now = now or datetime.now(timezone.utc)
        dur = (now - self._disconnect_at).total_seconds()
        return dur >= self.disconnect_alert_threshold_sec

    def mark_disconnect_alert_sent(self) -> None:
        """Callers invoke this after posting the disconnect alert."""
        self._disconnect_alert_sent = True

    def pop_reconnect_event(self) -> bool:
        """Returns True once (and resets) when a reconnect just succeeded.

        Call this right after ensure_connected(). If True, send a Slack
        recovery alert. Design: the bridge sets this flag when
        ensure_connected() successfully reconnects (transitions from
        disconnected → connected). The runner calls pop_reconnect_event()
        and owns the Slack notification.
        """
        val = self._reconnect_just_happened
        self._reconnect_just_happened = False
        return val

    def get_account_balance(self) -> float:
        if not self._connected:
            return 0.0
        info = self._mt5.account_info()
        return info.balance if info else 0.0

    # ------------------------------------------------------------------
    # Live bar access (Phase 4) — pulls OHLCV straight from MT5 so the
    # signal stack no longer has to alias H4 parquet caches as M5.
    # ------------------------------------------------------------------

    def supports_live_bars(self) -> bool:
        """True when the bridge can serve real live bars from MT5."""
        return bool(self._connected and self._mt5 is not None)

    def _resolve_timeframe(self, timeframe: str):
        """Map PRISM canonical timeframe label → MT5 constant."""
        attr = _MT5_TF_ATTR.get(timeframe)
        if attr is None:
            raise ValueError(
                f"Unknown timeframe {timeframe!r}. "
                f"Expected one of {list(_MT5_TF_ATTR)}"
            )
        mt5_tf = getattr(self._mt5, attr, None)
        if mt5_tf is None:
            raise ValueError(
                f"MT5 module does not expose {attr}. "
                f"Is MetaTrader5 installed and initialised?"
            )
        return mt5_tf

    def get_bars(
        self,
        instrument: str,
        timeframe: str,
        count: int = 500,
    ) -> "pd.DataFrame":
        """
        Pull the last ``count`` bars for ``instrument`` at ``timeframe``.

        Returns a DataFrame with the columns PRISM's signal stack expects:
        ``datetime`` (UTC, tz-aware), ``open``, ``high``, ``low``, ``close``,
        ``volume``. Returns an empty DataFrame on any MT5 error so callers
        can treat empty-vs-stale uniformly.

        Notes:
            * ``count`` defaults to 500 so the feature pipeline has enough
              history for 200-period EMAs + ATR rolling windows.
            * Volume is ``tick_volume`` (number of price updates). Exness
              doesn't expose real volume on FX; tick volume is the standard
              proxy used in every retail MT5 indicator.
        """
        if pd is None:
            raise RuntimeError("pandas is required for get_bars")
        if not self._connected or self._mt5 is None:
            logger.warning(
                "get_bars(%s, %s) called before MT5 connection — returning empty",
                instrument, timeframe,
            )
            return pd.DataFrame()

        symbol = self.resolve_symbol(instrument)
        try:
            mt5_tf = self._resolve_timeframe(timeframe)
            rates = self._mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
        except Exception as exc:
            logger.error("get_bars failed for %s %s: %s", instrument, timeframe, exc)
            return pd.DataFrame()

        if rates is None or len(rates) == 0:
            logger.warning("No bars returned for %s %s (symbol=%s)", instrument, timeframe, symbol)
            return pd.DataFrame()

        # MT5 rates: structured array with time, open, high, low, close,
        # tick_volume, spread, real_volume. Normalize to PRISM's column names.
        df = pd.DataFrame(rates)
        df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"tick_volume": "volume"})
        cols = ["datetime", "open", "high", "low", "close", "volume"]
        return df[cols].reset_index(drop=True)

    def deals_since_utc_midnight(
        self,
        now: Optional[datetime] = None,
        magic_number: int = MAGIC_NUMBER,
    ) -> list:
        """
        Return PRISM-magic deals closed since UTC 00:00 of ``now`` (default:
        now()). Each entry is a dict with ``{"ticket", "profit", "symbol",
        "time"}``.

        Used by ``DrawdownGuard`` to compute realized daily PnL from the
        canonical source (MT5 deal history) rather than guessing from
        local state. Filters by magic number so hand-placed or non-PRISM
        trades don't count against the kill-switch.

        Returns an empty list when disconnected, when MT5 errors, or when
        no PRISM deals exist for the period.
        """
        if not self._connected or self._mt5 is None:
            return []
        now = now or datetime.now(timezone.utc)
        midnight = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        try:
            deals = self._mt5.history_deals_get(midnight, now)
        except Exception as e:
            logger.warning("history_deals_get failed: %s", e)
            return []
        if not deals:
            return []
        out: list = []
        for d in deals:
            if getattr(d, "magic", None) != magic_number:
                continue
            out.append({
                "ticket": getattr(d, "ticket", None),
                "profit": float(getattr(d, "profit", 0.0) or 0.0),
                "symbol": getattr(d, "symbol", ""),
                "time": int(getattr(d, "time", 0)),
            })
        return out

    def bars_are_fresh(
        self,
        df: "pd.DataFrame",
        timeframe: str,
        now: Optional[datetime] = None,
        max_age_factor: float = 1.5,
    ) -> bool:
        """
        Return True if the latest bar is within ``max_age_factor × bar_period``
        of ``now`` (UTC). 1.5× means one full missed bar is tolerated before
        we reject the feed as stale.

        Returns False for empty DataFrames, unknown timeframes, or bars whose
        most recent timestamp is too old. Independent of self._connected so
        the runner can also apply it to cache-backed Mock bridges.
        """
        if pd is None or df is None or df.empty or "datetime" not in df.columns:
            return False
        period_min = BAR_PERIOD_MINUTES.get(timeframe)
        if period_min is None:
            return False
        now = now or datetime.now(timezone.utc)
        last = pd.to_datetime(df["datetime"].iloc[-1])
        if last.tzinfo is None:
            last = last.tz_localize("UTC")
        age_sec = (now - last.to_pydatetime()).total_seconds()
        max_age_sec = period_min * 60 * max_age_factor
        return 0 <= age_sec <= max_age_sec

    def count_open_positions(self, instrument: Optional[str] = None) -> int:
        if not self._connected:
            return 0
        symbol = self.resolve_symbol(instrument) if instrument else None
        positions = (
            self._mt5.positions_get(symbol=symbol)
            if symbol
            else self._mt5.positions_get()
        )
        return len(positions) if positions else 0

    # ------------------------------------------------------------------
    # Symbol resolution (Exness uses suffixed names like EURUSDm)
    # ------------------------------------------------------------------

    def resolve_symbol(self, instrument: str) -> str:
        """
        Resolve a PRISM instrument ID (e.g. "EURUSD") to the broker-specific
        symbol name (e.g. "EURUSDm" for Exness retail). Falls back to the
        original instrument string when not connected or no match is found.
        """
        if instrument in self._symbol_cache:
            return self._symbol_cache[instrument]
        if not self._connected or self._mt5 is None:
            return instrument
        try:
            # Fast path: exact symbol exists
            if self._mt5.symbol_info(instrument) is not None:
                self._symbol_cache[instrument] = instrument
                return instrument
            # Try known suffix candidates
            for suffix in EXNESS_SUFFIX_CANDIDATES:
                candidate = f"{instrument}{suffix}"
                if self._mt5.symbol_info(candidate) is not None:
                    self._symbol_cache[instrument] = candidate
                    logger.info(f"Resolved {instrument} → {candidate}")
                    return candidate
        except Exception as e:
            logger.warning(f"Symbol resolution failed for {instrument}: {e}")
        # No match — cache the raw name so we only log once
        self._symbol_cache[instrument] = instrument
        return instrument

    def _pick_filling_mode(self, symbol: str):
        """
        Read symbol_info.filling_mode and return a compatible order filling flag.
        filling_mode is a bitmask: bit 0 = FOK, bit 1 = IOC, bit 2 = RETURN.
        Preference order: IOC -> FOK -> RETURN. Falls back to IOC if unknown.
        """
        mt5 = self._mt5
        if mt5 is None:
            return None
        try:
            info = mt5.symbol_info(symbol)
            flags = getattr(info, "filling_mode", 0) if info is not None else 0
            # SYMBOL_FILLING_FOK = 1, SYMBOL_FILLING_IOC = 2 per MT5 docs
            if flags & 2:
                return mt5.ORDER_FILLING_IOC
            if flags & 1:
                return mt5.ORDER_FILLING_FOK
            return mt5.ORDER_FILLING_RETURN
        except Exception:
            return mt5.ORDER_FILLING_IOC

    def _pip_value_per_lot(self, symbol: str, instrument: str) -> float:
        """
        Prefer live symbol_info.trade_tick_value / trade_tick_size when connected;
        otherwise fall back to APPROX_PIP_VALUE_PER_LOT (retail approximation).
        """
        pip = PIP_SIZE.get(instrument, 0.0001)
        if self._connected and self._mt5 is not None:
            try:
                info = self._mt5.symbol_info(symbol)
                tick_value = getattr(info, "trade_tick_value", None) if info else None
                tick_size = getattr(info, "trade_tick_size", None) if info else None
                if tick_value and tick_size and tick_size > 0:
                    return float(tick_value) * (pip / float(tick_size))
            except Exception as e:
                logger.debug(f"symbol_info lookup failed for {symbol}: {e}")
        # Fallback: retail approximation
        if instrument == "XAUUSD":
            return APPROX_PIP_VALUE_PER_LOT["XAUUSD"]
        if "JPY" in instrument:
            return APPROX_PIP_VALUE_PER_LOT["USDJPY"]
        return APPROX_PIP_VALUE_PER_LOT["__DEFAULT__"]

    def calculate_lot_size(
        self,
        instrument: str,
        sl_price: float,
        entry_price: float,
        account_balance: float,
    ) -> float:
        """
        Risk-based lot sizing: risk 1% of balance on this trade.
        lot_size = (balance * risk_pct) / (sl_pips * pip_value_per_lot)

        When connected to MT5, pip_value_per_lot is derived from
        symbol_info.trade_tick_value / trade_tick_size. Offline we fall back to
        APPROX_PIP_VALUE_PER_LOT — acceptable for v0; label trades accordingly.
        """
        pip = PIP_SIZE.get(instrument, 0.0001)
        sl_pips = abs(entry_price - sl_price) / pip
        if sl_pips < 1:
            logger.warning(f"SL too tight: {sl_pips:.1f} pips — rejecting")
            return 0.0

        risk_amount = account_balance * RISK_PCT
        symbol = self.resolve_symbol(instrument)
        pip_value_per_lot = self._pip_value_per_lot(symbol, instrument)

        lot = risk_amount / (sl_pips * pip_value_per_lot)
        lot = max(0.01, min(lot, 10.0))
        lot = round(lot, 2)
        logger.info(
            f"Lot size: {lot} (risk={risk_amount:.2f}, sl={sl_pips:.1f} pips, "
            f"pip_value={pip_value_per_lot:.4f}, symbol={symbol})"
        )
        return lot

    def execute_signal(self, signal: SignalPacket) -> ExecutionResult:
        """
        Dispatch a signal according to the configured mode.

        * NOTIFY   -> never executes; returns status=NOTIFY
        * CONFIRM  -> returns status=PENDING_APPROVAL; caller must invoke
                      submit_order(signal) after human/Slack confirmation.
        * AUTO     -> calls submit_order(signal) immediately.
        """
        if self.mode == "NOTIFY":
            logger.info(
                f"NOTIFY mode: signal logged, not executed: {signal.instrument} {signal.direction}"
            )
            return ExecutionResult(
                success=False, ticket=None, error="NOTIFY mode",
                actual_entry=None, actual_sl=None, actual_tp=None,
                executed_at=None, status="NOTIFY",
            )

        if self.mode == "CONFIRM":
            logger.info(
                f"CONFIRM mode: awaiting approval for {signal.instrument} "
                f"{signal.direction} @ {signal.entry}"
            )
            return ExecutionResult(
                success=False, ticket=None, error=None,
                actual_entry=None, actual_sl=None, actual_tp=None,
                executed_at=None, status="PENDING_APPROVAL",
            )

        # AUTO
        return self.submit_order(signal)

    def submit_order(self, signal: SignalPacket) -> ExecutionResult:
        """
        Actually send the order to MT5. This is the approval surface that the
        Slack/OpenClaw confirm handler calls after the user clicks "approve".
        """
        if not self._connected:
            return ExecutionResult(
                success=False, ticket=None, error="MT5 not connected",
                actual_entry=None, actual_sl=None, actual_tp=None,
                executed_at=None, status="REJECTED",
            )

        open_count = self.count_open_positions()
        if open_count >= MAX_CONCURRENT:
            return ExecutionResult(
                success=False, ticket=None,
                error=f"Max concurrent trades reached ({MAX_CONCURRENT})",
                actual_entry=None, actual_sl=None, actual_tp=None,
                executed_at=None, status="REJECTED",
            )

        if signal.confidence < 0.60:
            return ExecutionResult(
                success=False, ticket=None,
                error=f"Confidence too low: {signal.confidence:.2f}",
                actual_entry=None, actual_sl=None, actual_tp=None,
                executed_at=None, status="REJECTED",
            )

        mt5 = self._mt5
        balance = self.get_account_balance()
        lot = self.calculate_lot_size(signal.instrument, signal.sl, signal.entry, balance)
        if lot <= 0:
            return ExecutionResult(
                success=False, ticket=None, error="Invalid lot size",
                actual_entry=None, actual_sl=None, actual_tp=None,
                executed_at=None, status="REJECTED",
            )

        symbol = self.resolve_symbol(signal.instrument)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return ExecutionResult(
                success=False, ticket=None,
                error=f"No tick data for {symbol}",
                actual_entry=None, actual_sl=None, actual_tp=None,
                executed_at=None, status="REJECTED",
            )

        order_type = mt5.ORDER_TYPE_BUY if signal.direction == "LONG" else mt5.ORDER_TYPE_SELL
        price = tick.ask if signal.direction == "LONG" else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": signal.sl,
            "tp": signal.tp2,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            # Short signal_id prefix keeps us under MT5's 31-char comment limit
            # while still being unique enough to reconcile Slack ↔ MT5 ticket.
            "comment": (
                f"PRISM_{(signal.signal_id or '')[:8]}_"
                f"{signal.direction}_{signal.confidence:.2f}"
            ),
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._pick_filling_mode(symbol),
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            error = str(result.retcode) if result else "None"
            logger.error(f"Order failed: {error}")
            return ExecutionResult(
                success=False, ticket=None, error=error,
                actual_entry=None, actual_sl=None, actual_tp=None,
                executed_at=None, status="REJECTED",
            )

        logger.info(f"Order executed: ticket={result.order} price={result.price} lot={lot}")
        return ExecutionResult(
            success=True,
            ticket=result.order,
            error=None,
            actual_entry=result.price,
            actual_sl=signal.sl,
            actual_tp=signal.tp2,
            executed_at=datetime.now(timezone.utc).isoformat(),
            status="EXECUTED",
        )

    def close_position(self, ticket: int) -> bool:
        """Close a specific position by ticket number."""
        if not self._connected:
            return False
        mt5 = self._mt5
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False
        pos = position[0]
        tick = mt5.symbol_info_tick(pos.symbol)
        close_price = tick.bid if pos.type == 0 else tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": close_price,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": "PRISM_CLOSE",
            "type_filling": self._pick_filling_mode(pos.symbol),
        }
        result = mt5.order_send(request)
        return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE


class MockMT5Bridge(MT5Bridge):
    """
    Mock bridge for testing without a live MT5 terminal.
    Returns fake ExecutionResult — useful for backtesting signal delivery.
    """

    def connect(self, **kwargs) -> bool:
        self._connected = True
        logger.info("MockMT5Bridge: connected (no real terminal)")
        return True

    def get_account_balance(self) -> float:
        return 1000.0  # Simulated $1000 account

    def count_open_positions(self, instrument=None) -> int:
        return 0

    def resolve_symbol(self, instrument: str) -> str:
        return instrument

    def supports_live_bars(self) -> bool:
        """
        Mock bridge cannot serve real live bars. The runner reads this to
        decide whether to (a) raise a DEMO MODE banner in Slack and (b) fall
        back to the parquet cache when the caller asks for H1/M5 — which are
        not actually available in Mock and will be aliased from H4.
        """
        return False

    def deals_since_utc_midnight(self, now=None, magic_number=MAGIC_NUMBER) -> list:
        """Mock bridge has no deal history; realized PnL tracked manually in tests."""
        return []

    def ensure_connected(self, now=None) -> bool:
        """Mock is always 'connected' — reconnect semantics are a live-bridge concern."""
        return True

    def pop_reconnect_event(self) -> bool:
        """Mock bridge never has a reconnect event (it's always connected)."""
        return False

    def should_alert_disconnect(self, now=None) -> bool:
        return False

    @property
    def disconnected_duration_sec(self):
        return None

    def get_bars(self, instrument: str, timeframe: str, count: int = 500):
        """
        Mock bars come from the parquet cache on disk. The cache only has
        4hour and daily files (written by the historical downloader), so
        asking for H1/M15/M5 returns the 4hour file with a loud warning.
        This keeps demo mode runnable without MT5 but callers should check
        ``supports_live_bars()`` before trusting the result as a real feed.
        """
        if pd is None:
            return None
        # Import inside the method so tests that don't exercise the cache
        # path (e.g. pure unit tests on get_bars semantics) don't pay for
        # the pandas round-trip.
        from pathlib import Path
        from prism.data.tiingo import INSTRUMENT_MAP

        tf_map = {
            "H4": "4hour",
            "D1": "daily",
            # H1/M15/M5 all fall through to 4hour — aliased.
            "H1": "4hour",
            "M15": "4hour",
            "M5": "4hour",
        }
        cache_tf = tf_map.get(timeframe, "4hour")
        cache_dir = Path("data/raw")
        ticker = INSTRUMENT_MAP.get(instrument, instrument)
        candidates = {ticker, ticker.lower(), ticker.upper()}
        paths = []
        for t in candidates:
            paths.extend(cache_dir.glob(f"tiingo_{t}_{cache_tf}_*.parquet"))
        if not paths:
            return pd.DataFrame()
        df = pd.read_parquet(sorted(paths)[0])
        # Normalise to the live-bars schema (datetime / ohlcv). The cache
        # files already have these columns but may use 'date' instead.
        if "datetime" not in df.columns and "date" in df.columns:
            df = df.rename(columns={"date": "datetime"})
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        return df.tail(count).reset_index(drop=True)

    def _pip_value_per_lot(self, symbol: str, instrument: str) -> float:
        # Preserve the original (approximate) retail values for deterministic tests.
        if instrument == "XAUUSD":
            return APPROX_PIP_VALUE_PER_LOT["XAUUSD"]
        if "JPY" in instrument:
            return APPROX_PIP_VALUE_PER_LOT["USDJPY"]
        return APPROX_PIP_VALUE_PER_LOT["__DEFAULT__"]

    def execute_signal(self, signal: SignalPacket) -> ExecutionResult:
        # In mock we always simulate an executed trade (regardless of mode)
        # so existing tests and backtests get deterministic outputs.
        logger.info(
            f"MockMT5Bridge: simulated execute {signal.instrument} {signal.direction} @ {signal.entry}"
        )
        return ExecutionResult(
            success=True,
            ticket=99999,
            error=None,
            actual_entry=signal.entry,
            actual_sl=signal.sl,
            actual_tp=signal.tp2,
            executed_at=datetime.now(timezone.utc).isoformat(),
            status="EXECUTED",
        )

    def submit_order(self, signal: SignalPacket) -> ExecutionResult:
        return self.execute_signal(signal)
