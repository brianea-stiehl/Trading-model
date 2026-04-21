"""
PRISM Signal Runner -- main loop.
Runs continuously, checks session + generates signals + delivers to Slack.

Usage:
    PRISM_SLACK_TOKEN=xoxb-... TIINGO_API_KEY=... FRED_API_KEY=... \\
        python prism/delivery/runner.py

Environment variables:
    PRISM_SLACK_TOKEN     -- Brian Corp PRISM Signals bot token
    PRISM_SLACK_CHANNEL   -- default: #prism-signals
    PRISM_EXECUTION_MODE  -- CONFIRM (default) | AUTO | NOTIFY
    TIINGO_API_KEY        -- for live news + data
    FRED_API_KEY          -- for macro features
    MT5_LOGIN             -- Exness account login
    MT5_SERVER            -- e.g. Exness-MT5Real
    MT5_PASSWORD          -- Exness account password
    PRISM_INSTRUMENTS     -- comma-separated, default: XAUUSD,EURUSD,GBPUSD
    PRISM_SCAN_INTERVAL   -- seconds between scans, default: 60
"""
import json
import logging
import os
import signal as signal_module
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful shutdown flag — set by SIGTERM / SIGINT handlers
# ---------------------------------------------------------------------------
_shutdown = False


def _handle_sigterm(signum, frame):
    """Set the shutdown flag so the main loop exits cleanly."""
    global _shutdown
    logger.info("SIGTERM received — PRISM shutting down gracefully")
    _shutdown = True


# ---------------------------------------------------------------------------
# Daily brief tracking
#
# _last_brief_date is persisted under ``PRISM_STATE_DIR`` (default: ``state/``)
# so a runner restart mid-day doesn't re-fire the 22:00 UTC brief a second
# time. The persistence is best-effort — if the directory is read-only we
# fall back to in-memory state and log a warning (still correct within a
# single process lifetime, which is the 99% case).
# ---------------------------------------------------------------------------
_last_brief_date: Optional[date] = None


def _state_dir() -> Path:
    """Resolve the directory where runner state is persisted."""
    return Path(os.environ.get("PRISM_STATE_DIR", "state"))


def _brief_state_file() -> Path:
    return _state_dir() / "last_brief_date.txt"


def _load_last_brief_date() -> Optional[date]:
    """Read the persisted last-brief date, if any. Silent on error."""
    path = _brief_state_file()
    try:
        if path.exists():
            raw = path.read_text().strip()
            if raw:
                return date.fromisoformat(raw)
    except (OSError, ValueError) as exc:
        logger.warning("Could not load last_brief_date from %s: %s", path, exc)
    return None


def _save_last_brief_date(d: date) -> None:
    """Persist the last-brief date. Best-effort — logs on failure."""
    path = _brief_state_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(d.isoformat())
    except OSError as exc:
        logger.warning("Could not persist last_brief_date to %s: %s", path, exc)


def _maybe_send_daily_brief(notifier, stats_accumulator: dict, now: datetime) -> None:
    """Fire send_daily_brief once per day at 22:00 UTC, then reset stats."""
    global _last_brief_date
    if now.hour == 22 and _last_brief_date != now.date():
        notifier.send_daily_brief(stats_accumulator)
        _last_brief_date = now.date()
        _save_last_brief_date(_last_brief_date)
        stats_accumulator.clear()


# ---------------------------------------------------------------------------
# Bridge factory
# ---------------------------------------------------------------------------

def _build_bridge(execution_mode: str):
    """Instantiate the appropriate MT5 bridge based on env credentials."""
    from prism.execution.mt5_bridge import MT5Bridge, MockMT5Bridge

    mt5_login = os.environ.get("MT5_LOGIN")
    if mt5_login:
        bridge = MT5Bridge(mode=execution_mode)
        connected = bridge.connect()
        if connected:
            logger.info("MT5Bridge connected in %s mode", execution_mode)
            return bridge
        logger.error("MT5 connection failed -- falling back to MockMT5Bridge")

    logger.info("Using MockMT5Bridge (demo mode) in %s mode", execution_mode)
    bridge = MockMT5Bridge(mode=execution_mode)
    bridge.connect()
    return bridge


# ---------------------------------------------------------------------------
# Cache path resolution
# ---------------------------------------------------------------------------

def _resolve_cache_paths(instrument: str, timeframe: str) -> list:
    """
    Resolve the parquet cache paths for an instrument + timeframe.

    Cache files are written by TiingoClient.get_ohlcv as
        tiingo_{ticker}_{timeframe}_{start}_{end}.parquet
    where ticker = INSTRUMENT_MAP[symbol] (e.g. XAUUSD -> GLD). We must look up
    the mapped ticker instead of globbing on the MT5 symbol name, otherwise
    XAUUSD never matches its GLD-named cache file. We also glob case-
    insensitively so FX pairs resolve whether the ticker is stored upper- or
    lowercase (the mapping flipped case across PRs).
    """
    from pathlib import Path
    from prism.data.tiingo import INSTRUMENT_MAP

    cache_dir = Path("data/raw")
    ticker = INSTRUMENT_MAP.get(instrument, instrument)

    candidates = {ticker, ticker.lower(), ticker.upper()}
    paths: list = []
    for t in candidates:
        paths.extend(cache_dir.glob(f"tiingo_{t}_{timeframe}_*.parquet"))
    return sorted(set(paths))


# ---------------------------------------------------------------------------
# Per-instrument scan
# ---------------------------------------------------------------------------

# Shown on every signal while the runner is falling back to H4 bars aliased
# as H1/M5 — i.e. when the bridge doesn't support live bars (MockMT5Bridge
# in demo mode). Real-bar paths never set this; the banner disappears the
# moment Exness credentials are wired.
_DEMO_WARNING = (
    "H4 bars are being aliased as H1/M5. FVG break-retest is disabled. "
    "Signals are directional only — verify manually before confirming."
)


# ---------------------------------------------------------------------------
# In-flight signal guard
#
# The scanner fires every ``scan_interval`` seconds inside a kill zone. Without
# a guard, the same underlying setup (same H4 bar, same direction) would
# produce an identical signal on every scan — flooding Slack and, worse,
# accidentally auto-executing duplicates in AUTO mode before the first trade
# has even filled. We suppress repeats by keying on (instrument, direction,
# H4 bar timestamp of the current signal). At most one signal per H4 bar
# per instrument per direction.
# ---------------------------------------------------------------------------
_last_signal_key: dict = {}


def _load_inflight_keys(state_dir: Path) -> dict:
    """Rehydrate in-flight dedup keys from disk. Returns {} on miss/corrupt.

    JSON serialises tuples as lists; this function converts them back to
    tuples so equality checks against _signal_key() results work correctly.
    """
    path = state_dir / "in_flight_keys.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            # Restore tuples that JSON serialised as lists
            return {k: tuple(v) if isinstance(v, list) else v for k, v in data.items()}
    except Exception as e:
        logger.warning("Corrupt in_flight_keys state at %s: %s — starting fresh", path, e)
    return {}


def _persist_inflight_keys(state_dir: Path) -> None:
    """Write current in-flight keys to disk."""
    path = state_dir / "in_flight_keys.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_last_signal_key, indent=2))
    except Exception as e:
        logger.warning("Failed to persist in_flight_keys: %s", e)


def _signal_key(instrument: str, signal, h4_df) -> tuple:
    """Build the dedup key for an (instrument, signal, H4 bar) tuple."""
    try:
        last_h4 = str(h4_df["datetime"].iloc[-1])
    except Exception:
        last_h4 = signal.signal_time  # fallback — still stable within a bar
    return (instrument, signal.direction, last_h4)


def _should_fire(instrument: str, signal, h4_df, now: Optional[datetime] = None) -> bool:
    """Return True if this signal is not a repeat of the last one we fired.

    Per-instrument keys are keyed by (instrument, direction, H4 bar timestamp)
    so a new UTC day's first bar always fires even if the direction is the same.
    The ``now`` parameter is injected by ``_scan_instrument`` (real clock) or
    test fixtures (controlled clock).
    """
    now = now or datetime.now(timezone.utc)
    key = _signal_key(instrument, signal, h4_df)
    stored = _last_signal_key.get(instrument)
    if stored == key:
        return False
    _last_signal_key[instrument] = key
    _persist_inflight_keys(_state_dir())
    return True


def _scan_instrument(
    instrument: str,
    notifier,
    bridge,
    now: datetime,
    stats: dict = None,
    approvers: Optional[set] = None,
    guard=None,
) -> None:
    """Scan one instrument, generate a signal if conditions are met, deliver to Slack."""
    import pandas as pd

    from prism.signal.generator import SignalGenerator
    from prism.delivery.confirm_handler import PollConfirmHandler, ConfirmationResult
    from prism.data.pipeline import PRISMFeaturePipeline

    if stats is None:
        stats = {}

    # -- Drawdown guard gate --
    # Refresh first (may reset into a new UTC day, may sync MT5 deals).
    # If tripped, send a one-shot Slack alert and skip signal generation.
    # We deliberately check this BEFORE any bar fetching so a halted PRISM
    # doesn't hammer the data provider for work it will ignore.
    if guard is not None:
        guard.refresh(now)
        if guard.is_tripped:
            if guard.needs_notification:
                ts = notifier.send_alert(guard.format_alert())
                if ts is not None or getattr(notifier, "client", None) is None:
                    guard.mark_notified()
            logger.info(
                "%s: Drawdown guard tripped — skipping (realized=$%.2f)",
                instrument, guard.snapshot.get("realized_pnl_usd", 0.0),
            )
            return

    # -- MT5 reconnect gate --
    # Heartbeat + on-failure reinit (exponential backoff). A no-op on Mock.
    # If the real bridge is currently disconnected AND cooldown says don't
    # retry yet, SKIP the scan rather than fall through to the demo cache
    # path — running live-mode strategy against stale parquet bars would
    # be worse than running nothing.
    #
    # getattr() guards against legacy bridge stubs (pre-Phase-4) that
    # don't implement the reconnect contract. Real MT5Bridge /
    # MockMT5Bridge both have ensure_connected; a missing method here
    # means "always connected" which is the correct interpretation for
    # the legacy stubs.
    ensure_fn = getattr(bridge, "ensure_connected", None)
    if ensure_fn is not None and not ensure_fn(now):
        if getattr(bridge, "should_alert_disconnect", lambda _n: False)(now):
            dur = getattr(bridge, "disconnected_duration_sec", None) or 0
            notifier.send_alert(
                f":warning: *PRISM can't reach MT5* — disconnected for {int(dur)}s. "
                f"Bot is blind; signals are paused until the link comes back."
            )
            bridge.mark_disconnect_alert_sent()
        logger.warning(
            "%s: MT5 link down (%ss) — skipping scan",
            instrument,
            int(getattr(bridge, "disconnected_duration_sec", None) or 0),
        )
        return

    # Recovery notification — one-shot, fires on first scan after reconnect
    pop_fn = getattr(bridge, "pop_reconnect_event", None)
    if pop_fn is not None and pop_fn():
        notifier.send_alert(
            ":white_check_mark: *PRISM reconnected to MT5* — resuming live bar feed."
        )

    # -- Data loading --
    # Live-bar path when the bridge is connected to MT5; cache-backed alias
    # path when running under MockMT5Bridge. The banner only fires on the
    # alias path, and we only enable FVG retest on real M5 bars.
    live = bridge.supports_live_bars()
    demo_warning: Optional[str] = None

    if live:
        h4_raw = bridge.get_bars(instrument, "H4", count=500)
        h1_raw = bridge.get_bars(instrument, "H1", count=500)
        entry_raw = bridge.get_bars(instrument, "M5", count=500)
        if h4_raw is None or h4_raw.empty:
            logger.warning("%s: No H4 bars from MT5 — skipping", instrument)
            return
        # Freshness guard — bail out if the feed is stale on any layer.
        # Stale bars = old prices = signals fire against data that doesn't
        # match the current market. Better to skip the scan than silently
        # trade a ghost.
        for raw, tf in [(h4_raw, "H4"), (h1_raw, "H1"), (entry_raw, "M5")]:
            if not bridge.bars_are_fresh(raw, tf, now=now):
                logger.warning(
                    "%s: %s bars are stale (latest=%s, now=%s) — skipping scan",
                    instrument, tf,
                    raw["datetime"].iloc[-1] if not raw.empty else "∅",
                    now.isoformat(),
                )
                return
        # Feature-engineer H4 in-memory. H1 and the entry layer are raw
        # OHLCV — ICC detection and FVG retest don't need features.
        pipeline = PRISMFeaturePipeline(instrument, timeframe="H4")
        h4_df = pipeline.build_features_from_bars(h4_raw)
        h1_df = h1_raw
        entry_df = entry_raw
        persist_fvg = True
    else:
        # Demo-mode fallback: Mock bridge serves aliased H4 bars for every
        # timeframe request. Fire the DEMO banner so the notifier surfaces
        # it, and disable FVG retest because we don't actually have M5.
        h4_df = bridge.get_bars(instrument, "H4", count=500)
        if h4_df is None or h4_df.empty:
            # Legacy cache path for environments where get_bars isn't
            # implemented on an older bridge — keep the runner bootable.
            paths = _resolve_cache_paths(instrument, "4hour") or _resolve_cache_paths(instrument, "daily")
            if not paths:
                logger.warning("%s: No cached data found -- skipping", instrument)
                return
            h4_df = pd.read_parquet(paths[0])
        h1_df = h4_df
        entry_df = h4_df
        persist_fvg = False
        demo_warning = _DEMO_WARNING
        logger.warning("%s: %s", instrument, _DEMO_WARNING)

    gen = SignalGenerator(instrument, persist_fvg=persist_fvg)
    signal = gen.generate(h4_df, h1_df, entry_df)

    if signal is None:
        logger.info("%s: No signal this scan", instrument)
        return

    if not _should_fire(instrument, signal, h4_df, now=now):
        logger.info(
            "%s: Signal %s suppressed — duplicate of last signal on this H4 bar",
            instrument, signal.direction,
        )
        return

    logger.info(
        "%s: Signal generated — %s confidence=%.2f id=%s",
        instrument, signal.direction, signal.confidence,
        getattr(signal, "signal_id", "n/a"),
    )
    stats["signals_fired"] = stats.get("signals_fired", 0) + 1

    # -- Delivery --
    ts = notifier.send_signal(
        signal,
        mode=bridge.mode,
        use_buttons=False,
        demo_warning=demo_warning,
    )
    if not ts:
        logger.error("%s: Failed to send signal to Slack", instrument)
        return

    if bridge.mode == "CONFIRM":
        # Reuse the notifier's WebClient instead of spinning a fresh one per
        # scan. Same token, same rate-limit pool.
        handler = PollConfirmHandler(
            notifier.client,
            notifier.channel,
            ts,
            approvers=approvers,
        )
        # Honour PRISM_CONFIRM_TIMEOUT_SEC set on the notifier so the Slack
        # context block ("auto-expires in N min") and the actual wait stay
        # in sync. ``should_stop`` is read on every poll + every sleep-second
        # so SIGTERM during a pending confirm aborts in under a second
        # instead of blocking for the full timeout.
        result = handler.wait(
            timeout_sec=notifier.confirm_timeout_sec,
            should_stop=lambda: _shutdown,
        )

        if result == ConfirmationResult.CONFIRMED:
            notifier.update_signal_status(ts, "CONFIRMED", signal)
            stats["confirmed"] = stats.get("confirmed", 0) + 1
            exec_result = bridge.submit_order(signal)
            if exec_result.success:
                notifier.update_signal_status(ts, "EXECUTED", signal)
                stats["executed"] = stats.get("executed", 0) + 1
                logger.info("%s: Executed — ticket=%s", instrument, exec_result.ticket)
            else:
                notifier.update_signal_status(ts, "FAILED", signal)
                logger.error("%s: Execution failed — %s", instrument, exec_result.error)

        elif result == ConfirmationResult.SKIPPED:
            notifier.update_signal_status(ts, "SKIPPED", signal)
            stats["skipped"] = stats.get("skipped", 0) + 1

        elif result == ConfirmationResult.SHUTDOWN:
            # Treat like SKIPPED but annotate the Slack message so operators
            # can tell the difference between "user declined" and "PRISM was
            # stopping". No MT5 execution in either case.
            notifier.update_signal_status(ts, "SKIPPED", signal)
            stats["skipped"] = stats.get("skipped", 0) + 1
            logger.info("%s: Signal dropped — runner shutting down", instrument)

        else:  # EXPIRED
            notifier.update_signal_status(ts, "EXPIRED", signal)
            stats["expired"] = stats.get("expired", 0) + 1

    elif bridge.mode == "AUTO":
        exec_result = bridge.execute_signal(signal)
        status = "EXECUTED" if exec_result.success else "FAILED"
        notifier.update_signal_status(ts, status, signal)
        if exec_result.success:
            logger.info("%s: Auto-executed — ticket=%s", instrument, exec_result.ticket)
            stats["executed"] = stats.get("executed", 0) + 1
        else:
            logger.error("%s: Auto-execution failed — %s", instrument, exec_result.error)

    # NOTIFY mode: no execution, message already sent above


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run() -> None:
    """Main signal runner loop."""
    # Register signal handlers before any blocking I/O
    signal_module.signal(signal_module.SIGTERM, _handle_sigterm)
    signal_module.signal(signal_module.SIGINT, _handle_sigterm)

    # Read env at run() time, not import time, so test fixtures work correctly
    instruments = os.environ.get("PRISM_INSTRUMENTS", "XAUUSD,EURUSD,GBPUSD").split(",")
    scan_interval = int(os.environ.get("PRISM_SCAN_INTERVAL", "60"))
    execution_mode = os.environ.get("PRISM_EXECUTION_MODE", "CONFIRM")
    # Slack user IDs allowed to approve a signal via reaction. If unset, ANY
    # reactor in the channel can approve — safe for demo, dangerous in prod.
    approvers_raw = os.environ.get("PRISM_APPROVERS", "")
    approvers = {u.strip() for u in approvers_raw.split(",") if u.strip()} or None
    if not approvers:
        logger.warning(
            "PRISM_APPROVERS not set — ANY reactor in the channel can approve "
            "trades. Set PRISM_APPROVERS=U01...,U02... to restrict."
        )

    from prism.delivery.slack_notifier import SlackNotifier
    from prism.delivery.session_filter import is_kill_zone, session_label
    from prism.model.predict import missing_model_files

    # Refuse to start if any instrument is missing its trained models.
    # Without this, SignalGenerator raises FileNotFoundError mid-scan,
    # the exception gets swallowed by the per-instrument try/except in the
    # scan loop, and PRISM runs forever in a kill zone producing no signals.
    # NOTIFY mode is exempt — it's a dry-run with no execution path.
    if execution_mode != "NOTIFY":
        missing = missing_model_files(instruments)
        if missing:
            logger.error(
                "Refusing to start in %s mode — missing model artefacts:\n  %s\n"
                "Run `python -m prism.model.retrain --instrument <SYMBOL>` for each, "
                "or set PRISM_EXECUTION_MODE=NOTIFY to dry-run.",
                execution_mode,
                "\n  ".join(str(p) for p in missing),
            )
            raise SystemExit(2)

    notifier = SlackNotifier()
    bridge = _build_bridge(execution_mode)

    # Daily drawdown kill-switch. Halts new entries once realized loss for
    # the current UTC day exceeds PRISM_MAX_DAILY_LOSS_PCT (default 3% of
    # start-of-day balance) OR PRISM_MAX_DAILY_LOSS_USD (optional absolute
    # cap). State persists under PRISM_STATE_DIR so a restart doesn't
    # reset the counter and re-enable trading after a halt.
    from prism.delivery.drawdown_guard import build_guard_from_env
    guard = build_guard_from_env(bridge, state_dir=_state_dir())

    # Stats accumulator — cleared after each daily brief
    stats: dict = {}

    # Rehydrate the daily-brief guard from disk so a restart mid-day (crash,
    # deploy, etc.) doesn't double-send the 22:00 UTC summary.
    global _last_brief_date, _last_signal_key
    _last_brief_date = _load_last_brief_date()
    if _last_brief_date is not None:
        logger.info(
            "Loaded last_brief_date=%s from %s — brief will not re-fire today",
            _last_brief_date.isoformat(),
            _brief_state_file(),
        )
    _last_signal_key = _load_inflight_keys(_state_dir())
    if _last_signal_key:
        logger.info(
            "Rehydrated in-flight signal keys: %s", list(_last_signal_key.keys())
        )

    logger.info(
        "PRISM runner started | instruments=%s | mode=%s | scan_interval=%ds | approvers=%s",
        instruments, execution_mode, scan_interval,
        "any" if not approvers else sorted(approvers),
    )

    while not _shutdown:
        now = datetime.now(timezone.utc)

        # Fire daily brief at 22:00 UTC
        _maybe_send_daily_brief(notifier, stats, now)

        if not is_kill_zone(now):
            logger.debug("Off kill zone (%s) -- sleeping %ds", session_label(now), scan_interval)
            time.sleep(scan_interval)
            continue

        logger.info("Kill zone active: %s -- scanning %s", session_label(now), instruments)

        for instrument in instruments:
            if _shutdown:
                break
            try:
                _scan_instrument(
                    instrument, notifier, bridge, now, stats,
                    approvers=approvers,
                    guard=guard,
                )
            except Exception as exc:
                logger.error("Error scanning %s: %s", instrument, exc, exc_info=True)

        time.sleep(scan_interval)

    logger.info("PRISM runner stopped")


if __name__ == "__main__":
    run()
