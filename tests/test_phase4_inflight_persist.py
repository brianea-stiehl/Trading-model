"""
PRISM Phase 4 — in-flight key persistence + MT5 reconnect recovery alert.

Locks in two behaviours:

Fix 1 — _last_signal_key round-trips through state/in_flight_keys.json
  * _should_fire() writes the key after every True return
  * _load_inflight_keys() rehydrates on startup
  * Corrupt / missing files start fresh (never raise)
  * Different key (new direction) replaces the old one and fires

Fix 2 — recovery notification after MT5 reconnect
  * pop_reconnect_event() is one-shot: True on first call after reconnect,
    False on every subsequent call until the next reconnect
  * ensure_connected() sets the flag on disconnect → connected transition
    (both the heartbeat-comes-back path and the reinit path)
  * Runner wiring: _scan_instrument calls pop_reconnect_event() after a
    successful ensure_connected() and fires notifier.send_alert() if True
  * MockMT5Bridge.pop_reconnect_event() always returns False
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers shared by both test classes
# ---------------------------------------------------------------------------

def _make_h4_df(ts: str = "2026-04-21 08:00:00+00:00") -> pd.DataFrame:
    """Minimal H4 DataFrame with a single bar at ``ts``."""
    return pd.DataFrame({
        "datetime": pd.to_datetime([ts], utc=True),
        "open": [1.0],
        "high": [1.1],
        "low": [0.9],
        "close": [1.05],
        "volume": [100],
    })


def _make_signal(direction: str = "LONG", signal_time: str = "2026-04-21T08:00:00+00:00"):
    sig = MagicMock()
    sig.direction = direction
    sig.signal_time = signal_time
    sig.confidence = 0.85
    sig.signal_id = "test-uuid-1234"
    return sig


class _FakeTerminalInfo:
    def __init__(self, connected=True):
        self.connected = connected


class _FakeAccountInfo:
    login = 123
    balance = 10_000.0
    server = "Test-MT5"


class _FakeMT5:
    """Programmable MT5 stub for reconnect tests."""
    def __init__(self, alive=True):
        self._terminal_alive = alive
        self._init_next = True
        self._account_info = _FakeAccountInfo()
        self.init_calls = 0

    def initialize(self, **kwargs):
        self.init_calls += 1
        return self._init_next

    def shutdown(self):
        pass

    def terminal_info(self):
        if self._terminal_alive is None:
            return None
        return _FakeTerminalInfo(connected=self._terminal_alive)

    def account_info(self):
        return self._account_info

    def last_error(self):
        return (1, "fake error")


def _connected_bridge(fake_mt5=None, **kwargs):
    """Return an MT5Bridge already wired to a fake MT5, in connected state."""
    from prism.execution.mt5_bridge import MT5Bridge
    fake = fake_mt5 or _FakeMT5()
    b = MT5Bridge(mode="CONFIRM", **kwargs)
    b._mt5 = fake
    b._connected = True
    b._last_connect_kwargs = {"login": 123, "password": "pw", "server": "Test-MT5"}
    return b, fake


# ===========================================================================
# Fix 1 — In-flight key persistence
# ===========================================================================

class TestInflightKeyPersistence:
    """_last_signal_key is backed by state/in_flight_keys.json."""

    def setup_method(self):
        """Isolate module-level dict between tests."""
        import prism.delivery.runner as runner
        runner._last_signal_key.clear()

    def test_should_fire_persists_key_to_disk(self, tmp_path, monkeypatch):
        """After _should_fire() returns True the key appears in in_flight_keys.json."""
        import prism.delivery.runner as runner

        monkeypatch.setattr(runner, "_state_dir", lambda: tmp_path)

        h4 = _make_h4_df("2026-04-21 08:00:00+00:00")
        sig = _make_signal("LONG")

        result = runner._should_fire("EURUSD", sig, h4)
        assert result is True

        key_file = tmp_path / "in_flight_keys.json"
        assert key_file.exists(), "in_flight_keys.json must be written after first fire"
        data = json.loads(key_file.read_text())
        assert "EURUSD" in data, "Instrument key must be present in persisted state"

    def test_should_fire_loads_from_disk_after_restart(self, tmp_path, monkeypatch):
        """Pre-populate in_flight_keys.json; _should_fire must return False for same signal."""
        import prism.delivery.runner as runner

        monkeypatch.setattr(runner, "_state_dir", lambda: tmp_path)

        # Build the exact key that _signal_key would produce
        h4 = _make_h4_df("2026-04-21 08:00:00+00:00")
        sig = _make_signal("LONG")
        expected_key = runner._signal_key("EURUSD", sig, h4)

        # Pre-populate disk as if a prior process wrote it
        key_file = tmp_path / "in_flight_keys.json"
        key_file.write_text(json.dumps({"EURUSD": list(expected_key)}))

        # Rehydrate (simulates what run() does at startup)
        loaded = runner._load_inflight_keys(tmp_path)
        runner._last_signal_key.update(loaded)

        # Same signal must be suppressed
        assert runner._should_fire("EURUSD", sig, h4) is False

    def test_corrupt_inflight_state_starts_fresh(self, tmp_path):
        """Corrupt JSON must return {} — never raise."""
        import prism.delivery.runner as runner

        key_file = tmp_path / "in_flight_keys.json"
        key_file.write_text("{this is not valid json!!!")

        result = runner._load_inflight_keys(tmp_path)
        assert result == {}, "Corrupt state must produce empty dict"

    def test_missing_inflight_state_starts_fresh(self, tmp_path):
        """Missing file must return {} — never raise."""
        import prism.delivery.runner as runner

        result = runner._load_inflight_keys(tmp_path)
        assert result == {}, "Missing file must produce empty dict"

    def test_different_key_replaces_old(self, tmp_path, monkeypatch):
        """A second signal with a different direction fires and overwrites the stored key."""
        import prism.delivery.runner as runner

        monkeypatch.setattr(runner, "_state_dir", lambda: tmp_path)

        h4 = _make_h4_df("2026-04-21 08:00:00+00:00")
        sig_long = _make_signal("LONG")
        sig_short = _make_signal("SHORT")

        assert runner._should_fire("EURUSD", sig_long, h4) is True

        # Different direction on the same bar — must fire
        result = runner._should_fire("EURUSD", sig_short, h4)
        assert result is True

        # Verify disk reflects the new key
        data = json.loads((tmp_path / "in_flight_keys.json").read_text())
        key = data["EURUSD"]
        assert "SHORT" in key, "Persisted key must reflect the new direction"

    def test_run_rehydrates_inflight_keys(self, tmp_path, monkeypatch):
        """run() rehydration: _last_signal_key is populated from in_flight_keys.json.

        We test the rehydration code path directly — _load_inflight_keys() +
        the global assignment — rather than stubbing all of run()'s I/O layer.
        This is the same code path run() executes at startup.
        """
        import prism.delivery.runner as runner

        # Pre-populate file with a prior-run key
        key_file = tmp_path / "in_flight_keys.json"
        pre_state = {"XAUUSD": ["XAUUSD", "LONG", "2026-04-21 08:00:00+00:00"]}
        key_file.write_text(json.dumps(pre_state))

        # _load_inflight_keys() is what run() calls at startup
        loaded = runner._load_inflight_keys(tmp_path)
        assert loaded, "_load_inflight_keys must return non-empty dict for pre-populated file"

        # Assign to the module global (as run() does via `global _last_signal_key`)
        runner._last_signal_key.clear()
        runner._last_signal_key.update(loaded)

        assert "XAUUSD" in runner._last_signal_key, \
            "After rehydration _last_signal_key must contain the pre-populated key"
        assert runner._last_signal_key["XAUUSD"] == ("XAUUSD", "LONG", "2026-04-21 08:00:00+00:00"), \
            "Rehydrated key must be a tuple matching _signal_key() output"
class TestReconnectAlert:
    """pop_reconnect_event() one-shot semantics + runner wiring."""

    # --- Bridge unit tests ---

    def test_pop_reconnect_event_false_initially(self):
        """Fresh bridge has no reconnect event pending."""
        from prism.execution.mt5_bridge import MT5Bridge
        b = MT5Bridge()
        assert b.pop_reconnect_event() is False

    def test_pop_reconnect_event_true_after_reconnect(self):
        """Manually set flag → pop returns True once, then False."""
        from prism.execution.mt5_bridge import MT5Bridge
        b = MT5Bridge()
        b._reconnect_just_happened = True

        assert b.pop_reconnect_event() is True, "First pop must return True"
        assert b.pop_reconnect_event() is False, "Second pop must return False (one-shot)"

    def test_ensure_connected_sets_flag_on_reconnect_heartbeat_path(self):
        """Heartbeat-comes-back path must set _reconnect_just_happened=True."""
        from datetime import timedelta

        b, fake = _connected_bridge()
        t0 = datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc)

        # Simulate outage: heartbeat fails
        fake._terminal_alive = False
        fake._init_next = False
        b.ensure_connected(t0)
        assert b._disconnect_at is not None

        # Recovery: heartbeat comes back (no reinit needed)
        fake._terminal_alive = True
        later = t0 + timedelta(seconds=15)
        result = b.ensure_connected(later)
        assert result is True
        assert b.pop_reconnect_event() is True, \
            "ensure_connected must set reconnect flag on heartbeat recovery"

    def test_ensure_connected_sets_flag_on_reinit_path(self):
        """Reinit-succeeds path must also set _reconnect_just_happened=True."""
        from datetime import timedelta

        b, fake = _connected_bridge()
        t0 = datetime(2026, 4, 21, 10, 0, tzinfo=timezone.utc)

        # Simulate outage
        fake._terminal_alive = False
        fake._init_next = False
        b.ensure_connected(t0)

        # Reinit succeeds AND terminal comes back
        def _ok_after_init(**kw):
            fake.init_calls += 1
            fake._terminal_alive = True
            return True
        fake.initialize = _ok_after_init

        later = t0 + timedelta(seconds=999)
        result = b.ensure_connected(later)
        assert result is True
        assert b.pop_reconnect_event() is True, \
            "ensure_connected must set reconnect flag on reinit recovery"

    # --- Runner wiring tests ---

    def test_runner_sends_alert_on_reconnect(self, monkeypatch):
        """_scan_instrument fires send_alert() when pop_reconnect_event() returns True."""
        import prism.delivery.runner as runner

        runner._last_signal_key.clear()

        class _RecoveringBridge:
            mode = "NOTIFY"
            _pop_count = 0

            def ensure_connected(self, now=None):
                return True

            def pop_reconnect_event(self):
                self._pop_count += 1
                return self._pop_count == 1  # True on first call only

            def supports_live_bars(self):
                return False

            def get_bars(self, *a, **kw):
                return pd.DataFrame()  # Empty → scan exits early

            def should_alert_disconnect(self, now=None):
                return False

            def disconnected_duration_sec(self):
                return None

        class _CapturingNotifier:
            def __init__(self):
                self.alerts = []
                self.channel = "#t"
                self.client = object()
                self.confirm_timeout_sec = 300

            def send_alert(self, text):
                self.alerts.append(text)
                return "ts-123"

            def send_signal(self, *a, **kw):
                return "ts-123"

            def update_signal_status(self, *a, **kw):
                pass

        class _CleanGuard:
            def refresh(self, now): pass
            @property
            def is_tripped(self): return False
            @property
            def needs_notification(self): return False
            @property
            def snapshot(self): return {"realized_pnl_usd": 0.0}

        bridge = _RecoveringBridge()
        notifier = _CapturingNotifier()
        guard = _CleanGuard()

        runner._scan_instrument(
            "EURUSD", notifier, bridge,
            datetime(2026, 4, 21, 8, 0, tzinfo=timezone.utc),
            guard=guard,
        )

        recovery_alerts = [a for a in notifier.alerts if "reconnected" in a.lower()]
        assert len(recovery_alerts) == 1, \
            f"Expected one reconnect alert, got: {notifier.alerts}"
        assert "MT5" in recovery_alerts[0]

    def test_no_alert_when_already_connected(self, monkeypatch):
        """pop_reconnect_event()=False must not trigger send_alert()."""
        import prism.delivery.runner as runner

        runner._last_signal_key.clear()

        class _StableBridge:
            mode = "NOTIFY"

            def ensure_connected(self, now=None):
                return True

            def pop_reconnect_event(self):
                return False  # already stable, no reconnect

            def supports_live_bars(self):
                return False

            def get_bars(self, *a, **kw):
                return pd.DataFrame()  # Empty → exits early

            def should_alert_disconnect(self, now=None):
                return False

            def disconnected_duration_sec(self):
                return None

        class _StrictNotifier:
            channel = "#t"
            client = object()
            confirm_timeout_sec = 300

            def send_alert(self, text):
                pytest.fail(f"send_alert must NOT fire for a stable connection: {text!r}")

            def send_signal(self, *a, **kw):
                return "ts-123"

            def update_signal_status(self, *a, **kw):
                pass

        class _CleanGuard:
            def refresh(self, now): pass
            @property
            def is_tripped(self): return False
            @property
            def needs_notification(self): return False
            @property
            def snapshot(self): return {"realized_pnl_usd": 0.0}

        runner._scan_instrument(
            "EURUSD", _StrictNotifier(), _StableBridge(),
            datetime(2026, 4, 21, 8, 0, tzinfo=timezone.utc),
            guard=_CleanGuard(),
        )
        # If we get here without send_alert firing, the test passes.

    def test_mock_bridge_pop_reconnect_event_always_false(self):
        """MockMT5Bridge.pop_reconnect_event() always returns False."""
        from prism.execution.mt5_bridge import MockMT5Bridge
        b = MockMT5Bridge()
        b.connect()
        assert b.pop_reconnect_event() is False
        assert b.pop_reconnect_event() is False  # idempotent
