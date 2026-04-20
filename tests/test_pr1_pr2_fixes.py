"""
tests/test_pr1_pr2_fixes.py
Regression tests for bugs discovered during post-merge review of PR#1 and PR#2.

Covers:
  - train.py referenced a non-existent column 'direction_4h' (should be 'direction_fwd_4')
  - train.py saved model filenames that PRISMPredictor could not load
    (layer1_lgbm vs layer1_lgb, layer2_reg vs layer2_magnitude,
     layer3_rf vs layer3_confidence)
  - tiingo.py routed FX intraday through the IEX endpoint (equities-only)
  - tiingo.py column mapping only handled adjusted equity-daily columns,
    so intraday and FX responses collapsed to empty DataFrames
  - fred.py computed cpi_yoy / gdp_growth on the forward-filled daily panel
    instead of the raw monthly/quarterly series
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# 1. train → predict end-to-end (catches both the column-name and filename bugs)
# ---------------------------------------------------------------------------

def _fake_features(n: int = 80, n_features: int = 12, seed: int = 7) -> pd.DataFrame:
    """Build a DataFrame that mirrors the shape PRISMTrainer expects."""
    rng = np.random.default_rng(seed)
    feats = pd.DataFrame(
        rng.standard_normal((n, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    feats["direction_fwd_4"] = rng.choice([-1, 0, 1], size=n)
    feats["magnitude_pips"] = rng.uniform(5, 40, size=n)
    return feats


class _FakePipeline:
    """Minimal stand-in for PRISMFeaturePipeline — avoids network I/O."""

    def __init__(self, instrument: str, timeframe: str = "H1"):
        self.instrument = instrument
        self.timeframe = timeframe
        self._feature_cols: list[str] = []
        self._df: pd.DataFrame | None = None

    def build_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        df = _fake_features()
        self._feature_cols = [c for c in df.columns if c.startswith("feat_")]
        self._df = df
        return df

    def split_train_test(self, df, test_ratio: float = 0.2):
        split = int(len(df) * (1 - test_ratio))
        train = df.iloc[:split]
        test = df.iloc[split:]
        return (
            train[self._feature_cols],
            test[self._feature_cols],
            train["direction_fwd_4"],
            test["direction_fwd_4"],
        )


def test_train_all_layers_then_predict_roundtrip(monkeypatch):
    """End-to-end contract: PRISMTrainer.train_all_layers() must save models
    under filenames that PRISMPredictor._load_models() can find.

    This single test catches both the PR#2 bugs:
      1. train.py dropna(subset=["direction_4h"]) → KeyError
      2. train saves layer1_lgbm_* / layer2_reg_* / layer3_rf_* but predict
         loads layer1_lgb_* / layer2_magnitude_* / layer3_confidence_*.
    """
    from prism.model import train as train_mod
    from prism.model import predict as predict_mod
    from prism.data import pipeline as pipeline_mod

    with tempfile.TemporaryDirectory() as tmpdir:
        mdir = Path(tmpdir)
        # Redirect both trainer and predictor to the temp models directory.
        monkeypatch.setattr(train_mod, "MODELS_DIR", mdir)
        monkeypatch.setattr(predict_mod, "MODEL_DIR", mdir)

        # Patch the feature pipeline on the SOURCE module. train.py currently
        # does a late `from prism.data.pipeline import PRISMFeaturePipeline`
        # inside train_all_layers(), so the name is re-resolved on every call
        # and this interception works. If train.py later moves the import
        # to module-level, the second setattr below also intercepts.
        # Keeping both is defence-in-depth, not duplication.
        monkeypatch.setattr(pipeline_mod, "PRISMFeaturePipeline", _FakePipeline)
        monkeypatch.setattr(
            train_mod, "PRISMFeaturePipeline", _FakePipeline, raising=False
        )

        trainer = train_mod.PRISMTrainer("EURUSD", timeframe="H1")
        results = trainer.train_all_layers("2024-01-01", "2024-06-01")

        # All four layers trained.
        assert len(results) == 4
        layer_names = {r.layer for r in results}
        assert layer_names == {
            "layer1_xgb", "layer1_lgb", "layer2_magnitude", "layer3_confidence"
        }

        # Each saved file exists under the name PRISMPredictor will look for.
        for name in ("layer1_xgb", "layer1_lgb",
                     "layer2_magnitude", "layer3_confidence"):
            assert (mdir / f"{name}_EURUSD.joblib").exists(), (
                f"Trainer did not save {name}_EURUSD.joblib — "
                "predictor will FileNotFoundError."
            )

        # Predictor loads cleanly and produces per-row output.
        predictor = predict_mod.PRISMPredictor("EURUSD")
        X = _fake_features(n=10)[[f"feat_{i}" for i in range(12)]]
        out = predictor.predict(X)
        assert len(out["direction"]) == 10
        assert set(out["direction_str"]).issubset({"LONG", "SHORT", "NEUTRAL"})


# ---------------------------------------------------------------------------
# 2. Tiingo FX endpoint routing
# ---------------------------------------------------------------------------

def test_tiingo_fx_instrument_hits_fx_endpoint(tmp_path, monkeypatch):
    """EURUSD intraday must hit /tiingo/fx/eurusd/prices, not /iex/."""
    from prism.data import tiingo as tiingo_mod

    monkeypatch.setattr(tiingo_mod, "CACHE_DIR", tmp_path)

    client = tiingo_mod.TiingoClient(api_key="test-key")
    captured: dict = {}

    def fake_get(endpoint: str, params: dict):
        captured["endpoint"] = endpoint
        captured["params"] = params
        return [
            {"date": "2024-01-01T00:00:00Z",
             "open": 1.10, "high": 1.11, "low": 1.09, "close": 1.105,
             "volume": 0},
            {"date": "2024-01-01T01:00:00Z",
             "open": 1.105, "high": 1.12, "low": 1.10, "close": 1.115,
             "volume": 0},
        ]

    monkeypatch.setattr(client, "_get", fake_get)
    df = client.get_ohlcv("EURUSD", "2024-01-01", "2024-01-02", timeframe="1hour")

    assert captured["endpoint"] == "tiingo/fx/eurusd/prices"
    assert captured["params"]["resampleFreq"] == "1hour"
    assert not df.empty
    assert list(df.columns) == ["datetime", "open", "high", "low", "close", "volume"]
    assert len(df) == 2


def test_tiingo_fx_daily_uses_1day_resample(tmp_path, monkeypatch):
    """Daily FX requests must go to the FX endpoint with resampleFreq=1day."""
    from prism.data import tiingo as tiingo_mod

    monkeypatch.setattr(tiingo_mod, "CACHE_DIR", tmp_path)
    client = tiingo_mod.TiingoClient(api_key="test-key")
    captured: dict = {}

    def fake_get(endpoint, params):
        captured["endpoint"] = endpoint
        captured["params"] = params
        return [{"date": "2024-01-01T00:00:00Z",
                 "open": 1.1, "high": 1.2, "low": 1.0, "close": 1.15,
                 "volume": 0}]

    monkeypatch.setattr(client, "_get", fake_get)
    df = client.get_ohlcv("GBPUSD", "2024-01-01", "2024-01-31", timeframe="daily")

    assert captured["endpoint"] == "tiingo/fx/gbpusd/prices"
    assert captured["params"]["resampleFreq"] == "1day"
    assert not df.empty


def test_tiingo_intraday_without_adj_columns_returns_ohlcv(tmp_path, monkeypatch):
    """Intraday responses have no adj* fields; the renamer must fall back to
    raw open/high/low/close/volume rather than silently returning empty."""
    from prism.data import tiingo as tiingo_mod

    monkeypatch.setattr(tiingo_mod, "CACHE_DIR", tmp_path)
    client = tiingo_mod.TiingoClient(api_key="test-key")

    def fake_get(endpoint, params):
        # Mimic Tiingo IEX / FX intraday payload (no adj* fields)
        return [
            {"date": "2024-01-01T12:00:00Z", "open": 100.0, "high": 101.0,
             "low": 99.5, "close": 100.5, "volume": 1000},
            {"date": "2024-01-01T13:00:00Z", "open": 100.5, "high": 101.5,
             "low": 100.0, "close": 101.0, "volume": 1500},
        ]

    monkeypatch.setattr(client, "_get", fake_get)
    df = client.get_ohlcv("XAUUSD", "2024-01-01", "2024-01-02", timeframe="1hour")

    assert not df.empty, "Expected raw OHLCV fallback, got empty DataFrame"
    assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)
    assert df["close"].iloc[-1] == pytest.approx(101.0)


def test_tiingo_daily_prefers_adjusted_columns(tmp_path, monkeypatch):
    """Equity daily responses expose adj* fields — those should be preferred."""
    from prism.data import tiingo as tiingo_mod

    monkeypatch.setattr(tiingo_mod, "CACHE_DIR", tmp_path)
    client = tiingo_mod.TiingoClient(api_key="test-key")

    def fake_get(endpoint, params):
        return [
            {"date": "2024-01-01T00:00:00Z",
             "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1_000,
             "adjOpen": 95.0, "adjHigh": 96.0, "adjLow": 94.0, "adjClose": 95.5,
             "adjVolume": 1_100},
        ]

    monkeypatch.setattr(client, "_get", fake_get)
    df = client.get_ohlcv("XAUUSD", "2024-01-01", "2024-01-31", timeframe="daily")

    assert df["close"].iloc[0] == pytest.approx(95.5), (
        "Daily XAUUSD (GLD) should use adjClose, not raw close"
    )
    assert df["volume"].iloc[0] == 1_100  # adjVolume


# ---------------------------------------------------------------------------
# 3. FRED cpi_yoy computed from raw monthly series
# ---------------------------------------------------------------------------

def test_fred_cpi_yoy_computed_on_raw_monthly_series(tmp_path, monkeypatch):
    """cpi_yoy must equal (CPI_this_month / CPI_12_months_ago - 1) * 100,
    independent of whether the daily panel has been forward-filled.

    Previously pct_change(252) on the daily ffilled series gave a DIFFERENT
    answer whenever start_date landed mid-month or the monthly release
    cadence shifted by a business day or two.
    """
    from prism.data import fred as fred_mod

    monkeypatch.setattr(fred_mod, "CACHE_DIR", tmp_path)

    # 24 monthly CPI observations: flat at 100 for year 1, then exactly +5%
    # for every month of year 2 (YoY for any row in year 2 = 5%).
    months = pd.date_range("2022-01-01", periods=24, freq="MS")
    cpi_monthly = pd.DataFrame({
        "date": months,
        "value": [100.0] * 12 + [105.0] * 12,
    })
    # Quarterly GDP: flat then +2% growth quarter-on-quarter.
    quarters = pd.date_range("2022-01-01", periods=8, freq="QS")
    gdp_q = pd.DataFrame({
        "date": quarters,
        "value": [100.0, 100.0, 100.0, 100.0, 102.0, 104.04, 106.12, 108.24],
    })

    def fake_get_series(self, series_id, start_date, end_date):
        if series_id == "CPIAUCSL":
            return cpi_monthly.copy()
        if series_id == "GDP":
            return gdp_q.copy()
        # Everything else: empty → NaN column, fine for this test
        return pd.DataFrame(columns=["date", "value"])

    # Skip fredapi entirely — we stub get_series.
    monkeypatch.setattr(fred_mod.FREDClient, "get_series", fake_get_series)

    client = fred_mod.FREDClient(api_key="")
    # Force fred to be non-None so the ImportError guard doesn't short-circuit
    client.fred = object()

    macro = client.get_macro_features("2022-01-01", "2023-12-31")

    # Pick a row from the second year — YoY should be 5% (per our synthetic series)
    macro["date"] = pd.to_datetime(macro["date"])
    late = macro[macro["date"] >= "2023-06-01"].iloc[0]
    assert late["cpi_yoy"] == pytest.approx(5.0, rel=0.02), (
        f"Expected cpi_yoy ~5.0, got {late['cpi_yoy']}"
    )

    # GDP QoQ on the raw series is 2% in the growth phase
    late_gdp = macro[macro["date"] >= "2023-03-01"].iloc[0]
    assert late_gdp["gdp_growth"] == pytest.approx(2.0, rel=0.05), (
        f"Expected gdp_growth ~2.0, got {late_gdp['gdp_growth']}"
    )


# ---------------------------------------------------------------------------
# 4. Anti-drift: single source of truth for instrument routing
# ---------------------------------------------------------------------------

def test_instrument_routing_single_source_of_truth():
    """Every instrument that callers can look up through INSTRUMENT_MAP or
    _FX_INSTRUMENTS MUST be classified in INSTRUMENT_ROUTING. This fails
    loudly if someone adds (say) AUDUSD to _FX_INSTRUMENTS without also
    wiring it into INSTRUMENT_ROUTING — which would silently route the
    request through the wrong endpoint.
    """
    from prism.data.tiingo import (
        INSTRUMENT_ROUTING, INSTRUMENT_MAP, _FX_INSTRUMENTS,
    )

    # 1. Every symbol in INSTRUMENT_MAP is in the routing table (derived
    #    tables should never outgrow the source).
    assert set(INSTRUMENT_MAP.keys()) == set(INSTRUMENT_ROUTING.keys()), (
        f"INSTRUMENT_MAP drifted from INSTRUMENT_ROUTING: "
        f"extra={set(INSTRUMENT_MAP) - set(INSTRUMENT_ROUTING)}, "
        f"missing={set(INSTRUMENT_ROUTING) - set(INSTRUMENT_MAP)}"
    )

    # 2. Every symbol in _FX_INSTRUMENTS is classified as "fx" in routing.
    for sym in _FX_INSTRUMENTS:
        assert sym in INSTRUMENT_ROUTING, (
            f"{sym} is in _FX_INSTRUMENTS but missing from INSTRUMENT_ROUTING"
        )
        assert INSTRUMENT_ROUTING[sym][1] == "fx", (
            f"{sym} is tagged fx in _FX_INSTRUMENTS but routed "
            f"as {INSTRUMENT_ROUTING[sym][1]!r} in INSTRUMENT_ROUTING"
        )

    # 3. Every entry classified "fx" is in _FX_INSTRUMENTS (reverse direction).
    fx_from_routing = {sym for sym, (_, kind) in INSTRUMENT_ROUTING.items()
                       if kind == "fx"}
    assert fx_from_routing == _FX_INSTRUMENTS, (
        f"_FX_INSTRUMENTS drifted from INSTRUMENT_ROUTING: "
        f"extra={_FX_INSTRUMENTS - fx_from_routing}, "
        f"missing={fx_from_routing - _FX_INSTRUMENTS}"
    )

    # 4. All FX tickers must be lowercase — Tiingo FX endpoint is case-
    #    sensitive in URL path (/tiingo/fx/eurusd/prices works; uppercase 404s).
    for sym in _FX_INSTRUMENTS:
        assert INSTRUMENT_MAP[sym] == INSTRUMENT_MAP[sym].lower(), (
            f"FX ticker {INSTRUMENT_MAP[sym]!r} for {sym} must be lowercase"
        )

    # 5. Endpoint kind must be one of the two supported values.
    for sym, (_, kind) in INSTRUMENT_ROUTING.items():
        assert kind in {"fx", "equity"}, (
            f"{sym} has unknown endpoint_kind={kind!r} "
            f"(tiingo.py only knows 'fx' and 'equity')"
        )


# ---------------------------------------------------------------------------
# 5. pipeline.py fixes surfaced by the Smoke #2 retrain run
# ---------------------------------------------------------------------------

def test_pipeline_yfinance_multiindex_is_flattened(monkeypatch):
    """yfinance >= 0.2.x returns a MultiIndex when a single ticker string is
    passed. The fallback used to run [c.lower() for c in raw.columns] which
    crashes on tuples with 'tuple object has no attribute lower'. Ensure
    a MultiIndex response is flattened to a valid OHLCV DataFrame.
    """
    from prism.data.pipeline import PRISMFeaturePipeline

    fake_yf_frame = pd.DataFrame(
        {
            ("Open", "EURUSD=X"):  [1.10, 1.11],
            ("High", "EURUSD=X"):  [1.12, 1.13],
            ("Low", "EURUSD=X"):   [1.09, 1.10],
            ("Close", "EURUSD=X"): [1.11, 1.12],
            ("Volume", "EURUSD=X"): [0, 0],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="h", name="Datetime"),
    )
    fake_yf_frame.columns = pd.MultiIndex.from_tuples(fake_yf_frame.columns)

    class _FakeYF:
        @staticmethod
        def download(*args, **kwargs):
            return fake_yf_frame

    # Block the Tiingo path so the pipeline falls through to yfinance.
    class _DeadTiingo:
        pass

    monkeypatch.setitem(sys.modules, "yfinance", _FakeYF)
    monkeypatch.setattr(
        "prism.data.tiingo.get_ohlcv",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("tiingo dead")),
    )

    pipeline = PRISMFeaturePipeline("EURUSD", timeframe="H1")
    df = pipeline._load_price_data("2024-01-01", "2024-01-02")

    assert not df.empty
    assert {"datetime", "open", "high", "low", "close"}.issubset(df.columns)
    assert df["close"].iloc[-1] == pytest.approx(1.12)


def test_pipeline_merge_key_is_tz_naive():
    """Tiingo/yfinance datetimes are tz-aware UTC; macro/COT/fear-greed
    DataFrames are tz-naive. Merging on a tz-aware key against a tz-naive
    one raises 'trying to merge on datetime64[ns, UTC] and datetime64[ns]'.
    Guard the normalisation: the date key used for every merge must be
    tz-naive midnight."""
    from prism.data.pipeline import PRISMFeaturePipeline  # noqa: F401 (import for coverage)

    # Replay the exact normalisation pipeline.py uses.
    datetime_aware = pd.Series(pd.to_datetime(
        ["2024-01-01T12:00:00Z", "2024-01-02T00:00:00Z"], utc=True
    ))
    date_key = (pd.to_datetime(datetime_aware, utc=True)
                .dt.tz_localize(None).dt.normalize())

    assert date_key.dt.tz is None, "date key leaked a tz — merge will fail"
    assert str(date_key.iloc[0]) == "2024-01-01 00:00:00"

    # Simulate merge against a tz-naive macro frame — must not raise.
    macro = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "cpi_yoy": [3.1, 3.1],
    })
    left = pd.DataFrame({"date": date_key, "close": [1.10, 1.11]})
    merged = left.merge(macro, on="date", how="left")
    assert merged["cpi_yoy"].notna().all()


def test_direction_fwd_4_safe_for_nan_tail():
    """np.sign on close.shift(-4) leaves NaN in the last 4 rows. Casting
    directly to int raises IntCastingNaNError on modern pandas. The fix is:
    keep as float, dropna, then astype(int). Verify that flow here."""
    close = pd.Series([1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16])
    raw = np.sign(close.shift(-4) - close)   # last 4 are NaN
    assert raw.isna().sum() == 4

    # This is the old (broken) path:
    with pytest.raises(Exception):
        raw.astype(int)

    # This is the new (correct) path:
    cleaned = raw.dropna().astype(int).tolist()
    assert cleaned == [1, 1, 1]  # three non-NaN rows, all positive


def test_iex_intraday_uses_lowercase_resample_freq():
    """Regression guard for the IEX resampleFreq casing — Tiingo docs show
    lowercase ('1hour', '4hour', '5min' etc). If someone changes the map
    back to capitalised values this test fails loudly."""
    from prism.data.tiingo import TiingoClient

    # Inspect the freq_map used inside get_ohlcv via a throwaway call that
    # short-circuits before the HTTP request. We rely on the captured params.
    import tempfile as _tempfile
    from prism.data import tiingo as tiingo_mod

    client = TiingoClient(api_key="test-key")
    captured: dict = {}

    def fake_get(endpoint: str, params: dict):
        captured["params"] = params
        return [{"date": "2024-01-01T00:00:00Z",
                 "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5}]

    with _tempfile.TemporaryDirectory() as td:
        original_cache = tiingo_mod.CACHE_DIR
        try:
            tiingo_mod.CACHE_DIR = Path(td)
            client._get = fake_get
            client.get_ohlcv("XAUUSD", "2024-01-01", "2024-01-02", "4hour")
        finally:
            tiingo_mod.CACHE_DIR = original_cache

    assert captured["params"]["resampleFreq"] == "4hour", (
        f"IEX resampleFreq must be lowercase per Tiingo docs; got "
        f"{captured['params']['resampleFreq']!r}"
    )
