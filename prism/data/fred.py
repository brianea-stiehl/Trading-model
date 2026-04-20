"""
prism/data/fred.py
FRED (Federal Reserve Economic Data) macro fetcher.
API key: env var FRED_API_KEY (free at fred.stlouisfed.org)
"""
from __future__ import annotations

import os
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)
CACHE_DIR = Path("data/raw")

SERIES = {
    "fed_funds_rate": "FEDFUNDS",
    "sofr": "SOFR",
    "cpi": "CPIAUCSL",
    "gdp": "GDP",
    "unemployment_rate": "UNRATE",
    "yield_10y": "DGS10",
    "yield_2y": "DGS2",
    "dxy": "DTWEXBGS",
    "vix": "VIXCLS",
}


class FREDClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            from fredapi import Fred
            self.fred = Fred(api_key=self.api_key) if self.api_key else None
        except ImportError:
            logger.warning("fredapi not installed. Run: pip install fredapi")
            self.fred = None

    def get_series(self, series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch a single FRED series. Returns DataFrame: date, value."""
        cache_file = CACHE_DIR / f"fred_{series_id}_{start_date}_{end_date}.parquet"
        if cache_file.exists():
            return pd.read_parquet(cache_file)

        if self.fred is None:
            logger.error("FRED client not initialized (missing key or fredapi package)")
            return pd.DataFrame(columns=["date", "value"])

        try:
            s = self.fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
            df = s.reset_index()
            df.columns = ["date", "value"]
            df["date"] = pd.to_datetime(df["date"])
            df.to_parquet(cache_file, index=False)
            return df
        except Exception as e:
            logger.error(f"FRED fetch failed for {series_id}: {e}")
            return pd.DataFrame(columns=["date", "value"])

    def get_macro_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Build combined macro feature DataFrame, daily frequency.

        Derived features (cpi_yoy, gdp_growth) are computed on the RAW native
        frequencies (CPI is monthly, GDP is quarterly) and then forward-filled
        onto the daily panel. The previous implementation computed pct_change
        on the ffilled daily series, which happened to approximate YoY only
        because the ffilled value stays constant for ~21 trading days — fragile
        and wrong if any row of the daily panel is missing.

        Columns: fed_funds_rate, fed_funds_delta_wk, sofr, cpi_yoy,
                 gdp_growth, unemployment_rate, yield_10y, yield_2y,
                 yield_spread, dxy, dxy_return_5d, vix
        """
        cache_file = CACHE_DIR / f"fred_macro_{start_date}_{end_date}.parquet"
        if cache_file.exists():
            return pd.read_parquet(cache_file)

        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        df = pd.DataFrame(index=date_range)
        df.index.name = "date"

        # Stash raw series for derived features that must be computed at native
        # frequency (e.g. YoY CPI from monthly observations).
        raw: dict = {}

        for col, series_id in SERIES.items():
            s = self.get_series(series_id, start_date, end_date)
            if s.empty:
                df[col] = float("nan")
                continue
            s_raw = s.set_index("date")["value"].sort_index()
            raw[col] = s_raw
            df[col] = s_raw.reindex(date_range).ffill().values

        # --- Derived features ---
        # Fed funds 5-day delta: daily panel is fine (rate is daily/weekly).
        df["fed_funds_delta_wk"] = df["fed_funds_rate"].diff(5)

        # CPI YoY: compute on the raw monthly series, then ffill to daily.
        if "cpi" in raw:
            cpi_yoy = raw["cpi"].pct_change(12) * 100   # 12 months = YoY
            df["cpi_yoy"] = cpi_yoy.reindex(date_range).ffill().values
        else:
            df["cpi_yoy"] = float("nan")

        # GDP growth: compute on the raw quarterly series, then ffill to daily.
        if "gdp" in raw:
            gdp_growth = raw["gdp"].pct_change(1) * 100  # 1 quarter = QoQ
            df["gdp_growth"] = gdp_growth.reindex(date_range).ffill().values
        else:
            df["gdp_growth"] = float("nan")

        df["yield_spread"] = df["yield_10y"] - df["yield_2y"]
        df["dxy_return_5d"] = df["dxy"].pct_change(5) * 100
        df = df.drop(columns=["cpi", "gdp"], errors="ignore")

        df = df.reset_index()
        df.to_parquet(cache_file, index=False)
        logger.info(f"FRED macro features built: {len(df)} rows, {len(df.columns)} columns")
        return df


def get_macro_features(start_date: str, end_date: str) -> pd.DataFrame:
    """Module-level convenience function."""
    return FREDClient().get_macro_features(start_date, end_date)
