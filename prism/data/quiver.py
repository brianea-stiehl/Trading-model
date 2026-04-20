"""
prism/data/quiver.py
Alternative data: CFTC COT reports, Fear & Greed, Quiver Quantitative.
QUIVER_API_KEY env var (free tier at quiverquant.com)
COT data is public — no key required.
"""
from __future__ import annotations

import os
import io
import logging
from pathlib import Path
import requests
import pandas as pd

logger = logging.getLogger(__name__)
CACHE_DIR = Path("data/raw")

# CFTC COT report market names
COT_MARKET_MAP = {
    "XAUUSD": "GOLD - COMMODITY EXCHANGE INC.",
    "EURUSD": "EURO FX - CHICAGO MERCANTILE EXCHANGE",
    "GBPUSD": "BRITISH POUND STERLING - CHICAGO MERCANTILE EXCHANGE",
    "USDJPY": "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE",
}

CFTC_COT_URL = "https://www.cftc.gov/dea/newcot/financial_lof.txt"


class QuiverClient:
    BASE_URL = "https://api.quiverquant.com/beta"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("QUIVER_API_KEY", "")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def get_cot_report(self, symbol: str) -> pd.DataFrame:
        """
        Fetch CFTC Commitments of Traders data.
        Falls back to public CFTC file (no API key needed).
        Returns: date, net_speculative (large spec net long), net_commercial
        """
        cache_file = CACHE_DIR / f"cot_{symbol}.parquet"
        if cache_file.exists():
            age_days = (pd.Timestamp.now() - pd.Timestamp(cache_file.stat().st_mtime, unit="s")).days
            if age_days < 7:  # COT is weekly, refresh after 7 days
                return pd.read_parquet(cache_file)

        market_name = COT_MARKET_MAP.get(symbol)
        if not market_name:
            logger.warning(f"No COT mapping for {symbol}")
            return pd.DataFrame(columns=["date", "net_speculative", "net_commercial"])

        try:
            resp = requests.get(CFTC_COT_URL, timeout=30)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text), low_memory=False)

            # Filter to target market
            mask = df["Market_and_Exchange_Names"].str.contains(market_name.split(" - ")[0], case=False, na=False)
            df = df[mask].copy()

            if df.empty:
                logger.warning(f"No COT data found for {symbol}")
                return pd.DataFrame(columns=["date", "net_speculative", "net_commercial"])

            df["date"] = pd.to_datetime(df["Report_Date_as_MM_DD_YYYY"], format="%m/%d/%Y", errors="coerce")
            df["net_speculative"] = pd.to_numeric(df.get("NonComm_Positions_Long_All", 0), errors="coerce") - \
                                    pd.to_numeric(df.get("NonComm_Positions_Short_All", 0), errors="coerce")
            df["net_commercial"] = pd.to_numeric(df.get("Comm_Positions_Long_All", 0), errors="coerce") - \
                                   pd.to_numeric(df.get("Comm_Positions_Short_All", 0), errors="coerce")

            result = df[["date", "net_speculative", "net_commercial"]].dropna().sort_values("date")
            result.to_parquet(cache_file, index=False)
            logger.info(f"COT data fetched for {symbol}: {len(result)} weeks")
            return result

        except Exception as e:
            logger.error(f"COT fetch failed for {symbol}: {e}")
            return pd.DataFrame(columns=["date", "net_speculative", "net_commercial"])

    def get_fear_greed(self) -> pd.DataFrame:
        """
        Fetch CNN Fear & Greed Index.
        Returns: date, fear_greed (0=extreme fear, 100=extreme greed)
        """
        cache_file = CACHE_DIR / "fear_greed.parquet"
        if cache_file.exists():
            age_h = (pd.Timestamp.now() - pd.Timestamp(cache_file.stat().st_mtime, unit="s")).total_seconds() / 3600
            if age_h < 24:
                return pd.read_parquet(cache_file)

        try:
            url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
            resp = requests.get(url, timeout=15,
                                headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            data = resp.json()
            scores = data.get("fear_and_greed_historical", {}).get("data", [])
            df = pd.DataFrame(scores)
            if df.empty:
                return pd.DataFrame(columns=["date", "fear_greed"])
            df["date"] = pd.to_datetime(df["x"], unit="ms").dt.date
            df["fear_greed"] = pd.to_numeric(df["y"], errors="coerce")
            result = df[["date", "fear_greed"]].dropna()
            result.to_parquet(cache_file, index=False)
            return result
        except Exception as e:
            logger.warning(f"Fear & Greed fetch failed: {e}")
            return pd.DataFrame(columns=["date", "fear_greed"])


def get_cot_report(symbol: str) -> pd.DataFrame:
    return QuiverClient().get_cot_report(symbol)

def get_fear_greed() -> pd.DataFrame:
    return QuiverClient().get_fear_greed()
