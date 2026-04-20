"""
prism/data/tiingo.py
Tiingo API price data fetcher.
API key: env var TIINGO_API_KEY
"""
from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from datetime import datetime
import requests
import pandas as pd

logger = logging.getLogger(__name__)

# Single source of truth for instrument routing. Each entry is
#     SYMBOL -> (tiingo_ticker, endpoint_kind)
# where `endpoint_kind` is either:
#     "fx"     -> /tiingo/fx/{ticker}/prices  (daily & intraday — resampleFreq=1day | 1hour | ...)
#     "equity" -> /tiingo/daily/{ticker}/prices  (daily)
#                 /iex/{ticker}/prices          (intraday — IEX, equities-only)
#
# Never maintain _FX_INSTRUMENTS / INSTRUMENT_MAP / YF_MAP as independent
# lookup tables — they drift (e.g. adding AUDUSD to one but not the other
# silently routes requests to the wrong endpoint). Derive everything below
# from this single map.
#
# XAUUSD is proxied by GLD (SPDR Gold ETF) because Tiingo doesn't expose spot
# gold directly. For spot-accurate gold bars, use the yfinance fallback
# ("GC=F" gold futures) — see YF_MAP.
INSTRUMENT_ROUTING: dict = {
    # symbol -> (tiingo_ticker, endpoint_kind)
    "XAUUSD": ("GLD",    "equity"),
    "EURUSD": ("eurusd", "fx"),   # Tiingo FX tickers are lowercase per docs
    "GBPUSD": ("gbpusd", "fx"),
    "USDJPY": ("usdjpy", "fx"),
}

# Derived lookup tables — DO NOT edit directly, edit INSTRUMENT_ROUTING.
INSTRUMENT_MAP: dict = {sym: ticker for sym, (ticker, _) in INSTRUMENT_ROUTING.items()}
_FX_INSTRUMENTS: set = {sym for sym, (_, kind) in INSTRUMENT_ROUTING.items() if kind == "fx"}


# yfinance ticker overrides — used when Tiingo is unavailable.
# These differ from INSTRUMENT_MAP because yfinance uses its own symbol conventions.
YF_MAP = {
    "XAUUSD": "GC=F",    # Gold futures — spot-accurate proxy on yfinance (GLD=X does not exist)
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
}
CACHE_DIR = Path("data/raw")

class TiingoClient:
    BASE_URL = "https://api.tiingo.com"
    RATE_LIMIT_DELAY = 1.2  # seconds between requests (50/min max)

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("TIINGO_API_KEY", "")
        if not self.api_key:
            logger.warning("TIINGO_API_KEY not set. Data fetches will fail.")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        })
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _get(self, endpoint: str, params: dict) -> list | dict:
        url = f"{self.BASE_URL}/{endpoint}"
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        time.sleep(self.RATE_LIMIT_DELAY)
        return resp.json()

    def get_ohlcv(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "daily",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars from Tiingo.
        symbol: MT5 symbol (e.g. 'XAUUSD') or Tiingo ticker
        timeframe: 'daily', '1hour', '15min', '5min', '1min'
        Returns DataFrame with columns: datetime, open, high, low, close, volume
        """
        ticker = INSTRUMENT_MAP.get(symbol, symbol.lower())
        is_fx = symbol in _FX_INSTRUMENTS
        cache_file = CACHE_DIR / f"tiingo_{ticker}_{timeframe}_{start_date}_{end_date}.parquet"

        if cache_file.exists():
            logger.info(f"Loading cached data: {cache_file}")
            return pd.read_parquet(cache_file)

        try:
            freq_map = {
                "1hour": "1hour", "4hour": "4hour",
                "15min": "15min", "5min": "5min", "1min": "1min",
            }
            resample = freq_map.get(timeframe, "1hour")

            if is_fx:
                # Tiingo FX endpoint covers both daily and intraday (resampleFreq=1day
                # for daily bars, or {1min,5min,15min,1hour,4hour} for intraday).
                params = {
                    "startDate": start_date,
                    "endDate": end_date,
                    "resampleFreq": "1day" if timeframe == "daily" else resample,
                }
                data = self._get(f"tiingo/fx/{ticker}/prices", params)
            elif timeframe == "daily":
                data = self._get(
                    f"tiingo/daily/{ticker}/prices",
                    {"startDate": start_date, "endDate": end_date, "format": "json"},
                )
            else:
                # Equity intraday via IEX — only valid for equity tickers (e.g. GLD).
                data = self._get(
                    f"iex/{ticker}/prices",
                    {
                        "startDate": start_date,
                        "endDate": end_date,
                        "resampleFreq": resample,
                        "columns": "open,high,low,close,volume",
                    },
                )

            df = pd.DataFrame(data)
            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return df

            # Prefer adjusted columns (equity daily), fall back to raw OHLCV
            # (FX and intraday responses have no adj* fields).
            adj_map = {"date": "datetime", "adjOpen": "open",
                       "adjHigh": "high", "adjLow": "low",
                       "adjClose": "close", "adjVolume": "volume"}
            raw_map = {"date": "datetime", "open": "open", "high": "high",
                       "low": "low", "close": "close", "volume": "volume"}
            if "adjClose" in df.columns:
                keep = [c for c in adj_map if c in df.columns]
                df = df[keep].rename(columns=adj_map)
            else:
                keep = [c for c in raw_map if c in df.columns]
                df = df[keep].rename(columns=raw_map)

            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            df = df.sort_values("datetime").reset_index(drop=True)
            df.to_parquet(cache_file, index=False)
            logger.info(f"Fetched {len(df)} bars for {ticker} ({timeframe})")
            return df

        except Exception as e:
            logger.error(f"Tiingo fetch failed for {ticker}: {e}")
            raise

    def get_news_sentiment(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch news sentiment scores from Tiingo.
        Returns DataFrame: date, sentiment_score (rolling 24h average)
        """
        ticker = INSTRUMENT_MAP.get(symbol, symbol.lower())
        cache_file = CACHE_DIR / f"tiingo_sentiment_{ticker}_{start_date}_{end_date}.parquet"

        if cache_file.exists():
            return pd.read_parquet(cache_file)

        try:
            data = self._get(
                "tiingo/news",
                {"tickers": ticker, "startDate": start_date, "endDate": end_date, "limit": 1000},
            )
            if not data:
                return pd.DataFrame(columns=["date", "sentiment_score"])

            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["publishedDate"]).dt.date
            # Average sentiment per day (Tiingo returns positive/negative scores)
            if "sentiment" in df.columns:
                df["sentiment_score"] = df["sentiment"].apply(
                    lambda x: x.get("compound", 0) if isinstance(x, dict) else 0
                )
            else:
                df["sentiment_score"] = 0.0

            daily = df.groupby("date")["sentiment_score"].mean().reset_index()
            daily.to_parquet(cache_file, index=False)
            return daily

        except Exception as e:
            logger.warning(f"Sentiment fetch failed for {ticker}: {e}. Returning empty.")
            return pd.DataFrame(columns=["date", "sentiment_score"])


def get_ohlcv(symbol: str, start_date: str, end_date: str, timeframe: str = "daily") -> pd.DataFrame:
    """Module-level convenience function."""
    return TiingoClient().get_ohlcv(symbol, start_date, end_date, timeframe)


def get_news_sentiment(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Module-level convenience function."""
    return TiingoClient().get_news_sentiment(symbol, start_date, end_date)
