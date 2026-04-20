"""
prism/data/pipeline.py
PRISM Feature Engineering Pipeline.
Builds the full feature matrix for ML training and inference.
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def _macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast = _ema(series, 12)
    slow = _ema(series, 26)
    macd = fast - slow
    signal = _ema(macd, 9)
    hist = macd - signal
    return macd, signal, hist

def _bollinger_pct(series: pd.Series, period: int = 20) -> pd.Series:
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (series - lower) / (upper - lower + 1e-10)

def _session(dt_index: pd.DatetimeIndex) -> pd.Series:
    """Encode trading session: 0=off, 1=tokyo, 2=london, 3=ny, 4=overlap"""
    hour = dt_index.hour
    session = pd.Series(0, index=dt_index)
    session[(hour >= 0) & (hour < 8)] = 1    # Tokyo
    session[(hour >= 8) & (hour < 12)] = 4   # London/Tokyo overlap
    session[(hour >= 12) & (hour < 16)] = 3  # NY
    session[(hour >= 16) & (hour < 20)] = 2  # London
    return session


class PRISMFeaturePipeline:
    """
    Builds the complete feature matrix for a given instrument.

    Usage:
        pipeline = PRISMFeaturePipeline("EURUSD", "H1")
        df = pipeline.build_features("2022-01-01", "2025-12-31")
        X_train, X_test, y_train, y_test = pipeline.train_test_split(df)
    """

    def __init__(self, instrument: str, timeframe: str = "H1"):
        self.instrument = instrument
        self.timeframe = timeframe
        self._scaler = None
        self._feature_cols: list[str] = []
        # Pip size for SL/TP calculations
        self.pip_size = 0.01 if any(x in instrument for x in ["XAU", "JPY"]) else 0.0001

    def build_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Build the full feature matrix.
        Returns DataFrame with all features + target columns.
        """
        # --- Price data ---
        # Try Tiingo first, fall back to yfinance
        df = self._load_price_data(start_date, end_date)
        if df.empty:
            raise ValueError(f"No price data for {self.instrument} ({start_date} to {end_date})")

        df = df.sort_values("datetime").reset_index(drop=True)
        close = df["close"]

        # --- Price features ---
        for n in [1, 5, 20, 50]:
            df[f"return_{n}"] = np.log(close / close.shift(n))

        df["atr_14"] = _atr(df, 14)
        df["atr_50"] = _atr(df, 50)
        df["hv_20"] = close.pct_change().rolling(20).std() * np.sqrt(252)

        for p in [9, 21, 50, 200]:
            df[f"ema_{p}"] = _ema(close, p)
        df["ema_9_slope"] = df["ema_9"].diff(3)
        df["ema_21_slope"] = df["ema_21"].diff(3)
        df["ema_trend"] = np.sign(df["ema_9"] - df["ema_21"])  # +1 / -1

        df["rsi_14"] = _rsi(close)
        df["macd"], df["macd_signal"], df["macd_hist"] = _macd(close)

        # Stochastic
        low_14 = df["low"].rolling(14).min()
        high_14 = df["high"].rolling(14).max()
        df["stoch_k"] = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        df["bb_pct"] = _bollinger_pct(close)
        df["daily_range_position"] = (close - df["low"]) / (df["high"] - df["low"] + 1e-10)

        if "volume" in df.columns:
            df["obv"] = (np.sign(close.diff()) * df["volume"]).cumsum()
            df["obv_change"] = df["obv"].pct_change(5)
        else:
            df["obv_change"] = 0.0

        # Session
        if pd.api.types.is_datetime64_any_dtype(df["datetime"]):
            df["session"] = _session(pd.DatetimeIndex(df["datetime"])).values
        df["day_of_week"] = pd.to_datetime(df["datetime"]).dt.dayofweek

        # Build a tz-naive midnight date key for merging against daily macro /
        # sentiment data. df["datetime"] is tz-aware UTC (from Tiingo FX/IEX
        # and the yfinance fallback); macro/cot/fear-greed dates are naive.
        # Merging a tz-aware key against a tz-naive one raises
        # "trying to merge on datetime64[ns, UTC] and datetime64[ns]" — so we
        # strip tz first and reuse the naive key for every merge below.
        _dt_utc = pd.to_datetime(df["datetime"], utc=True)
        _date_key = _dt_utc.dt.tz_localize(None).dt.normalize()

        # --- Macro features (FRED) ---
        try:
            from prism.data.fred import get_macro_features
            macro = get_macro_features(start_date, end_date)
            macro["date"] = pd.to_datetime(macro["date"]).dt.tz_localize(None)
            df["date"] = _date_key
            df = df.merge(macro, on="date", how="left")
            df = df.drop(columns=["date"], errors="ignore")
            for col in macro.columns:
                if col != "date" and col in df.columns:
                    df[col] = df[col].ffill()
        except Exception as e:
            logger.warning(f"FRED macro features unavailable: {e}")
            for col in ["fed_funds_rate", "fed_funds_delta_wk", "yield_spread",
                        "cpi_yoy", "dxy_return_5d", "vix"]:
                df[col] = np.nan

        # --- COT / sentiment features ---
        try:
            from prism.data.quiver import get_cot_report, get_fear_greed
            cot = get_cot_report(self.instrument)
            if not cot.empty:
                cot["date"] = pd.to_datetime(cot["date"]).dt.tz_localize(None)
                df["date"] = _date_key
                df = df.merge(cot[["date", "net_speculative"]], on="date", how="left")
                df["cot_net_speculative"] = df["net_speculative"].ffill()
                df = df.drop(columns=["date", "net_speculative"], errors="ignore")
            fg = get_fear_greed()
            if not fg.empty:
                fg["date"] = pd.to_datetime(fg["date"]).dt.tz_localize(None)
                df["date"] = _date_key
                df = df.merge(fg, on="date", how="left")
                df["fear_greed"] = df["fear_greed"].ffill()
                df = df.drop(columns=["date"], errors="ignore")
        except Exception as e:
            logger.warning(f"Alt data unavailable: {e}")
            df["cot_net_speculative"] = np.nan
            df["fear_greed"] = np.nan

        # --- Target variables ---
        # direction_fwd_4: sign of close 4 bars forward vs current close
        # (timeframe-agnostic — '4 bars ahead' on whatever pipeline timeframe
        # is configured, not necessarily 4 hours). The last 4 rows have no
        # future bar so np.sign returns NaN; keep as float here and let the
        # downstream dropna(subset=["direction_fwd_4"]) remove them. Modern
        # pandas raises IntCastingNaNError if we try to astype(int) first.
        df["direction_fwd_4"] = np.sign(close.shift(-4) - close)
        # magnitude: max favorable excursion next 20 bars (in pips)
        df["magnitude_pips"] = 0.0
        for i in range(len(df) - 20):
            future = df.iloc[i + 1:i + 21]
            if df["direction_fwd_4"].iloc[i] >= 0:
                df.at[df.index[i], "magnitude_pips"] = (future["high"].max() - close.iloc[i]) / self.pip_size
            else:
                df.at[df.index[i], "magnitude_pips"] = (close.iloc[i] - future["low"].min()) / self.pip_size

        # Drop rows with NaN targets, THEN cast to int — safe because NaN is gone.
        df = df.dropna(subset=["direction_fwd_4"]).reset_index(drop=True)
        df["direction_fwd_4"] = df["direction_fwd_4"].astype(int)

        # Store feature columns (exclude datetime, targets, raw OHLCV)
        exclude = {"datetime", "open", "high", "low", "close", "volume",
                   "direction_fwd_4", "magnitude_pips"}
        self._feature_cols = [c for c in df.columns if c not in exclude]

        logger.info(f"Feature matrix built: {len(df)} rows × {len(self._feature_cols)} features")
        return df

    def _load_price_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Try Tiingo first, fall back to yfinance."""
        tf_map = {"H4": "4hour", "H1": "1hour", "M15": "15min", "D1": "daily"}
        tiingo_tf = tf_map.get(self.timeframe, "1hour")

        try:
            from prism.data.tiingo import get_ohlcv
            df = get_ohlcv(self.instrument, start_date, end_date, tiingo_tf)
            if not df.empty:
                return df
        except Exception as e:
            logger.warning(f"Tiingo unavailable: {e}. Trying yfinance...")

        try:
            import yfinance as yf
            from prism.data.tiingo import YF_MAP
            ticker = YF_MAP.get(self.instrument, self.instrument + "=X")
            yf_tf = {"H4": "4h", "H1": "1h", "M15": "15m", "D1": "1d"}.get(self.timeframe, "1h")
            raw = yf.download(ticker, start=start_date, end=end_date, interval=yf_tf, progress=False)
            if raw.empty:
                return pd.DataFrame()
            raw = raw.reset_index()

            # yfinance >= 0.2.x returns a MultiIndex on columns when a single
            # ticker is passed as a string (e.g. [("Close", "EURUSD=X"), ...]).
            # Flatten to the field name only before lowercasing.
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0] if isinstance(c, tuple) else c
                               for c in raw.columns]
            raw.columns = [str(c).lower() for c in raw.columns]

            raw = raw.rename(columns={"index": "datetime", "date": "datetime",
                                       "datetime": "datetime"})
            raw["datetime"] = pd.to_datetime(raw["datetime"])
            cols = ["datetime", "open", "high", "low", "close"]
            if "volume" in raw.columns:
                cols.append("volume")
            return raw[cols].copy()
        except Exception as e:
            logger.error(f"yfinance also failed: {e}")
            return pd.DataFrame()

    def normalize(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """StandardScaler normalization per feature. Chronological fit only."""
        from sklearn.preprocessing import StandardScaler
        df = df.copy()
        if fit:
            self._scaler = StandardScaler()
            df[self._feature_cols] = self._scaler.fit_transform(
                df[self._feature_cols].fillna(0)
            )
        else:
            if self._scaler is None:
                raise RuntimeError("Call normalize(fit=True) first")
            df[self._feature_cols] = self._scaler.transform(
                df[self._feature_cols].fillna(0)
            )
        return df

    def split_train_test(
        self, df: pd.DataFrame, test_ratio: float = 0.2
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Chronological train/test split. NEVER shuffles.
        Returns: X_train, X_test, y_train, y_test
        """
        split_idx = int(len(df) * (1 - test_ratio))
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]
        X_train = train[self._feature_cols].fillna(0)
        X_test = test[self._feature_cols].fillna(0)
        y_train = train["direction_fwd_4"]
        y_test = test["direction_fwd_4"]
        logger.info(f"Train: {len(train)} rows | Test: {len(test)} rows | "
                    f"Features: {len(self._feature_cols)}")
        return X_train, X_test, y_train, y_test
