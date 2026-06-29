"""
QSES - Data Fetcher
Downloads OHLCV data via yfinance, caches to parquet.
Resamples to requested timeframe.

Special handling:
  USOIL (CL=F): fallback to USO ETF if CL=F returns no data
  EURUSD=X:     forex volume is always 0 -- logged explicitly, not silently ignored
"""
from __future__ import annotations
import os
import hashlib
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np

from ..config.settings import MARKETS, DATA_CACHE_DIR
from ..utils.logger import get_logger

logger = get_logger("DataFetcher")

# Markets where volume=0 is expected and should be logged, not errored
ZERO_VOLUME_MARKETS = {"EURUSD"}

# Markets with fallback tickers if primary fails
TICKER_FALLBACKS = {
    "CL=F": "USO",   # WTI Crude Futures -> USO ETF
}


class DataFetcher:

    def __init__(self, cache_dir: str = DATA_CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def fetch(
        self,
        market_name: str,
        timeframe: str,
        period_years: int,
    ) -> Optional[pd.DataFrame]:
        """
        Returns OHLCV DataFrame with lowercase column names.
        Timeframe: "3h" | "4h"
        """
        ticker = MARKETS.get(market_name)
        if not ticker:
            logger.error(f"Unknown market: {market_name}")
            return None

        cache_key  = self._cache_key(market_name, timeframe, period_years)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.parquet")

        if os.path.exists(cache_path):
            logger.info(f"[CACHE HIT] {market_name} {timeframe} {period_years}Y")
            df = pd.read_parquet(cache_path)
            df = self._validate(df, market_name)
            if df is not None:
                self._log_volume_status(df, market_name)
            return df

        logger.info(f"[DOWNLOAD] {ticker} {timeframe} {period_years}Y")
        df = self._download_with_fallback(ticker, market_name, timeframe, period_years)
        if df is None or len(df) < 50:
            logger.warning(f"Insufficient data for {market_name} (got {len(df) if df is not None else 0} bars)")
            return None

        df.to_parquet(cache_path)
        logger.info(f"[CACHED] {len(df)} bars -> {cache_path}")
        self._log_volume_status(df, market_name)
        return df

    def _download_with_fallback(
        self, ticker: str, market_name: str, timeframe: str, period_years: int
    ) -> Optional[pd.DataFrame]:
        """Try primary ticker, then fallback if primary returns empty/bad data."""
        df = self._download(ticker, timeframe, period_years)

        if (df is None or len(df) < 50) and ticker in TICKER_FALLBACKS:
            fallback = TICKER_FALLBACKS[ticker]
            logger.warning(
                f"Primary ticker {ticker} failed or returned <50 bars. "
                f"Trying fallback: {fallback}"
            )
            df = self._download(fallback, timeframe, period_years)
            if df is not None and len(df) >= 50:
                logger.info(f"[FALLBACK OK] Using {fallback} for {market_name}")

        return df

    def _download(self, ticker: str, timeframe: str, period_years: int) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed: pip install yfinance")
            return None

        end   = datetime.now()
        start = end - timedelta(days=period_years * 365 + 30)
        yf_period = self._yf_period(period_years)

        try:
            # All intervals go through 1h download + resample
            raw = yf.download(ticker, period=yf_period, interval="1h",
                              auto_adjust=True, progress=False)
        except Exception as e:
            logger.error(f"Download failed for {ticker}: {e}")
            return None

        if raw is None or len(raw) == 0:
            return None

        df = self._clean(raw)

        # Resample to requested timeframe
        tf_hours = self._tf_hours(timeframe)
        if tf_hours > 1:
            df = self._resample(df, tf_hours)

        # Trim to exact period
        cutoff = datetime.now() - timedelta(days=period_years * 365)
        df = df[df.index >= pd.Timestamp(cutoff, tz="UTC")]

        return df

    @staticmethod
    def _log_volume_status(df: pd.DataFrame, market_name: str) -> None:
        """Explicitly log volume=0 for forex markets so it is never silently ignored."""
        total_vol = df["volume"].sum()
        if total_vol == 0:
            logger.info(
                f"[VOLUME=0] {market_name}: forex market -- volume is always 0 from yfinance. "
                f"All algorithms will use range-based proxy instead of volume-weighted OFI."
            )
        else:
            avg_vol = df["volume"].mean()
            logger.info(f"[VOLUME OK] {market_name}: avg volume per bar = {avg_vol:,.0f}")

    @staticmethod
    def _clean(raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        # Handle both flat and MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() if isinstance(c, str) else str(c).lower()
                          for c in df.columns]
        required = {"open", "high", "low", "close", "volume"}
        available = set(df.columns)
        if not required.issubset(available):
            # yfinance sometimes drops 'volume' for forex -- fill with zeros
            for col in required - available:
                df[col] = 0.0
        df = df[["open", "high", "low", "close", "volume"]].dropna(subset=["open","high","low","close"])
        df = df[df["close"] > 0]
        df["volume"] = df["volume"].fillna(0.0)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df

    @staticmethod
    def _resample(df: pd.DataFrame, hours: int) -> pd.DataFrame:
        rule = f"{hours}h"
        resampled = df.resample(rule, label="right").agg({
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }).dropna(subset=["open","high","low","close"])
        return resampled

    @staticmethod
    def _yf_period(years: int) -> str:
        return "2y" if years >= 2 else "1y"

    @staticmethod
    def _tf_hours(tf: str) -> int:
        return {"3h": 3, "4h": 4}.get(tf, 1)

    @staticmethod
    def _validate(df: pd.DataFrame, name: str) -> Optional[pd.DataFrame]:
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            logger.error(f"Missing columns in {name}: {required - set(df.columns)}")
            return None
        return df

    @staticmethod
    def _cache_key(market: str, tf: str, yrs: int) -> str:
        raw = f"{market}_{tf}_{yrs}y"
        return hashlib.md5(raw.encode()).hexdigest()[:12] + f"_{market}_{tf}_{yrs}y"
