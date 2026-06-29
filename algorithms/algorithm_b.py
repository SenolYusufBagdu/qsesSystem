"""
QSES — Algorithm B: Adaptive Trend + Volatility Breakout

Approach: Donchian/Keltner hybrid breakout filtered by:
  - ADX trend strength
  - Volume surge confirmation
  - Hurst exponent (trending vs mean-reverting market filter)
  - Regime-adaptive position management

Rationale: Algorithm A is statistics-heavy with OU/OFI.
Algorithm B provides a classic trend-following complement —
different alpha source, low correlation with A, robust across
trending market conditions.
"""
from __future__ import annotations
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from .base import BaseAlgorithm


class AlgorithmB(BaseAlgorithm):
    """Adaptive Trend-Following with Breakout + ADX filter."""

    name = "AlgorithmB"

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        close  = df["close"].values
        high   = df["high"].values
        low    = df["low"].values
        volume = df["volume"].values if "volume" in df.columns else np.ones(len(close))
        n      = len(close)

        lookback     = params.get("lookback", 100)
        dc_len       = params.get("donchian_len", 20)
        adx_len      = params.get("adx_len", 14)
        adx_thresh   = params.get("adx_thresh", 25.0)
        vol_mult     = params.get("vol_surge_mult", 1.5)
        hurst_window = params.get("hurst_window", 100)
        hurst_thresh = params.get("hurst_thresh", 0.55)   # > 0.5 = trending
        wf_warm      = params.get("wf_window", 50)
        atr_len      = params.get("atr_len", 14)

        atr_arr = self._atr(df, atr_len)

        # ── Donchian Channel ─────────────────────────────────────────────────
        dc_high = self._rolling_max(high, dc_len)
        dc_low  = self._rolling_min(low,  dc_len)
        dc_mid  = (dc_high + dc_low) / 2

        # ── ADX ──────────────────────────────────────────────────────────────
        adx, di_plus, di_minus = self._adx(high, low, close, adx_len)

        # ── Volume Surge ─────────────────────────────────────────────────────
        # For zero-volume markets (forex), volume surge filter is bypassed.
        # Range expansion (ATR vs recent ATR) acts as the activity proxy instead.
        has_volume = volume.sum() > 0
        if has_volume:
            vol_ma    = self._rolling_mean(volume, 20)
            vol_surge = volume > vol_ma * vol_mult
        else:
            self.logger.info(
                "volume=0 detected in AlgorithmB: replacing vol_surge "
                "with ATR expansion proxy (bar range > 1.2x ATR MA)."
            )
            atr_b     = self._atr(df, params.get("atr_len", 14))
            atr_ma    = self._rolling_mean(atr_b, 20)
            crange_b  = high - low
            vol_surge = crange_b > atr_ma * vol_mult

        # ── Hurst Exponent (simplified R/S) ──────────────────────────────────
        hurst = self._hurst_rs(np.log(np.maximum(close, 1e-9)), hurst_window)

        # ── Keltner Midline trend filter ─────────────────────────────────────
        ema50 = self._ema_arr(close, 50)
        trend_up   = close > ema50
        trend_down = close < ema50

        # ── Signal Logic ─────────────────────────────────────────────────────
        warmed = np.arange(n) > (wf_warm + lookback)

        # Breakout: close exceeds prior Donchian high/low with confirmation
        long_break  = close > np.roll(dc_high, 1)
        short_break = close < np.roll(dc_low,  1)
        long_break[0] = short_break[0] = False

        long_cond = (warmed
                     & long_break
                     & (adx > adx_thresh)
                     & (di_plus > di_minus)
                     & vol_surge
                     & (hurst > hurst_thresh)
                     & trend_up)

        short_cond = (warmed
                      & short_break
                      & (adx > adx_thresh)
                      & (di_minus > di_plus)
                      & vol_surge
                      & (hurst > hurst_thresh)
                      & trend_down)

        sig = np.zeros(n, int)
        sig[long_cond]  =  1
        sig[short_cond] = -1

        return pd.Series(sig, index=df.index)

    def default_param_space(self) -> Dict[str, Any]:
        return {
            "lookback":        ("int",   50,  200),
            "donchian_len":    ("int",   10,  40),
            "adx_len":         ("int",   7,   21),
            "adx_thresh":      ("float", 18.0, 35.0, 1.0),
            "vol_surge_mult":  ("float", 1.0,  3.0,  0.1),
            "hurst_window":    ("int",   60,  150),
            "hurst_thresh":    ("float", 0.48, 0.70, 0.01),
            "exit_thresh":     ("float", -1.5, 0.0,  0.1),
            "atr_stop":        ("float", 1.0,  4.0,  0.25),
            "atr_tp":          ("float", 1.5,  6.0,  0.25),
            "kelly_frac":      ("float", 0.1,  0.5,  0.05),
        }

    def model_configs(self) -> List[Dict[str, Any]]:
        base = dict(lookback=100, wf_window=50, atr_len=14, kelly_cap=0.25)
        return [
            # M0 — Slow, selective, wide TP
            {**base, "donchian_len":30, "adx_len":21, "adx_thresh":30.0,
             "vol_surge_mult":2.0, "hurst_window":120, "hurst_thresh":0.60,
             "exit_thresh":-1.0, "atr_stop":2.5, "atr_tp":5.0, "kelly_frac":0.15},
            # M1 — Balanced
            {**base, "donchian_len":20, "adx_len":14, "adx_thresh":25.0,
             "vol_surge_mult":1.5, "hurst_window":100, "hurst_thresh":0.55,
             "exit_thresh":-0.5, "atr_stop":2.0, "atr_tp":3.5, "kelly_frac":0.25},
            # M2 — Fast, more signals
            {**base, "donchian_len":12, "adx_len":10, "adx_thresh":20.0,
             "vol_surge_mult":1.2, "hurst_window":60,  "hurst_thresh":0.52,
             "exit_thresh":-0.3, "atr_stop":1.5, "atr_tp":2.5, "kelly_frac":0.30},
            # M3 — Trend-only, tight Hurst gate
            {**base, "donchian_len":25, "adx_len":14, "adx_thresh":28.0,
             "vol_surge_mult":1.8, "hurst_window":140, "hurst_thresh":0.62,
             "exit_thresh":-0.8, "atr_stop":3.0, "atr_tp":4.5, "kelly_frac":0.20},
        ]

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _rolling_max(arr, w):
        out = np.full(len(arr), np.nan)
        for i in range(w-1, len(arr)):
            out[i] = np.max(arr[i-w+1:i+1])
        return out

    @staticmethod
    def _rolling_min(arr, w):
        out = np.full(len(arr), np.nan)
        for i in range(w-1, len(arr)):
            out[i] = np.min(arr[i-w+1:i+1])
        return out

    @staticmethod
    def _rolling_mean(arr, w):
        out = np.full(len(arr), np.nan)
        cs = np.nancumsum(arr)
        for i in range(w-1, len(arr)):
            out[i] = (cs[i] - (cs[i-w] if i >= w else 0)) / w
        return out

    @staticmethod
    def _ema_arr(arr, span):
        alpha = 2 / (span + 1)
        out = np.zeros(len(arr))
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
        return out

    def _adx(self, high, low, close, length):
        n = len(close)
        dm_plus  = np.zeros(n)
        dm_minus = np.zeros(n)
        tr_arr   = np.zeros(n)

        for i in range(1, n):
            h_diff = high[i]  - high[i-1]
            l_diff = low[i-1] - low[i]
            dm_plus[i]  = max(h_diff, 0) if h_diff > l_diff else 0
            dm_minus[i] = max(l_diff, 0) if l_diff > h_diff else 0
            tr_arr[i]   = max(high[i]-low[i],
                              abs(high[i]-close[i-1]),
                              abs(low[i]-close[i-1]))

        # Smoothed (Wilder)
        atr_s   = self._wilder(tr_arr, length)
        dmp_s   = self._wilder(dm_plus, length)
        dmm_s   = self._wilder(dm_minus, length)

        di_plus  = 100 * dmp_s / np.maximum(atr_s, 1e-9)
        di_minus = 100 * dmm_s / np.maximum(atr_s, 1e-9)
        dx       = 100 * np.abs(di_plus - di_minus) / np.maximum(di_plus + di_minus, 1e-9)
        adx      = self._wilder(dx, length)
        return adx, di_plus, di_minus

    @staticmethod
    def _wilder(arr, length):
        out = np.zeros(len(arr))
        out[length-1] = arr[:length].mean()
        for i in range(length, len(arr)):
            out[i] = (out[i-1] * (length-1) + arr[i]) / length
        return out

    @staticmethod
    def _hurst_rs(log_prices, window):
        """Simplified R/S Hurst exponent using rolling windows."""
        n = len(log_prices)
        hurst = np.full(n, 0.5)
        for i in range(window, n):
            chunk = log_prices[i-window:i]
            if np.std(chunk) < 1e-9:
                continue
            # Use 4 sub-period R/S estimates
            rs_vals = []
            for sub in [window//4, window//2, window*3//4, window]:
                sub = max(sub, 4)
                sub_chunk = chunk[:sub]
                mean = sub_chunk.mean()
                devs = np.cumsum(sub_chunk - mean)
                R = devs.max() - devs.min()
                S = sub_chunk.std(ddof=1) + 1e-9
                rs_vals.append((np.log(sub), np.log(R/S)))
            if len(rs_vals) >= 2:
                xs = np.array([v[0] for v in rs_vals])
                ys = np.array([v[1] for v in rs_vals])
                hurst[i] = np.polyfit(xs, ys, 1)[0]
        return hurst
