"""
QSES — Algorithm C: Regime-Gated Stat-Arb / Mean Reversion

Approach: Pure mean-reversion (Bollinger Band + RSI overbought/oversold)
  with strict regime gate — only trades in Low-Vol / ranging markets.
  Uses a Kalman-style spread filter and volume-profile support/resistance.

Rationale:
  - Algorithm A = composite quant (momentum + MR + OFI, all regimes)
  - Algorithm B = trend breakout (trending markets only)
  - Algorithm C = mean reversion (ranging/low-vol markets only)
  Together the three have low cross-correlation and provide genuine
  diversification across different market conditions.
"""
from __future__ import annotations
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from .base import BaseAlgorithm


class AlgorithmC(BaseAlgorithm):
    """Regime-Gated Mean Reversion — Bollinger + RSI + Kalman smoothing."""

    name = "AlgorithmC"

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        close  = df["close"].values
        high   = df["high"].values
        low    = df["low"].values
        volume = df["volume"].values if "volume" in df.columns else np.ones(len(close))
        n      = len(close)

        bb_len       = params.get("bb_len", 20)
        bb_std       = params.get("bb_std", 2.0)
        rsi_len      = params.get("rsi_len", 14)
        rsi_ob       = params.get("rsi_ob", 65.0)     # overbought
        rsi_os       = params.get("rsi_os", 35.0)     # oversold
        rv_len       = params.get("rv_len", 30)
        rv_ratio_lo  = params.get("rv_ratio_lo", 0.9)  # max RV ratio for MR regime
        kalman_q     = params.get("kalman_q", 0.01)    # process noise
        kalman_r     = params.get("kalman_r", 0.1)     # observation noise
        vol_len      = params.get("vol_len", 20)
        vol_profile  = params.get("vol_profile_len", 50)
        wf_warm      = params.get("wf_window", 50)
        lookback     = params.get("lookback", 100)
        atr_len      = params.get("atr_len", 14)
        min_zscore   = params.get("min_zscore", 1.5)   # entry threshold

        atr_arr = self._atr(df, atr_len)

        # ── Kalman-filtered price (process-noise adaptive smoothing) ──────────
        kf_price = self._kalman_filter(close, kalman_q, kalman_r)

        # ── Bollinger Bands on Kalman price ──────────────────────────────────
        bb_mid = self._rolling_mean_arr(kf_price, bb_len)
        bb_s   = self._rolling_std_arr(kf_price, bb_len)
        bb_up  = bb_mid + bb_std * bb_s
        bb_lo  = bb_mid - bb_std * bb_s

        # z-score of price relative to BB
        bb_zscore = (close - bb_mid) / np.maximum(bb_s, 1e-9)

        # ── RSI ───────────────────────────────────────────────────────────────
        rsi = self._rsi(close, rsi_len)

        # ── Regime Gate: only enter when market is ranging (low RV) ──────────
        log_ret   = np.diff(np.log(np.maximum(close, 1e-9)), prepend=np.nan)
        rv_curr   = self._rolling_std_arr(log_ret, rv_len)
        rv_long   = self._rolling_mean_arr(rv_curr, lookback)
        rv_ratio  = rv_curr / np.maximum(rv_long, 1e-9)
        mr_regime = rv_ratio < rv_ratio_lo  # True = low vol = MR favourable

        # ── Volume Profile: avoid entries near volume nodes (strong S/R) ─────
        # For zero-volume markets (forex), the volume filter is bypassed entirely.
        # The BB z-score and RSI thresholds are sufficient entry gates.
        has_volume = volume.sum() > 0
        if has_volume:
            vol_ma        = self._rolling_mean_arr(volume, vol_profile)
            vol_filter    = volume > vol_ma * 0.8
        else:
            self.logger.info(
                "volume=0 detected in AlgorithmC: volume profile filter bypassed."
            )
            vol_filter    = np.ones(n, dtype=bool)   # always True when no volume

        # ── Mean reversion signal ─────────────────────────────────────────────
        warmed = np.arange(n) > (wf_warm + lookback)

        # Long: price below lower BB, RSI oversold, low-vol regime
        long_cond = (warmed
                     & mr_regime
                     & (bb_zscore < -min_zscore)
                     & (rsi < rsi_os)
                     & vol_filter)

        # Short: price above upper BB, RSI overbought, low-vol regime
        short_cond = (warmed
                      & mr_regime
                      & (bb_zscore > min_zscore)
                      & (rsi > rsi_ob)
                      & vol_filter)

        sig = np.zeros(n, int)
        sig[long_cond]  =  1
        sig[short_cond] = -1

        return pd.Series(sig, index=df.index)

    def default_param_space(self) -> Dict[str, Any]:
        return {
            "lookback":      ("int",   60,  200),
            "bb_len":        ("int",   10,  40),
            "bb_std":        ("float", 1.5, 3.0,  0.1),
            "rsi_len":       ("int",   7,   21),
            "rsi_ob":        ("float", 60.0, 75.0, 1.0),
            "rsi_os":        ("float", 25.0, 40.0, 1.0),
            "rv_len":        ("int",   15,  60),
            "rv_ratio_lo":   ("float", 0.70, 1.05, 0.05),
            "kalman_q":      ("float", 0.001, 0.05, 0.005),
            "kalman_r":      ("float", 0.05,  0.5,  0.05),
            "min_zscore":    ("float", 1.0,   2.5,  0.1),
            "exit_thresh":   ("float", -0.5,  0.5,  0.1),
            "atr_stop":      ("float", 1.0,   3.5,  0.25),
            "atr_tp":        ("float", 1.0,   4.0,  0.25),
            "kelly_frac":    ("float", 0.10,  0.4,  0.05),
        }

    def model_configs(self) -> List[Dict[str, Any]]:
        base = dict(lookback=100, wf_window=50, atr_len=14, kelly_cap=0.25,
                    vol_len=20, vol_profile_len=50)
        return [
            # M0 — Conservative: strict z-score, wide stop
            {**base, "bb_len":25, "bb_std":2.5, "rsi_len":14,
             "rsi_ob":70.0, "rsi_os":30.0, "rv_len":40, "rv_ratio_lo":0.85,
             "kalman_q":0.005, "kalman_r":0.15, "min_zscore":2.0,
             "exit_thresh":0.0, "atr_stop":2.5, "atr_tp":2.0, "kelly_frac":0.15},
            # M1 — Balanced
            {**base, "bb_len":20, "bb_std":2.0, "rsi_len":14,
             "rsi_ob":65.0, "rsi_os":35.0, "rv_len":30, "rv_ratio_lo":0.90,
             "kalman_q":0.01,  "kalman_r":0.10, "min_zscore":1.5,
             "exit_thresh":0.0, "atr_stop":2.0, "atr_tp":2.5, "kelly_frac":0.20},
            # M2 — Aggressive: lower z threshold, more trades
            {**base, "bb_len":15, "bb_std":1.8, "rsi_len":10,
             "rsi_ob":62.0, "rsi_os":38.0, "rv_len":20, "rv_ratio_lo":0.95,
             "kalman_q":0.02,  "kalman_r":0.08, "min_zscore":1.2,
             "exit_thresh":0.2, "atr_stop":1.5, "atr_tp":2.0, "kelly_frac":0.30},
            # M3 — Ultra-selective: extreme z-scores only
            {**base, "bb_len":30, "bb_std":2.8, "rsi_len":21,
             "rsi_ob":72.0, "rsi_os":28.0, "rv_len":50, "rv_ratio_lo":0.80,
             "kalman_q":0.003, "kalman_r":0.20, "min_zscore":2.3,
             "exit_thresh":-0.3,"atr_stop":3.0, "atr_tp":1.5, "kelly_frac":0.15},
        ]

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _kalman_filter(obs: np.ndarray, q: float, r: float) -> np.ndarray:
        """Scalar Kalman filter — tracks the 'true' price."""
        n = len(obs)
        x = obs[0]
        p = 1.0
        out = np.zeros(n)
        for i in range(n):
            # Predict
            p_pred = p + q
            # Update
            k  = p_pred / (p_pred + r)
            x  = x + k * (obs[i] - x)
            p  = (1 - k) * p_pred
            out[i] = x
        return out

    @staticmethod
    def _rsi(close: np.ndarray, length: int) -> np.ndarray:
        n = len(close)
        rsi = np.full(n, 50.0)
        delta = np.diff(close, prepend=close[0])
        gain  = np.where(delta > 0, delta, 0.0)
        loss  = np.where(delta < 0, -delta, 0.0)

        avg_g = gain[:length].mean()
        avg_l = loss[:length].mean() + 1e-9

        for i in range(length, n):
            avg_g = (avg_g * (length-1) + gain[i]) / length
            avg_l = (avg_l * (length-1) + loss[i]) / length
            rs    = avg_g / max(avg_l, 1e-9)
            rsi[i] = 100 - 100 / (1 + rs)
        return rsi

    @staticmethod
    def _rolling_mean_arr(arr: np.ndarray, w: int) -> np.ndarray:
        out = np.full(len(arr), np.nan)
        for i in range(w-1, len(arr)):
            out[i] = np.nanmean(arr[i-w+1:i+1])
        return np.nan_to_num(out, nan=0.0)

    @staticmethod
    def _rolling_std_arr(arr: np.ndarray, w: int) -> np.ndarray:
        out = np.full(len(arr), np.nan)
        for i in range(w-1, len(arr)):
            out[i] = np.nanstd(arr[i-w+1:i+1], ddof=1)
        return np.nan_to_num(out, nan=0.0)
