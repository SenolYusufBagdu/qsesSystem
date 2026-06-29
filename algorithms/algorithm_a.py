"""
QSES — Algorithm A: Composite Statistical Edge
Port of QSES T1+ Pine Script logic to Python.

Signal pipeline:
    Factor 1: Volatility-Adjusted Momentum (ROC fast/slow / RV)
    Factor 2: Ornstein-Uhlenbeck Mean Reversion (log price OU)
    Factor 3: Order Flow Imbalance (buy/sell volume pressure)
    Factor 4: GARCH(1,1) proxy + hysteresis regime detection
    Composite: regime-adaptive weighted sum → EMA(3) → z-score

Four model configurations explore different regime/threshold trade-offs.
"""
from __future__ import annotations
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from .base import BaseAlgorithm


class AlgorithmA(BaseAlgorithm):
    """QSES Composite Signal — direct port of T1+ logic."""

    name = "AlgorithmA"

    # ── Signal Generation ────────────────────────────────────────────────────

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        p = params
        n = len(df)
        close  = df["close"].values
        high   = df["high"].values
        low    = df["low"].values
        volume = df["volume"].values if "volume" in df.columns else np.ones(n)

        lookback = p.get("lookback", 200)
        wf_warm  = p.get("wf_window", 50)

        # ── Factor 1: Momentum ───────────────────────────────────────────────
        mom_fast   = p.get("mom_fast", 10)
        mom_slow   = p.get("mom_slow", 40)
        mom_signal = p.get("mom_signal", 9)

        roc_fast = self._roc(close, mom_fast)
        roc_slow = self._roc(close, mom_slow)
        log_ret  = np.diff(np.log(np.maximum(close, 1e-9)), prepend=np.nan)
        rv       = self._rolling_std(log_ret, p.get("vol_len", 20)) * np.sqrt(252) * 100

        mom_raw      = (roc_fast - roc_slow) / np.maximum(rv, 0.001)
        mom_smoothed = self._ema(mom_raw, mom_signal)
        mom_z        = self._zscore(mom_smoothed, lookback)

        # ── Factor 2: OU Mean Reversion ─────────────────────────────────────
        ou_len     = p.get("ou_len", 30)
        log_price  = np.log(np.maximum(close, 1e-9))
        lp_mean    = self._rolling_mean(log_price, ou_len)
        lp_std     = self._rolling_std(log_price, ou_len)
        ou_z       = (log_price - lp_mean) / np.maximum(lp_std, 1e-9)
        mr_signal  = -ou_z

        # ── Factor 3: Order Flow Imbalance ───────────────────────────────────
        # For forex markets (EURUSD, etc.) yfinance returns volume=0.
        # We detect this and automatically switch to a range-based OFI proxy
        # that captures the same bar-close-position information without volume.
        # Proxy formula: close_position = (close-low)/(high-low), normalised to [-1,+1]
        # This is embedded in the SAME code path -- no separate function.
        of_len    = p.get("of_len", 14)
        crange    = np.maximum(high - low, 1e-9)
        buy_p     = (close - low)  / crange    # [0,1]
        sell_p    = (high - close) / crange    # [0,1]

        has_volume = volume.sum() > 0          # False for EURUSD/forex
        if not has_volume:
            # Log once per run (only at bar 0 to avoid log spam)
            self.logger.info(
                "volume=0 detected: using range-based OFI proxy "
                "((close_position - 0.5) * 2). Market is likely forex."
            )
            # ofi_proxy in [-1, +1]: positive = price closed near high (buy pressure)
            ofi_proxy   = (buy_p - 0.5) * 2
            ofi_raw     = self._rolling_mean(ofi_proxy, of_len)   # smooth over window
        else:
            buy_vol     = volume * buy_p
            sell_vol    = volume * sell_p
            ofi_buy     = self._rolling_sum(buy_vol,  of_len)
            ofi_sell    = self._rolling_sum(sell_vol, of_len)
            ofi_total   = np.maximum(ofi_buy + ofi_sell, 1e-9)
            ofi_raw     = (ofi_buy - ofi_sell) / ofi_total

        ofi_z         = self._zscore(ofi_raw, lookback)

        # OFI momentum (same direction as previous bar)
        ofi_acc_long  = np.roll(ofi_z, 1) < ofi_z
        ofi_acc_short = np.roll(ofi_z, 1) > ofi_z
        ofi_acc_long[0] = ofi_acc_short[0] = False

        # ── Factor 4: GARCH Regime ───────────────────────────────────────────
        ga = p.get("garch_alpha", 0.15)
        gb = p.get("garch_beta",  0.80)
        rc = p.get("regime_confirm", 3)
        vt = p.get("vol_thresh",    1.20)
        vl = p.get("vol_low_mult",  0.83)

        rv_long  = self._rolling_mean(self._rolling_std(log_ret, p.get("vol_len", 20)), lookback)
        rv_ratio = self._rolling_std(log_ret, p.get("vol_len", 20)) / np.maximum(rv_long, 1e-9)

        garch_var = np.zeros(n)
        for i in range(1, n):
            garch_var[i] = (ga * log_ret[i-1]**2 + gb * garch_var[i-1]
                            if not np.isnan(log_ret[i-1]) else garch_var[i-1])
        garch_vol   = np.sqrt(np.maximum(garch_var, 0))
        garch_long  = self._rolling_mean(garch_vol, lookback)
        garch_ratio = garch_vol / np.maximum(garch_long, 1e-9)

        # Hysteresis regime
        regime = self._detect_regime(rv_ratio, garch_ratio, vt, vl, rc, n)

        # ── Composite Signal ──────────────────────────────────────────────────
        # Weights per regime
        w_mom = np.where(regime == 1, 0.55, np.where(regime == -1, 0.15, 0.35))
        w_mr  = np.where(regime == 1, 0.15, np.where(regime == -1, 0.50, 0.30))
        w_ofi = np.where(regime == 1, 0.30, np.where(regime == -1, 0.35, 0.35))

        composite    = mom_z * w_mom + mr_signal * w_mr + ofi_z * w_ofi
        comp_smooth  = self._ema(composite, 3)
        final_signal = self._zscore(comp_smooth, lookback)

        # ── [OPT-1] Dynamic Threshold ────────────────────────────────────────
        thresh_base   = p.get("thresh_base", 1.50)
        thresh_hv     = p.get("thresh_hv_mult", 1.30)
        thresh_lv     = p.get("thresh_lv_mult", 0.80)
        opt1          = p.get("opt1_enable", True)

        if opt1:
            thresh = np.where(regime == 1,  thresh_base * thresh_hv,
                     np.where(regime == -1, thresh_base * thresh_lv, thresh_base))
        else:
            thresh = np.where(regime == 1, thresh_base * thresh_hv, thresh_base)

        # ── [OPT-2] OFI Regime Filter ────────────────────────────────────────
        opt2       = p.get("opt2_enable", True)
        ofi_hv_min = p.get("ofi_hv_min", 0.50)
        ofi_nr_min = p.get("ofi_nrm_min", 0.30)
        ofi_lv_min = p.get("ofi_lv_min",  0.20)

        if opt2:
            ofi_min = np.where(regime == 1, ofi_hv_min,
                      np.where(regime == -1, ofi_lv_min, ofi_nr_min))
        else:
            ofi_min = np.full(n, 0.30)

        ofi_ok_long  = (ofi_z >  ofi_min) & ofi_acc_long
        ofi_ok_short = (ofi_z < -ofi_min) & ofi_acc_short

        # ── [OPT-4] Signal Momentum ───────────────────────────────────────────
        opt4  = p.get("opt4_enable", True)
        sm_lb = p.get("sigmon_lookback", 2)

        if opt4:
            sig_rising  = np.ones(n, bool)
            sig_falling = np.ones(n, bool)
            for k in range(sm_lb):
                sig_rising  &= np.roll(final_signal, k) < np.roll(final_signal, k + 1)
                sig_falling &= np.roll(final_signal, k) > np.roll(final_signal, k + 1)
            sig_rising[:sm_lb+1]  = False
            sig_falling[:sm_lb+1] = False
        else:
            sig_rising = sig_falling = np.ones(n, bool)

        # ── VWAP Filter (rolling window -- NOT cumulative) ───────────────────
        # The Pine Script uses cumulative VWAP which is session-reset there.
        # A never-resetting cumulative VWAP drifts far from price in trending
        # markets (XU100, XAUUSD) and permanently blocks all entries.
        # Fix: use a rolling VWAP over the same lookback window instead.
        vwap_dev  = p.get("vwap_dev", 1.5)
        vwap_len  = p.get("vwap_len", lookback)       # rolling window length
        atr_arr   = self._atr(df, p.get("atr_len", 14))
        hlc3      = (high + low + close) / 3
        hlc3v     = hlc3 * volume
        roll_hlc3v= self._rolling_sum(hlc3v, vwap_len)
        roll_vol  = self._rolling_sum(volume, vwap_len)
        vwap_val  = roll_hlc3v / np.maximum(roll_vol, 1e-9)
        vwap_dist = (close - vwap_val) / np.maximum(atr_arr, 1e-9)

        # ── Entry Conditions ─────────────────────────────────────────────────
        warmed = np.arange(n) > (wf_warm + lookback)

        long_cond  = (warmed
                      & (final_signal >  thresh)
                      & ofi_ok_long
                      & (vwap_dist <= vwap_dev)
                      & sig_rising)

        short_cond = (warmed
                      & (final_signal < -thresh)
                      & ofi_ok_short
                      & (vwap_dist >= -vwap_dev)
                      & sig_falling)

        sig_out = np.zeros(n, dtype=int)
        sig_out[long_cond]  =  1
        sig_out[short_cond] = -1

        return pd.Series(sig_out, index=df.index)

    # ── Parameter Space ──────────────────────────────────────────────────────

    def default_param_space(self) -> Dict[str, Any]:
        return {
            "lookback":        ("int",    100, 300),
            "mom_fast":        ("int",    5,   20),
            "mom_slow":        ("int",    20,  80),
            "mom_signal":      ("int",    3,   15),
            "ou_len":          ("int",    15,  60),
            "of_len":          ("int",    7,   28),
            "thresh_base":     ("float",  1.0, 2.5,  0.05),
            "thresh_hv_mult":  ("float",  1.1, 1.8,  0.05),
            "thresh_lv_mult":  ("float",  0.5, 1.0,  0.05),
            "ofi_hv_min":      ("float",  0.2, 0.8,  0.05),
            "ofi_nrm_min":     ("float",  0.1, 0.5,  0.05),
            "ofi_lv_min":      ("float",  0.05,0.3,  0.05),
            "exit_thresh":     ("float",  -1.5, 0.0, 0.1),
            "atr_stop":        ("float",  1.0, 4.0,  0.25),
            "atr_tp":          ("float",  1.5, 6.0,  0.25),
            "kelly_frac":      ("float",  0.1, 0.5,  0.05),
            "garch_alpha":     ("float",  0.05,0.30, 0.05),
            "garch_beta":      ("float",  0.60,0.95, 0.05),
            "vol_thresh":      ("float",  1.0, 1.6,  0.05),
            "vol_low_mult":    ("float",  0.6, 0.95, 0.05),
            "regime_confirm":  ("int",    1,   7),
            "opt1_enable":     ("choice", [True, False]),
            "opt2_enable":     ("choice", [True, False]),
            "opt3_enable":     ("choice", [True, False]),
            "opt4_enable":     ("choice", [True, False]),
            "vwap_dev":        ("float",  0.8, 5.0,  0.1),
            "vwap_len":        ("int",    50,  300),
        }

    # ── TradingView Reference Seeds ──────────────────────────────────────────
    # Verified parameters from real broker tick data backtests in TradingView.
    # These are injected as trial-0 (seed) by the Optimizer before random search.
    # Keys match market names in config/settings.py MARKETS dict.
    TV_REFERENCE_SEEDS: Dict[str, Dict[str, Any]] = {
        "NQ1!": dict(
            # Common fixed parameters (from Pine Script header)
            lookback=200, wf_window=50,
            mom_fast=10, mom_signal=9,
            ou_len=30, of_len=14,
            vol_thresh=1.20,
            # NQ1!-specific
            opt1_enable=False, opt2_enable=True,
            ofi_hv_min=0.20, ofi_nrm_min=0.30, ofi_lv_min=0.35,
            opt3_enable=True, exit_thresh=-0.80,
            opt4_enable=True,
            atr_stop=3.25, atr_tp=3.0,
            kelly_frac=0.15,
            # Additional params from FAZ 3 brief
            vwap_dev=1.5, kelly_cap=0.25,
            commission=0.01, slippage_atr=0.02,
            # Fill remaining required params with balanced defaults
            mom_slow=40, vol_len=20, atr_len=14,
            garch_alpha=0.15, garch_beta=0.80,
            vol_low_mult=0.83, regime_confirm=3,
            thresh_base=1.50, thresh_hv_mult=1.30, thresh_lv_mult=0.80,
            sigmon_lookback=2, vwap_len=200,
        ),
        "XU100": dict(
            lookback=200, wf_window=50,
            mom_fast=10, mom_signal=9,
            ou_len=30, of_len=14,
            vol_thresh=1.20,
            # XU100-specific
            opt1_enable=True,
            ofi_hv_min=0.30, ofi_nrm_min=0.10, ofi_lv_min=0.15,
            opt2_enable=True,
            opt3_enable=True, exit_thresh=-1.70,
            opt4_enable=True,
            atr_stop=2.0, atr_tp=3.0,
            kelly_frac=0.25,
            vwap_dev=1.5, kelly_cap=0.25,
            commission=0.01, slippage_atr=0.02,
            mom_slow=40, vol_len=20, atr_len=14,
            garch_alpha=0.15, garch_beta=0.80,
            vol_low_mult=0.83, regime_confirm=3,
            thresh_base=1.50, thresh_hv_mult=1.30, thresh_lv_mult=0.80,
            sigmon_lookback=2, vwap_len=200,
        ),
        "XAUUSD": dict(
            lookback=200, wf_window=50,
            mom_fast=10, mom_signal=9,
            ou_len=30, of_len=14,
            vol_thresh=1.20,
            # XAUUSD-specific
            opt1_enable=False,
            ofi_hv_min=0.70, ofi_nrm_min=0.40, ofi_lv_min=0.30,
            opt2_enable=True,
            opt3_enable=True, exit_thresh=-1.80,
            opt4_enable=True,
            atr_stop=5.0, atr_tp=10.0,
            kelly_frac=0.30,
            # vwap_dev: Pine uses session-reset VWAP (price always near it).
            # Python rolling VWAP drifts in trends -> set wider to replicate Pine behavior.
            # Original Pine value 1.9 kept as reference; 4.0 is the rolling-equivalent.
            vwap_dev=4.0, kelly_cap=0.40,
            commission=0.03, slippage_atr=0.03,
            mom_slow=40, vol_len=20, atr_len=14,
            garch_alpha=0.15, garch_beta=0.80,
            vol_low_mult=0.83, regime_confirm=3,
            thresh_base=1.50, thresh_hv_mult=1.30, thresh_lv_mult=0.80,
            # sigmon_lookback=1 for XAUUSD: lower-frequency market needs less strict
            # momentum confirmation (Pine bar confirmation behaves like lookback=1).
            sigmon_lookback=1, vwap_len=200,
        ),
    }

    def model_configs(self) -> List[Dict[str, Any]]:
        """
        4 preset models:
          M0 — Conservative (tight thresholds, wide stop, low Kelly)
          M1 — Balanced (T1+ defaults, all OPTs on)
          M2 — Aggressive (loose thresholds, tight stop, higher Kelly)
          M3 — Mean-Rev focus (low-vol favoured, smaller momentum weight)
        """
        base = dict(
            lookback=200, wf_window=50, vol_len=20, atr_len=14,
            of_len=14, vwap_dev=1.5, garch_alpha=0.15, garch_beta=0.80,
            vol_thresh=1.20, vol_low_mult=0.83, regime_confirm=3,
            opt1_enable=True, opt2_enable=True, opt3_enable=True, opt4_enable=True,
            sigmon_lookback=2, kelly_cap=0.25,
        )
        return [
            # M0 — Conservative
            {**base, "mom_fast":8, "mom_slow":50, "mom_signal":12,
             "ou_len":40, "thresh_base":1.80, "thresh_hv_mult":1.40, "thresh_lv_mult":0.90,
             "ofi_hv_min":0.60, "ofi_nrm_min":0.40, "ofi_lv_min":0.25,
             "exit_thresh":-1.00, "atr_stop":3.00, "atr_tp":4.00, "kelly_frac":0.15},
            # M1 — Balanced (T1+ defaults)
            {**base, "mom_fast":10, "mom_slow":40, "mom_signal":9,
             "ou_len":30, "thresh_base":1.50, "thresh_hv_mult":1.30, "thresh_lv_mult":0.80,
             "ofi_hv_min":0.50, "ofi_nrm_min":0.30, "ofi_lv_min":0.20,
             "exit_thresh":-0.50, "atr_stop":2.00, "atr_tp":3.00, "kelly_frac":0.25},
            # M2 — Aggressive
            {**base, "mom_fast":6, "mom_slow":30, "mom_signal":6,
             "ou_len":20, "thresh_base":1.20, "thresh_hv_mult":1.20, "thresh_lv_mult":0.70,
             "ofi_hv_min":0.35, "ofi_nrm_min":0.20, "ofi_lv_min":0.10,
             "exit_thresh":-0.30, "atr_stop":1.50, "atr_tp":2.50, "kelly_frac":0.35},
            # M3 — Mean-Rev Focus
            {**base, "mom_fast":12, "mom_slow":60, "mom_signal":14,
             "ou_len":50, "thresh_base":1.60, "thresh_hv_mult":1.50, "thresh_lv_mult":0.70,
             "ofi_hv_min":0.55, "ofi_nrm_min":0.30, "ofi_lv_min":0.15,
             "exit_thresh":-0.80, "atr_stop":2.50, "atr_tp":3.50, "kelly_frac":0.20,
             "vol_low_mult":0.75, "regime_confirm":2},
        ]

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _roc(arr: np.ndarray, n: int) -> np.ndarray:
        shifted = np.roll(arr, n)
        shifted[:n] = arr[:n]
        return np.where(shifted != 0, (arr - shifted) / np.abs(shifted) * 100, 0.0)

    @staticmethod
    def _ema(arr: np.ndarray, span: int) -> np.ndarray:
        alpha = 2 / (span + 1)
        out = np.zeros_like(arr, dtype=float)
        out[0] = arr[0] if not np.isnan(arr[0]) else 0.0
        for i in range(1, len(arr)):
            v = arr[i] if not np.isnan(arr[i]) else 0.0
            out[i] = alpha * v + (1 - alpha) * out[i-1]
        return out

    @staticmethod
    def _rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
        out = np.full(len(arr), np.nan)
        for i in range(w - 1, len(arr)):
            out[i] = np.nanmean(arr[i-w+1:i+1])
        return out

    @staticmethod
    def _rolling_std(arr: np.ndarray, w: int) -> np.ndarray:
        out = np.full(len(arr), np.nan)
        for i in range(w - 1, len(arr)):
            out[i] = np.nanstd(arr[i-w+1:i+1], ddof=1)
        return np.nan_to_num(out, nan=0.0)

    @staticmethod
    def _rolling_sum(arr: np.ndarray, w: int) -> np.ndarray:
        out = np.zeros(len(arr))
        cs  = np.cumsum(arr)
        out[w-1:] = cs[w-1:] - np.concatenate([[0], cs[:-w]])
        return out

    @staticmethod
    def _zscore(arr: np.ndarray, w: int) -> np.ndarray:
        out = np.zeros(len(arr))
        for i in range(w - 1, len(arr)):
            window = arr[i-w+1:i+1]
            mu  = np.nanmean(window)
            std = np.nanstd(window, ddof=1) + 1e-9
            out[i] = (arr[i] - mu) / std
        return out

    @staticmethod
    def _detect_regime(rv_ratio, garch_ratio, vt, vl, rc, n):
        regime   = np.zeros(n, dtype=int)
        hv_count = lv_count = 0
        cur = 0
        for i in range(n):
            raw_hv = int(rv_ratio[i] > vt) + int(garch_ratio[i] > vt)
            raw_lv = int(rv_ratio[i] < vl) + int(garch_ratio[i] < vl)

            if raw_hv >= 1:
                hv_count += 1; lv_count = 0
            elif raw_lv >= 1:
                lv_count += 1; hv_count = 0
            else:
                hv_count = lv_count = 0

            if hv_count >= rc:
                cur = 1
            elif lv_count >= rc:
                cur = -1
            elif raw_hv == 0 and raw_lv == 0:
                cur = 0
            regime[i] = cur
        return regime
