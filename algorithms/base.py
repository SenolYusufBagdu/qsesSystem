"""
QSES - Base Algorithm
Every algorithm inherits from this. The engine calls only the public API:
    run()  →  BacktestMetrics

Intrabar SL/TP execution (Faz 5):
    _check_exit_intrabar() uses bar's high/low to detect SL/TP touches
    within the bar, returning the correct exit price (stop/tp level, not
    bar close). Gap-open handling: if bar opens beyond stop, exit at open.

Strategy Pattern contract — unchanged from v1.0:
    generate_signals(df, params)  →  pd.Series {1, -1, 0}
    default_param_space()         →  Dict
    model_configs()               →  List[Dict]  (exactly 4 items)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd

from ..core.types import BacktestMetrics
from ..core.metrics import compute_metrics
from ..config.settings import DEFAULT_COMMISSION_PCT, DEFAULT_SLIPPAGE_ATR, DEFAULT_INITIAL_EQUITY
from ..utils.logger import get_logger


class BaseAlgorithm(ABC):
    """
    Strategy Pattern base.
    Subclasses MUST implement: generate_signals, default_param_space, model_configs.
    Subclasses MUST NOT override: run(), _simulate(), _check_exit_intrabar().
    Algorithm A may override _check_exit_intrabar() only to pass extra signal
    context — the intrabar price logic itself stays in the base.
    """

    name: str = "BaseAlgorithm"

    def __init__(self):
        self.logger = get_logger(self.name)

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        commission_pct: float = DEFAULT_COMMISSION_PCT,
        slippage_atr_frac: float = DEFAULT_SLIPPAGE_ATR,
        initial_equity: float = DEFAULT_INITIAL_EQUITY,
    ) -> BacktestMetrics:
        """Full backtest. df must have: open, high, low, close, volume."""
        try:
            signals = self.generate_signals(df, params)
        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}")
            return BacktestMetrics()
        return self._simulate(df, signals, params, commission_pct,
                              slippage_atr_frac, initial_equity)

    # ── Abstract interface (Strategy Pattern contract) ─────────────────────────

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """Returns Series aligned with df.index. Values: 1=long, -1=short, 0=flat."""
        ...

    @abstractmethod
    def default_param_space(self) -> Dict[str, Any]:
        """
        Returns parameter space for the optimizer.
        Format: {"name": ("float", lo, hi, step) | ("int", lo, hi) | ("choice", [...])}
        """
        ...

    @abstractmethod
    def model_configs(self) -> List[Dict[str, Any]]:
        """Returns exactly 4 preset parameter dicts. Model 0=conservative, 3=aggressive."""
        ...

    # ── Simulation engine (shared, not overrideable) ───────────────────────────

    def _simulate(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        params: Dict[str, Any],
        commission_pct: float,
        slippage_atr_frac: float,
        initial_equity: float,
    ) -> BacktestMetrics:

        atr_stop  = params.get("atr_stop",  2.0)
        atr_tp    = params.get("atr_tp",    3.0)
        kelly_f   = params.get("kelly_frac", 0.25)
        kelly_cap = params.get("kelly_cap",  0.25)

        opens   = df["open"].values
        closes  = df["close"].values
        highs   = df["high"].values
        lows    = df["low"].values
        atrs    = self._atr(df, params.get("atr_len", 14))
        sigs    = signals.values

        equity       = initial_equity
        trades: List[Dict] = []
        equity_curve = [equity]

        pos       = 0      # 1=long, -1=short, 0=flat
        entry_px  = 0.0
        stop_px   = 0.0
        tp_px     = 0.0
        entry_bar = 0
        wins_seen = 0
        trades_seen = 0

        for i in range(1, len(closes)):
            atr  = atrs[i]
            slip = slippage_atr_frac * atr

            # ── Exit check ────────────────────────────────────────────────────
            if pos != 0:
                exited, exit_px, exit_reason = self._check_exit_intrabar(
                    pos=pos,
                    bar_open=opens[i],
                    bar_high=highs[i],
                    bar_low=lows[i],
                    bar_close=closes[i],
                    entry_px=entry_px,
                    stop_px=stop_px,
                    tp_px=tp_px,
                    signal=float(sigs[i]),
                    params=params,
                )
                if exited:
                    gross = pos * (exit_px - entry_px) / max(entry_px, 1e-9) * 100
                    net   = gross - 2 * commission_pct - slip / max(entry_px, 1e-9) * 100
                    hold  = i - entry_bar
                    trades.append({
                        "pnl_pct":     net,
                        "hold_bars":   max(hold, 1),
                        "direction":   pos,
                        "entry_bar":   entry_bar,
                        "exit_bar":    i,
                        "entry_px":    entry_px,
                        "exit_px":     exit_px,
                        "exit_reason": exit_reason,
                    })
                    trades_seen += 1
                    if net > 0:
                        wins_seen += 1
                    equity *= (1 + net / 100)
                    equity_curve.append(equity)
                    pos = 0

            # ── Entry check ───────────────────────────────────────────────────
            if pos == 0 and i >= 10:
                sig = int(sigs[i])
                if sig != 0:
                    wr    = wins_seen / trades_seen if trades_seen >= 10 else 0.5
                    rr    = atr_tp / max(atr_stop, 0.001)
                    kelly = (wr * rr - (1 - wr)) / max(rr, 0.001)
                    _size = min(max(kelly * kelly_f, 0.0), kelly_cap)   # noqa: F841

                    pos       = sig
                    entry_px  = closes[i] + slip * sig   # slippage on entry
                    stop_px   = entry_px - sig * atr * atr_stop
                    tp_px     = entry_px + sig * atr * atr_tp
                    entry_bar = i

        # Close open position at last bar
        if pos != 0:
            slip  = slippage_atr_frac * atrs[-1]
            c     = closes[-1]
            gross = pos * (c - entry_px) / max(entry_px, 1e-9) * 100
            net   = gross - 2 * commission_pct - slip / max(entry_px, 1e-9) * 100
            hold  = len(closes) - 1 - entry_bar
            trades.append({
                "pnl_pct":     net,
                "hold_bars":   max(hold, 1),
                "direction":   pos,
                "entry_bar":   entry_bar,
                "exit_bar":    len(closes) - 1,
                "entry_px":    entry_px,
                "exit_px":     c,
                "exit_reason": "end_of_data",
            })
            equity *= (1 + net / 100)
            equity_curve.append(equity)

        return compute_metrics(
            trades=trades,
            equity_curve=equity_curve,
            total_bars=len(closes),
            initial_equity=initial_equity,
        )

    # ── Intrabar SL/TP + Gap Logic ─────────────────────────────────────────────

    def _check_exit_intrabar(
        self,
        pos:        int,
        bar_open:   float,
        bar_high:   float,
        bar_low:    float,
        bar_close:  float,
        entry_px:   float,
        stop_px:    float,
        tp_px:      float,
        signal:     float,
        params:     Dict,
    ) -> Tuple[bool, float, str]:
        """
        Returns (exited: bool, exit_price: float, exit_reason: str).

        Intrabar execution rules:
        1. Gap-open: if bar opens past the stop (gap through), exit at open —
           not at stop_px. This is the most realistic slippage model.
        2. SL/TP both touched in same bar (high>=tp AND low<=sl):
           Priority = whichever level the open is closest to (directional
           bias proxy). For LONG: if open is closer to low -> SL hit first.
        3. SL or TP touched (not both): exit at the touched level.
        4. Signal exit (OPT-3): exit at bar close (no intrabar level to hit).
        5. No exit: return (False, 0.0, "").
        """
        exit_thresh = params.get("exit_thresh", -0.5)

        if pos == 1:  # ── LONG ───────────────────────────────────────────────
            # Gap-down through stop (open < stop)
            if bar_open <= stop_px:
                return True, bar_open, "gap_stop"

            sl_hit = bar_low  <= stop_px
            tp_hit = bar_high >= tp_px

            if sl_hit and tp_hit:
                # Both levels touched: use open proximity to decide which hit first
                dist_to_sl = abs(bar_open - stop_px)
                dist_to_tp = abs(bar_open - tp_px)
                if dist_to_sl <= dist_to_tp:
                    return True, stop_px, "sl_intrabar"
                else:
                    return True, tp_px, "tp_intrabar"

            if sl_hit:
                return True, stop_px, "sl_intrabar"
            if tp_hit:
                return True, tp_px,   "tp_intrabar"

            # Signal exit (OPT-3 graduated exit) — at close
            if signal < exit_thresh:
                return True, bar_close, "signal_exit"

        else:  # ── SHORT ──────────────────────────────────────────────────────
            # Gap-up through stop (open > stop)
            if bar_open >= stop_px:
                return True, bar_open, "gap_stop"

            sl_hit = bar_high >= stop_px
            tp_hit = bar_low  <= tp_px

            if sl_hit and tp_hit:
                dist_to_sl = abs(bar_open - stop_px)
                dist_to_tp = abs(bar_open - tp_px)
                if dist_to_sl <= dist_to_tp:
                    return True, stop_px, "sl_intrabar"
                else:
                    return True, tp_px, "tp_intrabar"

            if sl_hit:
                return True, stop_px, "sl_intrabar"
            if tp_hit:
                return True, tp_px,   "tp_intrabar"

            if signal > -exit_thresh:
                return True, bar_close, "signal_exit"

        return False, 0.0, ""

    # ── ATR (Wilder smoothing) ─────────────────────────────────────────────────

    @staticmethod
    def _atr(df: pd.DataFrame, length: int = 14) -> np.ndarray:
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values
        tr = np.maximum(h - l,
             np.maximum(np.abs(h - np.roll(c, 1)),
                        np.abs(l - np.roll(c, 1))))
        tr[0] = h[0] - l[0]
        atr = np.zeros(len(tr))
        atr[:length] = tr[:length].mean()
        for i in range(length, len(tr)):
            atr[i] = (atr[i-1] * (length - 1) + tr[i]) / length
        return atr
