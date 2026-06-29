"""
QSES - Metrics Engine
Computes all 17+ performance metrics from a list of trade records.
Pure functions -- no side effects.
"""
from __future__ import annotations
import math
from typing import List, Dict, Any, Optional

import numpy as np

from ..core.types import BacktestMetrics, MIN_VALID_TRADES
from ..config.settings import ROBUSTNESS_WEIGHTS


def compute_metrics(
    trades: List[Dict],
    equity_curve: List[float],
    total_bars: int,
    initial_equity: float = 100_000.0,
    periods_per_year: int = 252,
) -> BacktestMetrics:
    """
    Parameters
    ----------
    trades : list of dicts with keys: pnl_pct, hold_bars, direction
    equity_curve : list of equity values
    total_bars : total bars in the backtest window
    """
    m = BacktestMetrics()
    m.trades       = trades
    m.equity_curve = equity_curve

    # ── Sample validity gate ──────────────────────────────────────────────────
    if not trades:
        m.exclusion_reason = "zero_trades"
        m.is_valid_sample  = False
        return m

    if len(trades) < MIN_VALID_TRADES:
        m.total_trades     = len(trades)
        m.exclusion_reason = f"insufficient_trades (n={len(trades)}, min={MIN_VALID_TRADES})"
        m.is_valid_sample  = False
        # Still compute what we can for debugging, but flag clearly
        _compute_all(m, trades, equity_curve, total_bars, initial_equity, periods_per_year)
        return m

    m.is_valid_sample  = True
    m.exclusion_reason = ""
    _compute_all(m, trades, equity_curve, total_bars, initial_equity, periods_per_year)
    return m


def _compute_all(
    m: BacktestMetrics,
    trades: List[Dict],
    equity_curve: List[float],
    total_bars: int,
    initial_equity: float,
    periods_per_year: int,
) -> None:
    """Fills all metric fields. Called for both valid and invalid samples."""
    pnls  = np.array([t["pnl_pct"] for t in trades])
    holds = np.array([t["hold_bars"] for t in trades])

    wins   = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    m.total_trades     = len(trades)
    m.win_rate         = len(wins) / m.total_trades
    m.avg_win          = float(wins.mean())   if len(wins)   > 0 else 0.0
    m.avg_loss         = float(losses.mean()) if len(losses) > 0 else 0.0
    m.net_profit_pct   = float(pnls.sum())
    m.avg_holding_bars = float(holds.mean())

    # Risk/Reward
    if m.avg_loss != 0:
        m.risk_reward = abs(m.avg_win / m.avg_loss)

    # Profit Factor -- div-by-zero guard
    # If there are NO losing trades, profit_factor is mathematically undefined
    # (infinite). We set it to None and flag has_zero_losses rather than
    # producing a multi-billion phantom value.
    gross_profit = float(wins.sum())  if len(wins)   > 0 else 0.0
    gross_loss   = float(abs(losses.sum())) if len(losses) > 0 else 0.0

    if gross_loss == 0.0:
        if gross_profit > 0:
            m.profit_factor  = None   # infinite - flagged explicitly
            m.has_zero_losses = True
            m.exclusion_reason = (m.exclusion_reason + " div_by_zero_guard").strip()
        else:
            m.profit_factor = 0.0
    else:
        m.profit_factor = gross_profit / gross_loss

    # Expectancy
    m.expectancy = (m.win_rate * m.avg_win) + ((1 - m.win_rate) * m.avg_loss)

    # CAGR
    if len(equity_curve) >= 2:
        total_return = equity_curve[-1] / equity_curve[0] - 1.0
        years = len(equity_curve) / max(periods_per_year, 1)
        m.cagr = (1 + total_return) ** (1 / max(years, 1e-9)) - 1.0

    # Max Drawdown
    eq_arr = np.array(equity_curve) if equity_curve else np.array([initial_equity])
    running_max = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - running_max) / running_max * 100
    m.max_drawdown_pct = float(dd.min())

    # Sharpe (needs at least 5 trades for any estimate, 10 for validity)
    if len(pnls) >= 5:
        avg_r = pnls.mean()
        std_r = pnls.std(ddof=1) + 1e-10
        tpy   = periods_per_year / max(m.avg_holding_bars, 1.0)
        m.sharpe_ratio = float(avg_r / std_r * math.sqrt(tpy))

    # Sortino
    if len(pnls) >= 5:
        down     = pnls[pnls < 0]
        down_std = down.std(ddof=1) + 1e-10 if len(down) > 1 else 1e-10
        tpy      = periods_per_year / max(m.avg_holding_bars, 1.0)
        m.sortino_ratio = float(pnls.mean() / down_std * math.sqrt(tpy))

    # Recovery Factor
    if m.max_drawdown_pct != 0:
        m.recovery_factor = m.net_profit_pct / abs(m.max_drawdown_pct)

    # Exposure
    m.exposure_pct = int(holds.sum()) / max(total_bars, 1) * 100

    # Consecutive wins/losses
    m.max_consec_wins, m.max_consec_losses = _consecutive(pnls)

    # Robustness score (only meaningful for valid samples)
    if m.is_valid_sample:
        m.robustness_score = _robustness_score(m)


def _consecutive(pnls: np.ndarray):
    max_w = max_l = cur_w = cur_l = 0
    for p in pnls:
        if p > 0:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        max_w = max(max_w, cur_w)
        max_l = max(max_l, cur_l)
    return max_w, max_l


def _robustness_score(m: BacktestMetrics) -> float:
    """
    Composite robustness score [0, 100].
    Only called for is_valid_sample=True results.
    """
    w = ROBUSTNESS_WEIGHTS

    # Use profit_factor safely -- if None (no losses), cap at 3.0 for scoring
    pf_safe = m.profit_factor if m.profit_factor is not None else 3.0

    # Profit Consistency [0,100]
    pc = (m.win_rate * 60) + (min(pf_safe, 3) / 3 * 40)

    # Drawdown Stability [0,100]  (0% DD = 100, -30% DD = 0)
    dd_score = max(0.0, 100 + m.max_drawdown_pct * 100 / 30)

    # Sharpe component [0,100]
    sharpe_score = min(max(m.sharpe_ratio * 33.33, 0), 100)

    # Cross-market placeholder (overwritten by ranker)
    cross_score = 50.0

    score = (w["profit_consistency"]     * pc
           + w["drawdown_stability"]     * dd_score
           + w["sharpe"]                 * sharpe_score
           + w["cross_market_stability"] * cross_score)

    return round(score, 2)
