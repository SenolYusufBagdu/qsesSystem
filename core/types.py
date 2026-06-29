"""
QSES - Core Data Types
Immutable result containers and parameter schemas.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd


# Minimum trades required for a result to be statistically meaningful.
# Rationale: below 10 trades, win-rate and Sharpe estimates have very high
# variance. At 10 trades the confidence interval on a 60% WR is +/-30pp.
# 15 would be ideal but 10 is the practical minimum for real-market data
# where signal frequency is inherently limited.
MIN_VALID_TRADES: int = 10


@dataclass
class BacktestMetrics:
    """All performance metrics for a single backtest run."""
    win_rate:            float = 0.0
    total_trades:        int   = 0
    net_profit_pct:      float = 0.0
    cagr:                float = 0.0
    sharpe_ratio:        float = 0.0
    sortino_ratio:       float = 0.0
    profit_factor:       Optional[float] = 0.0   # None = no losing trades (div-by-zero guard)
    has_zero_losses:     bool  = False             # True when profit_factor would be infinite
    recovery_factor:     float = 0.0
    expectancy:          float = 0.0
    avg_win:             float = 0.0
    avg_loss:            float = 0.0
    risk_reward:         float = 0.0
    max_drawdown_pct:    float = 0.0
    avg_holding_bars:    float = 0.0
    exposure_pct:        float = 0.0
    max_consec_wins:     int   = 0
    max_consec_losses:   int   = 0
    robustness_score:    float = 0.0
    # Sample validity (set by compute_metrics)
    is_valid_sample:     bool  = False
    exclusion_reason:    str   = ""
    # Raw equity curve for visualisation
    equity_curve:        List[float] = field(default_factory=list)
    # Trade log
    trades:              List[Dict]  = field(default_factory=list)

    def is_valid(self) -> bool:
        """Returns False if the backtest produced no statistically meaningful output."""
        return self.is_valid_sample

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()
             if k not in ("equity_curve", "trades")}
        # Serialise Optional[float] profit_factor cleanly
        if d.get("profit_factor") is None:
            d["profit_factor"] = "inf (no losses)"
        return d


@dataclass
class BacktestResult:
    """A fully-labelled result: who ran it, on what data, with what params."""
    market:      str
    timeframe:   str
    period_yrs:  int
    algorithm:   str
    model_id:    int
    params:      Dict[str, Any]
    metrics:     BacktestMetrics
    # Price series for the charting overlay
    price_series:   Optional[pd.Series] = None
    signal_series:  Optional[pd.Series] = None

    @property
    def label(self) -> str:
        return (f"{self.market} | {self.timeframe} | {self.period_yrs}Y | "
                f"{self.algorithm} | M{self.model_id}")


@dataclass
class RankingRow:
    """One row in the final cross-market ranking table."""
    algorithm:           str
    model_id:            int
    params:              Dict[str, Any]
    markets_passed:      int
    markets_tested:      int
    avg_win_rate:        float
    avg_net_pnl:         float
    avg_sharpe:          float
    avg_max_dd:          float
    avg_robustness:      float
    stability_score:     float   # low std across markets = more stable
    selected:            bool = False

    @property
    def pass_ratio(self) -> float:
        return self.markets_passed / max(self.markets_tested, 1)
