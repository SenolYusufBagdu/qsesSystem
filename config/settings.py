"""
QSES — Configuration
All tunable constants live here. Nothing is hardcoded in the core.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any

# ─── Markets ────────────────────────────────────────────────────────────────
# yfinance tickers. Add new markets here — the engine picks them up automatically.
MARKETS: Dict[str, str] = {
    "NQ1!":   "NQ=F",
    "XU100":  "XU100.IS",
    "XAUUSD": "GC=F",
    "SP500":  "ES=F",
    "USOIL":  "CL=F",       # WTI Crude Oil Futures (fallback: USO ETF)
    "EURUSD": "EURUSD=X",   # Forex -- volume=0, OFI uses range-based proxy
}

# ─── Timeframes ─────────────────────────────────────────────────────────────
TIMEFRAMES: List[str] = ["3h", "4h"]

# ─── Backtest Periods ────────────────────────────────────────────────────────
PERIODS_YEARS: List[int] = [1, 2]

# ─── Algorithms ─────────────────────────────────────────────────────────────
# Keys match class names in algorithms/
ALGORITHMS: List[str] = ["AlgorithmA", "AlgorithmB", "AlgorithmC"]

# ─── Models per Algorithm ───────────────────────────────────────────────────
# 4 model configurations per algorithm (different param sets to explore)
NUM_MODELS: int = 4

# ─── Optimization ────────────────────────────────────────────────────────────
# Robustness score weights (must sum to 1.0)
ROBUSTNESS_WEIGHTS: Dict[str, float] = {
    "profit_consistency":   0.40,
    "drawdown_stability":   0.25,
    "sharpe":               0.20,
    "cross_market_stability": 0.15,
}

# Optimization objective — NOT max profit
OPTIMIZATION_METRIC: str = "robustness_score"

# Optimizer settings
OPTIMIZER_N_TRIALS: int = 50       # per market/timeframe/period combo
OPTIMIZER_N_JOBS: int   = -1       # -1 = all CPU cores

# ─── Risk Defaults ──────────────────────────────────────────────────────────
DEFAULT_COMMISSION_PCT:  float = 0.05   # per side
DEFAULT_SLIPPAGE_ATR:   float = 0.05
DEFAULT_INITIAL_EQUITY: float = 100_000.0

# ─── Ranking / Elimination ──────────────────────────────────────────────────
# A param set is eliminated if it fails on more than this fraction of markets
MAX_MARKET_FAIL_RATIO: float = 0.5

# Minimum acceptable Sharpe across markets
MIN_SHARPE_THRESHOLD: float = 0.3

# Minimum win rate
MIN_WIN_RATE: float = 0.40

# Maximum drawdown allowed
MAX_DRAWDOWN_PCT: float = -30.0

# ─── Paths ──────────────────────────────────────────────────────────────────
DATA_CACHE_DIR:    str = "data/cache"
RESULTS_DIR:       str = "results"
REPORTS_DIR:       str = "results/reports"
CHARTS_DIR:        str = "results/charts"

# ─── Plotting ────────────────────────────────────────────────────────────────
CHART_THEME: str = "plotly_dark"
CHART_WIDTH: int  = 1400
CHART_HEIGHT: int = 800
