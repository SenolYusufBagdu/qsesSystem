# QSES — Quant Statistical Edge System
## Cross-Market Robustness Optimizer · Python v1.0

---

### Architecture Overview

```
qses/
├── main.py                   # Entry point / CLI
├── engine.py                 # Orchestrates all 192 combinations
├── config/
│   └── settings.py           # All tunable constants (no hardcoding)
├── algorithms/
│   ├── base.py               # BaseAlgorithm (Strategy Pattern)
│   ├── algorithm_a.py        # QSES T1+ composite (Pine port)
│   ├── algorithm_b.py        # Adaptive trend + ADX breakout
│   ├── algorithm_c.py        # Regime-gated mean reversion
│   └── __init__.py           # Registry (add new algos here)
├── core/
│   ├── types.py              # Immutable result containers
│   └── metrics.py            # All 17+ metrics + robustness score
├── data/
│   └── fetcher.py            # yfinance downloader + parquet cache
├── optimization/
│   ├── optimizer.py          # Random search + local refinement
│   └── ranker.py             # Cross-market stability ranking
├── reporting/
│   └── reporter.py           # Plotly charts, heatmaps, HTML table
└── utils/
    └── logger.py             # Structured logging
```

---

### Combination Matrix

| Dimension    | Count | Values                              |
|:-------------|:-----:|:------------------------------------|
| Markets      | 4     | NQ1!, XU100, XAUUSD, SP500          |
| Algorithms   | 3     | A (QSES T1+), B (Trend), C (MR)    |
| Models       | 4     | Conservative → Aggressive           |
| Timeframes   | 2     | 3h, 4h                              |
| Periods      | 2     | 1Y, 2Y                              |
| **Total**    | **192** |                                   |

---

### Usage

```bash
# Full 192-backtest run
cd qses
python main.py

# Quick test: 2 markets, no optimization
python main.py --markets NQ1! XAUUSD --no-optimize

# Specific algorithm, more optimization trials
python main.py --algorithms AlgorithmA --n-trials 100

# Parallel execution (4 cores)
python main.py --n-jobs 4

# Single combination for debugging
python main.py --markets NQ1! --timeframes 3h --periods 1 --algorithms AlgorithmA --no-optimize
```

---

### Algorithms

#### Algorithm A — QSES T1+ Composite Signal
Direct Python port of the Pine Script T1+ logic.

**Signal pipeline:**
1. **Momentum** — Volatility-adjusted ROC (fast/slow) normalized by realized vol
2. **OU Mean Reversion** — Ornstein-Uhlenbeck process z-score of log price
3. **OFI** — Order Flow Imbalance (buy/sell volume pressure)
4. **GARCH Regime** — GARCH(1,1) proxy + hysteresis (High/Normal/Low vol)
5. **Composite** — Regime-adaptive weighted sum → EMA(3) → z-score

**4 Optimizations:**
- [OPT-1] Dynamic threshold (3 regimes × separate σ threshold)
- [OPT-2] OFI regime-aware minimum (HV=0.5, Normal=0.3, LV=0.2)
- [OPT-3] Graduated exit threshold (not aggressive 0σ exit)
- [OPT-4] Signal momentum confirmation (rising signal = valid entry)

#### Algorithm B — Adaptive Trend Breakout
Donchian/Keltner breakout filtered by:
- ADX trend strength filter
- Volume surge confirmation
- Hurst exponent (only enters trending markets)
- EMA trend direction filter

#### Algorithm C — Regime-Gated Mean Reversion
Bollinger Band + RSI mean reversion, strictly gated by:
- Low-volatility regime detection (RV ratio < threshold)
- Kalman-filtered price smoothing
- Volume profile confirmation

---

### Robustness Score Formula

```
Score = 40% × Profit Consistency
      + 25% × Drawdown Stability
      + 20% × Sharpe Component
      + 15% × Cross-Market Stability
```

Configurable in `config/settings.py` → `ROBUSTNESS_WEIGHTS`.

---

### Overfitting Guards

A configuration is **eliminated** if:
- It fails on > 50% of tested markets
- Win rate < 40%
- Max drawdown > -30%
- Sharpe ratio < 0.3
- Passes only 1-year period but not 2-year
- Trades < 5 (insufficient statistical basis)

The optimizer is explicitly prohibited from selecting for max profit.
Objective function: `robustness_score` (not net P&L).

---

### Extending the System

**Add a new market:**
```python
# config/settings.py
MARKETS["BTCUSD"] = "BTC-USD"
```

**Add a new algorithm:**
```python
# algorithms/algorithm_d.py
class AlgorithmD(BaseAlgorithm):
    name = "AlgorithmD"
    def generate_signals(self, df, params): ...
    def default_param_space(self): ...
    def model_configs(self): ...  # must return 4 configs

# algorithms/__init__.py
from .algorithm_d import AlgorithmD
REGISTRY["AlgorithmD"] = AlgorithmD
```

**Add a new timeframe:**
```python
# config/settings.py
TIMEFRAMES = ["3h", "4h", "1d"]
```

---

### Outputs

| File | Description |
|:-----|:------------|
| `results/reports/all_results.csv` | All 192 backtest metrics |
| `results/reports/ranking_table.html` | Cross-market ranking |
| `results/charts/heatmap_sharpe.html` | Market × Algorithm heatmap |
| `results/charts/cross_market_sharpe_ratio.html` | Bar chart |
| `results/charts/<label>.html` | Per-backtest price + equity chart |

---

### Metrics Reported

Win Rate · Total Trades · Net Profit % · CAGR · Sharpe · Sortino ·
Profit Factor · Recovery Factor · Expectancy · Avg Win · Avg Loss ·
Risk/Reward · Max Drawdown · Avg Hold Bars · Exposure % ·
Consecutive Wins · Consecutive Losses · **Robustness Score**
