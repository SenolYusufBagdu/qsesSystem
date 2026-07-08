"""
QSES — Quant Statistical Edge System
Python implementation of the T1+ composite signal framework.
"""
from .engine import BacktestEngine
from .algorithms import get_algorithm, REGISTRY
from .core.types import BacktestResult, BacktestMetrics, RankingRow
from .data.fetcher import DataFetcher
from .reporting.reporter import Reporter
from .optimization.optimizer import Optimizer
from .optimization.ranker import Ranker

__version__ = "1.5.0"
__all__ = [
    "BacktestEngine",
    "get_algorithm", "REGISTRY",
    "BacktestResult", "BacktestMetrics", "RankingRow",
    "DataFetcher", "Reporter", "Optimizer", "Ranker",
]
