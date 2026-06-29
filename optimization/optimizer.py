"""
QSES - Optimizer
Searches for parameter configurations that maximise robustness_score
(not net profit) across the full parameter space.

Two-stage:
  Stage 0: TradingView reference seed (if available for this market)
  Stage 1: Random search over the full param space (fast coverage)
  Stage 2: Local refinement around best-found configs

Objective is cross-market stability, NOT single-market max profit.
"""
from __future__ import annotations
import random
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

from ..algorithms.base import BaseAlgorithm
from ..core.types import BacktestMetrics
from ..config.settings import OPTIMIZER_N_TRIALS
from ..utils.logger import get_logger


class Optimizer:

    def __init__(self, algorithm: BaseAlgorithm, n_trials: int = OPTIMIZER_N_TRIALS, seed: int = 42):
        self.algo     = algorithm
        self.n_trials = n_trials
        self.seed     = seed
        self.logger   = get_logger(f"Optimizer[{algorithm.name}]")
        random.seed(seed)
        np.random.seed(seed)

    def optimize(
        self,
        df: "pd.DataFrame",
        base_params: Optional[Dict[str, Any]] = None,
        n_trials:    Optional[int] = None,
        market:      Optional[str] = None,    # used to look up TV seed
    ) -> Tuple[Dict[str, Any], BacktestMetrics]:
        """
        Run seed check + random search + local refinement.
        Returns (best_params, best_metrics).
        """
        n     = n_trials or self.n_trials
        space = self.algo.default_param_space()

        best_params  = base_params or {}
        best_metrics = self.algo.run(df, best_params) if base_params else BacktestMetrics()
        best_score   = best_metrics.robustness_score

        # ── Stage 0: TradingView reference seed ──────────────────────────────
        self.seed_result     = None
        self.seed_params     = None
        tv_seeds = getattr(self.algo, "TV_REFERENCE_SEEDS", {})

        if market and market in tv_seeds:
            seed_params  = tv_seeds[market].copy()
            seed_metrics = self.algo.run(df, seed_params)
            self.seed_result = seed_metrics
            self.seed_params = seed_params
            self.logger.info(
                f"[seed_reference_result] market={market} "
                f"WR={seed_metrics.win_rate:.1%} Sh={seed_metrics.sharpe_ratio:.2f} "
                f"Rob={seed_metrics.robustness_score:.0f} trades={seed_metrics.total_trades}"
            )
            if seed_metrics.is_valid() and seed_metrics.robustness_score > best_score:
                best_score   = seed_metrics.robustness_score
                best_params  = seed_params
                best_metrics = seed_metrics
        else:
            self.logger.debug(f"No TV seed available for market={market!r}")

        if n == 0:
            return best_params, best_metrics

        self.logger.info(f"Starting optimisation: {n} trials | space size={len(space)}")

        # ── Stage 1: Random search ───────────────────────────────────────────
        for trial in range(n):
            candidate = self._sample(space, base_params)
            metrics   = self.algo.run(df, candidate)

            if not metrics.is_valid():
                continue
            if self._fails_hard_filter(metrics):
                continue

            score = metrics.robustness_score
            if score > best_score:
                best_score   = score
                best_params  = candidate
                best_metrics = metrics
                self.logger.debug(
                    f"Trial {trial:3d}: score={score:.1f} WR={metrics.win_rate:.1%} "
                    f"Sh={metrics.sharpe_ratio:.2f} DD={metrics.max_drawdown_pct:.1f}%"
                )

        # ── Stage 2: Local refinement ────────────────────────────────────────
        if best_params:
            best_params, best_metrics = self._refine(df, best_params, best_metrics, space)

        self.logger.info(
            f"Optimisation done. Best score={best_score:.1f} | "
            f"WR={best_metrics.win_rate:.1%} Sh={best_metrics.sharpe_ratio:.2f}"
        )
        return best_params, best_metrics

    # ── Sampling ─────────────────────────────────────────────────────────────

    def _sample(self, space: Dict, base: Optional[Dict]) -> Dict[str, Any]:
        candidate = {}
        for key, spec in space.items():
            kind = spec[0]
            if kind == "float":
                lo, hi, step = spec[1], spec[2], spec[3]
                steps = int(round((hi - lo) / step)) + 1
                candidate[key] = round(lo + random.randint(0, steps-1) * step, 6)
            elif kind == "int":
                lo, hi = spec[1], spec[2]
                candidate[key] = random.randint(lo, hi)
            elif kind == "choice":
                candidate[key] = random.choice(spec[1])
        for k, v in (base or {}).items():
            if k not in space:
                candidate[k] = v
        return candidate

    def _refine(
        self,
        df: "pd.DataFrame",
        best_params: Dict,
        best_metrics: BacktestMetrics,
        space: Dict,
    ) -> Tuple[Dict, BacktestMetrics]:
        self.logger.debug("Local refinement phase")
        for key, spec in space.items():
            kind = spec[0]
            if kind not in ("float", "int"):
                continue
            val = best_params.get(key)
            if val is None:
                continue
            step      = spec[3] if kind == "float" else 1
            neighbours= [val - step, val + step]
            for nval in neighbours:
                nval      = max(spec[1], min(spec[2], nval))
                candidate = {**best_params, key: nval}
                metrics   = self.algo.run(df, candidate)
                if metrics.is_valid() and metrics.robustness_score > best_metrics.robustness_score:
                    best_params  = candidate
                    best_metrics = metrics
        return best_params, best_metrics

    # ── Hard Filters ─────────────────────────────────────────────────────────

    @staticmethod
    def _fails_hard_filter(m: BacktestMetrics) -> bool:
        from ..config.settings import MIN_WIN_RATE, MAX_DRAWDOWN_PCT, MIN_SHARPE_THRESHOLD
        if m.win_rate        < MIN_WIN_RATE:          return True
        if m.max_drawdown_pct< MAX_DRAWDOWN_PCT:      return True
        if m.sharpe_ratio    < MIN_SHARPE_THRESHOLD:  return True
        if m.total_trades    < 5:                     return True
        return False
