"""
QSES — Ranker
Aggregates results across all markets/timeframes/periods and
identifies the most stable, cross-market parameter configurations.

Key principle: Stability > Max Profit
  A config that earns 40% across ALL markets beats one that earns 200%
  on one market and -50% on another.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import numpy as np

from ..core.types import BacktestResult, RankingRow, BacktestMetrics
from ..config.settings import (
    MAX_MARKET_FAIL_RATIO, MIN_SHARPE_THRESHOLD,
    MIN_WIN_RATE, MAX_DRAWDOWN_PCT, ROBUSTNESS_WEIGHTS
)
from ..utils.logger import get_logger

logger = get_logger("Ranker")


class Ranker:
    """Aggregates BacktestResult records → ranked RankingRow list."""

    def rank(self, results: List[BacktestResult]) -> List[RankingRow]:
        """
        Group results by (algorithm, model_id, params_fingerprint).
        For each group: compute cross-market stability and select winners.
        """
        groups = self._group_by_config(results)
        rows   = []

        for key, group in groups.items():
            row = self._score_group(key, group)
            if row:
                rows.append(row)

        # Sort by stability_score DESC, then avg_sharpe DESC
        rows.sort(key=lambda r: (r.stability_score, r.avg_sharpe), reverse=True)

        # Mark selected (top configs that pass all gates)
        selected_count = 0
        for row in rows:
            if self._passes_selection_gate(row) and selected_count < 5:
                row.selected = True
                selected_count += 1

        logger.info(f"Ranked {len(rows)} configurations | {selected_count} selected")
        return rows

    # ── Grouping ──────────────────────────────────────────────────────────────

    @staticmethod
    def _group_by_config(results: List[BacktestResult]) -> Dict:
        groups = defaultdict(list)
        for r in results:
            key = (r.algorithm, r.model_id)
            groups[key].append(r)
        return groups

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_group(self, key: Tuple, group: List[BacktestResult]) -> Optional[RankingRow]:
        algo, model_id = key
        if not group:
            return None

        sharpes  = [r.metrics.sharpe_ratio   for r in group if r.metrics.is_valid()]
        win_rates= [r.metrics.win_rate        for r in group if r.metrics.is_valid()]
        net_pnls = [r.metrics.net_profit_pct  for r in group if r.metrics.is_valid()]
        dds      = [r.metrics.max_drawdown_pct for r in group if r.metrics.is_valid()]
        robs     = [r.metrics.robustness_score for r in group if r.metrics.is_valid()]

        if not sharpes:
            return None

        markets_tested = len(set(r.market for r in group))
        markets_passed = sum(1 for r in group
                             if r.metrics.is_valid()
                             and not self._fails_hard_filter(r.metrics))

        avg_sharpe  = float(np.mean(sharpes))
        avg_wr      = float(np.mean(win_rates))
        avg_pnl     = float(np.mean(net_pnls))
        avg_dd      = float(np.mean(dds))
        avg_rob     = float(np.mean(robs))

        # Stability: how consistent is Sharpe across markets?
        # Low std = more stable = better
        if len(sharpes) >= 2:
            cross_std    = float(np.std(sharpes, ddof=1))
            cross_score  = max(0.0, 100 - cross_std * 40)   # 0 std → 100
        else:
            cross_std    = 0.0
            cross_score  = 50.0

        # Inject cross_score into robustness (overrides placeholder 50)
        w = ROBUSTNESS_WEIGHTS
        # Re-score avg robustness with real cross-market component
        # The individual scores had cross_score=50; correct for the difference
        cross_correction = w["cross_market_stability"] * (cross_score - 50.0)
        stability_score  = avg_rob + cross_correction

        # Params: use first valid result
        params = group[0].params

        return RankingRow(
            algorithm        = algo,
            model_id         = model_id,
            params           = params,
            markets_passed   = markets_passed,
            markets_tested   = markets_tested,
            avg_win_rate     = avg_wr,
            avg_net_pnl      = avg_pnl,
            avg_sharpe       = avg_sharpe,
            avg_max_dd       = avg_dd,
            avg_robustness   = avg_rob,
            stability_score  = stability_score,
        )

    # ── Selection Gate ────────────────────────────────────────────────────────

    def _passes_selection_gate(self, row: RankingRow) -> bool:
        """
        A configuration is selected only if it:
        1. Passes on majority of markets
        2. Has acceptable average Sharpe
        3. Has acceptable average drawdown
        4. Has acceptable win rate
        """
        if row.pass_ratio < (1 - MAX_MARKET_FAIL_RATIO):
            return False
        if row.avg_sharpe < MIN_SHARPE_THRESHOLD:
            return False
        if row.avg_max_dd < MAX_DRAWDOWN_PCT:
            return False
        if row.avg_win_rate < MIN_WIN_RATE:
            return False
        return True

    @staticmethod
    def _fails_hard_filter(m: BacktestMetrics) -> bool:
        if m.win_rate        < MIN_WIN_RATE:         return True
        if m.max_drawdown_pct< MAX_DRAWDOWN_PCT:     return True
        if m.sharpe_ratio    < MIN_SHARPE_THRESHOLD: return True
        return False
