"""
QSES — Backtest Engine
Orchestrates all combinations:
  4 markets x 3 algorithms x 4 models x 2 timeframes x 2 periods = 192 runs

Each run is isolated. Failures do not stop other runs.
Parallel execution via joblib.
"""
from __future__ import annotations
import os
import time
import traceback
from typing import List, Optional, Callable
from itertools import product

from joblib import Parallel, delayed

from .algorithms import get_algorithm
from .core.types import BacktestResult, BacktestMetrics
from .data.fetcher import DataFetcher
from .optimization.optimizer import Optimizer
from .optimization.ranker import Ranker
from .reporting.reporter import Reporter
from .config.settings import (
    MARKETS, TIMEFRAMES, PERIODS_YEARS, ALGORITHMS, NUM_MODELS,
    OPTIMIZER_N_TRIALS, OPTIMIZER_N_JOBS, RESULTS_DIR
)
from .utils.logger import get_logger

logger = get_logger("BacktestEngine")


class BacktestEngine:
    """
    Runs the full QSES pipeline:
      1. Data fetch (cached)
      2. Run preset model configs
      3. Optimise parameters
      4. Rank and select robust configs
      5. Generate reports
    """

    def __init__(
        self,
        markets:     Optional[List[str]] = None,
        timeframes:  Optional[List[str]] = None,
        periods:     Optional[List[int]] = None,
        algorithms:  Optional[List[str]] = None,
        n_jobs:      int = 1,               # parallel workers
        optimize:    bool = True,
        n_opt_trials:int = OPTIMIZER_N_TRIALS,
        progress_callback: Optional[Callable] = None,
    ):
        self.markets   = markets   or list(MARKETS.keys())
        self.timeframes= timeframes or TIMEFRAMES
        self.periods   = periods   or PERIODS_YEARS
        self.algorithms= algorithms or ALGORITHMS
        self.n_jobs    = n_jobs
        self.optimize  = optimize
        self.n_opt_trials = n_opt_trials
        self.progress_cb  = progress_callback

        self.fetcher   = DataFetcher()
        self.ranker    = Ranker()
        self.reporter  = Reporter()

        os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Main entry point ─────────────────────────────────────────────────────

    def run_all(self) -> List[BacktestResult]:
        """Execute all combinations and return results."""
        combos = list(product(
            self.markets, self.timeframes, self.periods,
            self.algorithms, range(NUM_MODELS)
        ))

        total = len(combos)
        logger.info(f"Starting QSES run: {total} backtests")
        logger.info(f"  Markets:    {self.markets}")
        logger.info(f"  Timeframes: {self.timeframes}")
        logger.info(f"  Periods:    {self.periods}Y")
        logger.info(f"  Algorithms: {self.algorithms}")
        logger.info(f"  Models/algo:{NUM_MODELS}")
        logger.info(f"  Parallel:   n_jobs={self.n_jobs}")

        t0 = time.time()

        if self.n_jobs == 1:
            results       = []
            seed_comparisons = []
            for i, combo in enumerate(combos, 1):
                r, sc = self._safe_run_combo(combo, i, total)
                if r:
                    results.append(r)
                if sc:
                    seed_comparisons.extend(sc)
                if self.progress_cb:
                    self.progress_cb(i, total, r)
        else:
            raw = Parallel(n_jobs=self.n_jobs, verbose=5)(
                delayed(self._safe_run_combo)(combo, i+1, total)
                for i, combo in enumerate(combos)
            )
            results          = [r for r, _ in raw if r is not None]
            seed_comparisons = [sc for _, sc_list in raw if sc_list for sc in sc_list]

        elapsed = time.time() - t0
        logger.info(f"All {len(results)}/{total} backtests complete in {elapsed:.1f}s")

        # Rank
        ranking = self.ranker.rank(results)
        logger.info("Generating reports...")

        # Reports
        self.reporter.save_results_csv(results)
        self.reporter.heatmap_sharpe(results)
        self.reporter.bar_cross_market(results, "sharpe_ratio")
        self.reporter.bar_cross_market(results, "net_profit_pct")
        self.reporter.render_ranking_table(ranking)
        if seed_comparisons:
            self.reporter.save_seed_comparison(seed_comparisons)

        # Per-run charts (only for valid results)
        valid = [r for r in results if r.metrics.is_valid()]
        logger.info(f"Generating {len(valid)} individual charts...")
        for r in valid:
            try:
                self.reporter.chart_backtest(r)
            except Exception as e:
                logger.warning(f"Chart failed for {r.label}: {e}")

        self._print_summary(ranking)
        return results

    # ── Single combo ─────────────────────────────────────────────────────────

    def _safe_run_combo(self, combo, idx: int, total: int):
        market, tf, period, algo_name, model_id = combo
        label = f"[{idx:3d}/{total}] {market} {tf} {period}Y {algo_name} M{model_id}"
        try:
            return self._run_combo(market, tf, period, algo_name, model_id, label)
        except Exception as e:
            logger.error(f"{label} FAILED: {e}\n{traceback.format_exc()}")
            return None, []

    def _run_combo(
        self, market: str, tf: str, period: int,
        algo_name: str, model_id: int, label: str
    ):
        """Returns (BacktestResult, list_of_seed_comparison_dicts)."""
        seed_comparisons = []

        # 1. Fetch data
        df = self.fetcher.fetch(market, tf, period)
        if df is None or len(df) < 100:
            logger.warning(f"{label} - insufficient data ({len(df) if df is not None else 0} bars)")
            return BacktestResult(
                market=market, timeframe=tf, period_yrs=period,
                algorithm=algo_name, model_id=model_id,
                params={}, metrics=BacktestMetrics()
            ), seed_comparisons

        # 2. Get algorithm + preset params
        algo   = get_algorithm(algo_name)
        params = algo.model_configs()[model_id].copy()

        # 3. Run preset backtest
        metrics = algo.run(df, params)
        logger.info(
            f"{label} | preset : WR={metrics.win_rate:.1%} "
            f"Sh={metrics.sharpe_ratio:.2f} DD={metrics.max_drawdown_pct:.1f}% "
            f"trades={metrics.total_trades}"
        )

        # 4. Optimise (pass market so optimizer can use TV seed)
        if self.optimize and metrics.total_trades >= 5:
            opt = Optimizer(algo, n_trials=self.n_opt_trials)
            opt_params, opt_metrics = opt.optimize(df, base_params=params, market=market)
            if opt_metrics.robustness_score > metrics.robustness_score:
                params  = opt_params
                metrics = opt_metrics
                logger.info(
                    f"{label} | optimised: WR={metrics.win_rate:.1%} "
                    f"Sh={metrics.sharpe_ratio:.2f} Rob={metrics.robustness_score:.0f}"
                )

            # Build seed comparison record if a seed was tested
            if opt.seed_result is not None:
                sr = opt.seed_result
                winner = ("seed" if sr.robustness_score > opt_metrics.robustness_score
                          else "tie" if sr.robustness_score == opt_metrics.robustness_score
                          else "optimizer")
                seed_comparisons.append({
                    "market": market, "timeframe": tf, "period_yrs": period,
                    "algorithm": algo_name, "model_id": model_id,
                    "seed_wr":      sr.win_rate,      "seed_sharpe":  sr.sharpe_ratio,
                    "seed_pnl":     sr.net_profit_pct,"seed_dd":      sr.max_drawdown_pct,
                    "seed_rob":     sr.robustness_score,"seed_trades": sr.total_trades,
                    "opt_wr":       opt_metrics.win_rate,
                    "opt_sharpe":   opt_metrics.sharpe_ratio,
                    "opt_pnl":      opt_metrics.net_profit_pct,
                    "opt_dd":       opt_metrics.max_drawdown_pct,
                    "opt_rob":      opt_metrics.robustness_score,
                    "opt_trades":   opt_metrics.total_trades,
                    "winner":       winner,
                    "delta_rob":    opt_metrics.robustness_score - sr.robustness_score,
                })

        # 5. Build result
        return BacktestResult(
            market       = market,
            timeframe    = tf,
            period_yrs   = period,
            algorithm    = algo_name,
            model_id     = model_id,
            params       = params,
            metrics      = metrics,
            price_series = df["close"],
        ), seed_comparisons

    # ── Pretty Summary ────────────────────────────────────────────────────────

    @staticmethod
    def _print_summary(ranking):
        logger.info("\n" + "="*70)
        logger.info("QSES FINAL RANKING - TOP CONFIGS")
        logger.info("="*70)
        logger.info(f"{'#':<3} {'Algo':<14} {'M':<3} {'WR':>6} {'P&L':>8} "
                    f"{'Sharpe':>8} {'MaxDD':>8} {'Stab':>6} {'Sel':<6}")
        logger.info("-"*70)
        for i, r in enumerate(ranking[:10], 1):
            sel = "OK" if r.selected else ""
            logger.info(
                f"{i:<3} {r.algorithm:<14} {r.model_id:<3} "
                f"{r.avg_win_rate:>6.1%} {r.avg_net_pnl:>+7.1f}% "
                f"{r.avg_sharpe:>8.2f} {r.avg_max_dd:>7.1f}% "
                f"{r.stability_score:>6.0f} {sel:<6}"
            )
        logger.info("="*70)

        selected = [r for r in ranking if r.selected]
        if selected:
            logger.info("\nSELECTED CONFIGURATIONS (stable across markets):")
            for r in selected:
                logger.info(f"  {r.algorithm} M{r.model_id} | "
                             f"WR={r.avg_win_rate:.1%} Sh={r.avg_sharpe:.2f} "
                             f"DD={r.avg_max_dd:.1f}% Stab={r.stability_score:.0f}")
        else:
            logger.warning("No configurations passed all selection gates.")
