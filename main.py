#!/usr/bin/env python3
"""
QSES -- Main Entry Point

Usage examples:

  # Full 288-backtest run (production)
  python main.py

  # Quick single-market test
  python main.py --markets NQ1! --timeframes 3h --periods 1 --no-optimize

  # Specific algorithm
  python main.py --algorithms AlgorithmA --n-trials 30

  # Fast smoke test (no optimization)
  python main.py --markets NQ1! XAUUSD --no-optimize --n-trials 0

  # Faz 7: rolling walk-forward validation (all 6 markets, AlgorithmA, 4h, 2Y)
  python main.py --walk-forward --n-trials 50

  # Faz 7: walk-forward on the 3 markets that have no TV reference seed
  python main.py --walk-forward --markets SP500 USOIL EURUSD --n-trials 50
"""
import argparse
import sys
import os

# ── Windows UTF-8 fix (must be before any logger/print calls) ────────────────
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Add parent dir to path when running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qses import BacktestEngine
from qses.utils.logger import get_logger

logger = get_logger("Main")


def parse_args():
    p = argparse.ArgumentParser(description="QSES Backtest Engine")
    p.add_argument("--markets",     nargs="+", default=None,
                   help="Markets to test (default: all). E.g. NQ1! XAUUSD")
    p.add_argument("--timeframes",  nargs="+", default=None,
                   help="Timeframes (default: 3h 4h)")
    p.add_argument("--periods",     nargs="+", type=int, default=None,
                   help="Backtest periods in years (default: 1 2)")
    p.add_argument("--algorithms",  nargs="+", default=None,
                   help="Algorithms (default: all)")
    p.add_argument("--n-jobs",      type=int, default=1,
                   help="Parallel workers (-1=all cores)")
    p.add_argument("--no-optimize", action="store_true",
                   help="Skip parameter optimisation (faster)")
    p.add_argument("--n-trials",    type=int, default=None,
                   help="Optimisation trials per combination "
                        "(standard run) or per walk (--walk-forward)")
    p.add_argument("--walk-forward", action="store_true",
                   help="Run rolling walk-forward validation (Faz 6/7) "
                        "instead of the standard grid backtest")
    p.add_argument("--models",      nargs="+", type=int, default=None,
                   help="Model IDs 0-3 for --walk-forward (default: all 4)")
    p.add_argument("--n-walks",     type=int, default=None,
                   help="Number of rolling walks for --walk-forward "
                        "(default: settings.WALK_FORWARD_N_WALKS)")
    return p.parse_args()


def run_walk_forward(args) -> None:
    """
    Faz 6/7 entry point: rolling walk-forward validation.

    Unlike the standard grid backtest (which fans out over market x
    timeframe x period), walk-forward operates on ONE continuous price
    series per market (a single timeframe/period), then internally splits
    it into --n-walks chronological train/test slices. So --timeframes and
    --periods here take exactly one value each (first element used if more
    are passed), not a cartesian product.
    """
    from qses.data.fetcher import DataFetcher
    from qses.algorithms import get_algorithm
    from qses.optimization.walk_forward import WalkForwardEngine
    from qses.reporting import walk_forward_reporter as wfr
    from qses.config.settings import (
        WALK_FORWARD_MARKETS, WALK_FORWARD_N_WALKS, OPTIMIZER_N_TRIALS_WF,
        REPORTS_DIR,
    )

    os.makedirs(REPORTS_DIR, exist_ok=True)

    markets    = args.markets    or WALK_FORWARD_MARKETS
    algorithms = args.algorithms or ["AlgorithmA"]
    models     = args.models     or [0, 1, 2, 3]
    timeframe  = (args.timeframes or ["4h"])[0]
    period_yrs = (args.periods or [2])[0]
    n_walks    = args.n_walks or WALK_FORWARD_N_WALKS
    n_trials   = args.n_trials if args.n_trials is not None else OPTIMIZER_N_TRIALS_WF

    logger.info(
        f"Walk-forward: markets={markets} algorithms={algorithms} "
        f"models={models} timeframe={timeframe} period={period_yrs}Y "
        f"n_walks={n_walks} n_trials={n_trials}"
    )

    fetcher = DataFetcher()
    all_results = []

    for market in markets:
        df = fetcher.fetch(market, timeframe, period_yrs)
        if df is None:
            logger.warning(f"[SKIP] {market}: no data returned by fetcher")
            continue

        for algo_name in algorithms:
            algo   = get_algorithm(algo_name)
            engine = WalkForwardEngine(algo, n_walks=n_walks, n_trials=n_trials)

            for model_id in models:
                try:
                    result = engine.run(df, model_id, market)
                    all_results.append(result)
                except Exception as e:
                    logger.error(
                        f"[FAIL] {market}/{algo_name}/M{model_id}: {e}"
                    )

    if not all_results:
        logger.warning(
            "No walk-forward results produced (no data reachable / all "
            "markets failed). Nothing written to results/reports/."
        )
        return

    wfr.save_walk_detail(all_results)
    wfr.save_model_summary(all_results)
    wfr.save_bootstrap(all_results)
    wfr.save_rationale(all_results)
    wfr.print_model_table(all_results)

    passed = sum(
        1 for r in all_results
        if r.passes_decay_gate and r.passes_ruin_gate and r.passes_sample_gate
    )
    logger.info(
        f"Done. {passed}/{len(all_results)} (market, algo, model) combos "
        f"passed all walk-forward gates."
    )
    logger.info("Reports saved to: results/reports/")


def main():
    args = parse_args()

    logger.info("=" * 50)
    logger.info("  QSES -- Quant Statistical Edge System")
    logger.info("  Cross-Market Robustness Optimizer v1.5")
    logger.info("=" * 50)

    if args.walk_forward:
        run_walk_forward(args)
        return

    engine_kwargs = dict(
        markets    = args.markets,
        timeframes = args.timeframes,
        periods    = args.periods,
        algorithms = args.algorithms,
        n_jobs     = args.n_jobs,
        optimize   = not args.no_optimize,
    )
    if args.n_trials is not None:
        engine_kwargs["n_opt_trials"] = args.n_trials

    engine = BacktestEngine(**engine_kwargs)
    results = engine.run_all()

    valid = [r for r in results if r.metrics.is_valid()]
    logger.info(f"Done. {len(valid)}/{len(results)} backtests produced valid results.")
    logger.info("Reports saved to: results/reports/")
    logger.info("Charts  saved to: results/charts/")


if __name__ == "__main__":
    main()
