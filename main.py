#!/usr/bin/env python3
"""
QSES -- Main Entry Point

Usage examples:

  # Full 192-backtest run (production)
  python main.py

  # Quick single-market test
  python main.py --markets NQ1! --timeframes 3h --periods 1 --no-optimize

  # Specific algorithm
  python main.py --algorithms AlgorithmA --n-trials 30

  # Fast smoke test (no optimization)
  python main.py --markets NQ1! XAUUSD --no-optimize --n-trials 0
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
                   help="Optimisation trials per combination")
    return p.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 50)
    logger.info("  QSES -- Quant Statistical Edge System")
    logger.info("  Cross-Market Robustness Optimizer v1.0")
    logger.info("=" * 50)

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
