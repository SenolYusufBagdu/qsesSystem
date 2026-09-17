"""
QSES — Unit Tests: TV Reference Seed / Parameter Space Bound Consistency (EK-2, RCA-9)

RCA-9 (found while reviewing what "production ready" actually requires):
  Every one of AlgorithmA's TV_REFERENCE_SEEDS had at least one parameter
  value OUTSIDE that same algorithm's default_param_space() declared bounds:
      NQ1!:   ofi_lv_min=0.35   vs declared [0.05, 0.30]
      XU100:  exit_thresh=-1.7  vs declared [-1.5, 0.0]
      XAUUSD: exit_thresh=-1.8  vs declared [-1.5, 0.0]
      XAUUSD: atr_stop=5.0      vs declared [1.0, 4.0]
      XAUUSD: atr_tp=10.0       vs declared [1.5, 6.0]   <- worst offender

  optimizer.py's Stage 0 seed injection runs the seed UNVALIDATED (no bound
  check), and optimizer.py::_refine()'s neighbour-step computed
  `neighbours = [val - step, val + step]` directly from the (possibly
  out-of-bounds) seed value, THEN clamped each neighbour independently to
  [lo, hi]. For a sufficiently-out-of-bounds seed (e.g. XAUUSD's atr_tp=10.0
  against bounds [1.5, 6.0]) both neighbours clamp to the SAME boundary
  value (6.0), so local refinement could never actually search away from
  the raw seed -- it was a silent no-op. This is very likely why real-data
  XAUUSD 4h backtests showed avg_hold=84-124 bars: the "optimized" result
  was actually the raw, unrefined, out-of-bounds seed, never touched by
  search.

Fix:
  1. default_param_space() bounds widened for ofi_lv_min, exit_thresh,
     atr_stop so the real, TradingView-verified NQ1!/XU100/XAUUSD seed
     values are valid, searchable points.
  2. XAUUSD's atr_tp seed value itself changed 10.0 -> 5.0, consistent with
     research_journal.md's own prior conclusion ("Python-optimal deger
     3-5xATR") rather than widening the space to accommodate a value the
     project had already concluded doesn't transfer well to the Python
     execution model.
  3. optimizer.py::_refine() now clamps its OWN starting value to
     [spec[1], spec[2]] before computing neighbours, as defense-in-depth
     against any future seed (for this or any other algorithm) that isn't
     caught by test_all_tv_seeds_are_within_param_space_bounds below.

Each test uses concrete numbers -- no mocks.
"""
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from qses.algorithms import REGISTRY
from qses.core.types import BacktestMetrics
from qses.optimization.optimizer import Optimizer


def test_all_tv_seeds_are_within_param_space_bounds():
    """
    Systemic guard (RCA-9): for every registered algorithm that exposes
    TV_REFERENCE_SEEDS, every seed parameter that also appears in
    default_param_space() must fall within that parameter's declared
    [lo, hi] (float/int) or declared choices (choice). This is the single
    test that would have caught all five original RCA-9 violations at
    once, and prevents any future algorithm/market seed from silently
    drifting out of its own declared search space.
    """
    violations = []
    for algo_name, algo_cls in REGISTRY.items():
        algo = algo_cls()
        seeds = getattr(algo, "TV_REFERENCE_SEEDS", {})
        space = algo.default_param_space()
        for market, params in seeds.items():
            for key, value in params.items():
                if key not in space:
                    continue  # extra fixed params (e.g. commission) aren't searched, exempt
                spec = space[key]
                kind = spec[0]
                if kind in ("float", "int"):
                    lo, hi = spec[1], spec[2]
                    if not (lo <= value <= hi):
                        violations.append(
                            f"{algo_name}/{market}: {key}={value} outside "
                            f"declared bounds [{lo}, {hi}]"
                        )
                elif kind == "choice":
                    choices = spec[1]
                    if value not in choices:
                        violations.append(
                            f"{algo_name}/{market}: {key}={value} not in "
                            f"declared choices {choices}"
                        )

    assert not violations, (
        "RCA-9 regression: TV reference seed(s) outside their own declared "
        "param_space bounds:\n" + "\n".join(violations)
    )


def test_xauusd_atr_tp_seed_matches_documented_research_conclusion():
    """
    Specific regression for the worst offender: XAUUSD's atr_tp seed must
    stay within [1.5, 6.0] (default_param_space()'s declared bounds) and
    specifically at the documented Python-optimal value of 5.0 -- not the
    original out-of-bounds Pine/TV value of 10.0, which research_journal.md
    already independently concluded produces unrealistic ~124-bar holds
    that don't close within walk-forward test windows.
    """
    algo = REGISTRY["AlgorithmA"]()
    seed = algo.TV_REFERENCE_SEEDS["XAUUSD"]
    space = algo.default_param_space()
    lo, hi = space["atr_tp"][1], space["atr_tp"][2]

    assert lo <= seed["atr_tp"] <= hi
    assert seed["atr_tp"] == pytest.approx(5.0)
    assert seed["atr_tp"] != pytest.approx(10.0)


def test_refine_does_not_silently_no_op_when_starting_value_is_out_of_bounds():
    """
    Defense-in-depth regression: even if some future seed slips past
    test_all_tv_seeds_are_within_param_space_bounds (e.g. via a manual
    edit that isn't caught until the next test run), Optimizer._refine()
    itself must not silently do nothing. Before the fix, an out-of-bounds
    starting value produced two neighbours that both clamped to the same
    boundary -- a degenerate step that could never find a better,
    in-bounds candidate. After the fix, the starting value is clamped
    FIRST, so the two neighbours are genuinely distinct points and real
    search can happen.
    """
    class _FakeAlgo:
        """Deterministic fake: robustness peaks at x=9.0, strictly inside bounds."""
        name = "FakeAlgo"

        def run(self, df, params):
            x = params["x"]
            score = 100.0 - abs(x - 9.0) * 10.0
            return BacktestMetrics(
                win_rate=0.6, total_trades=20, sharpe_ratio=1.0,
                max_drawdown_pct=-5.0, net_profit_pct=10.0,
                robustness_score=score, is_valid_sample=True,
            )

    space = {"x": ("float", 0.0, 10.0, 1.0)}
    fake_algo = _FakeAlgo()
    opt = Optimizer(fake_algo, n_trials=1)

    # Deliberately out-of-bounds starting point (15.0, declared max is 10.0).
    starting_params = {"x": 15.0}
    starting_metrics = fake_algo.run(None, {"x": 10.0})  # score at the clamp boundary

    best_params, best_metrics = opt._refine(None, starting_params, starting_metrics, space)

    # Without the fix: neighbours would be [14.0, 16.0] -> both clamp to 10.0 ->
    # no distinct candidate is ever tried -> best_params stays {"x": 15.0} (out
    # of bounds!) or at best ties at 10.0, never reaching 9.0.
    # With the fix: starting value clamps to 10.0 first -> neighbours [9.0, 11.0]
    # -> 11.0 clamps to 10.0 (same as start, no improvement) but 9.0 is a
    # genuinely new, better-scoring, in-bounds candidate -> must be found.
    assert best_params["x"] == pytest.approx(9.0), (
        f"RCA-9 regression: local refinement failed to escape the "
        f"out-of-bounds starting value (got x={best_params['x']})"
    )
    assert 0.0 <= best_params["x"] <= 10.0
