"""
QSES — Unit Tests: Cross-Market Pass Ratio (EK-2, RCA-8 regression)

RCA-8 (found while reviewing a real 288-combination run):
  ranker.py computed:
      markets_tested = len(set(market for r in group))          # distinct markets, max 6
      markets_passed = sum(1 for r in group if valid and passes_hard_filter)  # ROW count, max 24
  Then pass_ratio = markets_passed / markets_tested mixed two different units
  (a market COUNT denominator against a combo-ROW-COUNT numerator). Since each
  market contributes up to 4 rows (2 timeframes x 2 periods), a config that
  succeeded on a SINGLE market across all 4 of its combos could score
  pass_ratio = 4/6 = 0.67 -- comfortably above the >=0.5 selection gate --
  while failing completely on the other 5 markets. This defeated the
  documented purpose of the gate ("Sadece 1 piyasada basarili, digerlerinde
  basarisiz -> elenir", research_journal.md / project README).

Fix: markets_passed now counts DISTINCT markets with at least one passing
combo, matching markets_tested's units. pass_ratio is now bounded in [0, 1]
and actually measures "fraction of markets where this config worked".

Each test uses concrete constructed BacktestResult objects -- no mocks.
"""
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from qses.core.types import BacktestResult, BacktestMetrics
from qses.optimization.ranker import Ranker


def _result(market, timeframe, period_yrs, valid, win_rate=0.6,
            max_drawdown_pct=-5.0, sharpe_ratio=1.0, net_profit_pct=10.0,
            robustness_score=70.0, algorithm="AlgorithmA", model_id=0):
    m = BacktestMetrics(
        win_rate=win_rate, total_trades=20 if valid else 2,
        max_drawdown_pct=max_drawdown_pct, sharpe_ratio=sharpe_ratio,
        net_profit_pct=net_profit_pct, robustness_score=robustness_score,
        is_valid_sample=valid,
        exclusion_reason="" if valid else "insufficient_trades (n=2, min=10)",
    )
    return BacktestResult(market=market, timeframe=timeframe, period_yrs=period_yrs,
                           algorithm=algorithm, model_id=model_id, params={}, metrics=m)


def test_pass_ratio_never_exceeds_one_for_single_strong_market():
    """
    RCA-8 core regression: ONE market (NQ1!) passes on all 4 of its
    (timeframe, period) combos; the other 5 markets have no valid results
    at all. Before the fix, markets_passed counted the 4 passing ROWS
    against markets_tested=6 distinct markets -> pass_ratio = 4/6 = 0.67,
    comfortably clearing the 0.5 gate despite total failure on 5/6 markets.
    After the fix, markets_passed must count the 1 distinct passing market
    -> pass_ratio = 1/6 ~= 0.167, correctly failing the gate.
    """
    group = []
    for tf in ["3h", "4h"]:
        for py in [1, 2]:
            group.append(_result("NQ1!", tf, py, valid=True,
                                  win_rate=0.7, sharpe_ratio=2.0, max_drawdown_pct=-3.0))
    for market in ["XU100", "XAUUSD", "SP500", "USOIL", "EURUSD"]:
        for tf in ["3h", "4h"]:
            for py in [1, 2]:
                group.append(_result(market, tf, py, valid=False))

    ranker = Ranker()
    row = ranker._score_group(("AlgorithmA", 0), group)

    assert row.markets_tested == 6
    assert row.markets_passed == 1, (
        f"Expected 1 distinct passing market, got markets_passed={row.markets_passed} "
        f"(RCA-8 regression: counting rows instead of distinct markets)"
    )
    assert row.pass_ratio == pytest.approx(1 / 6)
    assert row.pass_ratio <= 1.0
    assert not ranker._passes_selection_gate(row), (
        "A config passing on only 1 of 6 markets must NOT clear the "
        "cross-market selection gate, regardless of how many timeframe/period "
        "combos that single market contributed."
    )


def test_pass_ratio_correctly_passes_when_majority_of_markets_pass():
    """Sanity check in the other direction: 4 of 6 markets pass (each with
    just 1 valid combo) -> pass_ratio = 4/6 = 0.667 >= 0.5 -> gate passes."""
    group = []
    for market in ["NQ1!", "XAUUSD", "SP500", "USOIL"]:
        group.append(_result(market, "4h", 2, valid=True,
                              win_rate=0.7, sharpe_ratio=1.5, max_drawdown_pct=-4.0))
    for market in ["XU100", "EURUSD"]:
        group.append(_result(market, "4h", 2, valid=False))

    ranker = Ranker()
    row = ranker._score_group(("AlgorithmA", 0), group)

    assert row.markets_tested == 6
    assert row.markets_passed == 4
    assert row.pass_ratio == pytest.approx(4 / 6)
    assert ranker._passes_selection_gate(row)


def test_markets_passed_does_not_double_count_same_market_multiple_combos():
    """A market with several passing (timeframe, period) combos must only
    count ONCE toward markets_passed."""
    group = []
    for tf in ["3h", "4h"]:
        for py in [1, 2]:
            group.append(_result("XAUUSD", tf, py, valid=True,
                                  win_rate=0.65, sharpe_ratio=1.2, max_drawdown_pct=-4.0))
    ranker = Ranker()
    row = ranker._score_group(("AlgorithmA", 0), group)
    assert row.markets_tested == 1
    assert row.markets_passed == 1  # not 4


def test_real_world_reproduction_algorithmA_m0_xu100_total_failure():
    """
    Reproduction of an actual result from a real 288-combo run: AlgorithmA/M0
    had 8 valid+passing rows spread across 5 markets, with XU100 producing
    ZERO valid results in any of its 4 combos. The old code reported
    markets_passed=8 (row count) against markets_tested=6, i.e. pass_ratio
    =1.33 -- a nonsensical value above 1.0 that still trivially cleared the
    gate. The fixed code must report markets_passed=5 (distinct passing
    markets), giving a sane pass_ratio of 5/6.
    """
    group = []
    # 5 markets each contribute exactly the row counts seen in the real data
    # (NQ1!:2, XAUUSD:2, SP500:1, USOIL:2, EURUSD:1 = 8 total passing rows)
    passing_rows = {"NQ1!": 2, "XAUUSD": 2, "SP500": 1, "USOIL": 2, "EURUSD": 1}
    for market, n in passing_rows.items():
        for i in range(n):
            group.append(_result(market, "3h" if i == 0 else "4h", 2, valid=True,
                                  win_rate=0.8, sharpe_ratio=2.5, max_drawdown_pct=-2.5))
    # XU100: all 4 combos invalid (zero_trades / insufficient_trades)
    for tf in ["3h", "4h"]:
        for py in [1, 2]:
            group.append(_result("XU100", tf, py, valid=False))

    ranker = Ranker()
    row = ranker._score_group(("AlgorithmA", 0), group)

    assert sum(1 for r in group if r.metrics.is_valid()) == 8  # matches real data's 8 valid rows
    assert row.markets_tested == 6
    assert row.markets_passed == 5, (
        f"Expected 5 distinct passing markets (XU100 fully failed), "
        f"got {row.markets_passed}"
    )
    assert row.pass_ratio == pytest.approx(5 / 6)
    assert row.pass_ratio <= 1.0
