"""
QSES — Unit Tests: train_test_decay Formula (EK-2, RCA-7 regression)

RCA-7 (research_journal.md, Faz 6, açık madde):
  Old formula: decay = (avg_train - avg_test) / abs(avg_train)
  Bug: when avg_train is negative, the denominator's abs() strips the sign
  information that the numerator still depends on, so the same numeric
  degradation can register as a wildly different (and sometimes flipped)
  decay value depending only on the sign of avg_train — not on the actual
  train->test degradation.

Fix: decay = (avg_train - avg_test) / (abs(avg_train) + abs(avg_test) + DECAY_EPSILON)
  This is bounded, always defined (no div-by-zero), and its sign/magnitude
  depend only on the *relative* gap between train and test, never flipping
  because of avg_train's sign alone.

Each test uses concrete numbers and a real numerical assert — no mocks.
"""
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from qses.optimization.walk_forward import WalkForwardEngine, WalkForwardResult, WalkResult
from qses.config.settings import DECAY_EPSILON


def _make_result(train_sharpes, test_sharpes):
    """Build a minimal WalkForwardResult with only the fields _aggregate() needs."""
    walks = [
        WalkResult(
            walk_id=i, train_start="", train_end="", test_start="", test_end="",
            train_sharpe=tr, test_sharpe=te, test_wr=0.5, test_dd=-5.0,
            test_trades=10, test_pnl=1.0,
        )
        for i, (tr, te) in enumerate(zip(train_sharpes, test_sharpes))
    ]
    return WalkForwardResult(market="TEST", algorithm="AlgorithmA", model_id=0, walks=walks)


def test_decay_positive_train_positive_test_degradation():
    """Baseline sanity case: train=2.0, test=1.0 -> classic 50% relative degradation."""
    r = _make_result(train_sharpes=[2.0, 2.0], test_sharpes=[1.0, 1.0])
    WalkForwardEngine._aggregate(r)
    expected = (2.0 - 1.0) / (abs(2.0) + abs(1.0) + DECAY_EPSILON)
    assert r.train_test_decay == pytest.approx(expected, abs=1e-6)
    assert 0.0 < r.train_test_decay < 1.0


def test_decay_negative_train_does_not_flip_sign_pathologically():
    """
    RCA-7 core regression: avg_train is NEGATIVE (-0.2) and avg_test is
    slightly WORSE (-0.4). Under the old formula:
        (avg_train - avg_test) / abs(avg_train) = (-0.2 - (-0.4)) / 0.2 = 1.0
    -> reported as 100% decay (would be eliminated by MAX_SHARPE_DECAY=0.50)
    even though the absolute degradation is only 0.2 Sharpe.
    The new symmetric formula must stay well below the elimination threshold
    for a degradation this small.
    """
    r = _make_result(train_sharpes=[-0.2, -0.2], test_sharpes=[-0.4, -0.4])
    WalkForwardEngine._aggregate(r)
    old_formula_decay = (-0.2 - (-0.4)) / abs(-0.2)
    assert old_formula_decay == pytest.approx(1.0)
    assert r.train_test_decay < 0.5, (
        f"RCA-7 regression: small absolute degradation reported as "
        f"decay={r.train_test_decay:.3f} (old formula gave {old_formula_decay:.3f})"
    )


def test_decay_is_bounded_between_negative_one_and_one():
    """Symmetric formula must stay within (-1, 1) for any finite same-sign or mixed-sign input."""
    cases = [
        ([3.0], [-3.0]),
        ([-3.0], [3.0]),
        ([0.0], [0.0]),
        ([5.0], [5.0]),
        ([-1.0], [-1.0]),
    ]
    for tr, te in cases:
        r = _make_result(train_sharpes=tr, test_sharpes=te)
        WalkForwardEngine._aggregate(r)
        assert -1.0 <= r.train_test_decay <= 1.0, f"decay out of bounds for train={tr}, test={te}"


def test_decay_zero_when_train_equals_test():
    """No degradation at all -> decay must be exactly 0.0, regardless of sign."""
    for value in (1.5, -1.5, 0.0):
        r = _make_result(train_sharpes=[value], test_sharpes=[value])
        WalkForwardEngine._aggregate(r)
        assert r.train_test_decay == pytest.approx(0.0, abs=1e-6)


def test_decay_no_division_by_zero_when_both_zero():
    """Old formula returned a hardcoded 0.0 special-case for avg_train==0.
    New formula must not raise ZeroDivisionError and must still report 0 decay
    when train and test are both exactly 0."""
    r = _make_result(train_sharpes=[0.0, 0.0], test_sharpes=[0.0, 0.0])
    WalkForwardEngine._aggregate(r)  # must not raise
    assert r.train_test_decay == pytest.approx(0.0, abs=1e-3)


def test_consistency_score_capped_at_100_for_negative_decay():
    """When test outperforms train, decay goes negative; compute_wf_score's
    cons_score must be capped at 100, not overshoot past it."""
    r = _make_result(train_sharpes=[0.5, 0.5], test_sharpes=[1.5, 1.5])
    WalkForwardEngine._aggregate(r)
    assert r.train_test_decay < 0.0
    r.param_cv = {}
    r.unstable_params = []
    score = WalkForwardEngine.compute_wf_score(r)
    assert 0.0 <= score <= 100.0
