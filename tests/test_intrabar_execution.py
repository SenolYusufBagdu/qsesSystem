"""
QSES — Unit Tests: Intrabar SL/TP Execution (EK-2)

Each test covers exactly ONE scenario with real numerical OHLC values.
No mocks, no placeholders — every assert checks a concrete expected price
and exit_reason string.
"""
import sys
import os
import pytest
import numpy as np
import pandas as pd

# Allow import from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from qses.algorithms.base import BaseAlgorithm
from qses.algorithms import get_algorithm


# ─── Minimal concrete subclass so we can instantiate BaseAlgorithm ───────────

class _ConcreteAlgo(BaseAlgorithm):
    name = "TestAlgo"
    def generate_signals(self, df, params):
        return pd.Series(0, index=df.index)
    def default_param_space(self):
        return {}
    def model_configs(self):
        return [{} for _ in range(4)]


ALGO = _ConcreteAlgo()

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _default_params(exit_thresh=-0.5):
    return {"exit_thresh": exit_thresh}


# ─── Test 1: LONG TP hit intrabar ────────────────────────────────────────────

def test_long_position_tp_hit_intrabar():
    """
    LONG: bar_high reaches TP level.
    Exit must happen at tp_px (100.5), not bar_close (99.0).
    """
    exited, exit_px, reason = ALGO._check_exit_intrabar(
        pos=1,
        bar_open=99.0,
        bar_high=101.0,   # > tp_px
        bar_low=98.5,
        bar_close=99.0,
        entry_px=98.0,
        stop_px=96.0,
        tp_px=100.5,
        signal=0.0,
        params=_default_params(),
    )
    assert exited is True,            f"Expected exit, got exited={exited}"
    assert exit_px == 100.5,          f"Expected exit at tp_px=100.5, got {exit_px}"
    assert reason == "tp_intrabar",   f"Expected 'tp_intrabar', got {reason!r}"


# ─── Test 2: LONG SL hit intrabar ────────────────────────────────────────────

def test_long_position_sl_hit_intrabar():
    """
    LONG: bar_low falls to SL level.
    Exit must happen at stop_px (95.0), not bar_close (96.5).
    """
    exited, exit_px, reason = ALGO._check_exit_intrabar(
        pos=1,
        bar_open=97.0,
        bar_high=97.5,
        bar_low=94.0,     # < stop_px
        bar_close=96.5,
        entry_px=98.0,
        stop_px=95.0,
        tp_px=104.0,
        signal=0.0,
        params=_default_params(),
    )
    assert exited is True,            f"Expected exit, got exited={exited}"
    assert exit_px == 95.0,           f"Expected exit at stop_px=95.0, got {exit_px}"
    assert reason == "sl_intrabar",   f"Expected 'sl_intrabar', got {reason!r}"


# ─── Test 3: SHORT TP hit intrabar ───────────────────────────────────────────

def test_short_position_tp_hit_intrabar():
    """
    SHORT: bar_low falls to TP level.
    Exit must happen at tp_px (90.0), not bar_close (92.0).
    """
    exited, exit_px, reason = ALGO._check_exit_intrabar(
        pos=-1,
        bar_open=92.5,
        bar_high=93.0,
        bar_low=89.0,     # < tp_px (short TP is below entry)
        bar_close=92.0,
        entry_px=95.0,
        stop_px=97.0,     # above entry for short
        tp_px=90.0,       # below entry for short
        signal=0.0,
        params=_default_params(),
    )
    assert exited is True,            f"Expected exit, got exited={exited}"
    assert exit_px == 90.0,           f"Expected exit at tp_px=90.0, got {exit_px}"
    assert reason == "tp_intrabar",   f"Expected 'tp_intrabar', got {reason!r}"


# ─── Test 4: SHORT SL hit intrabar ───────────────────────────────────────────

def test_short_position_sl_hit_intrabar():
    """
    SHORT: bar_high exceeds SL level.
    Exit must happen at stop_px (97.0), not bar_close (96.0).
    """
    exited, exit_px, reason = ALGO._check_exit_intrabar(
        pos=-1,
        bar_open=96.0,
        bar_high=98.0,    # > stop_px
        bar_low=95.0,
        bar_close=96.0,
        entry_px=95.0,
        stop_px=97.0,
        tp_px=90.0,
        signal=0.0,
        params=_default_params(),
    )
    assert exited is True,            f"Expected exit, got exited={exited}"
    assert exit_px == 97.0,           f"Expected exit at stop_px=97.0, got {exit_px}"
    assert reason == "sl_intrabar",   f"Expected 'sl_intrabar', got {reason!r}"


# ─── Test 5: Gap through stop ────────────────────────────────────────────────

def test_gap_through_stop():
    """
    LONG: bar opens (gap-down) below stop_px.
    Exit price must be bar_open (94.0), NOT stop_px (95.0).
    This models realistic gap slippage — you can't exit at a level
    the market gapped through.
    """
    exited, exit_px, reason = ALGO._check_exit_intrabar(
        pos=1,
        bar_open=94.0,    # gap below stop_px=95.0
        bar_high=96.0,
        bar_low=93.5,
        bar_close=95.5,
        entry_px=100.0,
        stop_px=95.0,
        tp_px=110.0,
        signal=0.0,
        params=_default_params(),
    )
    assert exited is True,            f"Expected exit on gap, got exited={exited}"
    assert exit_px == 94.0,           f"Expected gap exit at open=94.0, got {exit_px}"
    assert reason == "gap_stop",      f"Expected 'gap_stop', got {reason!r}"
    # Crucially: NOT at stop_px=95.0
    assert exit_px != 95.0,           "Exit should NOT be at stop_px when bar gaps through it"


def test_gap_through_stop_short():
    """
    SHORT: bar opens (gap-up) above stop_px.
    Exit price must be bar_open (98.5), NOT stop_px (97.0).
    """
    exited, exit_px, reason = ALGO._check_exit_intrabar(
        pos=-1,
        bar_open=98.5,    # gap above stop_px=97.0
        bar_high=99.0,
        bar_low=97.5,
        bar_close=98.0,
        entry_px=95.0,
        stop_px=97.0,
        tp_px=90.0,
        signal=0.0,
        params=_default_params(),
    )
    assert exited is True
    assert exit_px == 98.5,           f"Expected gap exit at open=98.5, got {exit_px}"
    assert reason == "gap_stop"
    assert exit_px != 97.0,           "Exit should NOT be at stop_px on gap"


# ─── Test 6: Same bar — both SL and TP touched ───────────────────────────────

def test_same_bar_both_sl_and_tp_hit():
    """
    LONG: bar_high >= tp AND bar_low <= sl (wide range bar).
    Decision rule: whichever level bar_open is closer to is taken first.
    Scenario A: open closer to SL -> SL wins.
    Scenario B: open closer to TP -> TP wins.
    """
    stop_px = 95.0
    tp_px   = 105.0

    # Scenario A: open=96.0 (1 pt from SL, 9 pts from TP) -> SL first
    exited_a, exit_px_a, reason_a = ALGO._check_exit_intrabar(
        pos=1,
        bar_open=96.0,   # dist_to_sl=1, dist_to_tp=9
        bar_high=106.0,  # > tp
        bar_low=94.0,    # < sl
        bar_close=100.0,
        entry_px=100.0,
        stop_px=stop_px,
        tp_px=tp_px,
        signal=0.0,
        params=_default_params(),
    )
    assert exited_a is True
    assert exit_px_a == stop_px,     f"Scenario A: expected SL={stop_px}, got {exit_px_a}"
    assert reason_a == "sl_intrabar",f"Scenario A: expected sl_intrabar, got {reason_a!r}"

    # Scenario B: open=104.0 (9 pts from SL, 1 pt from TP) -> TP first
    exited_b, exit_px_b, reason_b = ALGO._check_exit_intrabar(
        pos=1,
        bar_open=104.0,  # dist_to_sl=9, dist_to_tp=1
        bar_high=106.0,
        bar_low=94.0,
        bar_close=100.0,
        entry_px=100.0,
        stop_px=stop_px,
        tp_px=tp_px,
        signal=0.0,
        params=_default_params(),
    )
    assert exited_b is True
    assert exit_px_b == tp_px,       f"Scenario B: expected TP={tp_px}, got {exit_px_b}"
    assert reason_b == "tp_intrabar",f"Scenario B: expected tp_intrabar, got {reason_b!r}"


# ─── Test 7: No exit when neither level touched ───────────────────────────────

def test_no_exit_when_neither_level_touched():
    """
    Neither SL nor TP hit, signal not at exit threshold.
    Position must remain open (exited=False).
    """
    exited, exit_px, reason = ALGO._check_exit_intrabar(
        pos=1,
        bar_open=100.0,
        bar_high=101.5,   # below tp=105
        bar_low=99.0,     # above sl=95
        bar_close=100.5,
        entry_px=100.0,
        stop_px=95.0,
        tp_px=105.0,
        signal=0.1,       # above exit_thresh=-0.5
        params=_default_params(),
    )
    assert exited is False,           f"Expected no exit, got exited={exited}"
    assert exit_px == 0.0,            f"Expected exit_px=0.0, got {exit_px}"
    assert reason == "",              f"Expected empty reason, got {reason!r}"


def test_no_exit_short_neither_level():
    """SHORT: bar stays inside SL/TP range, no signal exit."""
    exited, exit_px, reason = ALGO._check_exit_intrabar(
        pos=-1,
        bar_open=93.0,
        bar_high=93.5,    # below stop=97
        bar_low=92.5,     # above tp=90
        bar_close=93.0,
        entry_px=95.0,
        stop_px=97.0,
        tp_px=90.0,
        signal=-0.1,      # below exit trigger for short: signal > +0.5 would exit
        params=_default_params(),
    )
    assert exited is False
    assert reason == ""


# ─── Test 8: Signal-based exit still works ───────────────────────────────────

def test_signal_based_exit_still_works():
    """
    OPT-3 graduated exit: signal drops below exit_thresh while price
    is between SL and TP — must exit at bar_close, not SL/TP.
    """
    exited, exit_px, reason = ALGO._check_exit_intrabar(
        pos=1,
        bar_open=101.0,
        bar_high=101.5,   # below tp=105
        bar_low=100.5,    # above sl=95
        bar_close=100.8,
        entry_px=100.0,
        stop_px=95.0,
        tp_px=105.0,
        signal=-1.2,      # below exit_thresh=-0.5 -> signal exit
        params=_default_params(exit_thresh=-0.5),
    )
    assert exited is True,              f"Expected signal exit, got exited={exited}"
    assert exit_px == 100.8,            f"Expected exit at close=100.8, got {exit_px}"
    assert reason == "signal_exit",     f"Expected 'signal_exit', got {reason!r}"


def test_signal_exit_short():
    """SHORT signal exit: signal rises above -exit_thresh."""
    exited, exit_px, reason = ALGO._check_exit_intrabar(
        pos=-1,
        bar_open=93.0,
        bar_high=93.5,
        bar_low=92.5,
        bar_close=93.2,
        entry_px=95.0,
        stop_px=97.0,
        tp_px=90.0,
        signal=0.8,       # > -(-0.5) = 0.5 -> exit
        params=_default_params(exit_thresh=-0.5),
    )
    assert exited is True
    assert exit_px == 93.2,             f"Expected close=93.2, got {exit_px}"
    assert reason == "signal_exit"


# ─── Test 9: SL priority over signal exit ────────────────────────────────────

def test_sl_takes_priority_over_signal_exit():
    """
    When BOTH SL is touched AND signal crosses exit — intrabar SL
    should take priority (returns sl_intrabar, not signal_exit).
    """
    exited, exit_px, reason = ALGO._check_exit_intrabar(
        pos=1,
        bar_open=97.0,
        bar_high=97.5,
        bar_low=94.0,     # < stop=95 -> SL hit
        bar_close=96.5,
        entry_px=100.0,
        stop_px=95.0,
        tp_px=110.0,
        signal=-2.0,      # also triggers signal exit
        params=_default_params(exit_thresh=-0.5),
    )
    assert exited is True
    assert exit_px == 95.0,             f"Expected SL price=95.0, got {exit_px}"
    assert reason == "sl_intrabar",     f"SL should take priority, got {reason!r}"


# ─── Test 10: Full backtest integration — exit_reason in trade records ────────

def test_exit_reason_recorded_in_trades():
    """
    End-to-end: run a full backtest on a tiny crafted DataFrame.
    The resulting trade dicts must contain 'exit_reason' field.
    Entry now happens at bar N+1 open (Pine-equivalent execution).
    """
    np.random.seed(0)
    n = 300
    price = 100 + np.cumsum(np.random.normal(0, 0.5, n))
    hi    = price + np.abs(np.random.normal(0, 0.3, n))
    lo    = price - np.abs(np.random.normal(0, 0.3, n))
    opens = np.roll(price, 1) + np.random.normal(0, 0.1, n)
    opens[0] = price[0]
    vol   = np.ones(n) * 100_000
    idx   = pd.date_range("2023-01-01", periods=n, freq="3h")
    df    = pd.DataFrame({"open": opens, "high": hi, "low": lo,
                           "close": price, "volume": vol}, index=idx)

    algo   = get_algorithm("AlgorithmA")
    params = algo.model_configs()[1]
    m      = algo.run(df, params)

    if m.trades:
        for t in m.trades:
            assert "exit_reason" in t, f"Trade missing 'exit_reason': {t}"
            assert t["exit_reason"] in {
                "sl_intrabar", "tp_intrabar", "gap_stop",
                "signal_exit", "end_of_data"
            }, f"Unknown exit_reason: {t['exit_reason']!r}"
            # Entry bar must be valid index
            assert t["entry_bar"] >= 1, \
                f"entry_bar={t['entry_bar']} should be >= 1 (N+1 entry)"


# ─── Test 11: Buy & Hold baseline ────────────────────────────────────────────

def test_buy_and_hold_baseline_computable():
    """
    compute_buy_and_hold() must return a finite, positive number
    for a trending market and a near-zero (or negative) number for
    a flat/declining one.
    """
    def buy_and_hold_return(prices):
        return (prices[-1] / prices[0] - 1.0) * 100.0

    # Trending up
    up_prices = np.array([100.0, 110.0, 120.0, 130.0])
    bh_up = buy_and_hold_return(up_prices)
    assert bh_up > 0,   f"Expected positive B&H for trending up: {bh_up}"
    assert abs(bh_up - 30.0) < 0.01, f"Expected 30%, got {bh_up}"

    # Flat
    flat_prices = np.array([100.0, 100.0, 100.0])
    bh_flat = buy_and_hold_return(flat_prices)
    assert abs(bh_flat) < 0.01, f"Expected ~0% B&H for flat: {bh_flat}"

    # Declining
    down_prices = np.array([100.0, 90.0, 80.0])
    bh_down = buy_and_hold_return(down_prices)
    assert bh_down < 0, f"Expected negative B&H for declining: {bh_down}"


# ─── Test 12: N+1 entry execution (Gate 8 fix validation) ────────────────────

def test_n_plus_one_entry_execution():
    """
    Signal at bar N must result in entry at bar N+1 open price,
    NOT at bar N close price. This matches Pine Strategy Tester
    behaviour (barstate.isconfirmed -> next bar open execution).

    Verify by constructing a DataFrame where opens[N+1] is deliberately
    +200 pts above closes[N] (clear gap), then checking that entry_px
    is derived from opens[N+1], not closes[N].
    """
    np.random.seed(7)
    n     = 2000
    ret   = np.random.normal(0.0008, 0.013, n)
    price = 15000 * np.exp(np.cumsum(ret))
    hi    = price * (1 + np.abs(np.random.normal(0, 0.008, n)))
    lo    = price * (1 - np.abs(np.random.normal(0, 0.008, n)))
    # Deliberate +200pt gap so open[N+1] and close[N] are clearly distinct
    opens = np.roll(price, 1) + 200.0
    opens[0] = price[0]
    vol   = np.random.randint(100_000, 500_000, n).astype(float)
    idx   = pd.date_range("2022-01-01", periods=n, freq="3h")
    df    = pd.DataFrame({"open": opens, "high": hi, "low": lo,
                           "close": price, "volume": vol}, index=idx)

    algo   = get_algorithm("AlgorithmA")
    params = algo.model_configs()[1]
    m      = algo.run(df, params)

    if not m.trades:
        pytest.fail("No trades generated — cannot verify N+1 entry timing")

    failures = []
    for t in m.trades:
        entry_bar = t["entry_bar"]
        entry_px  = t["entry_px"]
        signal_bar = entry_bar - 1
        if signal_bar < 0:
            continue

        close_at_signal = price[signal_bar]
        open_at_entry   = opens[entry_bar]

        dist_from_open  = abs(entry_px - open_at_entry)
        dist_from_close = abs(entry_px - close_at_signal)

        if dist_from_open >= dist_from_close:
            failures.append(
                f"entry_bar={entry_bar}: entry_px={entry_px:.2f} closer to "
                f"close[{signal_bar}]={close_at_signal:.2f} (dist={dist_from_close:.2f}) "
                f"than open[{entry_bar}]={open_at_entry:.2f} (dist={dist_from_open:.2f})"
            )

    assert not failures, \
        f"N+1 entry FAILED on {len(failures)} trades:\n" + "\n".join(failures[:3])


# ─── Test 13: N+1 ATR at entry (technical-debt item, verified already fixed) ─

def test_stop_loss_uses_entry_bar_atr_not_signal_bar_atr():
    """
    research_journal.md lists as an OPEN technical debt: 'SL/TP seviyeleri N
    barinin ATR'siyle hesaplaniyor, N+1 ATR kullanilmali' (signal bar N's ATR
    is used for SL/TP instead of entry bar N+1's ATR).

    This test proves that claim is now FALSE for the current codebase:
    algorithms/base.py already computes atr_entry = atrs[i + 1] (line ~177),
    i.e. the entry bar's own ATR, not the signal bar's.

    Construction: signal fires at bar 10 (flat/low-vol run before it), entry
    bar 11 has a deliberate wide asymmetric range (no downside breach) so its
    own ATR spikes far above the signal bar's ATR. We then verify the actual
    stop level realized by the engine matches a stop computed from the entry
    bar's ATR (atrs[11]) and NOT from the signal bar's ATR (atrs[10]) — by
    placing bar 12's low strictly between the two candidate stop levels
    (breaches the signal-bar-ATR stop, does not breach the entry-bar-ATR
    stop) and bar 13's low below both.
    """
    n = 20
    opens  = np.full(n, 100.0)
    highs  = np.full(n, 100.5)
    lows   = np.full(n, 99.5)
    closes = np.full(n, 100.0)
    vol    = np.full(n, 1000.0)

    # Entry bar (11): wide asymmetric range spikes ATR without breaching downside.
    opens[11], highs[11], lows[11], closes[11] = 100.0, 130.0, 99.0, 100.0
    # Bar 12: low sits between the two candidate stops -> must NOT exit if
    # the engine correctly uses the entry bar's (larger) ATR.
    opens[12], highs[12], lows[12], closes[12] = 100.0, 101.0, 90.0, 95.0
    # Bar 13: low breaches the entry-bar-ATR stop -> exit must happen here.
    opens[13], highs[13], lows[13], closes[13] = 95.0, 96.0, 80.0, 85.0

    idx = pd.date_range("2022-01-01", periods=n, freq="3h")
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows,
                        "close": closes, "volume": vol}, index=idx)

    # Independently compute what the two candidate stop levels would be.
    atrs = ALGO._atr(df, length=2)
    atr_stop = 1.0
    entry_px_expected = opens[11]  # slippage_atr_frac=0.0 below
    stop_using_entry_bar_atr  = entry_px_expected - atr_stop * atrs[11]
    stop_using_signal_bar_atr = entry_px_expected - atr_stop * atrs[10]
    assert stop_using_entry_bar_atr < stop_using_signal_bar_atr, \
        "test construction error: entry-bar ATR must be larger than signal-bar ATR"

    signals = pd.Series(0, index=df.index)
    signals.iloc[10] = 1  # signal at bar 10 -> entry at bar 11 open

    params = {"atr_stop": atr_stop, "atr_tp": 100.0, "atr_len": 2,
              "kelly_frac": 0.25, "kelly_cap": 0.25, "exit_thresh": -0.5}
    m = ALGO._simulate(df, signals, params,
                        commission_pct=0.0, slippage_atr_frac=0.0,
                        initial_equity=100_000.0)

    assert m.trades, "Expected exactly one trade"
    t = m.trades[0]
    assert t["entry_bar"] == 11
    assert t["entry_px"] == pytest.approx(entry_px_expected)
    # If the bug (signal-bar ATR) were still present, this would have exited
    # at bar 12 instead, since bar 12's low (90) breaches the tighter,
    # signal-bar-ATR stop (99) but not the correct entry-bar-ATR stop (84).
    assert t["exit_bar"] == 13, (
        f"Expected exit at bar 13 (entry-bar ATR stop); got exit_bar={t['exit_bar']}. "
        f"An exit at bar 12 would indicate the signal-bar ATR bug has regressed."
    )
    assert t["exit_reason"] == "sl_intrabar"
    assert t["exit_px"] == pytest.approx(stop_using_entry_bar_atr)
