"""
QSES — Walk-Forward Reporter (Faz 6)
Generates:
  walk_forward_detail.csv     — per-walk metrics
  optimizer_funnel_{mkt}.csv  — Top20 -> Top5 -> final selection
  bootstrap_significance.csv  — p-values and CIs
  selection_rationale.md      — human-readable rationale per config
"""
from __future__ import annotations
import os
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from ..optimization.walk_forward import WalkForwardResult, WalkForwardEngine
from ..config.settings import REPORTS_DIR, MAX_SHARPE_DECAY, MAX_RUIN_PROBABILITY
from ..utils.logger import get_logger

logger = get_logger("WFReporter")


def _ensure_reports_dir() -> None:
    """Defensive: callers that use this module standalone (e.g. a notebook or
    the `python -c` snippet in the README) may not have created REPORTS_DIR
    yet. main.py's CLI path already does this once up front, but each
    save_* function guards for direct/standalone use too."""
    os.makedirs(REPORTS_DIR, exist_ok=True)


def save_walk_detail(results: List[WalkForwardResult]) -> str:
    _ensure_reports_dir()
    rows = []
    for r in results:
        for w in r.walks:
            rows.append({
                "market":        r.market,
                "algorithm":     r.algorithm,
                "model_id":      r.model_id,
                "walk_id":       w.walk_id,
                "train_start":   w.train_start,
                "train_end":     w.train_end,
                "test_start":    w.test_start,
                "test_end":      w.test_end,
                "train_sharpe":  w.train_sharpe,
                "test_sharpe":   w.test_sharpe,
                "test_wr":       round(w.test_wr, 4),
                "test_dd":       w.test_dd,
                "test_trades":   w.test_trades,
                "test_pnl":      w.test_pnl,
                "selected_params": str(w.selected_params),
            })
    out = os.path.join(REPORTS_DIR, "walk_forward_detail.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info(f"Walk detail CSV: {out} ({len(rows)} rows)")
    return out


def save_model_summary(results: List[WalkForwardResult]) -> str:
    _ensure_reports_dir()
    """Per-model summary table with all aggregate metrics."""
    rows = []
    for r in results:
        wf_score = WalkForwardEngine.compute_wf_score(r)
        rows.append({
            "market":             r.market,
            "algorithm":          r.algorithm,
            "model_id":           r.model_id,
            "avg_test_sharpe":    round(r.avg_test_sharpe, 4),
            "std_test_sharpe":    round(r.std_test_sharpe, 4),
            "worst_walk_dd":      round(r.worst_walk_dd, 4),
            "avg_test_wr":        round(r.avg_test_wr, 4),
            "total_test_trades":  r.total_test_trades,
            "train_test_decay":   round(r.train_test_decay, 4),
            "passes_decay_gate":  r.passes_decay_gate,
            "passes_ruin_gate":   r.passes_ruin_gate,
            "passes_sample_gate": r.passes_sample_gate,
            "mc_ruin_prob":       round(r.mc_ruin_prob, 4),
            "mc_worst_dd":        round(r.mc_worst_dd, 4),
            "mc_sharpe_ci_lo":    round(r.mc_sharpe_ci_lo, 4),
            "mc_sharpe_ci_hi":    round(r.mc_sharpe_ci_hi, 4),
            "bs_p_value":         round(r.bs_p_value, 4),
            "bs_significant":     r.bs_significant,
            "bs_sharpe_ci_lo":    round(r.bs_sharpe_ci_lo, 4),
            "bs_sharpe_ci_hi":    round(r.bs_sharpe_ci_hi, 4),
            "bh_sharpe_ci_lo":    round(r.bh_sharpe_ci_lo, 4),
            "bh_sharpe_ci_hi":    round(r.bh_sharpe_ci_hi, 4),
            "deflated_sharpe":    round(r.deflated_sharpe, 4),
            "unstable_params":    "|".join(r.unstable_params),
            "wf_score":           wf_score,
            "confidence_score":   r.confidence_score,
        })
    out = os.path.join(REPORTS_DIR, "walk_forward_summary.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info(f"WF summary CSV: {out}")
    return out


def save_bootstrap(results: List[WalkForwardResult]) -> str:
    _ensure_reports_dir()
    rows = []
    for r in results:
        rows.append({
            "market": r.market, "model_id": r.model_id,
            "strat_sharpe_ci_lo": round(r.bs_sharpe_ci_lo, 4),
            "strat_sharpe_ci_hi": round(r.bs_sharpe_ci_hi, 4),
            "bh_sharpe_ci_lo":    round(r.bh_sharpe_ci_lo, 4),
            "bh_sharpe_ci_hi":    round(r.bh_sharpe_ci_hi, 4),
            "p_value":            round(r.bs_p_value, 4),
            "significant_vs_bh":  r.bs_significant,
            "interpretation": (
                "Strategy statistically outperforms B&H (p<0.05)"
                if r.bs_significant else
                "Cannot distinguish from B&H at p=0.05"
            ),
        })
    out = os.path.join(REPORTS_DIR, "bootstrap_significance.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info(f"Bootstrap CSV: {out}")
    return out


def save_rationale(results: List[WalkForwardResult]) -> str:
    _ensure_reports_dir()
    out = os.path.join(REPORTS_DIR, "selection_rationale.md")
    algos = sorted(set(r.algorithm for r in results))
    lines = ["# QSES Walk-Forward Selection Rationale\n",
             f"Markets: {sorted(set(r.market for r in results))} | "
             f"Algorithms: {algos}\n\n"]
    for r in results:
        wf_score = WalkForwardEngine.compute_wf_score(r)
        passed   = r.passes_decay_gate and r.passes_ruin_gate and r.passes_sample_gate
        lines.append(f"## {r.market} — {r.algorithm} M{r.model_id} "
                     f"({'SELECTED' if passed else 'ELIMINATED'})\n\n")
        lines.append(f"{r.selection_rationale}\n\n")
        lines.append(f"**WalkForwardScore:** {wf_score:.1f}/100 | "
                     f"**Confidence:** {r.confidence_score:.0f}%\n\n")
        lines.append("| Walk | Train Sh | Test Sh | Test WR | Test DD | Trades |\n")
        lines.append("|------|----------|---------|---------|---------|--------|\n")
        for w in r.walks:
            lines.append(f"| {w.walk_id} | {w.train_sharpe:.2f} | "
                         f"{w.test_sharpe:.2f} | {w.test_wr:.1%} | "
                         f"{w.test_dd:.1f}% | {w.test_trades} |\n")
        lines.append("\n---\n\n")
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(lines)
    logger.info(f"Selection rationale: {out}")
    return out


def print_model_table(results: List[WalkForwardResult]) -> None:
    """EK-4 acceptance test format table."""
    algos = sorted(set(r.algorithm for r in results))
    header = f"{'Market':<8} {'Model':<6} {'Tr.Sh':>7} {'Te.Sh':>7} {'Decay':>7} {'Ruin%':>7} {'DSR':>6} {'WFS':>6} {'Status':<12}"
    print("\n" + "="*75)
    print(f"WALK-FORWARD RESULTS — {', '.join(algos)}")
    print("="*75)
    print(header)
    print("-"*75)
    for r in results:
        avg_train = float(np.mean([w.train_sharpe for w in r.walks])) if r.walks else 0
        passed    = r.passes_decay_gate and r.passes_ruin_gate and r.passes_sample_gate
        wfs       = WalkForwardEngine.compute_wf_score(r)
        status    = "PASS" if passed else "ELIMINATED"
        print(f"{r.market:<8} M{r.model_id:<5} {avg_train:>7.2f} "
              f"{r.avg_test_sharpe:>7.2f} {r.train_test_decay:>7.1%} "
              f"{r.mc_ruin_prob:>7.1%} {r.deflated_sharpe:>6.2f} "
              f"{wfs:>6.1f} {status:<12}")
    print("="*75)
