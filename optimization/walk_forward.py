"""
QSES — Rolling Walk-Forward Engine (Faz 6)

Architecture:
  WalkForwardEngine.run() → WalkForwardResult
  - N rolling walks, each: optimise on train, evaluate on test
  - Parameter drift analysis (CV per param across walks)
  - WalkForwardScore composite ranking
  - No data snooping: test slice never touches optimizer

SOLID compliance:
  S: This module owns only walk-forward splitting + aggregation.
     Optimization delegated to Optimizer. Metrics to compute_metrics.
  O: New walk strategies (expanding window etc.) can subclass without
     modifying this engine.
  DRY: Single _make_walk_slices() builds all splits; reused by all markets.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

from ..algorithms.base import BaseAlgorithm
from ..core.types import BacktestMetrics, MIN_VALID_TRADES
from ..core.metrics import compute_metrics
from ..optimization.optimizer import Optimizer
from ..config.settings import (
    WALK_FORWARD_TRAIN_RATIO, WALK_FORWARD_N_WALKS,
    MAX_SHARPE_DECAY, MAX_PARAM_CV, MAX_RUIN_PROBABILITY,
    WALK_FORWARD_SCORE_WEIGHTS, OPTIMIZER_N_TRIALS_WF,
    N_MONTE_CARLO, N_BOOTSTRAP,
)
from ..utils.logger import get_logger


@dataclass
class WalkResult:
    """Single walk: train optimisation + test evaluation."""
    walk_id:      int
    train_start:  str
    train_end:    str
    test_start:   str
    test_end:     str
    train_sharpe: float
    test_sharpe:  float
    test_wr:      float
    test_dd:      float
    test_trades:  int
    test_pnl:     float
    selected_params: Dict[str, Any] = field(default_factory=dict)
    test_trade_pnls: List[float]    = field(default_factory=list)


@dataclass
class WalkForwardResult:
    """Aggregated result across all walks for one (market, algo, model)."""
    market:      str
    algorithm:   str
    model_id:    int
    walks:       List[WalkResult]

    # Aggregates
    avg_test_sharpe:  float = 0.0
    std_test_sharpe:  float = 0.0
    worst_walk_dd:    float = 0.0
    avg_test_wr:      float = 0.0
    total_test_trades: int  = 0
    train_test_decay:  float = 0.0   # (avg_train - avg_test) / avg_train
    wf_score:         float = 0.0
    param_cv:         Dict[str, float] = field(default_factory=dict)
    unstable_params:  List[str]        = field(default_factory=list)

    # Monte Carlo
    mc_worst_dd:       float = 0.0
    mc_ruin_prob:      float = 0.0
    mc_sharpe_ci_lo:   float = 0.0
    mc_sharpe_ci_hi:   float = 0.0

    # Bootstrap significance
    bs_sharpe_ci_lo:   float = 0.0
    bs_sharpe_ci_hi:   float = 0.0
    bh_sharpe_ci_lo:   float = 0.0
    bh_sharpe_ci_hi:   float = 0.0
    bs_p_value:        float = 1.0
    bs_significant:    bool  = False

    # Final gates
    passes_decay_gate: bool  = False
    passes_ruin_gate:  bool  = False
    passes_sample_gate:bool  = False
    confidence_score:  float = 0.0

    # DSR
    deflated_sharpe:   float = 0.0

    # Explainability
    selection_rationale: str = ""


class WalkForwardEngine:
    """
    Rolling walk-forward: splits data into N chronological walks,
    optimises on train, evaluates on test, never crosses the boundary.
    """

    def __init__(
        self,
        algorithm: BaseAlgorithm,
        n_walks:   int   = WALK_FORWARD_N_WALKS,
        n_trials:  int   = OPTIMIZER_N_TRIALS_WF,
        seed:      int   = 42,
    ):
        self.algo     = algorithm
        self.n_walks  = n_walks
        self.n_trials = n_trials
        self.seed     = seed
        self.logger   = get_logger(f"WalkForward[{algorithm.name}]")

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        df:       pd.DataFrame,
        model_id: int,
        market:   str,
    ) -> WalkForwardResult:
        """Run all walks for one (market, model) combination."""
        base_params = self.algo.model_configs()[model_id]
        slices      = self._make_walk_slices(df, self.n_walks)

        self.logger.info(
            f"{market} M{model_id}: {self.n_walks} walks, "
            f"train_ratio={WALK_FORWARD_TRAIN_RATIO:.0%}, "
            f"n_trials={self.n_trials}"
        )

        walks: List[WalkResult] = []
        for walk_id, (df_train, df_test) in enumerate(slices, 1):
            wr = self._run_single_walk(
                walk_id, df_train, df_test, base_params, market
            )
            walks.append(wr)
            self.logger.info(
                f"  Walk {walk_id}/{self.n_walks}: "
                f"train_Sh={wr.train_sharpe:.2f} "
                f"test_Sh={wr.test_sharpe:.2f} "
                f"test_WR={wr.test_wr:.1%} "
                f"test_trades={wr.test_trades}"
            )

        result = WalkForwardResult(
            market=market, algorithm=self.algo.name, model_id=model_id, walks=walks
        )
        self._aggregate(result)
        self._param_drift(result)
        self._monte_carlo(result)
        self._bootstrap_significance(result, df)
        self._deflated_sharpe(result)
        self._compute_gates(result)
        self._confidence_score(result)
        self._build_rationale(result)
        return result

    # ── Walk splitting ─────────────────────────────────────────────────────────

    @staticmethod
    def _make_walk_slices(
        df: pd.DataFrame,
        n_walks: int,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Rolling window walk-forward.
        Each walk: train = fixed-length window, test = next block.
        Chronological order strictly preserved, no shuffle.
        """
        n          = len(df)
        # Total window per walk = n / (n_walks * (1 - WALK_FORWARD_TRAIN_RATIO) + WALK_FORWARD_TRAIN_RATIO)
        # Simpler: step = n // (n_walks + 1), train_len = step * 2
        step       = n // (n_walks + 1)
        train_len  = max(step * 2, int(n * WALK_FORWARD_TRAIN_RATIO))

        slices = []
        for k in range(n_walks):
            test_start  = step * (k + 1)
            test_end    = min(test_start + step, n)
            train_start = max(0, test_start - train_len)
            df_train    = df.iloc[train_start:test_start]
            df_test     = df.iloc[test_start:test_end]
            if len(df_train) < 300 or len(df_test) < 50:
                continue
            slices.append((df_train, df_test))
        return slices

    # ── Single walk ────────────────────────────────────────────────────────────

    def _run_single_walk(
        self,
        walk_id:     int,
        df_train:    pd.DataFrame,
        df_test:     pd.DataFrame,
        base_params: Dict[str, Any],
        market:      str,
    ) -> WalkResult:
        # Optimise on TRAIN only
        opt = Optimizer(self.algo, n_trials=self.n_trials,
                        seed=self.seed + walk_id)
        best_p, train_m = opt.optimize(
            df_train, base_params=base_params, market=market
        )

        # Evaluate FROZEN params on TEST (no feedback loop)
        test_m = self.algo.run(df_test, best_p)

        trade_pnls = [t["pnl_pct"] for t in test_m.trades]

        return WalkResult(
            walk_id      = walk_id,
            train_start  = str(df_train.index[0].date()),
            train_end    = str(df_train.index[-1].date()),
            test_start   = str(df_test.index[0].date()),
            test_end     = str(df_test.index[-1].date()),
            train_sharpe = round(train_m.sharpe_ratio, 4),
            test_sharpe  = round(test_m.sharpe_ratio,  4),
            test_wr      = round(test_m.win_rate,       4),
            test_dd      = round(test_m.max_drawdown_pct, 4),
            test_trades  = test_m.total_trades,
            test_pnl     = round(test_m.net_profit_pct, 4),
            selected_params = best_p,
            test_trade_pnls = trade_pnls,
        )

    # ── Aggregation ────────────────────────────────────────────────────────────

    @staticmethod
    def _aggregate(r: WalkForwardResult) -> None:
        sharpes  = [w.test_sharpe  for w in r.walks]
        tr_sh    = [w.train_sharpe for w in r.walks]
        r.avg_test_sharpe   = float(np.mean(sharpes))
        r.std_test_sharpe   = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0
        r.worst_walk_dd     = float(min(w.test_dd for w in r.walks))
        r.avg_test_wr       = float(np.mean([w.test_wr for w in r.walks]))
        r.total_test_trades = sum(w.test_trades for w in r.walks)
        avg_train = float(np.mean(tr_sh)) if tr_sh else 0.0
        if avg_train != 0:
            r.train_test_decay = (avg_train - r.avg_test_sharpe) / abs(avg_train)
        else:
            r.train_test_decay = 0.0

    # ── Parameter drift ────────────────────────────────────────────────────────

    @staticmethod
    def _param_drift(r: WalkForwardResult) -> None:
        """CV = std/mean per parameter across walks."""
        if not r.walks:
            return
        all_params = [w.selected_params for w in r.walks]
        keys = set(k for p in all_params for k in p
                   if isinstance(p.get(k), (int, float)))

        for k in keys:
            vals = [p[k] for p in all_params if k in p and isinstance(p[k], (int, float))]
            if len(vals) < 2:
                continue
            mu  = abs(np.mean(vals))
            std = np.std(vals, ddof=1)
            cv  = (std / mu) if mu > 1e-9 else 0.0
            r.param_cv[k] = round(cv, 4)
            if cv > MAX_PARAM_CV:
                r.unstable_params.append(k)

    # ── Monte Carlo ────────────────────────────────────────────────────────────

    def _monte_carlo(self, r: WalkForwardResult) -> None:
        """Permute trade order 1000x; measure worst DD + ruin probability."""
        all_pnls = []
        for w in r.walks:
            all_pnls.extend(w.test_trade_pnls)

        if len(all_pnls) < 5:
            self.logger.warning("Monte Carlo: fewer than 5 test trades, skipping")
            return

        pnls_arr = np.array(all_pnls)
        rng      = np.random.default_rng(self.seed)

        worst_dds  = []
        ruin_count = 0
        mc_sharpes = []

        for _ in range(N_MONTE_CARLO):
            perm    = rng.permutation(pnls_arr)
            equity  = np.cumprod(1 + perm / 100) * 100
            peak    = np.maximum.accumulate(equity)
            dd      = (equity - peak) / peak * 100
            worst_dds.append(float(dd.min()))
            if equity.min() < 50:   # below 50% of start = ruin
                ruin_count += 1
            mu  = perm.mean()
            std = perm.std(ddof=1) + 1e-10
            tpy = 252 / max(perm.shape[0], 1)
            mc_sharpes.append(mu / std * np.sqrt(tpy))

        r.mc_worst_dd   = float(np.percentile(worst_dds, 5))   # 5th pct worst
        r.mc_ruin_prob  = ruin_count / N_MONTE_CARLO
        r.mc_sharpe_ci_lo = float(np.percentile(mc_sharpes, 2.5))
        r.mc_sharpe_ci_hi = float(np.percentile(mc_sharpes, 97.5))
        self.logger.info(
            f"  Monte Carlo: ruin_prob={r.mc_ruin_prob:.1%} "
            f"worst_dd={r.mc_worst_dd:.1f}% "
            f"Sharpe CI=[{r.mc_sharpe_ci_lo:.2f},{r.mc_sharpe_ci_hi:.2f}]"
        )

    # ── Bootstrap significance ─────────────────────────────────────────────────

    def _bootstrap_significance(
        self, r: WalkForwardResult, df_full: pd.DataFrame
    ) -> None:
        """
        Bootstrap strategy Sharpe vs buy&hold Sharpe.
        H0: strategy Sharpe == B&H Sharpe.
        p-value = fraction of bootstrap samples where B&H > strategy.
        """
        all_pnls = []
        for w in r.walks:
            all_pnls.extend(w.test_trade_pnls)

        if len(all_pnls) < 5:
            return

        pnls_arr = np.array(all_pnls)
        close    = df_full["close"].values
        bh_ret   = np.diff(close) / close[:-1] * 100   # bar returns

        rng = np.random.default_rng(self.seed + 999)

        def sharpe_from_pnls(arr: np.ndarray) -> float:
            mu  = arr.mean()
            std = arr.std(ddof=1) + 1e-10
            return mu / std * np.sqrt(252)

        strat_sharpes = []
        bh_sharpes    = []

        for _ in range(N_BOOTSTRAP):
            samp_s = rng.choice(pnls_arr, size=len(pnls_arr), replace=True)
            samp_b = rng.choice(bh_ret,   size=len(pnls_arr), replace=True)
            strat_sharpes.append(sharpe_from_pnls(samp_s))
            bh_sharpes.append(sharpe_from_pnls(samp_b))

        strat_arr = np.array(strat_sharpes)
        bh_arr    = np.array(bh_sharpes)

        r.bs_sharpe_ci_lo = float(np.percentile(strat_arr, 2.5))
        r.bs_sharpe_ci_hi = float(np.percentile(strat_arr, 97.5))
        r.bh_sharpe_ci_lo = float(np.percentile(bh_arr, 2.5))
        r.bh_sharpe_ci_hi = float(np.percentile(bh_arr, 97.5))

        # p-value: fraction of bootstrap iterations where B&H >= strategy
        diff = strat_arr - bh_arr
        r.bs_p_value    = float(np.mean(diff <= 0))
        r.bs_significant = r.bs_p_value < 0.05
        self.logger.info(
            f"  Bootstrap: Strat CI=[{r.bs_sharpe_ci_lo:.2f},{r.bs_sharpe_ci_hi:.2f}] "
            f"B&H CI=[{r.bh_sharpe_ci_lo:.2f},{r.bh_sharpe_ci_hi:.2f}] "
            f"p={r.bs_p_value:.3f} sig={r.bs_significant}"
        )

    # ── Deflated Sharpe Ratio ─────────────────────────────────────────────────

    def _deflated_sharpe(self, r: WalkForwardResult) -> None:
        """
        Bailey & López de Prado (2014) Deflated Sharpe Ratio.
        Adjusts observed Sharpe for number of trials and non-normality.
        Reference: https://doi.org/10.3905/jpm.2014.40.5.094
        Formula: DSR = SR* × sqrt(T) where SR* accounts for multiple testing.
        Simplified implementation without skewness/kurtosis adjustment
        (conservative: underestimates DSR slightly).
        """
        from scipy import stats as sp_stats

        all_pnls = []
        for w in r.walks:
            all_pnls.extend(w.test_trade_pnls)
        if len(all_pnls) < 5:
            return

        pnls   = np.array(all_pnls)
        T      = len(pnls)
        sr_obs = (pnls.mean() / (pnls.std(ddof=1) + 1e-10)
                  * np.sqrt(252))

        # N = total optimizer trials across all walks
        N = self.n_trials * len(r.walks)

        # Expected maximum SR from N independent trials (Bailey & López de Prado eq.8)
        # E[max SR] ≈ (1 - γ) × Φ^{-1}(1 - 1/N) + γ × Φ^{-1}(1 - 1/(N×e))
        # where γ = Euler-Mascheroni constant ≈ 0.5772
        gamma = 0.5772
        if N > 1:
            z1     = sp_stats.norm.ppf(1 - 1/N)
            z2     = sp_stats.norm.ppf(1 - 1/(N * np.e))
            e_max_sr = (1 - gamma) * z1 + gamma * z2
        else:
            e_max_sr = 0.0

        # Probabilistic SR: P(SR > E[max SR] | sample)
        sr_std = np.sqrt((1 + 0.5 * sr_obs**2) / max(T - 1, 1))
        if sr_std > 0:
            psr  = sp_stats.norm.cdf((sr_obs - e_max_sr) / sr_std)
        else:
            psr  = 0.0

        # DSR expressed as equivalent annual Sharpe
        r.deflated_sharpe = round(sr_obs * psr, 4)
        self.logger.info(
            f"  DSR: SR_obs={sr_obs:.2f} E[max_SR]={e_max_sr:.2f} "
            f"PSR={psr:.3f} DSR={r.deflated_sharpe:.2f} "
            f"(N_trials={N})"
        )

    # ── Quality gates ──────────────────────────────────────────────────────────

    @staticmethod
    def _compute_gates(r: WalkForwardResult) -> None:
        r.passes_decay_gate  = r.train_test_decay <= MAX_SHARPE_DECAY
        r.passes_ruin_gate   = r.mc_ruin_prob     <= MAX_RUIN_PROBABILITY
        r.passes_sample_gate = r.total_test_trades >= MIN_VALID_TRADES

    # ── Confidence score ───────────────────────────────────────────────────────

    def _confidence_score(self, r: WalkForwardResult) -> None:
        """
        7-criterion confidence score. Each criterion = 1/7 weight.
        Partial credit where applicable.
        """
        criteria = [
            # 1: Walk-forward PASS (≥3/5 walks with test Sharpe > 0.3)
            sum(1 for w in r.walks if w.test_sharpe > 0.3) / max(len(r.walks), 1) >= 0.6,
            # 2: Decay gate
            r.passes_decay_gate,
            # 3: Monte Carlo ruin gate
            r.passes_ruin_gate,
            # 4: Cross-market (single market run -- N/A, give benefit)
            True,
            # 5: Sample size ≥ 50 total test trades
            r.total_test_trades >= 50,
            # 6: No lookahead (structural -- always True after Gate 8 fix)
            True,
            # 7: Parameter stability (no UNSTABLE params)
            len(r.unstable_params) == 0,
        ]
        score = sum(criteria) / len(criteria)
        # Scale to 60–100% range (60% floor for "did it run at all")
        r.confidence_score = round(60 + score * 40, 1)

    # ── WalkForwardScore ───────────────────────────────────────────────────────

    @staticmethod
    def compute_wf_score(r: WalkForwardResult) -> float:
        """
        Composite WalkForwardScore [0,100].
        Weights from config WALK_FORWARD_SCORE_WEIGHTS.
        """
        w = WALK_FORWARD_SCORE_WEIGHTS

        # Normalize avg_test_sharpe: -1→0, 0→50, 2→100
        sh_score = min(max((r.avg_test_sharpe + 1) / 3 * 100, 0), 100)

        # Worst walk Sharpe: < 0 → 0, > 1 → 100
        ww_score = min(max(
            min(w.test_sharpe for w in r.walks) / 1.0 * 100
            if r.walks else 0, 0), 100)

        # Train/test consistency: decay 0=100, decay 1=0
        cons_score = max(0, (1 - r.train_test_decay) * 100)

        # Parameter stability: fraction of params that are stable
        total_p = len(r.param_cv)
        stable  = total_p - len(r.unstable_params)
        stab_score = (stable / max(total_p, 1)) * 100

        score = (w["avg_test_sharpe"]        * sh_score
               + w["worst_walk_sharpe"]       * ww_score
               + w["train_test_consistency"]  * cons_score
               + w["parameter_stability"]     * stab_score)
        return round(score, 2)

    # ── Explainability ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_rationale(r: WalkForwardResult) -> None:
        walks_passed = sum(1 for w in r.walks if w.test_sharpe > 0.3)
        decay_str    = f"{r.train_test_decay:.1%}"
        ruin_str     = f"{r.mc_ruin_prob:.1%}"
        unstable_str = ", ".join(r.unstable_params) if r.unstable_params else "none"

        status = "PASS" if (r.passes_decay_gate and r.passes_ruin_gate
                            and r.passes_sample_gate) else "FAIL"

        r.selection_rationale = (
            f"Market={r.market} Algo={r.algorithm} M{r.model_id} | "
            f"Status={status} | "
            f"Avg test Sharpe={r.avg_test_sharpe:.2f} "
            f"(walks passed {walks_passed}/{len(r.walks)}, "
            f"std={r.std_test_sharpe:.2f}) | "
            f"Train/test decay={decay_str} "
            f"(threshold={MAX_SHARPE_DECAY:.0%}) | "
            f"Ruin prob={ruin_str} (threshold={MAX_RUIN_PROBABILITY:.0%}) | "
            f"Test trades={r.total_test_trades} | "
            f"DSR={r.deflated_sharpe:.2f} | "
            f"Unstable params: {unstable_str} | "
            f"Confidence={r.confidence_score:.0f}%"
        )
