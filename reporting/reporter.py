"""
QSES — Reporting Engine
Generates:
  1. Per-backtest chart: price (blue) + equity (red) + trade markers
  2. Heatmap: market × algorithm sharpe
  3. Cross-market comparison bar chart
  4. Parameter stability plot
  5. Final ranking table (HTML)
"""
from __future__ import annotations
import os
import json
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from ..core.types import BacktestResult, RankingRow
from ..config.settings import CHART_THEME, CHART_WIDTH, CHART_HEIGHT, CHARTS_DIR, REPORTS_DIR
from ..utils.logger import get_logger

logger = get_logger("Reporter")


class Reporter:

    def __init__(self):
        os.makedirs(CHARTS_DIR, exist_ok=True)
        os.makedirs(REPORTS_DIR, exist_ok=True)

    # ── Per-Backtest Chart ────────────────────────────────────────────────────

    def chart_backtest(self, result: BacktestResult) -> Optional[str]:
        """
        Main chart: blue = original price, red = equity curve.
        Also plots buy/sell markers, SL/TP lines, drawdown regions.
        Returns saved file path.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.error("plotly not installed")
            return None

        m  = result.metrics
        if not m.trades or not m.equity_curve:
            return None

        safe_label = result.label.replace(" | ", "_").replace("!", "_")
        out_path   = os.path.join(CHARTS_DIR, f"{safe_label}.html")

        # Price series (if available) else reconstruct from trades
        if result.price_series is not None:
            price_idx = result.price_series.index
            price_vals= result.price_series.values
        else:
            price_idx = list(range(len(m.equity_curve)))
            price_vals= np.linspace(100, 100 * (1 + m.net_profit_pct/100),
                                    len(m.equity_curve))

        # Build equity index aligned to price
        eq_idx  = np.linspace(0, len(price_idx)-1, len(m.equity_curve)).astype(int)
        eq_idx  = np.clip(eq_idx, 0, len(price_idx)-1)
        eq_x    = [price_idx[i] for i in eq_idx]
        eq_norm = np.array(m.equity_curve) / m.equity_curve[0] * price_vals[0]

        # Drawdown regions
        eq_arr  = np.array(m.equity_curve)
        peak    = np.maximum.accumulate(eq_arr)
        dd_pct  = (eq_arr - peak) / peak * 100

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            subplot_titles=["Price + Equity", "Drawdown %"],
            vertical_spacing=0.06
        )

        # ── Blue: original price ──────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=list(price_idx), y=price_vals,
            name="Market Price", line=dict(color="#4A90D9", width=1.5),
            opacity=0.85
        ), row=1, col=1)

        # ── Red: equity curve ─────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=eq_x, y=eq_norm,
            name="Strategy Equity", line=dict(color="#E05252", width=2),
        ), row=1, col=1)

        # ── Trade markers ─────────────────────────────────────────────────────
        buy_x, buy_y, sell_x, sell_y = [], [], [], []
        sl_x,  sl_y,  tp_x,  tp_y   = [], [], [], []

        for t in m.trades:
            eb = t.get("entry_bar", 0)
            xb = t.get("exit_bar",  0)
            if eb < len(price_idx) and xb < len(price_idx):
                ep = price_vals[min(eb, len(price_vals)-1)]
                xp = price_vals[min(xb, len(price_vals)-1)]
                if t["direction"] == 1:
                    buy_x.append(price_idx[eb])
                    buy_y.append(ep)
                    sell_x.append(price_idx[xb])
                    sell_y.append(xp)
                else:
                    sell_x.append(price_idx[eb])
                    sell_y.append(ep)
                    buy_x.append(price_idx[xb])
                    buy_y.append(xp)

        if buy_x:
            fig.add_trace(go.Scatter(x=buy_x, y=buy_y, mode="markers",
                name="Buy (Long)", marker=dict(symbol="triangle-up", color="#2ECC71",
                                          size=9, line=dict(width=1, color="#27AE60"))),
                row=1, col=1)
        if sell_x:
            fig.add_trace(go.Scatter(x=sell_x, y=sell_y, mode="markers",
                name="Sell (Short)", marker=dict(symbol="triangle-down", color="#E74C3C",
                                           size=9, line=dict(width=1, color="#C0392B"))),
                row=1, col=1)

        # ── Drawdown fill ─────────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=eq_x, y=dd_pct.tolist(),
            name="Drawdown %",
            fill="tozeroy",
            line=dict(color="#E05252", width=1),
            fillcolor="rgba(224,82,82,0.25)",
        ), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

        # ── Annotations: summary stats ────────────────────────────────────────
        stats_text = (
            f"WR: {m.win_rate:.1%}  |  Trades: {m.total_trades}  |  "
            f"Net P&L: {m.net_profit_pct:+.1f}%  |  Sharpe: {m.sharpe_ratio:.2f}  |  "
            f"Max DD: {m.max_drawdown_pct:.1f}%  |  Robustness: {m.robustness_score:.0f}"
        )
        fig.add_annotation(
            text=stats_text, xref="paper", yref="paper",
            x=0.01, y=1.02, showarrow=False,
            font=dict(size=11, color="#AAAAAA"),
            align="left"
        )

        fig.update_layout(
            template=CHART_THEME,
            title=dict(text=result.label, font=dict(size=14)),
            width=CHART_WIDTH, height=CHART_HEIGHT,
            legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0),
            margin=dict(l=60, r=40, t=80, b=40),
            hovermode="x unified",
        )

        fig.write_html(out_path)
        logger.info(f"Chart saved: {out_path}")
        return out_path

    # ── Heatmap ───────────────────────────────────────────────────────────────

    def heatmap_sharpe(self, results: List[BacktestResult], output_suffix: str = "") -> Optional[str]:
        """
        Heatmap: rows=markets, cols=algorithms, value=avg Sharpe.
        Cells backed by fewer than MIN_VALID_TRADES are shown in grey
        with an 'insufficient data (N=X)' annotation instead of a
        misleading colour signal.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        from ..core.types import MIN_VALID_TRADES

        markets = sorted(set(r.market    for r in results))
        algos   = sorted(set(r.algorithm for r in results))

        matrix      = np.full((len(markets), len(algos)), np.nan)
        text_matrix = [["" for _ in algos] for _ in markets]
        # Separate mask: True = cell has valid data
        valid_mask  = np.zeros((len(markets), len(algos)), dtype=bool)

        for i, mkt in enumerate(markets):
            for j, alg in enumerate(algos):
                all_sub   = [r for r in results if r.market == mkt and r.algorithm == alg]
                valid_sub = [r for r in all_sub if r.metrics.is_valid_sample]
                total_trades = sum(r.metrics.total_trades for r in all_sub)

                if valid_sub:
                    avg_sharpe       = np.mean([r.metrics.sharpe_ratio for r in valid_sub])
                    matrix[i, j]     = avg_sharpe
                    valid_mask[i, j] = True
                    text_matrix[i][j]= f"{avg_sharpe:.2f}"
                else:
                    # Grey cell — show trade count so user knows why it's blank
                    text_matrix[i][j] = f"n={total_trades}<br>(insuff.)"

        # Layer 1: valid cells (coloured)
        z_valid = np.where(valid_mask, matrix, np.nan)
        # Layer 2: invalid cells (grey overlay)
        z_grey  = np.where(~valid_mask, 0, np.nan)

        fig = go.Figure()

        fig.add_trace(go.Heatmap(
            z=z_valid, x=algos, y=markets,
            colorscale="RdYlGn", zmid=0,
            zmin=-1, zmax=2,
            text=[[text_matrix[i][j] if valid_mask[i][j] else "" for j in range(len(algos))]
                  for i in range(len(markets))],
            texttemplate="%{text}",
            colorbar=dict(title="Sharpe"),
            name="Valid",
            showscale=True,
        ))

        # Grey overlay for insufficient-data cells
        fig.add_trace(go.Heatmap(
            z=z_grey, x=algos, y=markets,
            colorscale=[[0, "rgba(80,80,80,0.6)"], [1, "rgba(80,80,80,0.6)"]],
            showscale=False,
            text=[[text_matrix[i][j] if not valid_mask[i][j] else "" for j in range(len(algos))]
                  for i in range(len(markets))],
            texttemplate="%{text}",
            textfont=dict(color="rgba(200,200,200,0.8)", size=10),
            name=f"Insufficient data (N < {MIN_VALID_TRADES} trades)",
            hovertemplate="<b>%{y} x %{x}</b><br>Insufficient sample<extra></extra>",
        ))

        fig.update_layout(
            template=CHART_THEME,
            title=f"Sharpe Ratio Heatmap - Market x Algorithm (grey = N < {MIN_VALID_TRADES} trades)",
            width=1000, height=500,
        )

        out = os.path.join(CHARTS_DIR, f"heatmap_sharpe{output_suffix}.html")
        fig.write_html(out)
        logger.info(f"Heatmap saved: {out}")
        return out

    # ── Cross-Market Comparison Bar Chart ────────────────────────────────────

    def bar_cross_market(self, results: List[BacktestResult], metric: str = "sharpe_ratio") -> Optional[str]:
        """
        Bar chart comparing metric across markets per algorithm.
        Bars backed by insufficient samples are rendered as semi-transparent
        with diagonal stripes and labelled in the legend separately.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        from ..core.types import MIN_VALID_TRADES

        markets = sorted(set(r.market    for r in results))
        algos   = sorted(set(r.algorithm for r in results))
        colors  = ["#4A90D9", "#2ECC71", "#E74C3C", "#F39C12"]

        fig = go.Figure()
        for j, alg in enumerate(algos):
            vals_valid   = []
            vals_invalid = []
            labels       = []

            for mkt in markets:
                all_sub   = [r for r in results if r.market == mkt and r.algorithm == alg]
                valid_sub = [r for r in all_sub if r.metrics.is_valid_sample]

                if valid_sub:
                    val = np.mean([getattr(r.metrics, metric) for r in valid_sub])
                    vals_valid.append(val)
                    vals_invalid.append(None)
                else:
                    # Compute raw metric anyway (for display), but flag it
                    all_w_trades = [r for r in all_sub if r.metrics.total_trades > 0]
                    val = (np.mean([getattr(r.metrics, metric) for r in all_w_trades])
                           if all_w_trades else 0.0)
                    vals_valid.append(None)
                    vals_invalid.append(val)

            # Valid bars (solid colour)
            fig.add_trace(go.Bar(
                name=alg, x=markets, y=vals_valid,
                marker_color=colors[j % len(colors)],
                legendgroup=alg,
            ))
            # Invalid bars (faded, pattern)
            fig.add_trace(go.Bar(
                name=f"{alg} (N<{MIN_VALID_TRADES}, unreliable)",
                x=markets, y=vals_invalid,
                marker=dict(
                    color=colors[j % len(colors)],
                    opacity=0.3,
                    pattern=dict(shape="/", fgcolor="white", size=6),
                ),
                legendgroup=f"{alg}_invalid",
                showlegend=any(v is not None for v in vals_invalid),
            ))

        fig.update_layout(
            template=CHART_THEME,
            title=f"Cross-Market Comparison - {metric} (faded = N<{MIN_VALID_TRADES} trades, unreliable)",
            barmode="group",
            width=CHART_WIDTH, height=500,
            yaxis_title=metric,
        )
        out = os.path.join(CHARTS_DIR, f"cross_market_{metric}.html")
        fig.write_html(out)
        return out

    # ── Final Ranking Table ───────────────────────────────────────────────────

    def render_ranking_table(self, rows: List[RankingRow]) -> str:
        """Generates a self-contained HTML ranking table."""
        out_path = os.path.join(REPORTS_DIR, "ranking_table.html")

        html_rows = ""
        for i, r in enumerate(rows):
            sel    = "[SELECTED]" if r.selected else ""
            sel_cls= "selected-row" if r.selected else ""
            html_rows += f"""
            <tr class="{sel_cls}">
                <td>{i+1}</td>
                <td>{r.algorithm}</td>
                <td>M{r.model_id}</td>
                <td>{r.avg_win_rate:.1%}</td>
                <td>{r.avg_net_pnl:+.1f}%</td>
                <td>{r.avg_sharpe:.2f}</td>
                <td>{r.avg_max_dd:.1f}%</td>
                <td>{r.avg_robustness:.0f}</td>
                <td>{r.stability_score:.0f}</td>
                <td>{r.markets_passed}/{r.markets_tested}</td>
                <td>{sel}</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>QSES - Ranking Table</title>
<style>
  body {{ font-family: 'Courier New', monospace; background: #0d0f1a; color: #ccc; padding: 20px; }}
  h1   {{ color: #4A90D9; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th   {{ background: #1a1d2e; color: #4A90D9; padding: 10px 12px; text-align: left; border-bottom: 2px solid #2a2d3e; }}
  td   {{ padding: 8px 12px; border-bottom: 1px solid #1a1d2e; }}
  tr:hover {{ background: #1a1d2e; }}
  .selected-row {{ background: #0d1f0d; border-left: 3px solid #2ECC71; }}
  .selected-row td {{ color: #2ECC71; }}
</style>
</head>
<body>
<h1>QSES - Cross-Market Ranking Report</h1>
<p style="color:#888">Ranked by Stability Score (cross-market Sharpe consistency + robustness).
NOT by maximum profit. Selected configs pass all quality gates.</p>
<table>
  <thead>
    <tr>
      <th>#</th><th>Algorithm</th><th>Model</th>
      <th>Avg WR</th><th>Avg P&L</th><th>Avg Sharpe</th>
      <th>Avg MaxDD</th><th>Robustness</th><th>Stability</th>
      <th>Markets</th><th>Status</th>
    </tr>
  </thead>
  <tbody>{html_rows}</tbody>
</table>
</body>
</html>"""

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"Ranking table saved: {out_path}")
        return out_path

    # ── Summary CSV ──────────────────────────────────────────────────────────

    def save_results_csv(self, results: List[BacktestResult]) -> str:
        rows = []
        for r in results:
            row = {
                "market":      r.market,
                "timeframe":   r.timeframe,
                "period_yrs":  r.period_yrs,
                "algorithm":   r.algorithm,
                "model_id":    r.model_id,
                # Sample validity columns (FAZ 2)
                "is_valid_sample":  r.metrics.is_valid_sample,
                "exclusion_reason": r.metrics.exclusion_reason,
            }
            row.update(r.metrics.to_dict())
            rows.append(row)

        df = pd.DataFrame(rows)

        # Reorder: validity columns right after identifiers, before metrics
        id_cols      = ["market", "timeframe", "period_yrs", "algorithm", "model_id"]
        validity_cols= ["is_valid_sample", "exclusion_reason"]
        metric_cols  = [c for c in df.columns if c not in id_cols + validity_cols]
        df = df[id_cols + validity_cols + metric_cols]

        out = os.path.join(REPORTS_DIR, "all_results.csv")
        df.to_csv(out, index=False)
        logger.info(f"Results CSV saved: {out}")

        # Log sample quality summary
        total   = len(df)
        valid_n = df["is_valid_sample"].sum()
        zero_n  = (df["total_trades"] == 0).sum()
        insuff_n= ((df["total_trades"] > 0) & (~df["is_valid_sample"])).sum()
        logger.info(f"Sample quality: {valid_n}/{total} valid | "
                    f"{zero_n} zero-trade | {insuff_n} insufficient (<{10} trades)")
        return out

    # ── Seed vs Optimized Comparison CSV (FAZ 3) ─────────────────────────────

    def save_seed_comparison(self, comparisons: List[Dict]) -> str:
        """
        Saves seed_vs_optimized_comparison.csv.

        Each entry in comparisons is a dict with keys:
            market, timeframe, period_yrs, algorithm, model_id,
            seed_wr, seed_sharpe, seed_pnl, seed_dd, seed_rob, seed_trades,
            opt_wr,  opt_sharpe,  opt_pnl,  opt_dd,  opt_rob,  opt_trades,
            winner (seed/optimizer/tie), delta_rob
        """
        import pandas as pd
        if not comparisons:
            return ""

        df = pd.DataFrame(comparisons)
        out = os.path.join(REPORTS_DIR, "seed_vs_optimized_comparison.csv")
        df.to_csv(out, index=False, float_format="%.4f")
        logger.info(f"Seed comparison CSV saved: {out} ({len(df)} rows)")

        # Summary log
        seed_wins = (df["winner"] == "seed").sum()
        opt_wins  = (df["winner"] == "optimizer").sum()
        ties      = (df["winner"] == "tie").sum()
        logger.info(f"Seed vs Optimizer: seed wins={seed_wins} | opt wins={opt_wins} | ties={ties}")
        return out
