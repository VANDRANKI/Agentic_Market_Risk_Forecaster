"""
Multi-agent risk analysis pipeline using CrewAI.

Four agents run in strict sequence, each with a well-defined role, deep domain
backstory, precise task instructions, and structured expected output:

  1. MarketMonitorAgent
     Role: Senior market surveillance analyst.
     Task: Interpret the portfolio's return distribution, volatility regime,
           and correlation structure. Flag whether current market conditions
           are elevated relative to the historical baseline.

  2. AnomalyDetectorAgent
     Role: Quantitative anomaly and tail-event specialist.
     Task: Interpret detected anomaly dates and their MAGNITUDE, compare
           Z-score vs Isolation Forest agreement, classify anomalies as
           isolated vs clustered, and infer likely economic context.

  3. RiskForecasterAgent
     Role: Market risk measurement expert (VaR/ES/GARCH).
     Task: Compare VaR estimates across 4 methods, compute and interpret
           the ES/VaR ratio for fat-tail severity, interpret GARCH persistence
           as a forward-looking signal, and deliver a plain-language verdict
           on whether the model passes or fails both backtests.

  4. PortfolioOptimizerAgent
     Role: Risk-aware portfolio construction specialist.
     Task: Compare current portfolio vs Max Sharpe vs Min Vol by specific
           WEIGHTS and risk metrics, quantify the volatility and Sharpe
           improvement available, and give a concrete ticker-level
           reallocation recommendation tied to the current vol regime.

Orchestration rules:
  - All computations are pre-done. The LLM only interprets and explains.
  - Agents 2, 3, 4 receive prior agents' outputs via CrewAI context chaining.
  - No em-dashes anywhere. No emojis. Precise, quantitative language.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Groq direct call helper
# ---------------------------------------------------------------------------


def _call_groq_agent(client, model: str, system: str, task_desc: str, prior_outputs: list) -> str:
    """Call Groq API directly for one agent step, passing prior outputs as context."""
    messages = [{"role": "system", "content": system}]
    user_content = task_desc
    if prior_outputs:
        prior_text = "\n\n".join(
            f"Previous analysis:\n{o}" for o in prior_outputs if o
        )
        user_content = prior_text + "\n\n" + task_desc
    messages.append({"role": "user", "content": user_content})
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=1000,
        )
        raw = response.choices[0].message.content or ""
        return raw.strip().replace("\u2014", " -- ").replace("\u2013", " - ")
    except Exception as exc:
        logger.error("Groq API call failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Context compilation
# ---------------------------------------------------------------------------


def compile_analysis_context(
    tickers: list[str],
    weights: dict[str, float],
    prices,
    returns,
    portfolio_returns,
    risk_metrics: dict,
    backtest_results: dict,
    portfolio_results: dict,
    anomalies: dict,
) -> dict:
    """
    Compile all pre-computed analytics into the single structured context dict
    that all 4 agents draw from.

    Nothing is computed here that was not already computed in the risk engine.
    This function only organizes and formats the pre-computed values.

    The context includes enriched anomaly data (return magnitudes),
    ES/VaR ratios, model divergence metrics, individual asset statistics,
    and full weight tables so agents have everything they need without gaps.
    """
    from risk.engine import return_stats, rolling_volatility

    r = portfolio_returns.dropna()
    stats = return_stats(r)

    # -- Individual asset stats --
    asset_stats = {}
    if hasattr(returns, "columns"):
        for col in returns.columns:
            s = returns[col].dropna()
            asset_stats[col] = {
                "mean_annual_pct": round(s.mean() * 252 * 100, 2),
                "std_annual_pct": round(s.std() * np.sqrt(252) * 100, 2),
                "skewness": round(float(__import__("scipy").stats.skew(s)), 3),
            }

    # -- Correlation matrix --
    corr_summary = {}
    if hasattr(returns, "corr"):
        corr = returns.corr()
        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corr_summary[f"{cols[i]}/{cols[j]}"] = round(float(corr.iloc[i, j]), 3)

    # -- Enriched anomaly data: include actual return on each anomaly day --
    def _enrich_anomaly_dates(date_list: list[str]) -> list[str]:
        enriched = []
        for ds in date_list:
            try:
                idx = pd.to_datetime(ds)
                if idx in r.index:
                    val = r.loc[idx]
                    enriched.append(f"{ds} ({val*100:.2f}%)")
                else:
                    enriched.append(ds)
            except Exception:
                enriched.append(ds)
        return enriched

    zscore_enriched = _enrich_anomaly_dates(anomalies.get("dates_zscore", [])[:10])
    if_enriched = _enrich_anomaly_dates(anomalies.get("dates_isolation_forest", [])[:10])

    # How many anomaly dates do BOTH methods agree on?
    z_set = set(anomalies.get("dates_zscore", [])[:20])
    if_set = set(anomalies.get("dates_isolation_forest", [])[:20])
    agreed = sorted(z_set & if_set)

    # -- VaR values (in %) --
    hv95 = risk_metrics.get("hist_var_95", 0) * 100
    pv95 = risk_metrics.get("param_var_95", 0) * 100
    mv95 = risk_metrics.get("mc_var_95", 0) * 100
    gv95 = (risk_metrics.get("garch_var_95") or 0) * 100

    hv99 = risk_metrics.get("hist_var_99", 0) * 100
    pv99 = risk_metrics.get("param_var_99", 0) * 100
    mv99 = risk_metrics.get("mc_var_99", 0) * 100
    gv99 = (risk_metrics.get("garch_var_99") or 0) * 100

    he95 = risk_metrics.get("hist_es_95", 0) * 100
    pe95 = risk_metrics.get("param_es_95", 0) * 100
    me95 = risk_metrics.get("mc_es_95", 0) * 100
    ge95 = (risk_metrics.get("garch_es_95") or 0) * 100

    # ES/VaR ratio: measures how much worse the average tail loss is vs the VaR cutoff
    # > 1.5 = fat tails requiring attention; > 2.0 = severe fat tails
    es_var_ratio_hist = round(he95 / hv95, 3) if hv95 > 0 else None
    es_var_ratio_param = round(pe95 / pv95, 3) if pv95 > 0 else None

    # Model divergence: max difference between 4 VaR 95% estimates
    var_95_vals = [v for v in [hv95, pv95, mv95, gv95] if v > 0]
    var_95_spread = round(max(var_95_vals) - min(var_95_vals), 3) if var_95_vals else 0

    # Historical vs GARCH divergence (tells you how much regime-conditional risk differs)
    garch_vs_hist_pct = round(gv95 - hv95, 3) if gv95 > 0 else None

    # -- Portfolio weights for all strategies --
    current_weights = (portfolio_results.get("current") or {}).get("weights", {})
    max_sharpe_weights = (portfolio_results.get("max_sharpe") or {}).get("weights", {})
    min_vol_weights = (portfolio_results.get("min_vol") or {}).get("weights", {})
    eq_wt_weights = (portfolio_results.get("equal_weight") or {}).get("weights", {})

    # Per-ticker weight delta: current vs Max Sharpe (shows what to buy/sell)
    all_tickers_opt = set(list(current_weights.keys()) + list(max_sharpe_weights.keys()))
    weight_delta_max_sharpe = {
        t: round(
            max_sharpe_weights.get(t, 0) - current_weights.get(t, 0), 4
        )
        for t in sorted(all_tickers_opt)
    }

    # -- Period --
    period_start = (
        str(prices.index[0].date()) if hasattr(prices.index[0], "date")
        else str(prices.index[0])
    )
    period_end = (
        str(prices.index[-1].date()) if hasattr(prices.index[-1], "date")
        else str(prices.index[-1])
    )

    # -- Worst 5 days with magnitudes --
    worst_days = r.nsmallest(5)
    worst_str = [
        f"{str(d.date()) if hasattr(d, 'date') else str(d)}: {v*100:.2f}%"
        for d, v in worst_days.items()
    ]

    # -- Best 3 days (useful for regime context) --
    best_days = r.nlargest(3)
    best_str = [
        f"{str(d.date()) if hasattr(d, 'date') else str(d)}: +{v*100:.2f}%"
        for d, v in best_days.items()
    ]

    # -- Backtest keys --
    kupiec = backtest_results.get("kupiec_95", {})
    christoffersen = backtest_results.get("christoffersen_95", {})

    return {
        # Identifiers
        "tickers": tickers,
        "weights": {k: round(v, 4) for k, v in weights.items()},
        "period_start": period_start,
        "period_end": period_end,
        "n_trading_days": len(r),

        # Portfolio return stats
        "portfolio_mean_daily": round(stats["mean_daily"], 6),
        "portfolio_mean_annual_pct": round(stats["mean_annual"] * 100, 2),
        "portfolio_std_daily": round(stats["std_daily"], 6),
        "portfolio_std_annual_pct": round(stats["std_annual"] * 100, 2),
        "portfolio_skewness": round(stats["skewness"], 3),
        "portfolio_excess_kurtosis": round(stats["excess_kurtosis"], 3),
        "portfolio_min_return_pct": round(stats["min_return"] * 100, 2),
        "portfolio_max_return_pct": round(stats["max_return"] * 100, 2),
        "recent_vol_20d_annual_pct": round(stats["recent_vol_20d"] * 100, 2),
        "sharpe_ratio_annual": round(stats["sharpe_ratio_annual"], 3),
        "worst_5_days": worst_str,
        "best_3_days": best_str,

        # Individual asset stats
        "asset_stats": asset_stats,
        "correlation_pairs": corr_summary,

        # Regime
        "current_regime": risk_metrics.get("regime", "unknown"),

        # Enriched anomaly data (with return magnitudes)
        "n_anomalies_zscore": anomalies.get("n_zscore", 0),
        "anomaly_dates_zscore_enriched": zscore_enriched,
        "n_anomalies_isolation_forest": anomalies.get("n_isolation_forest", 0),
        "anomaly_dates_if_enriched": if_enriched,
        "anomaly_dates_both_agreed": agreed[:5],
        "n_agreed_anomalies": len(agreed),

        # VaR (%)
        "hist_var_95_pct": round(hv95, 3),
        "param_var_95_pct": round(pv95, 3),
        "mc_var_95_pct": round(mv95, 3),
        "garch_var_95_pct": round(gv95, 3),

        "hist_var_99_pct": round(hv99, 3),
        "param_var_99_pct": round(pv99, 3),
        "mc_var_99_pct": round(mv99, 3),
        "garch_var_99_pct": round(gv99, 3),

        # ES (%)
        "hist_es_95_pct": round(he95, 3),
        "param_es_95_pct": round(pe95, 3),
        "mc_es_95_pct": round(me95, 3),
        "garch_es_95_pct": round(ge95, 3),

        # ES/VaR ratio (fat tail indicator)
        "es_var_ratio_historical": es_var_ratio_hist,
        "es_var_ratio_parametric": es_var_ratio_param,

        # Model divergence
        "var_95_method_spread_pct": var_95_spread,
        "garch_vs_hist_var_95_delta_pct": garch_vs_hist_pct,

        # 10-day
        "var_10d_95_pct": round(risk_metrics.get("var_10d_95", 0) * 100, 3),

        # Drawdown
        "max_drawdown_pct": round(risk_metrics.get("max_drawdown", 0) * 100, 2),

        # GARCH
        "garch_persistence": risk_metrics.get("garch_persistence"),
        "garch_long_run_vol_pct": round(
            (risk_metrics.get("garch_long_run_vol") or 0) * 100, 2
        ),

        # Backtest
        "n_exceedances_95": backtest_results.get("n_exceedances_95", 0),
        "expected_exceedances_95": backtest_results.get("expected_exceedances_95", 0),
        "kupiec_95_pval": kupiec.get("p_value"),
        "kupiec_95_reject": kupiec.get("reject_h0"),
        "kupiec_95_interpretation": kupiec.get("interpretation", ""),
        "christoffersen_95_pval": christoffersen.get("p_value"),
        "christoffersen_95_reject": christoffersen.get("reject_h0"),
        "christoffersen_95_pi01": christoffersen.get("pi01"),
        "christoffersen_95_pi11": christoffersen.get("pi11"),
        "christoffersen_95_interpretation": christoffersen.get("interpretation", ""),

        # Portfolio optimization with full weight tables
        "current_weights": current_weights,
        "current_vol_pct": round(
            (portfolio_results.get("current") or {}).get("volatility", 0) * 100, 2
        ),
        "current_return_pct": round(
            (portfolio_results.get("current") or {}).get("expected_return", 0) * 100, 2
        ),
        "current_sharpe": round(
            (portfolio_results.get("current") or {}).get("sharpe_ratio", 0), 3
        ),

        "max_sharpe_weights": max_sharpe_weights,
        "max_sharpe_vol_pct": round(
            (portfolio_results.get("max_sharpe") or {}).get("volatility", 0) * 100, 2
        ),
        "max_sharpe_return_pct": round(
            (portfolio_results.get("max_sharpe") or {}).get("expected_return", 0) * 100, 2
        ),
        "max_sharpe_sharpe": round(
            (portfolio_results.get("max_sharpe") or {}).get("sharpe_ratio", 0), 3
        ),

        "min_vol_weights": min_vol_weights,
        "min_vol_vol_pct": round(
            (portfolio_results.get("min_vol") or {}).get("volatility", 0) * 100, 2
        ),
        "min_vol_return_pct": round(
            (portfolio_results.get("min_vol") or {}).get("expected_return", 0) * 100, 2
        ),
        "min_vol_sharpe": round(
            (portfolio_results.get("min_vol") or {}).get("sharpe_ratio", 0), 3
        ),

        "weight_delta_max_sharpe": weight_delta_max_sharpe,
    }


# ---------------------------------------------------------------------------
# Context formatters (turn the context dict into LLM-readable text blocks)
# ---------------------------------------------------------------------------


def _fmt_market(ctx: dict) -> str:
    """
    Format market context block for the Market Monitor agent.
    Includes portfolio stats, individual asset stats, correlations, and worst days.
    """
    lines = [
        f"Portfolio: {', '.join(ctx['tickers'])}",
        f"Period: {ctx['period_start']} to {ctx['period_end']} "
        f"({ctx['n_trading_days']} trading days)",
        "",
        "Portfolio Level Statistics:",
        f"  Mean return: {ctx['portfolio_mean_daily']:.4f} daily / "
        f"{ctx['portfolio_mean_annual_pct']:.1f}% annualized",
        f"  Volatility: {ctx['portfolio_std_daily']:.4f} daily / "
        f"{ctx['portfolio_std_annual_pct']:.1f}% annualized",
        f"  Skewness: {ctx['portfolio_skewness']:.3f}  "
        f"(negative skew means fat LEFT tail -- losses are larger than gains of same frequency)",
        f"  Excess kurtosis: {ctx['portfolio_excess_kurtosis']:.3f}  "
        f"(> 0 means returns have fatter tails than a normal distribution)",
        f"  Min single-day return: {ctx['portfolio_min_return_pct']:.2f}%",
        f"  Max single-day return: {ctx['portfolio_max_return_pct']:.2f}%",
        f"  Annual Sharpe ratio: {ctx['sharpe_ratio_annual']:.3f}",
        "",
        f"Recent 20-day rolling annualized volatility: {ctx['recent_vol_20d_annual_pct']:.1f}%",
        f"Current volatility regime: {ctx['current_regime'].upper()}",
        "",
    ]

    if ctx.get("asset_stats"):
        lines.append("Individual Asset Statistics (annualized):")
        for ticker, s in ctx["asset_stats"].items():
            w = ctx["weights"].get(ticker, 0)
            lines.append(
                f"  {ticker} (weight {w*100:.1f}%): "
                f"return {s['mean_annual_pct']:.1f}%, "
                f"vol {s['std_annual_pct']:.1f}%, "
                f"skew {s['skewness']:.2f}"
            )
        lines.append("")

    if ctx.get("correlation_pairs"):
        lines.append("Pairwise Correlations:")
        for pair, val in ctx["correlation_pairs"].items():
            lines.append(f"  {pair}: {val:.3f}")
        lines.append("")

    lines.append("Worst 5 single-day losses:")
    for d in ctx.get("worst_5_days", []):
        lines.append(f"  {d}")

    lines.append("")
    lines.append("Best 3 single-day gains:")
    for d in ctx.get("best_3_days", []):
        lines.append(f"  {d}")

    return "\n".join(lines)


def _fmt_anomaly(ctx: dict) -> str:
    """
    Format anomaly context for the Anomaly Detector agent.
    Includes return magnitudes on anomaly dates and cross-method agreement.
    """
    lines = [
        f"Current volatility regime: {ctx['current_regime'].upper()}",
        f"Recent 20d annualized vol: {ctx['recent_vol_20d_annual_pct']:.1f}%",
        "",
        f"Z-score anomalies (threshold: 2.5 rolling std): {ctx['n_anomalies_zscore']} days",
        "  Dates and return magnitudes:",
    ]
    for d in ctx.get("anomaly_dates_zscore_enriched", []):
        lines.append(f"    {d}")

    lines += [
        "",
        f"Isolation Forest anomalies (contamination 5%): {ctx['n_anomalies_isolation_forest']} days",
        "  Dates and return magnitudes:",
    ]
    for d in ctx.get("anomaly_dates_if_enriched", []):
        lines.append(f"    {d}")

    agreed = ctx.get("anomaly_dates_both_agreed", [])
    lines += [
        "",
        f"Days flagged by BOTH methods ({ctx['n_agreed_anomalies']} total): "
        f"{', '.join(agreed) or 'none'}",
        "  (Agreement between two independent methods indicates the strongest anomalies.)",
        "",
        "Worst 5 portfolio days for context:",
    ]
    for d in ctx.get("worst_5_days", []):
        lines.append(f"  {d}")

    return "\n".join(lines)


def _fmt_risk(ctx: dict) -> str:
    """
    Format risk metrics for the Risk Forecaster agent.
    Includes all VaR/ES values, ES/VaR ratios, model divergence, and backtest results.
    """
    lines = [
        "=== 1-Day VaR at 95% Confidence (% of portfolio value lost) ===",
        f"  Historical Simulation: {ctx['hist_var_95_pct']:.3f}%",
        f"  Parametric (Normal): {ctx['param_var_95_pct']:.3f}%",
        f"  Monte Carlo Bootstrap: {ctx['mc_var_95_pct']:.3f}%",
        f"  GARCH(1,1) Conditional: {ctx['garch_var_95_pct']:.3f}%",
        f"  -- Spread across 4 methods: {ctx['var_95_method_spread_pct']:.3f}% "
        f"(wide spread = high model uncertainty)",
        f"  -- GARCH vs Historical delta: "
        f"{ctx.get('garch_vs_hist_var_95_delta_pct', 'N/A')}% "
        f"(positive = GARCH says forward risk is HIGHER than historical average)",
        "",
        "=== 1-Day Expected Shortfall at 95% (mean loss GIVEN breach of VaR) ===",
        f"  Historical: {ctx['hist_es_95_pct']:.3f}%",
        f"  Parametric: {ctx['param_es_95_pct']:.3f}%",
        f"  Monte Carlo: {ctx['mc_es_95_pct']:.3f}%",
        f"  GARCH: {ctx['garch_es_95_pct']:.3f}%",
        "",
        "=== ES/VaR Ratios (fat-tail indicator) ===",
        f"  Historical ES/VaR: {ctx.get('es_var_ratio_historical', 'N/A')}",
        f"  Parametric ES/VaR: {ctx.get('es_var_ratio_parametric', 'N/A')}",
        "  Interpretation: ratio > 1.25 = moderate fat tails, "
        "> 1.5 = severe fat tails, > 2.0 = extreme tail risk",
        "(For a perfect normal distribution this ratio is always ~1.25 at 95%.)",
        "",
        "=== 1-Day VaR at 99% Confidence ===",
        f"  Historical: {ctx['hist_var_99_pct']:.3f}%",
        f"  Parametric: {ctx['param_var_99_pct']:.3f}%",
        f"  Monte Carlo: {ctx['mc_var_99_pct']:.3f}%",
        f"  GARCH: {ctx['garch_var_99_pct']:.3f}%",
        "",
        f"10-Day VaR 95% (sqrt-of-time scaled from 1-day): {ctx['var_10d_95_pct']:.3f}%",
        f"Maximum drawdown over full period: {ctx['max_drawdown_pct']:.2f}%",
        "",
        "=== GARCH(1,1) Model ===",
        f"  Persistence (alpha+beta): {ctx.get('garch_persistence', 'N/A')}",
        f"  Long-run annualized vol: {ctx.get('garch_long_run_vol_pct', 'N/A')}%",
        "  Persistence interpretation: > 0.95 = very slow vol mean-reversion "
        "(shocks last weeks/months), < 0.80 = fast reversion (shocks fade within days)",
        "",
        "=== VaR Backtest Results ===",
        f"  Total days evaluated: approx "
        f"{round(ctx['expected_exceedances_95'] / 0.05) if ctx['expected_exceedances_95'] else 'N/A'}",
        f"  Actual exceedances: {ctx['n_exceedances_95']}",
        f"  Expected exceedances: {ctx['expected_exceedances_95']}",
        f"  Kupiec POF Test p-value: {ctx.get('kupiec_95_pval', 'N/A')} "
        f"-- {'REJECT H0 (model mis-calibrated)' if ctx.get('kupiec_95_reject') else 'PASS (calibration acceptable)'}",
        f"  Kupiec verdict: {ctx.get('kupiec_95_interpretation', '')}",
        "",
        f"  Christoffersen Independence Test p-value: {ctx.get('christoffersen_95_pval', 'N/A')} "
        f"-- {'REJECT H0 (breaches are clustered)' if ctx.get('christoffersen_95_reject') else 'PASS (breaches are independent)'}",
        f"  Conditional breach rate given prior breach (pi11): {ctx.get('christoffersen_95_pi11', 'N/A')}",
        f"  Baseline breach rate (pi01): {ctx.get('christoffersen_95_pi01', 'N/A')}",
        f"  Christoffersen verdict: {ctx.get('christoffersen_95_interpretation', '')}",
    ]
    return "\n".join(lines)


def _fmt_portfolio(ctx: dict) -> str:
    """
    Format portfolio optimization data for the Portfolio Optimizer agent.
    Includes full weight tables for all strategies, per-ticker weight deltas,
    and performance metrics side by side.
    """
    def fmt_weights(w: dict) -> str:
        if not w:
            return "  not available"
        return "\n".join(
            f"    {ticker}: {v*100:.1f}%"
            for ticker, v in sorted(w.items(), key=lambda x: -x[1])
            if v > 0.005
        )

    lines = [
        f"Volatility regime: {ctx['current_regime'].upper()}",
        f"Analysis period return: {ctx['portfolio_mean_annual_pct']:.1f}% annualized",
        "",
        "=== Current Portfolio ===",
        f"  Annual volatility: {ctx['current_vol_pct']:.1f}%",
        f"  Expected return: {ctx['current_return_pct']:.1f}%",
        f"  Sharpe ratio: {ctx['current_sharpe']:.3f}",
        "  Weights:",
        fmt_weights(ctx.get("current_weights", {})),
        "",
        "=== Max Sharpe Portfolio ===",
        f"  Annual volatility: {ctx['max_sharpe_vol_pct']:.1f}%",
        f"  Expected return: {ctx['max_sharpe_return_pct']:.1f}%",
        f"  Sharpe ratio: {ctx['max_sharpe_sharpe']:.3f}",
        "  Weights:",
        fmt_weights(ctx.get("max_sharpe_weights", {})),
        "",
        "=== Minimum Volatility Portfolio ===",
        f"  Annual volatility: {ctx['min_vol_vol_pct']:.1f}%",
        f"  Expected return: {ctx['min_vol_return_pct']:.1f}%",
        f"  Sharpe ratio: {ctx['min_vol_sharpe']:.3f}",
        "  Weights:",
        fmt_weights(ctx.get("min_vol_weights", {})),
        "",
        "=== Weight Changes to Move from Current to Max Sharpe ===",
        "  (positive = increase allocation, negative = decrease allocation)",
    ]
    for ticker, delta in sorted(
        ctx.get("weight_delta_max_sharpe", {}).items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    ):
        if abs(delta) > 0.005:
            sign = "+" if delta > 0 else ""
            lines.append(f"    {ticker}: {sign}{delta*100:.1f}%")

    lines += [
        "",
        "Performance comparison:",
        f"  Current Sharpe: {ctx['current_sharpe']:.3f} | "
        f"Max Sharpe: {ctx['max_sharpe_sharpe']:.3f} | "
        f"Min Vol: {ctx['min_vol_sharpe']:.3f}",
        f"  Current vol: {ctx['current_vol_pct']:.1f}% | "
        f"Max Sharpe vol: {ctx['max_sharpe_vol_pct']:.1f}% | "
        f"Min vol: {ctx['min_vol_vol_pct']:.1f}%",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent backstories (domain-specific, detailed)
# ---------------------------------------------------------------------------

_BASE_RULES = (
    "You are precise and factual. "
    "You always cite specific numbers from the data provided. "
    "You never guess or fabricate any number. "
    "You do not use em-dashes. "
    "You do not use emojis or icons. "
    "You write in clear, direct English sentences. "
    "You keep your response focused and under 280 words."
)

_BACKSTORY_MARKET_MONITOR = (
    "You are a senior market surveillance analyst at a tier-1 investment bank with 18 years "
    "of experience monitoring equity and ETF markets. You specialize in interpreting "
    "return distributions, identifying whether current volatility is elevated relative "
    "to historical norms, and explaining what skewness and excess kurtosis mean in "
    "practical risk terms. You know that negative skewness combined with high kurtosis "
    "is the classic signature of equity tail risk. You always compare recent 20-day "
    "realized volatility to the full-period average to assess whether risk is building. "
    "You flag high pairwise correlations between assets as a diversification warning "
    "because correlation spikes during crises reduce the benefit of diversification "
    "at exactly the wrong moment. "
    + _BASE_RULES
)

_BACKSTORY_ANOMALY_DETECTOR = (
    "You are a quantitative risk specialist with 15 years of experience in outlier "
    "detection and tail event analysis at a hedge fund. You understand the difference "
    "between statistical anomalies (extreme Z-scores) and economically meaningful "
    "events. You know that when two independent detection methods (Z-score and "
    "Isolation Forest) agree on an anomaly date, that event is much more likely to "
    "represent genuine market stress rather than noise. You always assess whether "
    "anomalies are clustered (suggesting a regime shift or sustained crisis) or "
    "isolated (suggesting a single event spike). For each anomaly you consider "
    "whether it coincides with known macroeconomic events: Fed decisions, earnings "
    "shocks, geopolitical events, or index rebalancing. You quantify severity by "
    "looking at the actual return magnitude on the anomaly date, not just its "
    "statistical rank. "
    + _BASE_RULES
)

_BACKSTORY_RISK_FORECASTER = (
    "You are a market risk measurement expert with 20 years of experience building "
    "and validating VaR models at a major risk management firm. You understand "
    "exactly why the four VaR methods produce different estimates: Historical "
    "Simulation preserves the empirical distribution but is backward-looking and "
    "slow to update; Parametric VaR assumes normality and systematically "
    "underestimates risk when excess kurtosis is positive; Monte Carlo Bootstrap "
    "captures empirical tail shape through resampling; GARCH(1,1) produces "
    "regime-conditional estimates that reflect today's volatility environment. "
    "You interpret the ES/VaR ratio as a fat-tail severity indicator: ratios "
    "above 1.5 at 95% confidence indicate the tail is heavier than normal and "
    "that ES-based capital requirements are materially higher than VaR suggests. "
    "You know that GARCH persistence above 0.95 means volatility shocks will "
    "persist for weeks and that a forward-looking risk manager should use the "
    "GARCH estimate rather than the historical average when persistence is high. "
    "For backtesting: Kupiec test failure means the breach count is wrong; "
    "Christoffersen failure means breaches cluster in time, which is the more "
    "dangerous failure because it means the model is blind during crisis periods. "
    + _BASE_RULES
)

_BACKSTORY_PORTFOLIO_OPTIMIZER = (
    "You are a risk-aware portfolio construction specialist with 17 years of "
    "experience at a quantitative asset manager. You understand the practical "
    "limitations of mean-variance optimization: it is highly sensitive to "
    "expected return estimates which are noisy, and it tends to produce "
    "concentrated portfolios. You know that the Minimum Volatility portfolio "
    "is often a better choice than Max Sharpe when confidence in return forecasts "
    "is low, because minimizing variance requires only the covariance matrix. "
    "You always evaluate portfolio changes in the context of the current volatility "
    "regime: in a HIGH regime, reducing individual stock concentration and adding "
    "diversifiers is prudent; in a LOW regime, there is more tolerance for "
    "concentrated factor bets. You recommend concrete ticker-level weight changes "
    "by looking at the delta table between current and optimal weights, and you "
    "explain the risk-return trade-off of each significant change. You always "
    "flag if the current portfolio has a materially lower Sharpe ratio than the "
    "optimized alternatives, and quantify the improvement explicitly. "
    + _BASE_RULES
)


# ---------------------------------------------------------------------------
# Main crew runner
# ---------------------------------------------------------------------------


def run_risk_analysis_crew(context: dict) -> dict:
    """
    Run the 4-agent sequential analysis on the pre-compiled context.

    Uses the Groq API directly (no crewai) to avoid the crewai 1.9.3 / openai
    version conflict on Streamlit Cloud. Each agent is a groq.chat.completions
    call. Context chaining is replicated by passing prior agent outputs as
    prefixed text in the next agent's user message.

    Returns dict with keys: market_monitor, anomaly_detector,
    risk_forecaster, portfolio_optimizer.
    Returns empty strings if Groq is unavailable.
    """
    empty = {
        "market_monitor": "",
        "anomaly_detector": "",
        "risk_forecaster": "",
        "portfolio_optimizer": "",
    }

    # ----------------------------------------------------------------
    # Active implementation: direct Groq API calls
    # To restore the CrewAI implementation, see the commented block below.
    # ----------------------------------------------------------------

    try:
        from groq import Groq
    except ImportError:
        logger.error("groq package not installed. Run: pip install groq")
        return empty

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not set. Agent analysis will be skipped.")
        return empty

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    client = Groq(api_key=api_key)

    # Agent 1: Market Monitor -- no prior context
    monitor_out = _call_groq_agent(
        client=client,
        model=model,
        system=_BACKSTORY_MARKET_MONITOR,
        task_desc=(
            "You are given the following market data for a user-defined portfolio. "
            "Write a structured market summary covering exactly these 4 points:\n\n"
            "1. VOLATILITY REGIME: State the current regime (LOW/MID/HIGH) and compare "
            "the recent 20-day vol to the full-period annualized vol. Is risk building or subsiding?\n"
            "2. RETURN DISTRIBUTION: Comment on mean, volatility, skewness (fat left tail?), "
            "and excess kurtosis (fatter tails than normal?). State whether the distribution "
            "is dangerous for VaR models that assume normality.\n"
            "3. CORRELATION STRUCTURE: Identify the highest pairwise correlations. "
            "High correlation reduces diversification benefit during drawdowns.\n"
            "4. NOTABLE EVENTS: Comment on the 5 worst return days. Were they isolated or clustered?\n\n"
            "Do NOT use em-dashes. Do NOT use emojis. Use specific numbers from the data.\n\n"
            f"MARKET DATA:\n{_fmt_market(context)}"
        ),
        prior_outputs=[],
    )

    # Agent 2: Anomaly Detector -- receives monitor output as context
    anomaly_out = _call_groq_agent(
        client=client,
        model=model,
        system=_BACKSTORY_ANOMALY_DETECTOR,
        task_desc=(
            "You are given anomaly detection results for the portfolio. "
            "Write an anomaly analysis covering exactly these 4 points:\n\n"
            "1. DETECTION COUNTS: State how many anomalies each method found and "
            "how many dates both methods agreed on. Agreement increases confidence.\n"
            "2. SEVERITY: For the most extreme anomaly dates, state the actual "
            "return magnitude (provided in the data). A -8% day is fundamentally "
            "different from a -2% day even if both are anomalies statistically.\n"
            "3. CLUSTERING vs ISOLATION: Are anomalies spread across the period "
            "or bunched together? Clustering signals a regime change or sustained crisis.\n"
            "4. ECONOMIC CONTEXT: For anomaly dates you can place in context "
            "(e.g., early 2020, late 2022), name the likely market driver. "
            "For dates without obvious context, state they appear isolated.\n\n"
            "Do NOT use em-dashes. Do NOT use emojis. Reference specific dates and return values.\n\n"
            f"ANOMALY DATA:\n{_fmt_anomaly(context)}"
        ),
        prior_outputs=[monitor_out],
    )

    # Agent 3: Risk Forecaster -- receives monitor + anomaly outputs as context
    risk_out = _call_groq_agent(
        client=client,
        model=model,
        system=_BACKSTORY_RISK_FORECASTER,
        task_desc=(
            "You are given the complete VaR, ES, GARCH, and backtest results. "
            "Write a risk assessment covering exactly these 5 points:\n\n"
            "1. VaR METHOD COMPARISON: The 4 methods will disagree. Explain WHY. "
            "If GARCH > Historical, it means current vol is above the long-run average. "
            "If Parametric < Historical, it means the return distribution has fatter tails "
            "than a normal distribution (confirmed by excess kurtosis > 0).\n"
            "2. ES/VaR RATIO: Use the historical ES/VaR ratio provided. State clearly "
            "whether the tail is normal (ratio ~1.25), moderate (1.25-1.5), severe (1.5-2.0), "
            "or extreme (>2.0). This ratio tells you how bad losses get GIVEN a VaR breach.\n"
            "3. GARCH SIGNAL: State the persistence value and what it means. "
            "High persistence (>0.95) means today's elevated volatility will persist for weeks "
            "and the GARCH VaR is the most relevant forward-looking estimate.\n"
            "4. BACKTEST VERDICT: State clearly whether Kupiec PASSES or FAILS and why. "
            "Then state whether Christoffersen PASSES or FAILS and why. "
            "A Christoffersen failure is more concerning than a Kupiec failure.\n"
            "5. OVERALL RISK VERDICT: Synthesize into one of: "
            "LOW RISK / MODERATE RISK / ELEVATED RISK / HIGH RISK, with one-sentence justification.\n\n"
            "Do NOT use em-dashes. Do NOT use emojis. Cite specific numbers.\n\n"
            f"RISK DATA:\n{_fmt_risk(context)}"
        ),
        prior_outputs=[monitor_out, anomaly_out],
    )

    # Agent 4: Portfolio Optimizer -- receives monitor + risk outputs as context
    portfolio_out = _call_groq_agent(
        client=client,
        model=model,
        system=_BACKSTORY_PORTFOLIO_OPTIMIZER,
        task_desc=(
            "You are given the current portfolio and two optimized alternatives. "
            "Write a portfolio recommendation covering exactly these 4 points:\n\n"
            "1. PERFORMANCE COMPARISON: Compare current vs Max Sharpe vs Min Vol "
            "on three metrics: annual volatility, expected return, and Sharpe ratio. "
            "Quantify the Sharpe ratio gap between current and optimal. "
            "If the Max Sharpe Sharpe ratio is materially higher, say by how much.\n"
            "2. WEIGHT ANALYSIS: Look at the weight delta table (current to Max Sharpe). "
            "Identify the 2-3 largest weight changes. Explain what each change achieves "
            "(e.g., reducing a high-vol single name, adding a lower-vol diversifier).\n"
            "3. REGIME-AWARE ASSESSMENT: Given the current volatility regime "
            f"({context['current_regime'].upper()}), state whether the current "
            "portfolio concentration is appropriate. In a HIGH vol regime, "
            "concentrated single-name exposure is riskier. In a LOW regime, "
            "there is more room for active positions.\n"
            "4. RECOMMENDATION: Give a clear, specific recommendation. "
            "Name specific tickers to reduce and increase and by how much (from the delta table). "
            "If the current portfolio is already near-optimal, say so with reasoning.\n\n"
            "Do NOT use em-dashes. Do NOT use emojis. Cite specific numbers and ticker names.\n\n"
            f"PORTFOLIO DATA:\n{_fmt_portfolio(context)}"
        ),
        prior_outputs=[monitor_out, risk_out],
    )

    return {
        "market_monitor": monitor_out,
        "anomaly_detector": anomaly_out,
        "risk_forecaster": risk_out,
        "portfolio_optimizer": portfolio_out,
    }

    # ----------------------------------------------------------------
    # CrewAI implementation -- kept for reference, currently inactive.
    # To reactivate: restore crewai to requirements.txt, swap the
    # active block above for this block, and restore _create_llm_crewai.
    # ----------------------------------------------------------------

    # try:
    #     from crewai import Agent, Task, Crew, Process
    # except ImportError:
    #     logger.error("crewai not installed.")
    #     return empty
    #
    # llm = _create_llm_crewai()   # needs crewai.LLM wrapping litellm/groq
    # if llm is None:
    #     return empty
    #
    # market_monitor = Agent(
    #     role="Market Monitor",
    #     goal=(
    #         "Summarize the portfolio's current market conditions, volatility regime, "
    #         "return distribution characteristics, correlation structure, and the most "
    #         "notable return events. Flag whether the current environment is elevated "
    #         "or subdued relative to the historical baseline."
    #     ),
    #     backstory=_BACKSTORY_MARKET_MONITOR,
    #     llm=llm,
    #     verbose=False,
    #     allow_delegation=False,
    # )
    # anomaly_detector = Agent(
    #     role="Anomaly Detector",
    #     goal=(
    #         "Interpret the list of anomaly dates and their return magnitudes. "
    #         "Compare what both detection methods (Z-score and Isolation Forest) "
    #         "found, with emphasis on dates where both agree. Classify anomalies "
    #         "as clustered vs isolated and infer the likely economic drivers. "
    #         "State clearly whether the anomaly pattern is a systemic risk signal "
    #         "or isolated noise."
    #     ),
    #     backstory=_BACKSTORY_ANOMALY_DETECTOR,
    #     llm=llm,
    #     verbose=False,
    #     allow_delegation=False,
    # )
    # risk_forecaster = Agent(
    #     role="Risk Forecaster",
    #     goal=(
    #         "Interpret the four VaR estimates and explain why they differ. "
    #         "Compute and explain the ES/VaR ratio as a fat-tail severity indicator. "
    #         "Interpret GARCH persistence as a forward-looking signal about how long "
    #         "current volatility will persist. Deliver a clear, specific verdict on "
    #         "both backtest results (Kupiec and Christoffersen) in plain language. "
    #         "State the overall risk level: acceptable, elevated, or critical."
    #     ),
    #     backstory=_BACKSTORY_RISK_FORECASTER,
    #     llm=llm,
    #     verbose=False,
    #     allow_delegation=False,
    # )
    # portfolio_optimizer = Agent(
    #     role="Portfolio Optimizer",
    #     goal=(
    #         "Compare the current portfolio versus Max Sharpe and Min Volatility "
    #         "alternatives using full weight tables. Quantify the Sharpe ratio and "
    #         "volatility improvement available. Recommend specific ticker-level "
    #         "weight changes from the delta table. Tie the recommendation to the "
    #         "current volatility regime and the risk forecaster's findings."
    #     ),
    #     backstory=_BACKSTORY_PORTFOLIO_OPTIMIZER,
    #     llm=llm,
    #     verbose=False,
    #     allow_delegation=False,
    # )
    # monitor_task = Task(
    #     description=(
    #         "You are given the following market data for a user-defined portfolio. "
    #         "Write a structured market summary covering exactly these 4 points:\n\n"
    #         "1. VOLATILITY REGIME: State the current regime (LOW/MID/HIGH) and compare "
    #         "the recent 20-day vol to the full-period annualized vol. Is risk building or subsiding?\n"
    #         "2. RETURN DISTRIBUTION: Comment on mean, volatility, skewness (fat left tail?), "
    #         "and excess kurtosis (fatter tails than normal?). State whether the distribution "
    #         "is dangerous for VaR models that assume normality.\n"
    #         "3. CORRELATION STRUCTURE: Identify the highest pairwise correlations. "
    #         "High correlation reduces diversification benefit during drawdowns.\n"
    #         "4. NOTABLE EVENTS: Comment on the 5 worst return days. Were they isolated or clustered?\n\n"
    #         "Do NOT use em-dashes. Do NOT use emojis. Use specific numbers from the data.\n\n"
    #         f"MARKET DATA:\n{_fmt_market(context)}"
    #     ),
    #     expected_output=(
    #         "A 4-section structured market summary (200-280 words) covering: "
    #         "volatility regime assessment with specific vol numbers, "
    #         "return distribution analysis with skewness/kurtosis interpretation, "
    #         "correlation observations, and worst-day event comments. "
    #         "Every claim is supported by a specific number from the data. "
    #         "No em-dashes. No emojis."
    #     ),
    #     agent=market_monitor,
    # )
    # anomaly_task = Task(
    #     description=(
    #         "You are given anomaly detection results for the portfolio. "
    #         "Write an anomaly analysis covering exactly these 4 points:\n\n"
    #         "1. DETECTION COUNTS: State how many anomalies each method found and "
    #         "how many dates both methods agreed on. Agreement increases confidence.\n"
    #         "2. SEVERITY: For the most extreme anomaly dates, state the actual "
    #         "return magnitude (provided in the data). A -8% day is fundamentally "
    #         "different from a -2% day even if both are anomalies statistically.\n"
    #         "3. CLUSTERING vs ISOLATION: Are anomalies spread across the period "
    #         "or bunched together? Clustering signals a regime change or sustained crisis.\n"
    #         "4. ECONOMIC CONTEXT: For anomaly dates you can place in context "
    #         "(e.g., early 2020, late 2022), name the likely market driver. "
    #         "For dates without obvious context, state they appear isolated.\n\n"
    #         "Do NOT use em-dashes. Do NOT use emojis. Reference specific dates and return values.\n\n"
    #         f"ANOMALY DATA:\n{_fmt_anomaly(context)}"
    #     ),
    #     expected_output=(
    #         "A 4-section anomaly analysis (200-280 words) covering: "
    #         "detection method counts and agreement, "
    #         "severity assessment with actual return magnitudes on worst anomaly days, "
    #         "clustering vs isolation classification, "
    #         "and economic context for the most significant events. "
    #         "No em-dashes. No emojis."
    #     ),
    #     agent=anomaly_detector,
    #     context=[monitor_task],
    # )
    # risk_task = Task(
    #     description=(
    #         "You are given the complete VaR, ES, GARCH, and backtest results. "
    #         "Write a risk assessment covering exactly these 5 points:\n\n"
    #         "1. VaR METHOD COMPARISON: The 4 methods will disagree. Explain WHY. "
    #         "If GARCH > Historical, it means current vol is above the long-run average. "
    #         "If Parametric < Historical, it means the return distribution has fatter tails "
    #         "than a normal distribution (confirmed by excess kurtosis > 0).\n"
    #         "2. ES/VaR RATIO: Use the historical ES/VaR ratio provided. State clearly "
    #         "whether the tail is normal (ratio ~1.25), moderate (1.25-1.5), severe (1.5-2.0), "
    #         "or extreme (>2.0). This ratio tells you how bad losses get GIVEN a VaR breach.\n"
    #         "3. GARCH SIGNAL: State the persistence value and what it means. "
    #         "High persistence (>0.95) means today's elevated volatility will persist for weeks "
    #         "and the GARCH VaR is the most relevant forward-looking estimate.\n"
    #         "4. BACKTEST VERDICT: State clearly whether Kupiec PASSES or FAILS and why. "
    #         "Then state whether Christoffersen PASSES or FAILS and why. "
    #         "A Christoffersen failure is more concerning than a Kupiec failure.\n"
    #         "5. OVERALL RISK VERDICT: Synthesize into one of: "
    #         "LOW RISK / MODERATE RISK / ELEVATED RISK / HIGH RISK, with one-sentence justification.\n\n"
    #         "Do NOT use em-dashes. Do NOT use emojis. Cite specific numbers.\n\n"
    #         f"RISK DATA:\n{_fmt_risk(context)}"
    #     ),
    #     expected_output=(
    #         "A 5-section risk interpretation (230-280 words) covering: "
    #         "VaR method comparison with specific numbers and reasons for divergence, "
    #         "ES/VaR ratio interpretation, "
    #         "GARCH persistence interpretation as a forward vol signal, "
    #         "plain-language Kupiec and Christoffersen verdicts, "
    #         "and a final risk level verdict with justification. "
    #         "No em-dashes. No emojis."
    #     ),
    #     agent=risk_forecaster,
    #     context=[monitor_task, anomaly_task],
    # )
    # portfolio_task = Task(
    #     description=(
    #         "You are given the current portfolio and two optimized alternatives. "
    #         "Write a portfolio recommendation covering exactly these 4 points:\n\n"
    #         "1. PERFORMANCE COMPARISON: Compare current vs Max Sharpe vs Min Vol "
    #         "on three metrics: annual volatility, expected return, and Sharpe ratio. "
    #         "Quantify the Sharpe ratio gap between current and optimal. "
    #         "If the Max Sharpe Sharpe ratio is materially higher, say by how much.\n"
    #         "2. WEIGHT ANALYSIS: Look at the weight delta table (current to Max Sharpe). "
    #         "Identify the 2-3 largest weight changes. Explain what each change achieves "
    #         "(e.g., reducing a high-vol single name, adding a lower-vol diversifier).\n"
    #         "3. REGIME-AWARE ASSESSMENT: Given the current volatility regime "
    #         "({context['current_regime'].upper()}), state whether the current "
    #         "portfolio concentration is appropriate. In a HIGH vol regime, "
    #         "concentrated single-name exposure is riskier. In a LOW regime, "
    #         "there is more room for active positions.\n"
    #         "4. RECOMMENDATION: Give a clear, specific recommendation. "
    #         "Name specific tickers to reduce and increase and by how much (from the delta table). "
    #         "If the current portfolio is already near-optimal, say so with reasoning.\n\n"
    #         "Do NOT use em-dashes. Do NOT use emojis. Cite specific numbers and ticker names.\n\n"
    #         f"PORTFOLIO DATA:\n{_fmt_portfolio(context)}"
    #     ),
    #     expected_output=(
    #         "A 4-section portfolio recommendation (230-280 words) covering: "
    #         "performance comparison with explicit numbers, "
    #         "weight analysis with 2-3 specific changes and their purpose, "
    #         "regime-aware risk assessment, "
    #         "and a concrete ticker-level reallocation recommendation. "
    #         "No em-dashes. No emojis."
    #     ),
    #     agent=portfolio_optimizer,
    #     context=[monitor_task, risk_task],
    # )
    # crew = Crew(
    #     agents=[market_monitor, anomaly_detector, risk_forecaster, portfolio_optimizer],
    #     tasks=[monitor_task, anomaly_task, risk_task, portfolio_task],
    #     process=Process.sequential,
    #     verbose=False,
    # )
    # try:
    #     result = crew.kickoff()
    #     outputs = {k: "" for k in empty}
    #     tasks_output = getattr(result, "tasks_output", None)
    #     if tasks_output and len(tasks_output) >= 4:
    #         for i, key in enumerate(outputs.keys()):
    #             raw = getattr(tasks_output[i], "raw", "") or ""
    #             outputs[key] = raw.strip().replace("\u2014", " -- ").replace("\u2013", " - ")
    #     elif hasattr(result, "raw"):
    #         raw = str(result.raw).strip().replace("\u2014", " -- ").replace("\u2013", " - ")
    #         outputs["portfolio_optimizer"] = raw
    #     return outputs
    # except Exception as exc:
    #     logger.error("CrewAI kickoff failed: %s", exc)
    #     return empty
