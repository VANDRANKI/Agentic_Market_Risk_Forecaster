"""
Analysis pipeline orchestration.

This module ties together the data layer, risk engine, and agent layer
into a single run_full_analysis() function that the Streamlit app calls.

The pipeline runs in this order:
  1. Fetch prices via DataProvider
  2. Compute log returns
  3. Compute portfolio-level returns
  4. Compute all VaR/ES estimates (4 methods x 2 confidence levels)
  5. Run Kupiec and Christoffersen backtests
  6. Run portfolio optimization (Max Sharpe, Min Vol, Equal Weight)
  7. Detect anomalies (Z-score and Isolation Forest)
  8. Compile a structured context dict
  9. (Optional) Run CrewAI agent crew for LLM explanations

All exceptions are caught per step so a failure in one step
(e.g. GARCH fails to converge) does not crash the whole analysis.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_full_analysis(
    tickers: list[str],
    weights: dict[str, float],
    lookback_days: int,
    confidence_level: float = 0.95,
    run_agents: bool = True,
    n_mc_sims: int = 10_000,
) -> dict:
    """
    Execute the full market risk analysis pipeline.

    Parameters
    ----------
    tickers          : list of ticker symbols
    weights          : dict mapping ticker -> raw weight (will be normalized)
    lookback_days    : number of calendar days to look back for data
    confidence_level : primary VaR/ES confidence (0.95 or 0.99)
    run_agents       : whether to run the CrewAI LLM analysis
    n_mc_sims        : number of Monte Carlo simulations

    Returns
    -------
    dict with keys: prices, returns, portfolio_returns, risk_metrics,
                    backtest_results, portfolio_results, anomalies,
                    agent_outputs, errors
    """
    from data.provider import DataProvider
    from risk.engine import (
        compute_returns,
        compute_portfolio_returns,
        historical_var, historical_es,
        variance_covariance_var, variance_covariance_es,
        monte_carlo_var, monte_carlo_es,
        scale_var_to_horizon,
        compute_drawdown, max_drawdown,
        current_regime,
        detect_anomalies_zscore,
        detect_anomalies_isolation_forest,
    )
    from risk.backtest import (
        count_exceedances, kupiec_test, christoffersen_test, rolling_var_series,
    )
    from risk.garch import garch_var_forecast
    from risk.portfolio import run_all_optimizations
    from agents.crew import run_risk_analysis_crew, compile_analysis_context

    errors: list[str] = []

    # ------------------------------------------------------------------
    # 1. Fetch prices
    # ------------------------------------------------------------------
    provider = DataProvider()
    end_date = datetime.today().strftime("%Y-%m-%d")
    # Add a 60-day buffer for weekends/holidays
    start_date = (
        datetime.today() - timedelta(days=lookback_days + 90)
    ).strftime("%Y-%m-%d")

    try:
        prices = provider.fetch_prices(tickers, start=start_date, end=end_date)
    except Exception as exc:
        raise RuntimeError(f"Data fetch failed: {exc}") from exc

    # Trim to exact lookback (in trading days)
    if len(prices) > lookback_days:
        prices = prices.iloc[-lookback_days:]

    # ------------------------------------------------------------------
    # 2. Compute returns
    # ------------------------------------------------------------------
    returns = compute_returns(prices, method="log")

    # Normalize weights and compute portfolio returns
    w_sum = sum(weights.values())
    norm_weights = {k: v / w_sum for k, v in weights.items()}
    # Only keep tickers that are in the returns DataFrame
    norm_weights = {k: v for k, v in norm_weights.items() if k in returns.columns}
    if not norm_weights:
        raise ValueError("No valid tickers after data fetch.")

    portfolio_returns = compute_portfolio_returns(returns, norm_weights)

    # ------------------------------------------------------------------
    # 3. Risk metrics
    # ------------------------------------------------------------------
    risk_metrics: dict = {}

    alpha_primary = 1 - confidence_level  # e.g. 0.05 for 95%
    alt_alpha = 0.01 if alpha_primary != 0.01 else 0.05

    pr = portfolio_returns.dropna()

    # Historical
    risk_metrics["hist_var_95"] = historical_var(pr, 0.05)
    risk_metrics["hist_es_95"] = historical_es(pr, 0.05)
    risk_metrics["hist_var_99"] = historical_var(pr, 0.01)
    risk_metrics["hist_es_99"] = historical_es(pr, 0.01)

    # Parametric
    risk_metrics["param_var_95"] = variance_covariance_var(pr, 0.05)
    risk_metrics["param_es_95"] = variance_covariance_es(pr, 0.05)
    risk_metrics["param_var_99"] = variance_covariance_var(pr, 0.01)
    risk_metrics["param_es_99"] = variance_covariance_es(pr, 0.01)

    # Monte Carlo
    try:
        risk_metrics["mc_var_95"] = monte_carlo_var(pr, 0.05, n_mc_sims)
        risk_metrics["mc_es_95"] = monte_carlo_es(pr, 0.05, n_mc_sims)
        risk_metrics["mc_var_99"] = monte_carlo_var(pr, 0.01, n_mc_sims)
        risk_metrics["mc_es_99"] = monte_carlo_es(pr, 0.01, n_mc_sims)
    except Exception as exc:
        errors.append(f"Monte Carlo failed: {exc}")
        risk_metrics["mc_var_95"] = risk_metrics["hist_var_95"]
        risk_metrics["mc_es_95"] = risk_metrics["hist_es_95"]
        risk_metrics["mc_var_99"] = risk_metrics["hist_var_99"]
        risk_metrics["mc_es_99"] = risk_metrics["hist_es_99"]

    # GARCH
    try:
        garch_95 = garch_var_forecast(pr, alpha=0.05, horizon=1)
        garch_99 = garch_var_forecast(pr, alpha=0.01, horizon=1)
        if garch_95:
            risk_metrics["garch_var_95"] = garch_95["var"]
            risk_metrics["garch_es_95"] = garch_95["es"]
            risk_metrics["garch_persistence"] = garch_95["persistence"]
            risk_metrics["garch_long_run_vol"] = (
                garch_95.get("annualized_long_run_vol", None)
            )
        if garch_99:
            risk_metrics["garch_var_99"] = garch_99["var"]
            risk_metrics["garch_es_99"] = garch_99["es"]
    except Exception as exc:
        errors.append(f"GARCH failed: {exc}")
        risk_metrics["garch_var_95"] = None
        risk_metrics["garch_es_95"] = None
        risk_metrics["garch_var_99"] = None
        risk_metrics["garch_es_99"] = None

    # 10-day VaR (sqrt-of-time scaled from 1-day historical)
    risk_metrics["var_10d_95"] = scale_var_to_horizon(risk_metrics["hist_var_95"], 10)

    # Drawdown
    risk_metrics["drawdown_series"] = compute_drawdown(pr)
    risk_metrics["max_drawdown"] = max_drawdown(pr)

    # Regime
    try:
        risk_metrics["regime"] = current_regime(pr)
    except Exception:
        risk_metrics["regime"] = "unknown"

    # ------------------------------------------------------------------
    # 4. Backtests
    # ------------------------------------------------------------------
    backtest_results: dict = {}

    try:
        rolling_var = rolling_var_series(pr, alpha=0.05, window=min(252, len(pr) // 2))
        exc_data = count_exceedances(pr, rolling_var)
        backtest_results["n_exceedances_95"] = exc_data["n_exceedances"]
        backtest_results["exceedance_rate_95"] = exc_data["exceedance_rate"]
        backtest_results["exceedance_dates_95"] = exc_data["exceedance_dates"]
        backtest_results["expected_exceedances_95"] = round(exc_data["n_observations"] * 0.05)

        kupiec_95 = kupiec_test(
            exc_data["n_exceedances"],
            exc_data["n_observations"],
            alpha=0.05,
        )
        backtest_results["kupiec_95"] = kupiec_95

        christoffersen_95 = christoffersen_test(pr, rolling_var, alpha=0.05)
        backtest_results["christoffersen_95"] = christoffersen_95

        # Store rolling VaR series for charting
        backtest_results["rolling_var_95"] = rolling_var
    except Exception as exc:
        errors.append(f"Backtesting failed: {exc}")
        backtest_results["kupiec_95"] = {}
        backtest_results["christoffersen_95"] = {}

    # ------------------------------------------------------------------
    # 5. Portfolio optimization
    # ------------------------------------------------------------------
    portfolio_results: dict = {}
    try:
        portfolio_results = run_all_optimizations(prices, norm_weights)
    except Exception as exc:
        errors.append(f"Portfolio optimization failed: {exc}")

    # ------------------------------------------------------------------
    # 6. Anomaly detection
    # ------------------------------------------------------------------
    anomalies: dict = {}
    try:
        z_flags = detect_anomalies_zscore(pr, threshold=2.5, window=60)
        anomaly_idx_z = pr.index[z_flags]
        anomalies["n_zscore"] = int(z_flags.sum())
        anomalies["dates_zscore"] = [
            str(d.date()) if hasattr(d, "date") else str(d)
            for d in anomaly_idx_z
        ]
        anomalies["zscore_flags"] = z_flags
    except Exception as exc:
        errors.append(f"Z-score anomaly detection failed: {exc}")

    try:
        if_flags = detect_anomalies_isolation_forest(pr)
        anomaly_idx_if = pr.index[if_flags]
        anomalies["n_isolation_forest"] = int(if_flags.sum())
        anomalies["dates_isolation_forest"] = [
            str(d.date()) if hasattr(d, "date") else str(d)
            for d in anomaly_idx_if
        ]
        anomalies["isolation_forest_flags"] = if_flags
    except Exception as exc:
        errors.append(f"Isolation Forest anomaly detection failed: {exc}")

    # ------------------------------------------------------------------
    # 7. Compile context for agents
    # ------------------------------------------------------------------
    try:
        agent_context = compile_analysis_context(
            tickers=list(norm_weights.keys()),
            weights=norm_weights,
            prices=prices,
            returns=returns,
            portfolio_returns=portfolio_returns,
            risk_metrics=risk_metrics,
            backtest_results=backtest_results,
            portfolio_results=portfolio_results,
            anomalies=anomalies,
        )
    except Exception as exc:
        errors.append(f"Context compilation failed: {exc}")
        agent_context = {}

    # ------------------------------------------------------------------
    # 8. Run agents (optional)
    # ------------------------------------------------------------------
    agent_outputs: dict = {}
    if run_agents and agent_context:
        try:
            agent_outputs = run_risk_analysis_crew(agent_context)
        except Exception as exc:
            errors.append(f"Agent crew failed: {exc}")

    if errors:
        for e in errors:
            logger.warning("Analysis pipeline warning: %s", e)

    return {
        "prices": prices,
        "returns": returns,
        "portfolio_returns": portfolio_returns,
        "norm_weights": norm_weights,
        "risk_metrics": risk_metrics,
        "backtest_results": backtest_results,
        "portfolio_results": portfolio_results,
        "anomalies": anomalies,
        "agent_context": agent_context,
        "agent_outputs": agent_outputs,
        "errors": errors,
    }
