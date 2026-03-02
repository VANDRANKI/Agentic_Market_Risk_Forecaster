"""
Portfolio construction and optimization using PyPortfolioOpt.

Three strategies are compared for each run:
  1. Max Sharpe     -- maximize risk-adjusted return
  2. Min Volatility -- minimize portfolio variance
  3. Equal Weight   -- simple 1/N benchmark

All three are shown side-by-side in the UI so the user can see the
trade-off between expected return, volatility, and tail risk.

Note: PyPortfolioOpt functions take a DataFrame of PRICES (not returns).
They internally compute returns and covariance.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_pypfopt_import():
    try:
        from pypfopt import EfficientFrontier, risk_models, expected_returns
        return EfficientFrontier, risk_models, expected_returns
    except ImportError:
        logger.error("PyPortfolioOpt not installed. Run: pip install PyPortfolioOpt")
        return None, None, None


def max_sharpe_portfolio(
    prices: pd.DataFrame,
    risk_free_rate: float = 0.04,
) -> Optional[dict]:
    """
    Compute the portfolio that maximizes the Sharpe ratio.

    Uses mean historical return as the expected return estimate and
    sample covariance for the risk estimate.

    Parameters
    ----------
    prices          : DataFrame of adjusted close prices (rows=dates, cols=tickers)
    risk_free_rate  : annualized risk-free rate (default 4%)

    Returns
    -------
    dict with keys: weights, expected_return, volatility, sharpe_ratio, method
    Returns None on failure.
    """
    EfficientFrontier, risk_models, expected_returns = _safe_pypfopt_import()
    if EfficientFrontier is None:
        return None

    try:
        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.sample_cov(prices)

        ef = EfficientFrontier(mu, S)
        ef.max_sharpe(risk_free_rate=risk_free_rate)
        weights = ef.clean_weights()
        perf = ef.portfolio_performance(risk_free_rate=risk_free_rate, verbose=False)

        return {
            "weights": {k: round(v, 6) for k, v in weights.items() if v > 1e-5},
            "expected_return": round(float(perf[0]), 6),
            "volatility": round(float(perf[1]), 6),
            "sharpe_ratio": round(float(perf[2]), 6),
            "method": "Max Sharpe",
        }
    except Exception as exc:
        logger.warning("Max Sharpe optimization failed: %s", exc)
        return None


def min_volatility_portfolio(
    prices: pd.DataFrame,
    risk_free_rate: float = 0.04,
) -> Optional[dict]:
    """
    Compute the minimum variance portfolio.

    Minimizes portfolio volatility regardless of expected return.
    This is the rational choice when you have no confidence in return
    forecasts (which is most of the time).

    Parameters
    ----------
    prices          : DataFrame of adjusted close prices
    risk_free_rate  : annualized risk-free rate for Sharpe calculation (default 4%)

    Returns
    -------
    dict with keys: weights, expected_return, volatility, sharpe_ratio, method
    Returns None on failure.
    """
    EfficientFrontier, risk_models, expected_returns = _safe_pypfopt_import()
    if EfficientFrontier is None:
        return None

    try:
        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.sample_cov(prices)

        ef = EfficientFrontier(mu, S)
        ef.min_volatility()
        weights = ef.clean_weights()
        perf = ef.portfolio_performance(risk_free_rate=risk_free_rate, verbose=False)

        return {
            "weights": {k: round(v, 6) for k, v in weights.items() if v > 1e-5},
            "expected_return": round(float(perf[0]), 6),
            "volatility": round(float(perf[1]), 6),
            "sharpe_ratio": round(float(perf[2]), 6),
            "method": "Min Volatility",
        }
    except Exception as exc:
        logger.warning("Min Volatility optimization failed: %s", exc)
        return None


def equal_weight_portfolio(
    prices: pd.DataFrame,
    risk_free_rate: float = 0.04,
) -> dict:
    """
    Compute equal-weight (1/N) portfolio performance.

    This is the simplest possible portfolio and often surprisingly hard
    to beat out-of-sample. It serves as a benchmark for the optimized portfolios.

    Parameters
    ----------
    prices          : DataFrame of adjusted close prices
    risk_free_rate  : annualized risk-free rate for Sharpe calculation

    Returns
    -------
    dict with keys: weights, expected_return, volatility, sharpe_ratio, method
    """
    EfficientFrontier, risk_models, expected_returns = _safe_pypfopt_import()

    tickers = list(prices.columns)
    n = len(tickers)
    w = 1.0 / n
    weights = {t: round(w, 6) for t in tickers}

    if EfficientFrontier is None:
        return {"weights": weights, "method": "Equal Weight"}

    try:
        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.sample_cov(prices)

        ef = EfficientFrontier(mu, S)
        ef.set_weights(weights)
        perf = ef.portfolio_performance(risk_free_rate=risk_free_rate, verbose=False)

        return {
            "weights": weights,
            "expected_return": round(float(perf[0]), 6),
            "volatility": round(float(perf[1]), 6),
            "sharpe_ratio": round(float(perf[2]), 6),
            "method": "Equal Weight",
        }
    except Exception as exc:
        logger.warning("Equal weight performance calc failed: %s", exc)
        return {"weights": weights, "method": "Equal Weight"}


def run_all_optimizations(
    prices: pd.DataFrame,
    current_weights: dict[str, float],
    risk_free_rate: float = 0.04,
) -> dict:
    """
    Run all three portfolio strategies and add the current (user) portfolio.

    Parameters
    ----------
    prices           : DataFrame of adjusted close prices
    current_weights  : user-specified portfolio weights
    risk_free_rate   : annualized risk-free rate

    Returns
    -------
    dict with keys: current, max_sharpe, min_vol, equal_weight
    Each value is a portfolio dict (or None on failure).
    """
    EfficientFrontier, risk_models, expected_returns = _safe_pypfopt_import()

    # Current portfolio stats
    current_perf = None
    if EfficientFrontier is not None:
        try:
            mu = expected_returns.mean_historical_return(prices)
            S = risk_models.sample_cov(prices)

            # Normalize weights to sum to 1
            w_sum = sum(current_weights.values())
            normalized = {k: v / w_sum for k, v in current_weights.items()}

            ef = EfficientFrontier(mu, S)
            ef.set_weights(normalized)
            perf = ef.portfolio_performance(
                risk_free_rate=risk_free_rate, verbose=False
            )
            current_perf = {
                "weights": {k: round(v, 6) for k, v in normalized.items()},
                "expected_return": round(float(perf[0]), 6),
                "volatility": round(float(perf[1]), 6),
                "sharpe_ratio": round(float(perf[2]), 6),
                "method": "Current Portfolio",
            }
        except Exception as exc:
            logger.warning("Current portfolio performance calc failed: %s", exc)

    return {
        "current": current_perf,
        "max_sharpe": max_sharpe_portfolio(prices, risk_free_rate),
        "min_vol": min_volatility_portfolio(prices, risk_free_rate),
        "equal_weight": equal_weight_portfolio(prices, risk_free_rate),
    }
