"""
GARCH(p,q) volatility modeling for conditional VaR and ES forecasting.

Why GARCH for risk:
  Historical and parametric VaR assume constant volatility, which is wrong.
  Real returns exhibit volatility clustering: large moves follow large moves.
  GARCH(1,1) captures this by making today's variance a function of:
    - yesterday's squared return (ARCH term, alpha)
    - yesterday's variance forecast (GARCH term, beta)

  The sum alpha + beta is the "persistence" parameter.
  High persistence (e.g. 0.97) means volatility shocks last a long time.
  Low persistence (e.g. 0.70) means volatility reverts to its mean quickly.

Minimum data requirement: at least 100 observations for reliable estimation.
Recommended: 500+ observations (2 years of daily data).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Minimum number of observations required for GARCH
_MIN_OBS = 100


def fit_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
) -> Optional[dict]:
    """
    Fit a GARCH(p,q) model with constant mean and normal innovations.

    Returns None if there is not enough data or fitting fails.

    Parameters
    ----------
    returns : daily return series in decimal form (e.g. 0.01 = 1%)
    p       : ARCH order (number of lagged squared residuals)
    q       : GARCH order (number of lagged variances)

    Returns
    -------
    dict with keys:
        result      -- the fitted arch.ARCHModelResult
        aic         -- Akaike information criterion
        bic         -- Bayesian information criterion
        persistence -- alpha[1] + beta[1] (only meaningful for GARCH(1,1))
        params      -- parameter dict
        annualized_long_run_vol -- sqrt(252) * long-run daily volatility
    """
    try:
        from arch import arch_model
    except ImportError:
        logger.error("arch package not installed. Run: pip install arch")
        return None

    r = returns.dropna()
    if len(r) < _MIN_OBS:
        logger.warning(
            "GARCH fitting skipped: only %d observations (need %d).",
            len(r), _MIN_OBS,
        )
        return None

    try:
        # Scale to percentage returns for numerical stability
        scaled = r * 100.0

        model = arch_model(
            scaled,
            vol="Garch",
            p=p,
            q=q,
            dist="normal",
            mean="Constant",
            rescale=False,
        )
        result = model.fit(disp="off", show_warning=False)

        alpha = float(result.params.get("alpha[1]", 0.0))
        beta = float(result.params.get("beta[1]", 0.0))
        omega = float(result.params.get("omega", 0.0))
        persistence = alpha + beta

        # Long-run (unconditional) daily variance from GARCH(1,1)
        if persistence < 1.0 and (1 - persistence) > 0:
            long_run_var_scaled = omega / (1 - persistence)
        else:
            long_run_var_scaled = float(scaled.var())

        long_run_daily_vol = np.sqrt(long_run_var_scaled) / 100.0

        return {
            "result": result,
            "aic": float(result.aic),
            "bic": float(result.bic),
            "persistence": round(persistence, 6),
            "params": {k: float(v) for k, v in result.params.items()},
            "annualized_long_run_vol": round(long_run_daily_vol * np.sqrt(252), 6),
        }

    except Exception as exc:
        logger.warning("GARCH fitting failed: %s", exc)
        return None


def garch_var_forecast(
    returns: pd.Series,
    alpha: float = 0.05,
    horizon: int = 1,
    p: int = 1,
    q: int = 1,
) -> Optional[dict]:
    """
    Forecast next-day (or N-day) VaR and ES using a fitted GARCH(p,q) model.

    The conditional variance for the next period comes directly from the
    GARCH model's one-step-ahead forecast. This is more accurate than the
    square-root-of-time scaling for short horizons.

    Parameters
    ----------
    returns : daily return series
    alpha   : tail probability (e.g. 0.05 for 95% VaR)
    horizon : forecast horizon in trading days
    p, q    : GARCH orders

    Returns
    -------
    dict with keys: var, es, conditional_sigma, persistence, horizon
    Returns None if fitting fails.
    """
    fit = fit_garch(returns, p=p, q=q)
    if fit is None:
        return None

    result = fit["result"]

    try:
        forecasts = result.forecast(horizon=horizon, reindex=False)
        # conditional variance is in (scaled returns)^2 units
        cond_var_scaled = float(forecasts.variance.values[-1, horizon - 1])
        cond_sigma = np.sqrt(cond_var_scaled) / 100.0  # back to decimal returns

        mu_scaled = float(result.params.get("mu", 0.0))
        mu = mu_scaled / 100.0

        z = stats.norm.ppf(alpha)
        var = float(-(mu + z * cond_sigma))
        es = float(-(mu - cond_sigma * stats.norm.pdf(z) / alpha))

        return {
            "var": round(max(var, 0.0), 6),
            "es": round(max(es, 0.0), 6),
            "conditional_sigma": round(cond_sigma, 6),
            "conditional_sigma_annual": round(cond_sigma * np.sqrt(252), 6),
            "persistence": fit["persistence"],
            "horizon": horizon,
            "annualized_long_run_vol": fit["annualized_long_run_vol"],
        }

    except Exception as exc:
        logger.warning("GARCH forecast failed: %s", exc)
        return None


def garch_conditional_vol_series(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
) -> Optional[pd.Series]:
    """
    Return the in-sample conditional volatility series from a fitted GARCH model.

    Useful for charting how the model estimates volatility evolved over the
    backtest period and for regime-aware analysis.

    Returns
    -------
    pd.Series of daily conditional volatility (annualized), same index as returns.
    Returns None if fitting fails.
    """
    fit = fit_garch(returns, p=p, q=q)
    if fit is None:
        return None

    result = fit["result"]
    # conditional_volatility is in scaled-return units
    cond_vol_scaled = result.conditional_volatility
    cond_vol_daily = cond_vol_scaled / 100.0
    cond_vol_annual = cond_vol_daily * np.sqrt(252)

    series = pd.Series(cond_vol_annual.values, index=returns.dropna().index[-len(cond_vol_annual):])
    series.name = "garch_cond_vol_annual"
    return series
