"""
Core risk computation functions.

All functions operate on pandas Series/DataFrames of daily returns
(not prices). Returns are in decimal form (0.01 = 1%).

VaR and ES are reported as positive numbers representing maximum expected loss.
Example: VaR = 0.025 means the portfolio is expected to lose at most 2.5%
at the given confidence level on a given day.
"""

from typing import Literal
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Returns computation
# ---------------------------------------------------------------------------


def compute_returns(
    prices: pd.Series | pd.DataFrame,
    method: Literal["log", "simple"] = "log",
) -> pd.Series | pd.DataFrame:
    """
    Compute daily returns from a price series or DataFrame.

    Parameters
    ----------
    prices : pd.Series or pd.DataFrame of adjusted close prices
    method : "log" for log returns, "simple" for percentage returns

    Returns
    -------
    Returns series/DataFrame with the same type as input (first row dropped).
    """
    if method == "log":
        result = np.log(prices / prices.shift(1)).dropna()
    else:
        result = prices.pct_change().dropna()
    return result


# ---------------------------------------------------------------------------
# Portfolio returns
# ---------------------------------------------------------------------------


def compute_portfolio_returns(
    returns: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """
    Compute portfolio-level daily returns given asset returns and weights.

    Weights are automatically normalized to sum to 1.

    Parameters
    ----------
    returns : DataFrame of asset returns (columns = tickers)
    weights : dict mapping ticker -> weight

    Returns
    -------
    pd.Series of daily portfolio returns
    """
    w = pd.Series(weights)
    w = w / w.sum()  # normalize
    common = returns.columns.intersection(w.index)
    if len(common) == 0:
        raise ValueError("No overlap between returns columns and weights keys.")
    return (returns[common] * w[common]).sum(axis=1)


# ---------------------------------------------------------------------------
# Historical simulation VaR and ES
# ---------------------------------------------------------------------------


def historical_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Historical simulation VaR at confidence level (1 - alpha).

    alpha = 0.05 gives 95% VaR. Returns a positive number (loss magnitude).
    This is the empirical (1-alpha) quantile of the loss distribution.

    Parameters
    ----------
    returns : daily return series
    alpha   : tail probability, e.g. 0.05 for 95% confidence

    Returns
    -------
    float VaR >= 0
    """
    return float(-np.percentile(returns.dropna(), alpha * 100))


def historical_es(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Expected Shortfall (CVaR) via historical simulation.

    ES is the mean loss conditional on exceeding VaR. It is always >= VaR
    and gives a better picture of tail severity.

    Parameters
    ----------
    returns : daily return series
    alpha   : tail probability

    Returns
    -------
    float ES >= 0
    """
    var = historical_var(returns, alpha)
    tail = returns[returns <= -var]
    if len(tail) == 0:
        return var
    return float(-tail.mean())


# ---------------------------------------------------------------------------
# Parametric (variance-covariance) VaR and ES
# ---------------------------------------------------------------------------


def variance_covariance_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Parametric VaR assuming normally distributed returns.

    Uses the formula: VaR = -(mu + z_alpha * sigma) where z_alpha is the
    alpha-quantile of the standard normal distribution.

    This method underestimates tail risk for fat-tailed return distributions
    (which is most real-world equity data).

    Parameters
    ----------
    returns : daily return series
    alpha   : tail probability

    Returns
    -------
    float VaR >= 0
    """
    mu = float(returns.mean())
    sigma = float(returns.std())
    z = stats.norm.ppf(alpha)
    return float(-(mu + z * sigma))


def variance_covariance_es(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Parametric ES under the normality assumption.

    Uses: ES = -(mu - sigma * phi(z_alpha) / alpha) where phi is the
    standard normal PDF.

    Parameters
    ----------
    returns : daily return series
    alpha   : tail probability

    Returns
    -------
    float ES >= 0
    """
    mu = float(returns.mean())
    sigma = float(returns.std())
    z = stats.norm.ppf(alpha)
    return float(-(mu - sigma * stats.norm.pdf(z) / alpha))


# ---------------------------------------------------------------------------
# Monte Carlo VaR and ES
# ---------------------------------------------------------------------------


def monte_carlo_var(
    returns: pd.Series,
    alpha: float = 0.05,
    n_sims: int = 10_000,
    horizon: int = 1,
    seed: int = 42,
) -> float:
    """
    Monte Carlo VaR via bootstrap resampling of historical returns.

    Bootstraps n_sims paths of length `horizon` days, then computes the
    alpha-quantile of the simulated horizon P&L distribution.

    Using bootstrap (not parametric simulation) preserves the empirical
    distribution including fat tails, skewness, and autocorrelation structure.

    Parameters
    ----------
    returns : daily return series
    alpha   : tail probability
    n_sims  : number of simulated paths
    horizon : number of trading days in the horizon
    seed    : random seed for reproducibility

    Returns
    -------
    float VaR >= 0
    """
    rng = np.random.default_rng(seed)
    r = returns.dropna().values
    simulated = rng.choice(r, size=(n_sims, horizon), replace=True)
    horizon_returns = simulated.sum(axis=1)
    return float(-np.percentile(horizon_returns, alpha * 100))


def monte_carlo_es(
    returns: pd.Series,
    alpha: float = 0.05,
    n_sims: int = 10_000,
    horizon: int = 1,
    seed: int = 42,
) -> float:
    """
    Monte Carlo ES via bootstrap resampling.

    Parameters
    ----------
    returns : daily return series
    alpha   : tail probability
    n_sims  : number of simulated paths
    horizon : number of trading days in the horizon

    Returns
    -------
    float ES >= 0
    """
    rng = np.random.default_rng(seed)
    r = returns.dropna().values
    simulated = rng.choice(r, size=(n_sims, horizon), replace=True)
    horizon_returns = simulated.sum(axis=1)
    var = -np.percentile(horizon_returns, alpha * 100)
    tail = horizon_returns[horizon_returns <= -var]
    if len(tail) == 0:
        return float(var)
    return float(-tail.mean())


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------


def scale_var_to_horizon(var_1d: float, horizon: int = 10) -> float:
    """
    Scale a 1-day VaR to an N-day VaR using the square-root-of-time rule.

    Assumes i.i.d. returns. For GARCH processes, this underestimates risk
    in high-volatility regimes because it ignores volatility clustering.

    Parameters
    ----------
    var_1d  : 1-day VaR (positive number)
    horizon : number of trading days

    Returns
    -------
    float N-day VaR
    """
    return float(var_1d * np.sqrt(horizon))


# ---------------------------------------------------------------------------
# Drawdown analysis
# ---------------------------------------------------------------------------


def compute_drawdown(returns: pd.Series) -> pd.Series:
    """
    Compute the drawdown series from a return series.

    Drawdown at time t is (cumulative wealth at t - running peak) / running peak.
    Values are <= 0.

    Returns
    -------
    pd.Series of drawdown values (same index as returns)
    """
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    return (cum - peak) / peak


def max_drawdown(returns: pd.Series) -> float:
    """
    Return the maximum (deepest) drawdown over the full period.

    A return of -0.30 means the portfolio fell 30% from its peak at some point.

    Returns
    -------
    float <= 0
    """
    return float(compute_drawdown(returns).min())


# ---------------------------------------------------------------------------
# Volatility and regime classification
# ---------------------------------------------------------------------------


def rolling_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Rolling annualized realized volatility.

    Annualizes by multiplying daily std by sqrt(252).

    Parameters
    ----------
    returns : daily return series
    window  : rolling window in trading days (default 20 = 1 month)

    Returns
    -------
    pd.Series of annualized rolling volatility
    """
    return returns.rolling(window).std() * np.sqrt(252)


def regime_classifier(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Classify each trading day into a volatility regime based on rolling
    realized volatility percentile.

    Regime boundaries:
      - "low"  : rolling vol <= 33rd percentile of full history
      - "mid"  : 33rd < rolling vol <= 67th percentile
      - "high" : rolling vol > 67th percentile

    Parameters
    ----------
    returns : daily return series
    window  : rolling window for volatility estimation

    Returns
    -------
    pd.Series[str] with values "low", "mid", or "high"
    """
    rv = rolling_volatility(returns, window).dropna()

    p33 = rv.quantile(0.33)
    p67 = rv.quantile(0.67)

    def _label(v: float) -> str:
        if v <= p33:
            return "low"
        if v <= p67:
            return "mid"
        return "high"

    return rv.apply(_label)


def current_regime(returns: pd.Series, window: int = 20) -> str:
    """
    Return the volatility regime for the most recent trading day.

    Returns
    -------
    str: "low", "mid", or "high"
    """
    regimes = regime_classifier(returns, window)
    return str(regimes.iloc[-1]) if not regimes.empty else "unknown"


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------


def detect_anomalies_zscore(
    returns: pd.Series,
    threshold: float = 2.5,
    window: int = 60,
) -> pd.Series:
    """
    Flag days where the return deviates more than `threshold` rolling standard
    deviations from the rolling mean.

    Returns
    -------
    pd.Series[bool] -- True on anomaly days
    """
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    z_scores = (returns - rolling_mean) / rolling_std
    return z_scores.abs() > threshold


def detect_anomalies_isolation_forest(
    returns: pd.Series,
    contamination: float = 0.05,
    window: int = 5,
) -> pd.Series:
    """
    Flag anomalous days using Isolation Forest on rolling return features.

    Features: return, absolute return, rolling 5-day return sum.

    Returns
    -------
    pd.Series[bool] -- True on anomaly days
    """
    from sklearn.ensemble import IsolationForest

    r = returns.dropna()
    features = pd.DataFrame(
        {
            "return": r,
            "abs_return": r.abs(),
            "rolling_5": r.rolling(window).sum(),
        }
    ).dropna()

    iso = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
    )
    preds = iso.fit_predict(features)
    anomaly_flags = pd.Series(preds == -1, index=features.index)

    # Align with original index
    result = pd.Series(False, index=r.index)
    result.loc[anomaly_flags.index] = anomaly_flags
    return result


# ---------------------------------------------------------------------------
# Return statistics summary
# ---------------------------------------------------------------------------


def return_stats(returns: pd.Series) -> dict:
    """
    Compute a comprehensive dictionary of return statistics.

    All volatility/VaR metrics are in daily decimal form unless noted.

    Returns
    -------
    dict with keys: mean_daily, mean_annual, std_daily, std_annual,
                    skewness, excess_kurtosis, min_return, max_return,
                    recent_vol_20d, sharpe_ratio_annual
    """
    r = returns.dropna()
    mean_d = float(r.mean())
    std_d = float(r.std())
    rv_20 = float(rolling_volatility(r, 20).iloc[-1]) if len(r) >= 20 else std_d * np.sqrt(252)
    sharpe = (mean_d * 252) / (std_d * np.sqrt(252)) if std_d > 0 else 0.0

    return {
        "mean_daily": mean_d,
        "mean_annual": float(mean_d * 252),
        "std_daily": std_d,
        "std_annual": float(std_d * np.sqrt(252)),
        "skewness": float(stats.skew(r)),
        "excess_kurtosis": float(stats.kurtosis(r)),
        "min_return": float(r.min()),
        "max_return": float(r.max()),
        "recent_vol_20d": rv_20,
        "sharpe_ratio_annual": sharpe,
        "n_observations": len(r),
    }
