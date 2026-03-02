"""
VaR backtesting: Kupiec (1995) and Christoffersen (1998) tests.

These tests answer two questions:
  1. Kupiec: Is the proportion of VaR exceedances close to the expected alpha?
  2. Christoffersen: Are exceedances independent (not clustered in time)?

Both are required for a complete VaR model validation because a model
can pass Kupiec (correct count) but fail Christoffersen (all breaches
happen in a cluster, signaling the model is slow to update during crises).
"""

import numpy as np
import pandas as pd
from scipy import stats


def count_exceedances(
    returns: pd.Series,
    var_series: pd.Series,
) -> dict:
    """
    Count days where actual portfolio loss exceeds the predicted VaR.

    An exceedance occurs when: return < -VaR  (loss exceeds the forecast).

    Parameters
    ----------
    returns    : realized daily portfolio returns
    var_series : predicted 1-day VaR for each day (positive number = loss)

    Returns
    -------
    dict with keys: n_exceedances, n_observations, exceedance_rate,
                    exceedance_dates (list of date strings)
    """
    aligned = returns.align(var_series, join="inner")
    r, v = aligned[0], aligned[1]

    violations = r < -v
    n_exc = int(violations.sum())
    n_obs = len(r)

    exc_dates = [str(d.date()) if hasattr(d, "date") else str(d)
                 for d in r.index[violations]]

    return {
        "n_exceedances": n_exc,
        "n_observations": n_obs,
        "exceedance_rate": round(n_exc / n_obs, 6) if n_obs > 0 else 0.0,
        "exceedance_dates": exc_dates,
    }


def kupiec_test(
    n_exceptions: int,
    n_observations: int,
    alpha: float = 0.05,
) -> dict:
    """
    Kupiec (1995) Proportion of Failures (POF) test.

    Tests whether the observed exceedance rate equals the expected rate alpha.
    Uses a likelihood ratio test statistic that follows chi-squared(1) under H0.

    H0: p_hat = alpha  (model is correctly calibrated)
    H1: p_hat != alpha

    Reject H0 (p-value < 0.05) means the VaR model is mis-calibrated:
    either too conservative (too few breaches) or too optimistic (too many).

    Parameters
    ----------
    n_exceptions  : actual number of VaR breaches
    n_observations: total number of evaluation days
    alpha         : confidence level tail probability (e.g. 0.05 for 95% VaR)

    Returns
    -------
    dict with keys: statistic, p_value, reject_h0, expected_exceptions,
                    actual_exceptions, test_name, interpretation
    """
    if n_observations == 0:
        return {
            "statistic": None, "p_value": None, "reject_h0": None,
            "expected_exceptions": 0, "actual_exceptions": 0,
            "test_name": "Kupiec POF", "interpretation": "Insufficient data."
        }

    p_hat = n_exceptions / n_observations
    p = alpha

    # Log-likelihood ratio statistic
    eps = 1e-10  # avoid log(0)
    if p_hat < eps:
        lr = -2.0 * (n_observations * np.log(max(1 - p, eps)))
    elif p_hat > 1 - eps:
        lr = -2.0 * (n_observations * np.log(max(p, eps)))
    else:
        lr = -2.0 * (
            n_exceptions * np.log(p / p_hat)
            + (n_observations - n_exceptions) * np.log((1 - p) / (1 - p_hat))
        )

    p_value = float(1 - stats.chi2.cdf(lr, df=1))
    reject = p_value < 0.05
    expected = round(n_observations * alpha)

    if reject:
        if n_exceptions > expected:
            interp = (
                f"Model underestimates risk. "
                f"Observed {n_exceptions} breaches vs {expected} expected. "
                f"Actual loss exceeds VaR too often."
            )
        else:
            interp = (
                f"Model is overly conservative. "
                f"Only {n_exceptions} breaches vs {expected} expected. "
                f"VaR is set too high, tying up unnecessary capital."
            )
    else:
        interp = (
            f"Model is correctly calibrated. "
            f"{n_exceptions} breaches vs {expected} expected "
            f"(p-value {p_value:.3f} >= 0.05)."
        )

    return {
        "statistic": round(lr, 4),
        "p_value": round(p_value, 4),
        "reject_h0": reject,
        "expected_exceptions": expected,
        "actual_exceptions": n_exceptions,
        "test_name": "Kupiec POF",
        "interpretation": interp,
    }


def christoffersen_test(
    returns: pd.Series,
    var_series: pd.Series,
    alpha: float = 0.05,
) -> dict:
    """
    Christoffersen (1998) Independence (LR_ind) test.

    Tests whether VaR exceedances are independent over time.
    If breaches cluster (e.g., 5 breaches in a row during a crash), the model
    fails to adapt quickly enough and the independence assumption is violated.

    H0: Exceedances are independent across consecutive days.
    H1: Exceedances exhibit first-order Markov clustering.

    Reject H0 means exceedances are clustered, which indicates the model
    does not respond to changing market conditions (no volatility updating).

    Parameters
    ----------
    returns    : realized daily portfolio returns
    var_series : predicted 1-day VaR (positive loss)
    alpha      : tail probability

    Returns
    -------
    dict with keys: statistic, p_value, reject_h0, pi01, pi11,
                    test_name, interpretation
    """
    aligned = returns.align(var_series, join="inner")
    r, v = aligned[0], aligned[1]

    violations = (r < -v).astype(int).values

    # Count first-order Markov transition counts
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(violations)):
        prev, curr = int(violations[i - 1]), int(violations[i])
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1

    eps = 1e-10
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11 + eps)

    def _log_lik(p01: float, p11: float) -> float:
        ll = 0.0
        if n00 > 0:
            ll += n00 * np.log(max(1 - p01, eps))
        if n01 > 0:
            ll += n01 * np.log(max(p01, eps))
        if n10 > 0:
            ll += n10 * np.log(max(1 - p11, eps))
        if n11 > 0:
            ll += n11 * np.log(max(p11, eps))
        return ll

    lr = float(-2.0 * (_log_lik(pi, pi) - _log_lik(pi01, pi11)))
    lr = max(lr, 0.0)  # numerical safety
    p_value = float(1 - stats.chi2.cdf(lr, df=1))
    reject = p_value < 0.05

    if reject:
        interp = (
            f"Exceedances are clustered (p-value {p_value:.3f} < 0.05). "
            f"The model does not update fast enough for changing volatility. "
            f"Conditional breach rate was {pi11:.1%} given a previous breach "
            f"vs {pi01:.1%} baseline."
        )
    else:
        interp = (
            f"Exceedances appear independent (p-value {p_value:.3f} >= 0.05). "
            f"No significant clustering of VaR breaches detected."
        )

    return {
        "statistic": round(lr, 4),
        "p_value": round(p_value, 4),
        "reject_h0": reject,
        "pi01": round(pi01, 4),
        "pi11": round(pi11, 4),
        "test_name": "Christoffersen Independence",
        "interpretation": interp,
    }


def rolling_var_series(
    returns: pd.Series,
    alpha: float = 0.05,
    window: int = 252,
) -> pd.Series:
    """
    Compute a rolling historical VaR series for backtesting.

    For each day t, VaR is estimated from the prior `window` days.
    This simulates the out-of-sample performance of a 1-year lookback model.

    Parameters
    ----------
    returns : daily return series
    alpha   : tail probability
    window  : estimation window (252 = 1 trading year)

    Returns
    -------
    pd.Series of rolling VaR estimates (positive loss values)
    """

    def _hist_var(window_returns: pd.Series) -> float:
        return float(-np.percentile(window_returns, alpha * 100))

    return returns.rolling(window).apply(_hist_var, raw=False).shift(1).dropna()
