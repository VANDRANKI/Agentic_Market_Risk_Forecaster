"""
Risk metrics display components for the Streamlit UI.

Each function renders a specific section of the risk analysis output.
All functions receive pre-computed results from app/analysis.py.
"""

import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Color coding helpers
# ---------------------------------------------------------------------------


def _regime_color(regime: str) -> str:
    mapping = {"low": "#10b981", "mid": "#f59e0b", "high": "#ef4444"}
    return mapping.get(regime.lower(), "#94a3b8")


def _backtest_color(reject: bool) -> str:
    return "#ef4444" if reject else "#10b981"


def _pval_label(p_value, reject: bool) -> str:
    status = "FAIL" if reject else "PASS"
    return f"{status}  (p = {p_value:.3f})"


# ---------------------------------------------------------------------------
# Summary metric cards (top of page)
# ---------------------------------------------------------------------------


def render_summary_cards(risk_metrics: dict, confidence_level: float = 0.95) -> None:
    """
    Render four key metric cards at the top of the main area.

    Shows: primary VaR, primary ES, max drawdown, volatility regime.
    """
    cl = int(confidence_level * 100)
    hist_var = risk_metrics.get("hist_var_95" if cl == 95 else "hist_var_99", 0)
    hist_es = risk_metrics.get("hist_es_95" if cl == 95 else "hist_es_99", 0)
    mdd = risk_metrics.get("max_drawdown", 0)
    regime = risk_metrics.get("regime", "unknown")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card risk-high">
                <div class="metric-label">{cl}% VaR (1-day)</div>
                <div class="metric-value">{hist_var * 100:.2f}%</div>
                <div class="metric-sub">Historical Simulation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card risk-high">
                <div class="metric-label">{cl}% ES (1-day)</div>
                <div class="metric-value">{hist_es * 100:.2f}%</div>
                <div class="metric-sub">Expected Shortfall</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card risk-medium">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value">{mdd * 100:.1f}%</div>
                <div class="metric-sub">Peak-to-Trough</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    regime_color = _regime_color(regime)
    with col4:
        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 4px solid {regime_color};">
                <div class="metric-label">Volatility Regime</div>
                <div class="metric-value" style="color:{regime_color};">
                    {regime.upper()}
                </div>
                <div class="metric-sub">20-day rolling vol</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# VaR/ES comparison table
# ---------------------------------------------------------------------------


def render_var_table(risk_metrics: dict) -> None:
    """Render a table comparing VaR and ES across all 4 methods."""
    st.markdown("#### VaR and ES by Method (1-day, % portfolio value)")

    methods = ["Historical", "Parametric", "Monte Carlo", "GARCH"]
    keys = [
        ("hist_var_95", "hist_es_95", "hist_var_99", "hist_es_99"),
        ("param_var_95", "param_es_95", "param_var_99", "param_es_99"),
        ("mc_var_95", "mc_es_95", "mc_var_99", "mc_es_99"),
        ("garch_var_95", "garch_es_95", "garch_var_99", "garch_es_99"),
    ]

    rows = []
    for method, (v95, e95, v99, e99) in zip(methods, keys):
        gv95 = risk_metrics.get(v95)
        ge95 = risk_metrics.get(e95)
        gv99 = risk_metrics.get(v99)
        ge99 = risk_metrics.get(e99)
        rows.append({
            "Method": method,
            "VaR 95%": f"{gv95*100:.3f}%" if gv95 is not None else "N/A",
            "ES 95%": f"{ge95*100:.3f}%" if ge95 is not None else "N/A",
            "VaR 99%": f"{gv99*100:.3f}%" if gv99 is not None else "N/A",
            "ES 99%": f"{ge99*100:.3f}%" if ge99 is not None else "N/A",
        })

    df = pd.DataFrame(rows).set_index("Method")
    st.dataframe(df, use_container_width=True)

    # 10-day VaR callout
    var_10d = risk_metrics.get("var_10d_95", 0)
    st.markdown(
        f"**10-Day VaR (95%, sqrt-of-time scaled):** `{var_10d*100:.3f}%`  "
        f"Assumes i.i.d. returns. GARCH-based 10-day VaR would be higher in "
        f"a high-vol regime because it accounts for volatility clustering."
    )


# ---------------------------------------------------------------------------
# Backtesting results
# ---------------------------------------------------------------------------


def render_backtest_results(backtest_results: dict) -> None:
    """Render Kupiec and Christoffersen test results in a clear panel."""
    st.markdown("#### VaR Backtest Results")

    st.markdown(
        "Both tests use the rolling 1-year VaR estimated from historical data. "
        "**Pass** means the model's calibration is statistically acceptable at 5% significance."
    )

    kupiec = backtest_results.get("kupiec_95", {})
    chri = backtest_results.get("christoffersen_95", {})

    n_exc = backtest_results.get("n_exceedances_95", 0)
    n_exp = backtest_results.get("expected_exceedances_95", 0)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Actual Exceedances", n_exc)
    with col2:
        st.metric("Expected Exceedances", n_exp)
    with col3:
        rate = backtest_results.get("exceedance_rate_95", 0)
        st.metric("Exceedance Rate", f"{rate*100:.2f}%")

    st.markdown("")

    col_a, col_b = st.columns(2)
    with col_a:
        k_reject = kupiec.get("reject_h0", False)
        k_color = _backtest_color(k_reject)
        k_label = _pval_label(kupiec.get("p_value", 0), k_reject)
        st.markdown(
            f"""
            <div class="backtest-card" style="border-left: 4px solid {k_color};">
                <strong>Kupiec POF Test</strong><br>
                <span style="color:{k_color}; font-size:1.1rem;">{k_label}</span><br>
                <small>{kupiec.get('interpretation', '')}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        c_reject = chri.get("reject_h0", False)
        c_color = _backtest_color(c_reject)
        c_label = _pval_label(chri.get("p_value", 0), c_reject)
        st.markdown(
            f"""
            <div class="backtest-card" style="border-left: 4px solid {c_color};">
                <strong>Christoffersen Independence Test</strong><br>
                <span style="color:{c_color}; font-size:1.1rem;">{c_label}</span><br>
                <small>{chri.get('interpretation', '')}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Exceedance dates
    exc_dates = backtest_results.get("exceedance_dates_95", [])
    if exc_dates:
        with st.expander(f"View all {len(exc_dates)} exceedance dates"):
            for d in exc_dates:
                st.markdown(f"- {d}")


# ---------------------------------------------------------------------------
# Portfolio optimization results
# ---------------------------------------------------------------------------


def render_portfolio_results(portfolio_results: dict, current_weights: dict) -> None:
    """Render the three optimized portfolios side by side."""
    st.markdown("#### Portfolio Optimization Comparison")

    current = portfolio_results.get("current")
    max_sharpe = portfolio_results.get("max_sharpe")
    min_vol = portfolio_results.get("min_vol")
    eq_wt = portfolio_results.get("equal_weight")

    strategies = []
    if current:
        strategies.append(("Current Portfolio", current))
    if max_sharpe:
        strategies.append(("Max Sharpe", max_sharpe))
    if min_vol:
        strategies.append(("Min Volatility", min_vol))
    if eq_wt:
        strategies.append(("Equal Weight", eq_wt))

    if not strategies:
        st.warning("Portfolio optimization results unavailable.")
        return

    cols = st.columns(len(strategies))
    for col, (name, data) in zip(cols, strategies):
        with col:
            vol_pct = data.get("volatility", 0) * 100
            ret_pct = data.get("expected_return", 0) * 100
            sharpe = data.get("sharpe_ratio", 0)

            st.markdown(f"**{name}**")
            st.metric("Annual Volatility", f"{vol_pct:.1f}%")
            st.metric("Expected Return", f"{ret_pct:.1f}%")
            st.metric("Sharpe Ratio", f"{sharpe:.3f}")

            st.markdown("**Weights:**")
            weights = data.get("weights", {})
            for ticker, w in sorted(weights.items(), key=lambda x: -x[1]):
                if w > 0.005:
                    bar_width = int(w * 100)
                    st.markdown(
                        f"<div style='display:flex;align-items:center;gap:8px;'>"
                        f"<span style='width:60px;font-size:0.85rem;'>{ticker}</span>"
                        f"<div style='background:#3b82f6;height:10px;width:{bar_width}%;border-radius:3px;'></div>"
                        f"<span style='font-size:0.85rem;'>{w*100:.1f}%</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )


# ---------------------------------------------------------------------------
# GARCH details
# ---------------------------------------------------------------------------


def render_garch_details(risk_metrics: dict) -> None:
    """Render GARCH model details in an expander."""
    garch_persistence = risk_metrics.get("garch_persistence")
    garch_lr_vol = risk_metrics.get("garch_long_run_vol")
    garch_var = risk_metrics.get("garch_var_95")

    if garch_var is None:
        st.info("GARCH model was not available for this analysis (insufficient data or fitting error).")
        return

    with st.expander("GARCH(1,1) Model Details"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Conditional VaR 95%", f"{garch_var*100:.3f}%")
        with col2:
            if garch_persistence is not None:
                st.metric("Persistence (alpha+beta)", f"{garch_persistence:.4f}")
        with col3:
            if garch_lr_vol is not None:
                st.metric("Long-run Annual Vol", f"{garch_lr_vol*100:.1f}%")

        if garch_persistence is not None:
            if garch_persistence > 0.95:
                msg = (
                    f"Persistence of {garch_persistence:.3f} is very high. "
                    "Volatility shocks decay slowly, meaning today's high vol "
                    "regime is likely to persist for many days. "
                    "Historical VaR will underestimate forward risk in this regime."
                )
                st.warning(msg)
            elif garch_persistence > 0.85:
                msg = (
                    f"Persistence of {garch_persistence:.3f} indicates moderate "
                    "volatility clustering. Shocks last but revert within weeks."
                )
                st.info(msg)
            else:
                msg = (
                    f"Persistence of {garch_persistence:.3f} indicates fast "
                    "mean-reversion in volatility. Shocks dissipate quickly."
                )
                st.success(msg)
