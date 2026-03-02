"""
Agentic Market Risk Forecaster -- Streamlit Application

Entry point: streamlit run app/main.py

Architecture:
  - Sidebar collects user inputs (tickers, weights, lookback, confidence).
  - "Run Analysis" triggers the full pipeline in app/analysis.py.
  - Results stored in st.session_state to avoid re-runs on widget interaction.
  - Five tabs display: Overview, Risk Analysis, Backtesting, Portfolio, Agent Insights.
"""

import os
import sys
import yaml
import logging
from pathlib import Path

# Add project root to Python path
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))

# Load environment variables before anything else
from dotenv import load_dotenv
load_dotenv(_root / ".env")

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be the very first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Market Risk Forecaster",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* ---- Space background ---- */
    .stApp {
        background-color: #020a14;
        background-image:
            /* Small stars */
            radial-gradient(1px 1px at 7%  5%,  rgba(255,255,255,0.90), transparent),
            radial-gradient(1px 1px at 13% 18%, rgba(255,255,255,0.70), transparent),
            radial-gradient(1px 1px at 21% 32%, rgba(255,255,255,0.85), transparent),
            radial-gradient(1px 1px at 35% 45%, rgba(255,255,255,0.75), transparent),
            radial-gradient(1px 1px at 50% 8%,  rgba(255,255,255,0.90), transparent),
            radial-gradient(1px 1px at 63% 22%, rgba(255,255,255,0.80), transparent),
            radial-gradient(1px 1px at 77% 14%, rgba(255,255,255,0.85), transparent),
            radial-gradient(1px 1px at 90% 30%, rgba(255,255,255,0.70), transparent),
            radial-gradient(1px 1px at 3%  62%, rgba(255,255,255,0.80), transparent),
            radial-gradient(1px 1px at 17% 90%, rgba(255,255,255,0.75), transparent),
            radial-gradient(1px 1px at 31% 85%, rgba(255,255,255,0.85), transparent),
            radial-gradient(1px 1px at 45% 95%, rgba(255,255,255,0.70), transparent),
            radial-gradient(1px 1px at 59% 80%, rgba(255,255,255,0.80), transparent),
            radial-gradient(1px 1px at 73% 88%, rgba(255,255,255,0.75), transparent),
            radial-gradient(1px 1px at 87% 92%, rgba(255,255,255,0.85), transparent),
            radial-gradient(1px 1px at 11% 48%, rgba(255,255,255,0.65), transparent),
            radial-gradient(1px 1px at 44% 17%, rgba(255,255,255,0.80), transparent),
            radial-gradient(1px 1px at 78% 6%,  rgba(255,255,255,0.75), transparent),
            radial-gradient(1px 1px at 4%  35%, rgba(255,255,255,0.70), transparent),
            radial-gradient(1px 1px at 55% 55%, rgba(255,255,255,0.65), transparent),
            radial-gradient(1px 1px at 68% 75%, rgba(255,255,255,0.80), transparent),
            radial-gradient(1px 1px at 82% 60%, rgba(255,255,255,0.75), transparent),
            /* Medium stars */
            radial-gradient(1.5px 1.5px at 28% 11%, rgba(255,255,255,0.85), transparent),
            radial-gradient(1.5px 1.5px at 42% 28%, rgba(255,255,255,0.90), transparent),
            radial-gradient(1.5px 1.5px at 70% 55%, rgba(255,255,255,0.80), transparent),
            radial-gradient(1.5px 1.5px at 96% 68%, rgba(255,255,255,0.75), transparent),
            radial-gradient(1.5px 1.5px at 24% 73%, rgba(255,255,255,0.85), transparent),
            radial-gradient(1.5px 1.5px at 52% 72%, rgba(255,255,255,0.90), transparent),
            radial-gradient(1.5px 1.5px at 80% 75%, rgba(255,255,255,0.80), transparent),
            radial-gradient(1.5px 1.5px at 19% 57%, rgba(255,255,255,0.70), transparent),
            radial-gradient(1.5px 1.5px at 88% 52%, rgba(255,255,255,0.75), transparent),
            /* Bright large stars */
            radial-gradient(2px 2px at 57% 38%, rgba(255,255,255,0.95), transparent),
            radial-gradient(2px 2px at 83% 42%, rgba(255,255,255,0.90), transparent),
            radial-gradient(2px 2px at 9%  78%, rgba(255,255,255,0.85), transparent),
            radial-gradient(2px 2px at 38% 60%, rgba(255,255,255,0.90), transparent),
            radial-gradient(2px 2px at 66% 65%, rgba(255,255,255,0.85), transparent),
            radial-gradient(2px 2px at 93% 82%, rgba(255,255,255,0.80), transparent),
            radial-gradient(2px 2px at 67% 48%, rgba(255,255,255,0.90), transparent),
            radial-gradient(2px 2px at 99% 15%, rgba(255,255,255,0.85), transparent),
            /* Subtle nebula glows */
            radial-gradient(ellipse 55% 35% at 25% 45%, rgba(59,130,246,0.07), transparent),
            radial-gradient(ellipse 45% 55% at 75% 35%, rgba(139,92,246,0.06), transparent),
            radial-gradient(ellipse 30% 40% at 60% 70%, rgba(16,185,129,0.04), transparent),
            /* Deep space base gradient */
            radial-gradient(ellipse at top center, #0d1b3e 0%, #030b1a 55%, #000000 100%);
        background-attachment: fixed;
        min-height: 100vh;
    }

    /* ---- Sidebar: glass panel over the starfield ---- */
    section[data-testid="stSidebar"] {
        background: rgba(5, 12, 28, 0.88) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(59, 130, 246, 0.12) !important;
    }

    /* ---- Main content: transparent so stars show through gaps ---- */
    .main .block-container {
        background: transparent;
    }

    /* ---- Global ---- */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* ---- Metric cards ---- */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 10px;
        min-height: 100px;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #f1f5f9;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 4px;
    }
    .risk-high  { border-left: 4px solid #ef4444; }
    .risk-medium{ border-left: 4px solid #f59e0b; }
    .risk-low   { border-left: 4px solid #10b981; }

    /* ---- Backtest cards ---- */
    .backtest-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
        line-height: 1.6;
    }

    /* ---- Agent cards ---- */
    .agent-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 18px 20px;
    }

    /* ---- Divider ---- */
    hr { border-color: #1e293b; }

    /* ---- Tabs styling ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 6px 16px;
        font-weight: 500;
    }

    /* ---- Section headers ---- */
    h4 { color: #f1f5f9 !important; margin-top: 1.2rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load configs
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_config():
    settings_path = _root / "configs" / "settings.yaml"
    universe_path = _root / "configs" / "universe.yaml"
    settings = yaml.safe_load(settings_path.read_text()) if settings_path.exists() else {}
    universe = yaml.safe_load(universe_path.read_text()) if universe_path.exists() else {}
    return settings, universe


settings, universe = _load_config()

_lookback_map: dict = settings.get("data", {}).get("lookback_options", {
    "1y": 252, "2y": 504, "3y": 756, "5y": 1260
})
_predefined = universe.get("predefined_portfolios", [])
_default_tickers = universe.get("default_portfolio", {}).get(
    "tickers", ["SPY", "QQQ", "IWM", "AAPL", "MSFT"]
)
_default_weights = universe.get("default_portfolio", {}).get(
    "weights", [0.25, 0.20, 0.15, 0.20, 0.20]
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar() -> dict:
    """
    Render the full sidebar and return the user's input configuration.
    """
    st.sidebar.markdown("## Portfolio Setup")

    # --- Predefined portfolio selector ---
    predefined_names = ["Custom"] + [p["name"] for p in _predefined]
    selected_preset = st.sidebar.selectbox("Load preset portfolio", predefined_names, index=0)

    if selected_preset != "Custom":
        preset = next(p for p in _predefined if p["name"] == selected_preset)
        init_tickers = preset["tickers"]
        init_weights = preset["weights"]
    else:
        init_tickers = _default_tickers
        init_weights = _default_weights

    # --- Ticker input ---
    ticker_str = st.sidebar.text_area(
        "Tickers (one per line)",
        value="\n".join(init_tickers),
        height=120,
        help="Enter valid Yahoo Finance symbols. Example: SPY, QQQ, AAPL",
    )
    tickers = [t.strip().upper() for t in ticker_str.split("\n") if t.strip()]

    if not tickers:
        st.sidebar.error("Enter at least one ticker.")
        st.stop()

    # --- Weights input ---
    st.sidebar.markdown("**Portfolio Weights** (will be normalized)")
    weights_raw = {}
    n = len(tickers)

    equal_weight_btn = st.sidebar.button("Set Equal Weights")

    if "sidebar_weights" not in st.session_state or equal_weight_btn:
        st.session_state["sidebar_weights"] = {t: round(1.0 / n, 4) for t in tickers}

    # Sync session state if tickers changed
    existing = st.session_state.get("sidebar_weights", {})
    for ticker in tickers:
        if ticker not in existing:
            existing[ticker] = round(1.0 / n, 4)
    st.session_state["sidebar_weights"] = {t: existing.get(t, 1.0 / n) for t in tickers}

    for ticker in tickers:
        val = st.sidebar.number_input(
            ticker,
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state["sidebar_weights"].get(ticker, 1.0 / n)),
            step=0.05,
            format="%.2f",
            key=f"w_{ticker}",
        )
        weights_raw[ticker] = val

    total_w = sum(weights_raw.values())
    if total_w <= 0:
        st.sidebar.error("Total weight must be > 0.")
        st.stop()

    st.sidebar.markdown(f"*Total weight: {total_w:.2f} (auto-normalized to 1.00)*")

    st.sidebar.markdown("---")
    st.sidebar.markdown("## Analysis Parameters")

    lookback_label = st.sidebar.selectbox(
        "Lookback Period",
        list(_lookback_map.keys()),
        index=list(_lookback_map.keys()).index(
            settings.get("data", {}).get("default_lookback", "2y")
        ),
    )
    lookback_days = _lookback_map[lookback_label]

    confidence_level = st.sidebar.selectbox(
        "Primary Confidence Level",
        [0.95, 0.99],
        format_func=lambda x: f"{int(x*100)}%",
    )

    run_agents = st.sidebar.checkbox(
        "Run Agent Analysis (LLM)",
        value=True,
        help=(
            "Runs the 4 CrewAI agents that explain the results using Groq LLM. "
            "This takes 30-90 seconds. Disable for a faster pure-quant run."
        ),
    )

    st.sidebar.markdown("---")
    run_btn = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

    return {
        "tickers": tickers,
        "weights": weights_raw,
        "lookback_days": lookback_days,
        "confidence_level": confidence_level,
        "run_agents": run_agents,
        "run_btn": run_btn,
    }


# ---------------------------------------------------------------------------
# Landing page (shown before first run)
# ---------------------------------------------------------------------------


def render_landing() -> None:
    st.markdown(
        """
        <div style="text-align:center; padding: 60px 20px 40px;">
            <h1 style="font-size:2.4rem; font-weight:800; color:#f1f5f9;">
                Agentic Market Risk Forecaster
            </h1>
            <p style="font-size:1.1rem; color:#94a3b8; max-width:620px; margin:16px auto;">
                A multi-agent quantitative risk system that computes VaR, ES, and
                drawdowns across 4 methods, runs formal VaR backtests,
                optimizes portfolio allocations, and explains everything
                through an AI agent team powered by Groq.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    cards = [
        ("Risk Engine", "Historical, Parametric, Monte Carlo, and GARCH VaR/ES across 95% and 99% confidence levels."),
        ("VaR Backtests", "Kupiec POF and Christoffersen independence tests to validate model calibration."),
        ("Portfolio Optimization", "Max Sharpe, Min Volatility, and Equal Weight strategies compared side by side."),
        ("AI Agent Analysis", "4 CrewAI agents explain every metric in plain language using Groq's LLMs."),
    ]

    for col, (title, desc) in zip([col1, col2, col3, col4], cards):
        with col:
            st.markdown(
                f"""
                <div class="metric-card risk-low" style="min-height:140px;">
                    <div class="metric-label">{title}</div>
                    <div style="color:#cbd5e1; font-size:0.88rem; line-height:1.55;">
                        {desc}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <p style="text-align:center; color:#64748b; margin-top:30px; font-size:0.88rem;">
            Configure your portfolio in the sidebar and click "Run Analysis" to begin.
        </p>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main results rendering
# ---------------------------------------------------------------------------


def render_results(results: dict, confidence_level: float) -> None:
    """Render all analysis tabs from the results dict."""
    from app.components.risk_display import (
        render_summary_cards,
        render_var_table,
        render_backtest_results,
        render_portfolio_results,
        render_garch_details,
    )
    from app.components.charts import (
        chart_returns_over_time,
        chart_cumulative_returns,
        chart_drawdown,
        chart_var_exceedance,
        chart_rolling_volatility,
        chart_var_comparison,
        chart_portfolio_weights_comparison,
        chart_correlation_heatmap,
    )
    from app.components.agent_panel import render_all_agent_outputs

    risk = results["risk_metrics"]
    backtest = results["backtest_results"]
    portfolio = results["portfolio_results"]
    anomalies = results["anomalies"]
    pr = results["portfolio_returns"]
    ret = results["returns"]
    prices = results["prices"]
    agent_out = results.get("agent_outputs", {})
    norm_weights = results.get("norm_weights", {})

    # Show any non-critical errors
    errors = results.get("errors", [])
    if errors:
        with st.expander(f"{len(errors)} non-critical warning(s) during analysis"):
            for e in errors:
                st.warning(e)

    # -- Summary cards --
    st.markdown("### Portfolio Risk Summary")
    render_summary_cards(risk, confidence_level)

    st.markdown("---")

    # -- Tabs --
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Risk Analysis",
        "Backtesting",
        "Portfolio",
        "Volatility",
        "Agent Insights",
    ])

    # ----------------------------------------------------------------
    # Tab 1: Risk Analysis
    # ----------------------------------------------------------------
    with tab1:
        st.markdown("#### Returns and Cumulative Performance")
        z_flags = anomalies.get("zscore_flags")
        col_r, col_cum = st.columns(2)
        with col_r:
            st.plotly_chart(
                chart_returns_over_time(pr, z_flags),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        with col_cum:
            st.plotly_chart(
                chart_cumulative_returns(pr),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        st.markdown("#### Drawdown")
        dd_series = risk.get("drawdown_series")
        if dd_series is not None:
            st.plotly_chart(
                chart_drawdown(dd_series),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        render_var_table(risk)
        render_garch_details(risk)

        # Correlation heatmap
        if len(ret.columns) > 1:
            st.markdown("#### Asset Return Correlations")
            st.plotly_chart(
                chart_correlation_heatmap(ret),
                use_container_width=True,
                config={"displayModeBar": False},
            )

    # ----------------------------------------------------------------
    # Tab 2: Backtesting
    # ----------------------------------------------------------------
    with tab2:
        render_backtest_results(backtest)

        rolling_var = backtest.get("rolling_var_95")
        if rolling_var is not None and not rolling_var.empty:
            st.markdown("#### VaR Exceedance Chart")
            st.plotly_chart(
                chart_var_exceedance(pr, rolling_var),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        # VaR comparison bar chart
        st.markdown("#### Method Comparison")
        import pandas as pd
        methods = ["Historical", "Parametric", "Monte Carlo", "GARCH"]
        var_data = {
            "Historical": [risk.get("hist_var_95", 0)*100, risk.get("hist_var_99", 0)*100],
            "Parametric": [risk.get("param_var_95", 0)*100, risk.get("param_var_99", 0)*100],
            "Monte Carlo": [risk.get("mc_var_95", 0)*100, risk.get("mc_var_99", 0)*100],
            "GARCH": [
                (risk.get("garch_var_95") or 0)*100,
                (risk.get("garch_var_99") or 0)*100,
            ],
        }
        var_df = pd.DataFrame(var_data, index=["95% VaR", "99% VaR"])
        st.plotly_chart(
            chart_var_comparison(var_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # ----------------------------------------------------------------
    # Tab 3: Portfolio
    # ----------------------------------------------------------------
    with tab3:
        render_portfolio_results(portfolio, norm_weights)

        st.markdown("---")
        st.markdown("#### Weight Comparison Chart")
        all_p = {
            k: v.get("weights", {})
            for k, v in portfolio.items()
            if v and v.get("weights")
        }
        display_names = {
            "current": "Current",
            "max_sharpe": "Max Sharpe",
            "min_vol": "Min Vol",
            "equal_weight": "Equal Wt",
        }
        all_p_display = {display_names.get(k, k): v for k, v in all_p.items()}
        if all_p_display:
            st.plotly_chart(
                chart_portfolio_weights_comparison(all_p_display),
                use_container_width=True,
                config={"displayModeBar": False},
            )

    # ----------------------------------------------------------------
    # Tab 4: Volatility
    # ----------------------------------------------------------------
    with tab4:
        st.markdown("#### Rolling Volatility and Regime Analysis")

        from risk.garch import garch_conditional_vol_series
        garch_vol = None
        try:
            garch_vol = garch_conditional_vol_series(pr)
        except Exception:
            pass

        st.plotly_chart(
            chart_rolling_volatility(pr, garch_vol),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        st.markdown("#### Regime Distribution")
        from risk.engine import regime_classifier
        try:
            regimes = regime_classifier(pr)
            regime_counts = regimes.value_counts()
            total = len(regimes)
            col_l, col_m, col_h = st.columns(3)
            for col, regime, color in [
                (col_l, "low", "#10b981"),
                (col_m, "mid", "#f59e0b"),
                (col_h, "high", "#ef4444"),
            ]:
                count = regime_counts.get(regime, 0)
                pct = count / total * 100 if total > 0 else 0
                with col:
                    st.markdown(
                        f"""
                        <div class="metric-card" style="border-left:4px solid {color};">
                            <div class="metric-label">{regime.upper()} Vol Regime</div>
                            <div class="metric-value" style="color:{color};">{pct:.1f}%</div>
                            <div class="metric-sub">{count} of {total} trading days</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        except Exception:
            st.info("Regime classification unavailable.")

    # ----------------------------------------------------------------
    # Tab 5: Agent Insights
    # ----------------------------------------------------------------
    with tab5:
        st.markdown("#### AI Agent Analysis")
        st.markdown(
            "Four specialized agents analyzed your portfolio using Groq's LLM. "
            "Each agent focuses on a different dimension of risk."
        )
        st.markdown("")
        render_all_agent_outputs(agent_out)


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------


def main() -> None:
    # Header
    st.markdown(
        """
        <div style="padding: 8px 0 16px;">
            <h2 style="margin:0; color:#f1f5f9; font-size:1.5rem; font-weight:700;">
                Agentic Market Risk Forecaster
            </h2>
            <p style="margin:2px 0 0; color:#64748b; font-size:0.85rem;">
                Multi-method VaR/ES analysis with AI-powered explanations
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    config = render_sidebar()

    # Session state init
    if "results" not in st.session_state:
        st.session_state["results"] = None
    if "last_config" not in st.session_state:
        st.session_state["last_config"] = None

    # Run analysis on button click
    if config["run_btn"]:
        st.session_state["last_config"] = {
            "tickers": config["tickers"],
            "confidence_level": config["confidence_level"],
        }
        with st.spinner("Running analysis... This may take a minute if agent analysis is enabled."):
            try:
                from app.analysis import run_full_analysis
                results = run_full_analysis(
                    tickers=config["tickers"],
                    weights=config["weights"],
                    lookback_days=config["lookback_days"],
                    confidence_level=config["confidence_level"],
                    run_agents=config["run_agents"],
                )
                st.session_state["results"] = results
                st.session_state["last_config"] = {
                    "tickers": config["tickers"],
                    "confidence_level": config["confidence_level"],
                }
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                st.session_state["results"] = None

    # Render results or landing
    if st.session_state["results"] is not None:
        cl = st.session_state["last_config"].get("confidence_level", 0.95)
        render_results(st.session_state["results"], confidence_level=cl)
    else:
        render_landing()


if __name__ == "__main__":
    main()
