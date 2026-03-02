"""
Agent insights display components.

Renders the LLM output from each of the 4 CrewAI agents in
styled cards with color-coded borders and agent role labels.
"""

import streamlit as st


# Agent metadata: display name, border color, role subtitle
_AGENT_META = {
    "market_monitor": {
        "title": "Market Monitor",
        "subtitle": "Recent market conditions and volatility regime",
        "color": "#3b82f6",
        "icon_label": "MM",
    },
    "anomaly_detector": {
        "title": "Anomaly Detector",
        "subtitle": "Unusual events and tail risk signals",
        "color": "#f59e0b",
        "icon_label": "AD",
    },
    "risk_forecaster": {
        "title": "Risk Forecaster",
        "subtitle": "VaR/ES interpretation and model comparison",
        "color": "#ef4444",
        "icon_label": "RF",
    },
    "portfolio_optimizer": {
        "title": "Portfolio Optimizer",
        "subtitle": "Allocation analysis and recommendations",
        "color": "#10b981",
        "icon_label": "PO",
    },
}


def render_agent_card(agent_key: str, content: str) -> None:
    """
    Render a single agent's analysis in a styled card.

    Parameters
    ----------
    agent_key : one of the 4 agent keys
    content   : the agent's text output (plain string)
    """
    meta = _AGENT_META.get(agent_key, {
        "title": agent_key.replace("_", " ").title(),
        "subtitle": "",
        "color": "#3b82f6",
        "icon_label": "?",
    })

    color = meta["color"]
    title = meta["title"]
    subtitle = meta["subtitle"]

    if not content or not content.strip():
        content = (
            "Analysis unavailable. Check that your GROQ_API_KEY is set in .env "
            "and that the agent run was enabled."
        )
        color = "#475569"

    # Clean up any accidental em-dashes in LLM output
    content = content.replace("\u2014", " -- ").replace("\u2013", " - ")

    st.markdown(
        f"""
        <div class="agent-card" style="border-left: 4px solid {color}; margin-bottom: 20px;">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
                <div style="background:{color}; color:#fff; font-weight:700;
                            font-size:0.75rem; padding:4px 10px; border-radius:4px;
                            letter-spacing:0.05em; white-space:nowrap;">
                    {title}
                </div>
                <div style="color:#94a3b8; font-size:0.85rem;">{subtitle}</div>
            </div>
            <div style="color:#e2e8f0; font-size:0.93rem; line-height:1.65;
                        white-space:pre-wrap;">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_all_agent_outputs(agent_outputs: dict) -> None:
    """
    Render all four agent outputs in order.

    Parameters
    ----------
    agent_outputs : dict with keys market_monitor, anomaly_detector,
                    risk_forecaster, portfolio_optimizer
    """
    if not agent_outputs or all(not v for v in agent_outputs.values()):
        st.info(
            "No agent analysis available. Enable 'Run Agent Analysis' in the sidebar "
            "and ensure your GROQ_API_KEY is set in the .env file."
        )
        return

    order = [
        "market_monitor",
        "anomaly_detector",
        "risk_forecaster",
        "portfolio_optimizer",
    ]

    for key in order:
        content = agent_outputs.get(key, "")
        render_agent_card(key, content)
