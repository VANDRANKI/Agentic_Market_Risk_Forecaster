"""
Plotly chart functions for the Streamlit app.

All charts use the plotly_dark template to match the dark theme.
Colors follow the project palette:
  - Blue  #3b82f6  (neutral / primary)
  - Green #10b981  (positive / safe)
  - Amber #f59e0b  (warning)
  - Red   #ef4444  (risk / loss)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Chart color constants
BLUE = "#3b82f6"
GREEN = "#10b981"
AMBER = "#f59e0b"
RED = "#ef4444"
LIGHT_GRAY = "#94a3b8"
DARK_BG = "#0f1117"
CARD_BG = "#1a1f2e"


def _base_layout(title: str, height: int = 420) -> dict:
    return dict(
        template="plotly_dark",
        title=dict(text=title, font=dict(size=15, color="#f1f5f9")),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=LIGHT_GRAY, size=12),
        height=height,
        margin=dict(l=50, r=20, t=50, b=40),
        xaxis=dict(showgrid=True, gridcolor="#1e293b", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#1e293b", zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    )


def chart_returns_over_time(
    portfolio_returns: pd.Series,
    anomaly_flags: pd.Series = None,
    title: str = "Portfolio Daily Returns",
) -> go.Figure:
    """Bar chart of daily returns, with anomaly days highlighted."""
    r = portfolio_returns.dropna()

    colors = [GREEN if v >= 0 else RED for v in r.values]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=r.index,
            y=r.values * 100,
            marker_color=colors,
            name="Daily Return",
            hovertemplate="%{x|%Y-%m-%d}<br>Return: %{y:.2f}%<extra></extra>",
        )
    )

    # Overlay anomaly markers
    if anomaly_flags is not None:
        anom = r[anomaly_flags.reindex(r.index, fill_value=False)]
        if not anom.empty:
            fig.add_trace(
                go.Scatter(
                    x=anom.index,
                    y=anom.values * 100,
                    mode="markers",
                    marker=dict(color=AMBER, size=9, symbol="diamond"),
                    name="Anomaly",
                    hovertemplate="%{x|%Y-%m-%d}<br>Anomaly: %{y:.2f}%<extra></extra>",
                )
            )

    layout = _base_layout(title)
    layout["yaxis"]["title"] = "Return (%)"
    fig.update_layout(**layout)
    return fig


def chart_cumulative_returns(
    portfolio_returns: pd.Series,
    title: str = "Cumulative Portfolio Returns",
) -> go.Figure:
    """Area chart of cumulative portfolio returns."""
    r = portfolio_returns.dropna()
    cum = (1 + r).cumprod() - 1

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cum.index,
            y=cum.values * 100,
            mode="lines",
            line=dict(color=BLUE, width=2),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.15)",
            name="Cumulative Return",
            hovertemplate="%{x|%Y-%m-%d}<br>Cumulative: %{y:.2f}%<extra></extra>",
        )
    )

    layout = _base_layout(title)
    layout["yaxis"]["title"] = "Cumulative Return (%)"
    fig.update_layout(**layout)
    return fig


def chart_drawdown(
    drawdown_series: pd.Series,
    title: str = "Portfolio Drawdown",
) -> go.Figure:
    """Area chart of portfolio drawdown from peak."""
    dd = drawdown_series.dropna()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd.values * 100,
            mode="lines",
            line=dict(color=RED, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.20)",
            name="Drawdown",
            hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%<extra></extra>",
        )
    )

    layout = _base_layout(title)
    layout["yaxis"]["title"] = "Drawdown (%)"
    layout["yaxis"]["tickformat"] = ".1f"
    fig.update_layout(**layout)
    return fig


def chart_var_exceedance(
    portfolio_returns: pd.Series,
    rolling_var: pd.Series,
    alpha: float = 0.05,
    title: str = "VaR Exceedances (Historical Simulation)",
) -> go.Figure:
    """
    Scatter plot of daily returns with the rolling VaR boundary and
    exceedance days highlighted.
    """
    r = portfolio_returns.dropna()
    aligned = r.align(rolling_var, join="inner")
    returns_al, var_al = aligned[0], aligned[1]

    violations = returns_al < -var_al

    fig = go.Figure()

    # All returns as gray dots
    fig.add_trace(
        go.Scatter(
            x=returns_al.index,
            y=returns_al.values * 100,
            mode="markers",
            marker=dict(color=LIGHT_GRAY, size=4, opacity=0.6),
            name="Daily Return",
            hovertemplate="%{x|%Y-%m-%d}<br>Return: %{y:.2f}%<extra></extra>",
        )
    )

    # VaR line (negative, because it's a loss threshold)
    fig.add_trace(
        go.Scatter(
            x=var_al.index,
            y=-var_al.values * 100,
            mode="lines",
            line=dict(color=AMBER, width=1.5, dash="dash"),
            name=f"{int((1-alpha)*100)}% VaR (negative)",
            hovertemplate="%{x|%Y-%m-%d}<br>-VaR: %{y:.2f}%<extra></extra>",
        )
    )

    # Exceedance days in red
    exc_returns = returns_al[violations]
    if not exc_returns.empty:
        fig.add_trace(
            go.Scatter(
                x=exc_returns.index,
                y=exc_returns.values * 100,
                mode="markers",
                marker=dict(color=RED, size=7, symbol="x"),
                name="VaR Exceedance",
                hovertemplate="%{x|%Y-%m-%d}<br>Breach: %{y:.2f}%<extra></extra>",
            )
        )

    layout = _base_layout(title, height=440)
    layout["yaxis"]["title"] = "Return (%)"
    fig.update_layout(**layout)
    return fig


def chart_rolling_volatility(
    returns: pd.Series,
    garch_vol: pd.Series = None,
    window: int = 20,
    title: str = "Rolling Volatility",
) -> go.Figure:
    """
    Line chart comparing rolling realized volatility with GARCH conditional volatility.
    """
    from risk.engine import rolling_volatility
    rv = rolling_volatility(returns, window).dropna()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=rv.index,
            y=rv.values * 100,
            mode="lines",
            line=dict(color=BLUE, width=1.5),
            name=f"Rolling {window}d Vol (annualized)",
            hovertemplate="%{x|%Y-%m-%d}<br>Vol: %{y:.1f}%<extra></extra>",
        )
    )

    if garch_vol is not None and not garch_vol.empty:
        gv = garch_vol.dropna()
        fig.add_trace(
            go.Scatter(
                x=gv.index,
                y=gv.values * 100,
                mode="lines",
                line=dict(color=AMBER, width=1.5, dash="dot"),
                name="GARCH Conditional Vol",
                hovertemplate="%{x|%Y-%m-%d}<br>GARCH Vol: %{y:.1f}%<extra></extra>",
            )
        )

    layout = _base_layout(title)
    layout["yaxis"]["title"] = "Annualized Volatility (%)"
    fig.update_layout(**layout)
    return fig


def chart_var_comparison(var_table: pd.DataFrame, title: str = "VaR Comparison by Method") -> go.Figure:
    """
    Grouped bar chart comparing VaR estimates across methods and confidence levels.

    Parameters
    ----------
    var_table : DataFrame with columns as methods, rows as confidence levels
    """
    methods = list(var_table.columns)
    bars_colors = [BLUE, GREEN, AMBER, RED]

    fig = go.Figure()
    for i, method in enumerate(methods):
        fig.add_trace(
            go.Bar(
                name=method,
                x=list(var_table.index),
                y=var_table[method].values,
                marker_color=bars_colors[i % len(bars_colors)],
                hovertemplate=f"{method}<br>%{{x}}: %{{y:.3f}}%<extra></extra>",
            )
        )

    layout = _base_layout(title)
    layout["barmode"] = "group"
    layout["yaxis"]["title"] = "VaR (%)"
    fig.update_layout(**layout)
    return fig


def chart_portfolio_weights_comparison(
    portfolios: dict,
    title: str = "Portfolio Weight Comparison",
) -> go.Figure:
    """
    Stacked bar chart comparing asset weights across portfolio strategies.

    Parameters
    ----------
    portfolios : dict mapping strategy_name -> {ticker: weight}
    """
    all_tickers = sorted({
        ticker
        for weights in portfolios.values()
        for ticker in weights.keys()
    })

    colors_list = [BLUE, GREEN, AMBER, RED, "#8b5cf6", "#ec4899", "#06b6d4"]
    fig = go.Figure()

    for i, ticker in enumerate(all_tickers):
        values = [
            portfolios[strat].get(ticker, 0) * 100
            for strat in portfolios.keys()
        ]
        fig.add_trace(
            go.Bar(
                name=ticker,
                x=list(portfolios.keys()),
                y=values,
                marker_color=colors_list[i % len(colors_list)],
                hovertemplate=f"{ticker}: %{{y:.1f}}%<extra></extra>",
            )
        )

    layout = _base_layout(title)
    layout["barmode"] = "stack"
    layout["yaxis"]["title"] = "Weight (%)"
    layout["yaxis"]["range"] = [0, 100]
    fig.update_layout(**layout)
    return fig


def chart_correlation_heatmap(
    returns: pd.DataFrame,
    title: str = "Asset Correlation Matrix",
) -> go.Figure:
    """Heatmap of pairwise return correlations."""
    corr = returns.corr()
    tickers = list(corr.columns)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=tickers,
            y=tickers,
            colorscale=[
                [0.0, "#ef4444"],
                [0.5, "#1e293b"],
                [1.0, "#3b82f6"],
            ],
            zmin=-1,
            zmax=1,
            text=[[f"{corr.iloc[i, j]:.2f}" for j in range(len(tickers))] for i in range(len(tickers))],
            texttemplate="%{text}",
            textfont=dict(size=11),
            hoverongaps=False,
            hovertemplate="%{y} / %{x}: %{z:.3f}<extra></extra>",
        )
    )

    layout = _base_layout(title, height=380)
    layout["xaxis"]["side"] = "bottom"
    layout.pop("xaxis", None)
    layout.pop("yaxis", None)
    fig.update_layout(**layout)
    return fig
