# Agentic Market Risk Forecaster

**Live dashboard:** https://agentic-market-risk-forecaster-prabhu.streamlit.app/

A multi-agent quantitative risk system that computes VaR and Expected Shortfall
across four methods, runs formal statistical backtests, optimizes portfolio
allocations, and explains every result through a team of four AI agents powered
by Groq and CrewAI.

Every number in this system is computed by the risk engine. The AI agents
interpret and explain the pre-computed results. The math is always correct
regardless of LLM quality, because the LLM never does the math.

---

## What I Built

**Quantitative risk layer:**
- Value at Risk (VaR) and Expected Shortfall (ES) via four methods:
  Historical Simulation, Parametric (Normal), Monte Carlo Bootstrap, GARCH(1,1)
- Dual confidence levels: 95% and 99%
- 1-day and 10-day horizons (10-day uses sqrt-of-time scaling)
- ES/VaR ratio computed across methods as a fat-tail severity indicator
- GARCH persistence used as a forward-looking vol signal

**VaR backtesting (model validation):**
- Kupiec (1995) Proportion of Failures test -- answers: is the breach count correct?
- Christoffersen (1998) Independence test -- answers: are breaches clustered?
- Rolling VaR exceedance chart for visual inspection across the full backtest window

**Portfolio construction:**
- Maximum Sharpe Ratio (mean-variance efficient)
- Minimum Volatility (covariance-only, no return forecast needed)
- Equal Weight benchmark
- Per-ticker weight delta table: shows exactly what to buy/sell to move from
  current to optimal

**Volatility and regime analysis:**
- Rolling 20-day realized volatility vs GARCH(1,1) conditional volatility
- Regime classification: LOW / MID / HIGH volatility (33rd/67th percentile)
- Z-score anomaly detection (rolling 2.5-sigma threshold)
- Isolation Forest anomaly detection (5% contamination)
- Both methods cross-referenced: agreement signals the strongest anomaly events

**AI agent team (4 agents, sequential pipeline):**

| Agent | Role | What It Analyzes |
|-------|------|-----------------|
| Market Monitor | Market surveillance specialist | Regime, distribution shape, correlations, worst events |
| Anomaly Detector | Tail event specialist | Anomaly magnitudes, clustering, economic context |
| Risk Forecaster | VaR/ES/GARCH expert | Method divergence, ES/VaR ratio, GARCH persistence, backtest verdicts |
| Portfolio Optimizer | Allocation specialist | Weight deltas, Sharpe gaps, regime-aware reallocation |

Each agent has a deep domain backstory, a structured 4-5 point task, and
receives prior agents' outputs via CrewAI context chaining.

---

## Architecture (5 Layers)

```
Layer 1  Data         data/provider.py         yfinance + Alpha Vantage fallback
                                                Parquet cache (24h TTL)

Layer 2  Risk         risk/engine.py            VaR, ES, returns, drawdown,
                                                regime, anomaly detection
                      risk/backtest.py          Kupiec, Christoffersen tests,
                                                rolling VaR series
                      risk/garch.py             GARCH(p,q) fitting and forecasting
                      risk/portfolio.py         PyPortfolioOpt wrappers

Layer 3  Agents       agents/crew.py            4 CrewAI agents + context
                                                compilation + orchestration

Layer 4  App          app/analysis.py           Full pipeline orchestration
                      app/main.py               Streamlit entry point
                      app/components/           Charts, display, agent panels

Layer 5  Infra        tests/                    pytest unit tests
                      configs/                  YAML config files
                      .streamlit/config.toml    Dark theme
```

---

## Project Structure

```
market-risk-agents/
  app/
    components/
      agent_panel.py       Renders 4 agent output cards in the UI
      charts.py            Plotly chart functions (dark theme)
      risk_display.py      Risk metric display and backtest result panels
    analysis.py            Orchestrates the full analysis pipeline
    main.py                Streamlit entry point (streamlit run app/main.py)
  agents/
    crew.py                4 CrewAI agents, context compilation, orchestration
  risk/
    engine.py              VaR, ES, returns, drawdown, regime, anomaly detection
    backtest.py            Kupiec and Christoffersen tests, rolling VaR
    garch.py               GARCH(p,q) fitting and conditional vol forecasting
    portfolio.py           Portfolio optimization via PyPortfolioOpt
  data/
    provider.py            DataProvider: yfinance + Alpha Vantage + Parquet cache
    raw/                   Cached price data in Parquet (gitignored)
  configs/
    settings.yaml          Risk parameters, lookback options, LLM model config
    universe.yaml          Default portfolios and preset tickers
  tests/
    test_risk_engine.py    Unit tests for all risk engine functions
    test_backtest.py       Unit tests for Kupiec and Christoffersen tests
    test_data.py           Unit tests for data provider cache logic
  .streamlit/
    config.toml            Dark theme configuration
  .env                     API keys (gitignored, never committed)
  .env.example             Template showing required keys
  requirements.txt
  README.md
```

---

## Getting Started

### 1. Install dependencies

**Option A (Windows, recommended):** double-click or run:
```bash
install.bat
```

**Option B (manual, two steps required):**
```bash
pip install -r requirements.txt
pip install "openai>=2.8.0" litellm --upgrade
```

The second step is required because crewai 1.9.3 pins openai to 1.x in its metadata,
but works correctly with openai 2.x at runtime. Running both steps ensures litellm
can route requests to Groq.

### 2. Set API keys

Copy `.env.example` to `.env` and add your keys:

```
GROQ_API_KEY=gsk_...
ALPHA_VANTAGE_API_KEY=...
```

Free keys:
- Groq: https://console.groq.com (free tier, fast inference)
- Alpha Vantage: https://www.alphavantage.co/support/#api-key (free, 500 calls/day)

Alpha Vantage is the fallback only. The system works with yfinance alone.

### 3. Run the app

```bash
python -m streamlit run app/main.py
```

The app opens at http://localhost:8501.

---

## How to Use It

1. In the sidebar, choose a preset portfolio (US Indices, Tech Heavy, Diversified)
   or type custom tickers one per line.
2. Set weights for each ticker (or click "Set Equal Weights").
3. Choose a lookback period and confidence level.
4. Check "Run Agent Analysis" if you want LLM explanations (adds ~60 seconds).
5. Click "Run Analysis".

Results appear across five tabs:

- **Risk Analysis**: Returns chart, cumulative performance, drawdown, VaR/ES table
  comparing all 4 methods, correlation heatmap
- **Backtesting**: Kupiec and Christoffersen test results with plain-language
  interpretation, VaR exceedance chart, method comparison bar chart
- **Portfolio**: Max Sharpe vs Min Vol vs current vs equal-weight comparison,
  weight charts, per-strategy metrics
- **Volatility**: Rolling realized vol vs GARCH conditional vol, regime
  distribution (LOW/MID/HIGH breakdown by day count)
- **Agent Insights**: 4 structured LLM analyses, each in a labeled card

---

## Design Decisions

**Why four VaR methods side by side?**

Each method makes different assumptions. Showing them together makes the model
uncertainty visible: if GARCH says 3.5% and historical says 2.0%, the 1.5%
spread IS the information. It tells you the current vol regime is above average
and that a risk manager relying only on historical VaR is under-estimating
forward risk.

**Why Kupiec AND Christoffersen?**

A model can pass Kupiec (correct total breach count across a year) but fail
Christoffersen (all 12 breaches happen in a 2-week crisis window). The
Christoffersen failure is the more dangerous one because it means the model
is systematically wrong exactly when it matters most.

**Why pre-compute everything before passing to agents?**

This keeps all quantitative results correct and deterministic. The LLM never
does arithmetic. It receives exact numbers and explains them. This architecture
means even a mediocre LLM response still displays correct numbers in the UI.

**Why GARCH for forward-looking risk?**

Historical VaR treats today's return distribution as representative of the
future. If today's volatility is elevated (GARCH persistence > 0.95), that
elevated regime will persist for days or weeks. GARCH(1,1) conditions on
today's variance and gives a forward-looking estimate that reflects the current
regime rather than the long-run average.

**Why is the ES/VaR ratio important?**

VaR tells you the minimum loss in the worst alpha% of scenarios. ES tells you
the average loss in those same scenarios. The ES/VaR ratio measures how much
worse the tail is beyond the VaR cutoff. For a normal distribution this ratio
is always about 1.25 at 95%. Real equity returns have ratios of 1.4-2.0+,
which is why ES-based capital requirements (Basel III, FRTB) are higher than
pure VaR calculations imply.

---

## Configuration

Edit `configs/settings.yaml` to change:
- `llm.model`: Groq model for agents (default `llama-3.3-70b-versatile`)
- `llm.reasoning_model`: Deeper reasoning model (default `qwen-qwq-32b`)
- `risk.monte_carlo_sims`: Number of MC simulations (default 10,000)
- `risk.garch_p` / `risk.garch_q`: GARCH orders (default 1/1)
- `risk.anomaly_z_threshold`: Z-score cutoff for anomaly detection (default 2.5)

Edit `configs/universe.yaml` to add new preset portfolios.

---

## Running Tests

```bash
pytest tests/ -v
```

To also run live API tests (requires API keys and network):

```bash
ENABLE_LIVE_TESTS=1 pytest tests/ -v
```

The unit tests cover:
- All VaR/ES methods with synthetic normal data and ground-truth checks
- Kupiec and Christoffersen tests with constructed exceedance scenarios
- Cache logic (save, load, freshness, date slicing)
- Drawdown, regime classifier, rolling volatility

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| App | Streamlit |
| Market data | yfinance (primary), Alpha Vantage (fallback) |
| Data cache | Parquet via PyArrow |
| Risk engine | NumPy, SciPy (custom implementations) |
| GARCH | arch library (Kevin Sheppard) |
| Portfolio optimization | PyPortfolioOpt |
| Anomaly detection | scikit-learn (Isolation Forest) |
| AI agents | CrewAI |
| LLM inference | Groq API |
| Visualization | Plotly |
| Configuration | PyYAML, python-dotenv |

---
