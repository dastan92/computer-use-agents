# Kalshi AI Trader

An autonomous prediction market trading system for [Kalshi](https://kalshi.com) — the only CFTC-regulated US event contract exchange. The system uses **Claude Sonnet 4.6** as its analytical engine to estimate probabilities on real-world events and automatically place trades when it finds an edge between its estimate and the market price.

## Project Summary

### What This Is

This is a complete, end-to-end trading bot that scans Kalshi's prediction markets (elections, economic indicators, weather, interest rates, etc.), asks Claude to estimate the true probability of each event using structured forecasting methodology, compares that estimate to what the market is pricing, and places trades when it detects a meaningful discrepancy. It also includes a full backtesting engine and cost/P&L tracking dashboard.

The system is roughly **2,700 lines of Python** across **8 modules**, controlled by a CLI with **6 commands**. It's designed to run with approximately $2,000 in Kalshi trading capital and $20–50 in Anthropic API credits.

### The Core Trading Engine

The heart of the system is `kalshi_client.py` — a complete Kalshi REST API v2 client that handles authentication via RSA-PSS cryptographic signatures, market data retrieval, order placement, portfolio management, and historical data collection. This is the layer that actually talks to Kalshi's servers.

On top of that sits `analyst.py`, which wraps Claude Sonnet 4.6 with carefully engineered "superforecaster" prompting. When the bot encounters a market like "Will unemployment exceed 4.5% in March?", it instructs Claude to reason through the problem using structured Fermi estimation: establish a base rate, update with recent evidence, steelman the opposing view, then arrive at a calibrated probability. This methodology is borrowed from IARPA's Good Judgment Project, which produced the most accurate geopolitical forecasters ever measured.

`strategy.py` ties it together as the live trading brain. It scans Kalshi for active markets, filters candidates by liquidity and time-to-expiry, sends them to Claude for analysis, and executes trades using **fractional Kelly criterion** position sizing — the mathematically optimal bet sizing formula that maximizes long-term growth while managing ruin risk. We use quarter-Kelly (25% of the theoretical optimum) to be conservative, with hard caps on position sizes and daily loss limits.

### Backtesting with Anti-Contamination

One of the biggest differentiators from existing Kalshi bots (we researched competitors like OctagonAI and ryanfrigo's bot) is the **backtesting engine** in `backtester.py`. This replays historically settled markets — events that already have known outcomes — and simulates what the bot would have done. It measures calibration (are our 70% predictions right 70% of the time?), calculates Brier scores, and reports simulated P&L.

We went further and built **anti-contamination safeguards** — a critical concern when using LLMs for prediction. Since Claude was trained on internet data, it might "remember" outcomes of high-profile events. The backtester includes:

- **Time window filter**: skips markets settled before Claude's training cutoff
- **Contamination scorer**: flags events likely present in training data
- **Blind mode**: withholds the current market price from Claude, preventing anchoring bias
- **Anonymize mode**: strips identifying names and dates from market titles to reduce recall
- **Contamination comparison**: runs filtered vs. unfiltered backtests side-by-side to quantify data leakage

### Cost and Performance Tracking

`tracker.py` is a SQLite-based tracking layer that logs every action the bot takes. Every Claude API call records the model, token counts, estimated cost, and which markets were analyzed. Every trade records the signal, confidence, edge, position size, and eventual outcome. The system computes **daily summaries** showing API costs, number of trades, win/loss record, trading P&L, net P&L (after subtracting API costs and exchange fees), and a cumulative running total.

There's also a **daily API budget cap** — if the bot has already spent its configured limit on Claude calls today, it stops trading until tomorrow, preventing runaway costs.

### Configuration

`config.py` centralizes all settings via environment variables and dataclass defaults:

- **Kalshi API**: credentials, demo vs. live mode
- **Claude**: model, temperature, token limits
- **Trading**: max position size (5% of balance), Kelly fraction (0.25), min edge threshold (5%), stop-loss (15%), take-profit (20%)
- **Scheduling**: continuous/hourly/daily/manual modes, preferred market categories, trading hours window (UTC), daily API cost cap
- **Backtesting**: initial balance, commission/slippage assumptions, anti-contamination toggles

### Data Models

`models.py` defines Pydantic models for the entire data pipeline: `Market`, `Event`, `Trade`, `MarketAnalysis`, `TradeSignal`, `BacktestTrade`, and `BacktestResult`. Everything is strongly typed with validation.

## Quick Start

```bash
# Clone and install
git clone <repo-url> && cd kalshi-ai-trader
pip install -e .

# Configure credentials
cp .env.example .env
# Edit .env with your Kalshi API key and Anthropic API key

# Scan markets (see what's tradeable)
python -m kalshi_trader scan

# Deep-analyze a single market
python -m kalshi_trader analyze TICKER-HERE

# Backtest against settled markets
python -m kalshi_trader backtest --limit 50

# Backtest with anti-contamination
python -m kalshi_trader backtest --blind --anonymize

# Run contamination comparison
python -m kalshi_trader backtest --check-contamination

# Dry-run trading (no real money)
python -m kalshi_trader trade

# LIVE trading (real money — be careful)
python -m kalshi_trader trade --live

# View daily cost & P&L report
python -m kalshi_trader costs
python -m kalshi_trader costs --days 30
```

## Architecture

```
kalshi_trader/
  config.py         — All configuration (env vars + dataclass defaults)
  models.py         — Pydantic data models for markets, trades, signals
  kalshi_client.py  — Kalshi REST API v2 client (RSA-PSS auth, orders, data)
  analyst.py        — Claude Sonnet 4.6 integration (superforecaster prompts)
  strategy.py       — Live trading strategy (scan, filter, analyze, execute)
  backtester.py     — Backtesting engine with anti-contamination safeguards
  tracker.py        — SQLite cost/trade/P&L tracking and daily reports
  main.py           — CLI entry point (scan, trade, backtest, collect, analyze, costs)
```

## How Money Flows

Your money goes into your Kalshi account directly. Kalshi is a CFTC-regulated exchange. The bot interacts via API but never holds your funds — it places orders on your behalf, same as clicking buttons on their website. The API key grants trading permission, not withdrawal access.

You need two accounts:
1. **Kalshi** — trading capital ($2,000 recommended starting balance)
2. **Anthropic** — API credits for Claude analysis ($20–50 prepaid, typically $1–5/day)

## What's Next

The system is fully built. Remaining steps:
1. Create Kalshi account and generate API key + RSA private key
2. Fund Anthropic account with API credits
3. Run a backtest on real historical data to validate
4. Start in dry-run mode to observe signals without risking capital
5. Go live with a small position size and scale up based on results
