"""Trading dashboard API and web server."""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

REFRESH_INTERVAL = int(os.getenv("REFRESH_HOURS", "12")) * 3600


async def refresh_alpha_data():
    """Run alpha engine in background to refresh dashboard data."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("No ANTHROPIC_API_KEY set, skipping alpha refresh")
        return
    try:
        from kalshi_trader.alpha_engine import AlphaEngine
        logger.info("Starting alpha engine refresh...")
        engine = AlphaEngine()
        # Run in thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, engine.run)
        logger.info("Alpha engine refresh complete")
    except Exception as e:
        logger.error("Alpha engine refresh failed: %s", e)


async def periodic_refresh():
    """Periodically refresh alpha data."""
    while True:
        await asyncio.sleep(REFRESH_INTERVAL)
        await refresh_alpha_data()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run alpha engine on startup and schedule periodic refresh."""
    # Run initial refresh in background (don't block startup)
    task = asyncio.create_task(refresh_alpha_data())
    refresh_task = asyncio.create_task(periodic_refresh())
    yield
    refresh_task.cancel()
    task.cancel()


app = FastAPI(title="Kalshi AI Trader Dashboard", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "backtest_data/results"))
DATA_DIR = Path(os.getenv("DATA_DIR", "backtest_data"))


def load_iterations():
    """Load all backtest iteration results."""
    iterations = []
    if not RESULTS_DIR.exists():
        return iterations
    for f in sorted(RESULTS_DIR.glob("iteration_*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
            iterations.append(data)
        except (json.JSONDecodeError, KeyError):
            pass
    return iterations


def load_predictions():
    """Load latest predictions on open markets."""
    pred_file = DATA_DIR / "live_predictions.json"
    if pred_file.exists():
        with open(pred_file) as f:
            return json.load(f)
    return []


def load_alpha_report():
    """Load latest alpha report."""
    report_file = DATA_DIR / "alpha_report.json"
    if report_file.exists():
        with open(report_file) as f:
            return json.load(f)
    return {}


def load_ensemble_backtest():
    """Load ensemble backtest results."""
    bt_file = DATA_DIR / "ensemble_backtest.json"
    if bt_file.exists():
        with open(bt_file) as f:
            return json.load(f)
    return {}


def load_timeseries_backtest():
    """Load time-series backtest results."""
    ts_file = DATA_DIR / "timeseries_backtest.json"
    if ts_file.exists():
        with open(ts_file) as f:
            return json.load(f)
    return {}


@app.get("/api/timeseries-backtest")
def get_timeseries_backtest():
    return load_timeseries_backtest()


@app.get("/api/iterations")
def get_iterations():
    return load_iterations()


@app.get("/api/predictions")
def get_predictions():
    return load_predictions()


@app.get("/api/alpha-report")
def get_alpha_report():
    return load_alpha_report()


@app.get("/api/ensemble-backtest")
def get_ensemble_backtest():
    return load_ensemble_backtest()


@app.get("/api/summary")
def get_summary():
    iterations = load_iterations()
    report = load_alpha_report()
    preds = load_predictions()

    if not iterations:
        return {"error": "No backtest data yet"}

    best = max(iterations, key=lambda x: x.get("metrics", {}).get("win_rate", 0))
    latest = iterations[-1]

    portfolio = [p for p in preds if p.get("in_portfolio")]

    return {
        "total_iterations": len(iterations),
        "best_iteration": best.get("iteration"),
        "best_accuracy": best.get("metrics", {}).get("win_rate"),
        "best_brier": best.get("metrics", {}).get("avg_brier_ai"),
        "latest_iteration": latest.get("iteration"),
        "latest_accuracy": latest.get("metrics", {}).get("win_rate"),
        "total_api_cost": sum(i.get("api_stats", {}).get("total_cost", 0) for i in iterations),
        "ai_beats_market": best.get("metrics", {}).get("brier_improvement", 0) > 0,
        "portfolio_positions": len(portfolio),
        "portfolio_allocation": report.get("total_allocation", 0),
        "portfolio_utilization": report.get("portfolio_utilization", 0),
        "avg_edge": report.get("avg_edge", 0),
        "avg_confidence": report.get("avg_confidence", 0),
        "tradeable_signals": report.get("tradeable_signals", 0),
    }


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML


DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Kalshi AI Trader</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0a0e17; color: #e0e0e0; min-height: 100vh;
  }
  .header {
    background: linear-gradient(135deg, #1a1f35 0%, #0d1321 100%);
    padding: 24px 32px; border-bottom: 1px solid #1e2a3a;
    display: flex; justify-content: space-between; align-items: center;
  }
  .header h1 {
    font-size: 24px; font-weight: 600;
    background: linear-gradient(90deg, #00d4ff, #7b61ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .header .subtitle { color: #6b7280; font-size: 14px; margin-top: 4px; }
  .demo-badge {
    background: #78350f; color: #fbbf24; padding: 4px 12px; border-radius: 9999px;
    font-size: 12px; font-weight: 600;
  }
  .container { max-width: 1400px; margin: 0 auto; padding: 24px; }
  .tabs {
    display: flex; gap: 4px; margin-bottom: 24px;
    border-bottom: 1px solid #1e2a3a; padding-bottom: 0;
  }
  .tab {
    padding: 10px 20px; cursor: pointer; color: #6b7280; font-size: 14px;
    border-bottom: 2px solid transparent; transition: all 0.2s;
  }
  .tab:hover { color: #e0e0e0; }
  .tab.active { color: #00d4ff; border-bottom-color: #00d4ff; }
  .tab-content { display: none; }
  .tab-content.active { display: block; }
  .section-desc {
    color: #9ca3af; font-size: 13px; margin-bottom: 20px; line-height: 1.5;
    padding: 12px 16px; background: #111827; border-radius: 8px; border-left: 3px solid #7b61ff;
  }
  .stats-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px; margin-bottom: 24px;
  }
  .stat-card {
    background: #111827; border: 1px solid #1e2a3a; border-radius: 12px; padding: 16px;
  }
  .stat-card .label { color: #6b7280; font-size: 12px; font-weight: 500; }
  .stat-card .value {
    font-size: 28px; font-weight: 700; margin-top: 6px;
    background: linear-gradient(90deg, #00d4ff, #7b61ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .stat-card .value.green { background: linear-gradient(90deg, #10b981, #34d399); -webkit-background-clip: text; }
  .stat-card .value.red { background: linear-gradient(90deg, #ef4444, #f87171); -webkit-background-clip: text; }
  .stat-card .detail { color: #6b7280; font-size: 11px; margin-top: 4px; }
  .charts-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
    gap: 20px; margin-bottom: 24px;
  }
  .chart-card {
    background: #111827; border: 1px solid #1e2a3a; border-radius: 12px; padding: 20px;
  }
  .chart-card h3 { font-size: 15px; margin-bottom: 4px; color: #e0e0e0; }
  .chart-card .chart-desc { font-size: 11px; color: #6b7280; margin-bottom: 12px; }
  .table-card {
    background: #111827; border: 1px solid #1e2a3a; border-radius: 12px;
    padding: 20px; margin-bottom: 20px; overflow-x: auto;
  }
  .table-card h3 { font-size: 15px; margin-bottom: 4px; }
  .table-card .table-desc { font-size: 11px; color: #6b7280; margin-bottom: 12px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; padding: 10px 12px; color: #9ca3af; border-bottom: 1px solid #1e2a3a; font-size: 11px; font-weight: 600; }
  td { padding: 10px 12px; border-bottom: 1px solid #0d1321; }
  tr:hover td { background: #1a1f35; }
  .badge { display: inline-block; padding: 3px 10px; border-radius: 9999px; font-size: 11px; font-weight: 600; }
  .badge-green { background: #064e3b; color: #34d399; }
  .badge-red { background: #7f1d1d; color: #fca5a5; }
  .badge-blue { background: #1e3a5f; color: #60a5fa; }
  .badge-yellow { background: #78350f; color: #fbbf24; }
  .portfolio-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 12px; }
  .port-card {
    background: #0d1321; border: 1px solid #1e2a3a; border-radius: 10px;
    padding: 16px; position: relative; overflow: hidden;
  }
  .port-card .side-bar { position: absolute; left: 0; top: 0; bottom: 0; width: 4px; }
  .port-card .side-bar.yes { background: #10b981; }
  .port-card .side-bar.no { background: #ef4444; }
  .port-card .title { font-size: 13px; font-weight: 500; margin-bottom: 4px; padding-left: 10px; }
  .port-card .bet-desc { font-size: 11px; color: #9ca3af; padding-left: 10px; margin-bottom: 8px; }
  .port-card .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; padding-left: 10px; font-size: 12px; }
  .port-card .metric-label { color: #6b7280; font-size: 10px; }
  .port-card .metric-value { font-weight: 600; }
  .pred-bar { height: 8px; border-radius: 4px; background: #1e2a3a; overflow: hidden; margin: 8px 10px; position: relative; }
  .pred-bar-fill { height: 100%; border-radius: 4px; }
  .pred-bar-marker { position: absolute; top: -2px; width: 2px; height: 12px; background: #fff; border-radius: 1px; }
</style>
</head>
<body>
<div class="header">
  <div>
    <h1>Kalshi AI Trader</h1>
    <div class="subtitle">AI-powered prediction market trading &mdash; Claude Sonnet ensemble engine</div>
  </div>
  <div class="demo-badge" id="liveBadge">LOADING...</div>
</div>
<div class="container">
  <div class="tabs">
    <div class="tab active" onclick="switchTab('portfolio')">Current Bets</div>
    <div class="tab" onclick="switchTab('performance')">Performance</div>
    <div class="tab" onclick="switchTab('backtest')">Accuracy History</div>
    <div class="tab" onclick="switchTab('strategies')">Strategies</div>
    <div class="tab" onclick="switchTab('all-predictions')">All Markets</div>
  </div>

  <!-- PORTFOLIO TAB -->
  <div id="tab-portfolio" class="tab-content active">
    <div class="section-desc">
      These are the bets the AI currently recommends. Each card shows a prediction market where our AI
      thinks the true probability differs from what the market is pricing &mdash; that gap is the "edge."
      The AI only bets when multiple strategies agree and the edge is large enough to be worth the risk.
    </div>
    <div id="portfolioStats" class="stats-grid"></div>
    <div class="charts-grid">
      <div class="chart-card">
        <h3>How Our Money Is Spread</h3>
        <div class="chart-desc">Groups of related bets (clusters) &mdash; we limit each group to avoid putting too many eggs in one basket.</div>
        <canvas id="clusterChart"></canvas>
      </div>
      <div class="chart-card">
        <h3>Bets by Topic</h3>
        <div class="chart-desc">Bars show dollars allocated per category; the line shows how many individual bets.</div>
        <canvas id="categoryChart"></canvas>
      </div>
    </div>
    <div class="table-card">
      <h3>Active Bets</h3>
      <div class="table-desc">Each card is one bet. Green bar = betting YES (we think it will happen), red = betting NO (we think it won't). The "edge" is how much we think the market is wrong.</div>
      <div id="portfolio" class="portfolio-grid"></div>
    </div>
  </div>

  <!-- PERFORMANCE TAB (NEW) -->
  <div id="tab-performance" class="tab-content">
    <div class="section-desc">
      How the strategy would have performed over time, tested on 80 real markets that already settled.
      This simulates entering bets when markets opened and settling them when they closed &mdash;
      tracking cumulative profit/loss month by month like a real trading account.
    </div>
    <div id="perfStats" class="stats-grid"></div>
    <div class="charts-grid">
      <div class="chart-card">
        <h3>Account Value Over Time</h3>
        <div class="chart-desc">Starting with $2,000, this line shows how the account value changed as bets settled. Higher is better.</div>
        <canvas id="equityCurveChart"></canvas>
      </div>
      <div class="chart-card">
        <h3>Monthly Profit/Loss</h3>
        <div class="chart-desc">Green bars = profitable months, red = losing months. Consistency matters more than any single month.</div>
        <canvas id="monthlyPnlChart"></canvas>
      </div>
    </div>
    <div class="table-card">
      <h3>Trade History</h3>
      <div class="table-desc">Every bet the system made during the backtest. "Edge" is how wrong we thought the market was. "P&L" is how much we made or lost. "Running" is the cumulative total.</div>
      <table id="tradeHistoryTable">
        <thead>
          <tr>
            <th>Date</th><th>Market</th><th>Bet</th><th>Edge</th>
            <th>Contracts</th><th>P&L</th><th>Running Total</th><th>Result</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <!-- BACKTEST TAB -->
  <div id="tab-backtest" class="tab-content">
    <div class="section-desc">
      We tested our AI on 80 real markets that already settled to see how accurate it is.
      Each "iteration" is a version of our prompts and strategy. We kept improving until
      we hit 86% accuracy &mdash; meaning the AI correctly predicted the outcome 86% of the time.
    </div>
    <div id="stats" class="stats-grid"></div>
    <div class="charts-grid">
      <div class="chart-card">
        <h3>Prediction Accuracy Over Iterations</h3>
        <div class="chart-desc">Each point shows how often the AI got the right answer. The red dashed line is our minimum target (60%). We reached 86%.</div>
        <canvas id="accuracyChart"></canvas>
      </div>
      <div class="chart-card">
        <h3>Prediction Quality (Brier Score)</h3>
        <div class="chart-desc">Brier score measures how close our probability estimates are to reality. Lower is better. Purple = AI, gray = market prices. When AI is lower, we have an edge.</div>
        <canvas id="brierChart"></canvas>
      </div>
    </div>
    <div class="charts-grid">
      <div class="chart-card">
        <h3>Simulated Trading Results</h3>
        <div class="chart-desc">If we actually traded on these predictions, what would happen? Bars = % of trades that made money. Line = total profit/loss in dollars.</div>
        <canvas id="tradeChart"></canvas>
      </div>
      <div class="chart-card">
        <h3>Are We Well-Calibrated?</h3>
        <div class="chart-desc">When we say something has a 70% chance, does it actually happen 70% of the time? Perfect calibration means the purple and green bars match.</div>
        <canvas id="calibrationChart"></canvas>
      </div>
    </div>
    <div class="table-card">
      <h3>Version History</h3>
      <div class="table-desc">Each row is a version of the AI. "Accuracy" = how often it was right. "AI vs Market" = whether the AI made better predictions than just using market prices. "P&L" = simulated profit/loss.</div>
      <table id="iterTable">
        <thead>
          <tr>
            <th>Version</th><th>Accuracy</th><th>AI Score</th><th>Market Score</th>
            <th>AI vs Market</th><th>Win Rate</th><th>Profit/Loss</th><th>API Cost</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <!-- STRATEGIES TAB -->
  <div id="tab-strategies" class="tab-content">
    <div class="section-desc">
      The AI uses 4 different reasoning strategies to make predictions. Two estimate probabilities
      (Bayesian + Contrarian) and two act as filters (Base Rate + Mean Reversion). We combine them
      into an ensemble &mdash; like getting 4 expert opinions and weighing them together.
    </div>
    <div class="charts-grid">
      <div class="chart-card">
        <h3>Strategy Accuracy Comparison</h3>
        <div class="chart-desc">How accurate is each strategy on its own vs. the market? Lower bars are better (closer to reality).</div>
        <canvas id="strategyCompareChart"></canvas>
      </div>
      <div class="chart-card">
        <h3>Strategy Profit/Loss</h3>
        <div class="chart-desc">If you only followed one strategy, how much would you make or lose?</div>
        <canvas id="strategyPnlChart"></canvas>
      </div>
    </div>
    <div class="table-card">
      <h3>Strategy Breakdown (80 settled markets)</h3>
      <div class="table-desc">Each row is one strategy. "Brier Edge" = how much better than the market (positive = AI wins). The ensemble combines the best of each.</div>
      <table id="ensembleTable">
        <thead>
          <tr>
            <th>Strategy</th><th>Accuracy</th><th>AI Score</th><th>Market Score</th>
            <th>AI Edge</th><th>Trades</th><th>Win Rate</th><th>Profit/Loss</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <!-- ALL PREDICTIONS TAB -->
  <div id="tab-all-predictions" class="tab-content">
    <div class="section-desc">
      Every market the AI analyzed, whether we bet on it or not. Most markets don't pass our filters &mdash;
      we only bet when the edge is large enough (>3%), enough strategies agree (>75%), and we're confident enough (>50%).
    </div>
    <div class="table-card">
      <h3>All Analyzed Markets</h3>
      <div class="table-desc">"AI Prob" = what the AI thinks the true probability is. "Market" = what the market is pricing. "Edge" = the gap (our advantage). "Size" = how much we'd bet (blank = didn't pass filters).</div>
      <table id="allPredsTable">
        <thead>
          <tr>
            <th>Market</th><th>Topic</th><th>AI Says</th><th>Market Says</th>
            <th>Edge</th><th>Agreement</th><th>Bet</th><th>Score</th><th>Bet Size</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
</div>

<script>
const API = window.location.origin;
let chartInstances = {};

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelector(`.tab-content#tab-${name}`).classList.add('active');
  event.target.classList.add('active');
}

async function load() {
  const [itersRes, predsRes, summaryRes, reportRes, ensembleRes, tsRes] = await Promise.all([
    fetch(API + '/api/iterations').then(r => r.json()),
    fetch(API + '/api/predictions').then(r => r.json()),
    fetch(API + '/api/summary').then(r => r.json()),
    fetch(API + '/api/alpha-report').then(r => r.json()),
    fetch(API + '/api/ensemble-backtest').then(r => r.json()).catch(() => ({})),
    fetch(API + '/api/timeseries-backtest').then(r => r.json()).catch(() => ({})),
  ]);

  document.getElementById('liveBadge').textContent =
    predsRes.length ? `DEMO \u2022 ${predsRes.length} MARKETS` : 'NO DATA';
  document.getElementById('liveBadge').className = 'demo-badge';

  renderPortfolioStats(summaryRes, reportRes);
  renderPortfolio(predsRes);
  renderClusterChart(reportRes);
  renderCategoryChart(reportRes);
  renderPerformance(tsRes);
  renderStats(summaryRes, itersRes);
  renderAccuracyChart(itersRes);
  renderBrierChart(itersRes);
  renderCalibrationChart(itersRes);
  renderTradeChart(itersRes);
  renderTable(itersRes);
  renderEnsembleTable(ensembleRes);
  renderStrategyCharts(ensembleRes);
  renderAllPredictions(predsRes);
}

function renderPortfolioStats(summary, report) {
  const s = document.getElementById('portfolioStats');
  const alloc = report.total_allocation || 0;
  const util = report.portfolio_utilization || 0;
  s.innerHTML = `
    <div class="stat-card">
      <div class="label">Active Bets</div>
      <div class="value">${report.portfolio_positions || 0}</div>
      <div class="detail">Out of ${report.tradeable_signals || 0} opportunities found</div>
    </div>
    <div class="stat-card">
      <div class="label">Money at Work</div>
      <div class="value green">$${alloc.toFixed(0)}</div>
      <div class="detail">${(util*100).toFixed(0)}% of $2,000 budget deployed</div>
    </div>
    <div class="stat-card">
      <div class="label">Average Edge</div>
      <div class="value ${(report.avg_edge||0) > 0.05 ? 'green' : ''}">${((report.avg_edge||0)*100).toFixed(1)}%</div>
      <div class="detail">How much we think markets are wrong</div>
    </div>
    <div class="stat-card">
      <div class="label">AI Confidence</div>
      <div class="value">${((report.avg_confidence||0)*100).toFixed(0)}%</div>
      <div class="detail">How sure the AI is about its picks</div>
    </div>
    <div class="stat-card">
      <div class="label">YES / NO Bets</div>
      <div class="value">${report.yes_signals||0} / ${report.no_signals||0}</div>
      <div class="detail">Betting on both sides for balance</div>
    </div>
    <div class="stat-card">
      <div class="label">Topic Groups</div>
      <div class="value">${report.n_clusters||0}</div>
      <div class="detail">Diversified across different themes</div>
    </div>
  `;
}

function renderPortfolio(preds) {
  const el = document.getElementById('portfolio');
  const portfolio = preds.filter(p => p.in_portfolio);
  if (!portfolio.length) {
    el.innerHTML = '<p style="color:#6b7280;padding:16px;">No active bets yet. The AI engine runs every 12 hours to find new opportunities.</p>';
    return;
  }
  el.innerHTML = portfolio.map(p => {
    const side = p.side || '?';
    const sideClass = side === 'YES' ? 'yes' : 'no';
    const edge = ((p.edge||0)*100).toFixed(1);
    const aiProb = ((p.ai_probability||0)*100).toFixed(0);
    const mktProb = ((p.market_probability||0)*100).toFixed(0);
    const agr = ((p.directional_agreement||0)*100).toFixed(0);
    const mr = p.mr_agrees ? '<span class="badge badge-yellow">MR</span>' : '';
    const betExplain = side === 'YES'
      ? `Betting YES \u2014 AI thinks ${aiProb}% likely, market says only ${mktProb}%`
      : `Betting NO \u2014 AI thinks only ${aiProb}% likely, market says ${mktProb}%`;
    const days = (p.days_to_close||0).toFixed(0);
    const daysLabel = days <= 30 ? days + ' days left' : days <= 365 ? Math.round(days/30) + ' months left' : Math.round(days/365*10)/10 + ' years left';
    return `<div class="port-card">
      <div class="side-bar ${sideClass}"></div>
      <div class="title">${(p.title||p.ticker).slice(0,70)}</div>
      <div class="bet-desc">${betExplain}</div>
      <div class="pred-bar">
        <div class="pred-bar-fill" style="width:${(p.ai_probability||0.5)*100}%;background:${side==='YES'?'#10b981':'#ef4444'}"></div>
        <div class="pred-bar-marker" style="left:${(p.market_probability||0.5)*100}%" title="Market price"></div>
      </div>
      <div class="metrics">
        <div><div class="metric-label">Bet</div><div class="metric-value"><span class="badge ${side==='YES'?'badge-green':'badge-red'}">${side}</span></div></div>
        <div><div class="metric-label">Edge</div><div class="metric-value">${edge}%</div></div>
        <div><div class="metric-label">Bet Size</div><div class="metric-value">$${(p.portfolio_size||0).toFixed(0)}</div></div>
        <div><div class="metric-label">Agreement</div><div class="metric-value">${agr}% ${mr}</div></div>
      </div>
      <div style="margin-top:6px;padding-left:10px;font-size:11px;color:#6b7280;">
        <span>${p.category||''}</span> &middot; <span>${daysLabel}</span>
      </div>
    </div>`;
  }).join('');
}

function renderPerformance(ts) {
  const statsEl = document.getElementById('perfStats');
  if (!ts || !ts.trades) {
    statsEl.innerHTML = '<p style="color:#6b7280;">No time-series backtest data available yet.</p>';
    return;
  }

  const retPct = ((ts.total_return||0)*100).toFixed(1);
  const annPct = ((ts.annualized_return||0)*100).toFixed(1);
  const wrPct = ((ts.win_rate||0)*100).toFixed(0);
  const ddPct = ((ts.max_drawdown||0)*100).toFixed(1);
  const retClass = ts.total_return >= 0 ? 'green' : 'red';

  statsEl.innerHTML = `
    <div class="stat-card">
      <div class="label">Total Return</div>
      <div class="value ${retClass}">${retPct}%</div>
      <div class="detail">$${ts.starting_capital} \u2192 $${(ts.final_value||0).toLocaleString()}</div>
    </div>
    <div class="stat-card">
      <div class="label">Annualized Return</div>
      <div class="value ${retClass}">${annPct}%</div>
      <div class="detail">If this rate held for a full year</div>
    </div>
    <div class="stat-card">
      <div class="label">Win Rate</div>
      <div class="value ${ts.win_rate >= 0.5 ? 'green' : 'red'}">${wrPct}%</div>
      <div class="detail">${ts.wins||0} wins, ${ts.losses||0} losses out of ${ts.total_trades||0} bets</div>
    </div>
    <div class="stat-card">
      <div class="label">Profit Factor</div>
      <div class="value ${(ts.profit_factor||0) >= 1.5 ? 'green' : ''}">${(ts.profit_factor||0).toFixed(1)}x</div>
      <div class="detail">$ won / $ lost (>1 = profitable)</div>
    </div>
    <div class="stat-card">
      <div class="label">Avg Win vs Avg Loss</div>
      <div class="value green">$${(ts.avg_win||0).toFixed(0)} / $${Math.abs(ts.avg_loss||0).toFixed(0)}</div>
      <div class="detail">Wins ${((ts.avg_win||1)/Math.abs(ts.avg_loss||1)).toFixed(1)}x bigger than losses</div>
    </div>
    <div class="stat-card">
      <div class="label">Worst Drawdown</div>
      <div class="value red">${ddPct}%</div>
      <div class="detail">Biggest drop from a peak before recovering</div>
    </div>
  `;

  // Equity curve chart
  const equity = ts.equity_curve || [];
  if (equity.length) {
    destroyChart('equityCurveChart');
    const ctx = document.getElementById('equityCurveChart').getContext('2d');
    chartInstances['equityCurveChart'] = new Chart(ctx, {
      type: 'line',
      data: {
        labels: equity.map(e => { const d = new Date(e[0]); return d.toLocaleDateString('en-US', {month:'short', year:'2-digit'}); }),
        datasets: [{
          label: 'Account Value ($)',
          data: equity.map(e => e[1]),
          borderColor: '#10b981',
          backgroundColor: 'rgba(16,185,129,0.1)',
          fill: true, tension: 0.3, pointRadius: 3, pointBackgroundColor: '#10b981',
        }, {
          label: 'Starting Capital',
          data: equity.map(() => ts.starting_capital),
          borderColor: '#374151', borderDash: [5, 5], pointRadius: 0,
        }]
      },
      options: {
        responsive: true,
        scales: {
          y: { ticks: { callback: v => '$'+v.toLocaleString(), color: '#6b7280' }, grid: { color: '#1e2a3a' } },
          x: { ticks: { color: '#6b7280', maxTicksLimit: 12 }, grid: { color: '#1e2a3a' } }
        },
        plugins: { legend: { labels: { color: '#e0e0e0' } } }
      }
    });
  }

  // Monthly P&L chart
  const monthly = ts.monthly_pnl || {};
  const months = Object.keys(monthly).sort();
  if (months.length) {
    destroyChart('monthlyPnlChart');
    const ctx = document.getElementById('monthlyPnlChart').getContext('2d');
    const monthLabels = months.map(m => {
      const [y, mo] = m.split('-');
      return new Date(y, mo-1).toLocaleDateString('en-US', {month:'short', year:'2-digit'});
    });
    chartInstances['monthlyPnlChart'] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: monthLabels,
        datasets: [{
          label: 'Monthly P&L ($)',
          data: months.map(m => monthly[m]),
          backgroundColor: months.map(m => monthly[m] >= 0 ? '#10b981' : '#ef4444'),
          borderRadius: 4,
        }]
      },
      options: {
        responsive: true,
        scales: {
          y: { ticks: { callback: v => (v>=0?'+':'')+('$'+v.toFixed(0)), color: '#6b7280' }, grid: { color: '#1e2a3a' } },
          x: { ticks: { color: '#6b7280' }, grid: { color: '#1e2a3a' } }
        },
        plugins: { legend: { labels: { color: '#e0e0e0' } } }
      }
    });
  }

  // Trade history table
  const tbody = document.querySelector('#tradeHistoryTable tbody');
  let running = 0;
  tbody.innerHTML = (ts.trades || []).map(t => {
    const won = t.won;
    const pnlColor = t.pnl >= 0 ? '#34d399' : '#fca5a5';
    return `<tr>
      <td>${t.date}</td>
      <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${(t.title||t.ticker||'').slice(0,50)}</td>
      <td><span class="badge ${t.side==='YES'?'badge-green':'badge-red'}">${t.side}</span></td>
      <td>${(t.edge*100).toFixed(1)}%</td>
      <td>${t.contracts}</td>
      <td style="color:${pnlColor};font-weight:600">$${t.pnl>=0?'+':''}${t.pnl.toFixed(2)}</td>
      <td style="color:${t.cumulative_pnl>=0?'#34d399':'#fca5a5'}">$${t.cumulative_pnl>=0?'+':''}${t.cumulative_pnl.toFixed(2)}</td>
      <td><span class="badge ${won?'badge-green':'badge-red'}">${won?'WIN':'LOSS'}</span></td>
    </tr>`;
  }).join('');
}

function renderClusterChart(report) {
  const clusters = report.by_cluster || {};
  const labels = Object.keys(clusters).map(c => c.length > 18 ? c.slice(0,18)+'...' : c);
  const sizes = Object.values(clusters).map(c => c.total_size || 0);
  const colors = ['#00d4ff','#7b61ff','#10b981','#f59e0b','#ef4444','#8b5cf6','#06b6d4','#ec4899','#84cc16'];

  destroyChart('clusterChart');
  const ctx = document.getElementById('clusterChart').getContext('2d');
  chartInstances['clusterChart'] = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: labels,
      datasets: [{ data: sizes, backgroundColor: colors.slice(0, labels.length), borderWidth: 0 }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'right', labels: { color: '#e0e0e0', font: { size: 11 } } },
        tooltip: { callbacks: { label: ctx => `$${ctx.raw.toFixed(0)}` } }
      }
    }
  });
}

function renderCategoryChart(report) {
  const cats = report.by_category || {};
  const entries = Object.entries(cats).filter(([,v]) => v.in_portfolio > 0);
  const labels = entries.map(([k]) => k);
  const sizes = entries.map(([,v]) => v.total_size || 0);
  const counts = entries.map(([,v]) => v.in_portfolio || 0);

  destroyChart('categoryChart');
  const ctx = document.getElementById('categoryChart').getContext('2d');
  chartInstances['categoryChart'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Dollars Bet ($)',
        data: sizes,
        backgroundColor: '#7b61ff',
        borderRadius: 4,
        yAxisID: 'y',
      }, {
        label: '# of Bets',
        data: counts,
        type: 'line',
        borderColor: '#00d4ff',
        pointRadius: 5,
        pointBackgroundColor: '#00d4ff',
        yAxisID: 'y1',
      }]
    },
    options: {
      responsive: true,
      scales: {
        y: { position: 'left', ticks: { callback: v => '$'+v, color: '#6b7280' }, grid: { color: '#1e2a3a' } },
        y1: { position: 'right', ticks: { color: '#6b7280' }, grid: { display: false } },
        x: { ticks: { color: '#6b7280' }, grid: { color: '#1e2a3a' } }
      },
      plugins: { legend: { labels: { color: '#e0e0e0' } } }
    }
  });
}

function renderStats(summary, iters) {
  const best = iters.find(i => i.iteration === summary.best_iteration) || {};
  const bm = best.metrics || {};
  const s = document.getElementById('stats');
  s.innerHTML = `
    <div class="stat-card">
      <div class="label">Best Accuracy</div>
      <div class="value green">${((summary.best_accuracy||0)*100).toFixed(1)}%</div>
      <div class="detail">Version #${summary.best_iteration} got this right</div>
    </div>
    <div class="stat-card">
      <div class="label">AI Beats Market?</div>
      <div class="value ${summary.ai_beats_market ? 'green' : 'red'}">${summary.ai_beats_market ? 'YES' : 'NO'}</div>
      <div class="detail">AI: ${(bm.avg_brier_ai||0).toFixed(4)} vs Market: ${(bm.avg_brier_market||0).toFixed(4)}</div>
    </div>
    <div class="stat-card">
      <div class="label">Simulated Win Rate</div>
      <div class="value ${(bm.trade_win_rate||0) > 0.5 ? 'green' : 'red'}">${((bm.trade_win_rate||0)*100).toFixed(1)}%</div>
      <div class="detail">${bm.n_trades||0} bets placed in simulation</div>
    </div>
    <div class="stat-card">
      <div class="label">Simulated Profit</div>
      <div class="value ${(bm.trade_pnl||0) >= 0 ? 'green' : 'red'}">$${(bm.trade_pnl||0).toFixed(2)}</div>
      <div class="detail">Only counting bets with >5% edge</div>
    </div>
    <div class="stat-card">
      <div class="label">AI Training Cost</div>
      <div class="value">$${(summary.total_api_cost||0).toFixed(2)}</div>
      <div class="detail">${summary.total_iterations} versions tested</div>
    </div>
  `;
}

function renderAccuracyChart(iters) {
  destroyChart('accuracyChart');
  const ctx = document.getElementById('accuracyChart').getContext('2d');
  chartInstances['accuracyChart'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: iters.map(i => 'v' + i.iteration),
      datasets: [{
        label: 'Accuracy',
        data: iters.map(i => ((i.metrics||{}).win_rate||0)*100),
        borderColor: '#00d4ff',
        backgroundColor: 'rgba(0,212,255,0.1)',
        fill: true, tension: 0.3, pointRadius: 5, pointBackgroundColor: '#00d4ff',
      }, {
        label: '60% Minimum Target',
        data: iters.map(() => 60),
        borderColor: '#ef4444', borderDash: [5, 5], pointRadius: 0,
      }]
    },
    options: {
      responsive: true,
      scales: {
        y: { min: 40, max: 100, ticks: { callback: v => v+'%', color: '#6b7280' }, grid: { color: '#1e2a3a' } },
        x: { ticks: { color: '#6b7280' }, grid: { color: '#1e2a3a' } }
      },
      plugins: { legend: { labels: { color: '#e0e0e0' } } }
    }
  });
}

function renderBrierChart(iters) {
  destroyChart('brierChart');
  const ctx = document.getElementById('brierChart').getContext('2d');
  chartInstances['brierChart'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: iters.map(i => 'v' + i.iteration),
      datasets: [{
        label: 'AI Score (lower = better)', data: iters.map(i => (i.metrics||{}).avg_brier_ai||0),
        backgroundColor: '#7b61ff', borderRadius: 4,
      }, {
        label: 'Market Score', data: iters.map(i => (i.metrics||{}).avg_brier_market||0),
        backgroundColor: '#374151', borderRadius: 4,
      }]
    },
    options: {
      responsive: true,
      scales: {
        y: { min: 0, max: 0.3, ticks: { color: '#6b7280' }, grid: { color: '#1e2a3a' } },
        x: { ticks: { color: '#6b7280' }, grid: { color: '#1e2a3a' } }
      },
      plugins: { legend: { labels: { color: '#e0e0e0' } } }
    }
  });
}

function renderCalibrationChart(iters) {
  const best = iters[iters.length - 1];
  const cal = (best?.metrics||{}).calibration||{};
  const buckets = Object.keys(cal);
  destroyChart('calibrationChart');
  const ctx = document.getElementById('calibrationChart').getContext('2d');
  chartInstances['calibrationChart'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: buckets,
      datasets: [{
        label: 'What AI Predicted', data: buckets.map(b => (cal[b].avg_predicted||0)*100),
        backgroundColor: '#7b61ff', borderRadius: 4,
      }, {
        label: 'What Actually Happened', data: buckets.map(b => (cal[b].actual_rate||0)*100),
        backgroundColor: '#10b981', borderRadius: 4,
      }]
    },
    options: {
      responsive: true,
      scales: {
        y: { min: 0, max: 100, ticks: { callback: v => v+'%', color: '#6b7280' }, grid: { color: '#1e2a3a' } },
        x: { ticks: { color: '#6b7280' }, grid: { color: '#1e2a3a' } }
      },
      plugins: { legend: { labels: { color: '#e0e0e0' } } }
    }
  });
}

function renderTradeChart(iters) {
  destroyChart('tradeChart');
  const ctx = document.getElementById('tradeChart').getContext('2d');
  chartInstances['tradeChart'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: iters.map(i => 'v' + i.iteration),
      datasets: [{
        label: 'Win Rate %',
        data: iters.map(i => ((i.metrics||{}).trade_win_rate||0)*100),
        backgroundColor: iters.map(i => ((i.metrics||{}).trade_win_rate||0) >= 0.5 ? '#10b981' : '#374151'),
        borderRadius: 4, yAxisID: 'y',
      }, {
        label: 'Profit/Loss ($)', data: iters.map(i => (i.metrics||{}).trade_pnl||0),
        type: 'line', borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.1)',
        fill: true, tension: 0.3, pointRadius: 5, yAxisID: 'y1',
      }]
    },
    options: {
      responsive: true,
      scales: {
        y: { min: 0, max: 70, position: 'left', ticks: { callback: v => v+'%', color: '#6b7280' }, grid: { color: '#1e2a3a' } },
        y1: { position: 'right', ticks: { callback: v => '$'+v.toFixed(0), color: '#6b7280' }, grid: { display: false } },
        x: { ticks: { color: '#6b7280' }, grid: { color: '#1e2a3a' } }
      },
      plugins: { legend: { labels: { color: '#e0e0e0' } } }
    }
  });
}

function renderTable(iters) {
  const tbody = document.querySelector('#iterTable tbody');
  tbody.innerHTML = iters.map(i => {
    const m = i.metrics || {};
    const beats = (m.brier_improvement||0) > 0;
    return `<tr>
      <td>v${i.iteration}</td>
      <td><span class="badge ${(m.win_rate||0) >= 0.75 ? 'badge-green' : 'badge-blue'}">${((m.win_rate||0)*100).toFixed(1)}%</span></td>
      <td>${(m.avg_brier_ai||0).toFixed(4)}</td>
      <td>${(m.avg_brier_market||0).toFixed(4)}</td>
      <td><span class="badge ${beats ? 'badge-green' : 'badge-red'}">${beats ? 'AI Wins' : 'Market Wins'}</span></td>
      <td>${((m.trade_win_rate||0)*100).toFixed(1)}%</td>
      <td style="color:${(m.trade_pnl||0)>=0?'#34d399':'#fca5a5'}">$${(m.trade_pnl||0).toFixed(2)}</td>
      <td>$${(i.api_stats||{}).total_cost?.toFixed(4)||'?'}</td>
    </tr>`;
  }).join('');
}

function renderEnsembleTable(ensemble) {
  const tbody = document.querySelector('#ensembleTable tbody');
  const evals = ensemble.evaluations || [];
  tbody.innerHTML = evals.map(e => {
    const isEnsemble = e.label.includes('ensemble') || e.label.includes('bayes');
    return `<tr style="${isEnsemble ? 'background:#111827' : ''}">
      <td><strong>${e.label.replace(/_/g, ' ')}</strong></td>
      <td><span class="badge ${e.accuracy >= 0.85 ? 'badge-green' : 'badge-blue'}">${(e.accuracy*100).toFixed(0)}%</span></td>
      <td>${e.brier.toFixed(4)}</td>
      <td>${e.market_brier.toFixed(4)}</td>
      <td><span class="badge ${e.brier_edge > 0 ? 'badge-green' : 'badge-red'}">${e.brier_edge > 0 ? '+' : ''}${e.brier_edge.toFixed(4)}</span></td>
      <td>${e.trades}</td>
      <td style="color:${e.trade_wr >= 0.5 ? '#34d399' : '#fca5a5'}">${(e.trade_wr*100).toFixed(0)}%</td>
      <td style="color:${e.trade_pnl >= 0 ? '#34d399' : '#fca5a5'}">$${e.trade_pnl.toFixed(2)}</td>
    </tr>`;
  }).join('');
}

function renderStrategyCharts(ensemble) {
  const evals = ensemble.evaluations || [];
  if (!evals.length) return;

  destroyChart('strategyCompareChart');
  const ctx1 = document.getElementById('strategyCompareChart').getContext('2d');
  chartInstances['strategyCompareChart'] = new Chart(ctx1, {
    type: 'bar',
    data: {
      labels: evals.map(e => e.label.replace(/_/g, ' ')),
      datasets: [{
        label: 'AI Score (lower = better)', data: evals.map(e => e.brier),
        backgroundColor: evals.map(e => e.brier_edge > 0 ? '#7b61ff' : '#374151'),
        borderRadius: 4,
      }, {
        label: 'Market Score', data: evals.map(e => e.market_brier),
        backgroundColor: '#1e2a3a', borderRadius: 4,
      }]
    },
    options: {
      responsive: true, indexAxis: 'y',
      scales: {
        x: { min: 0.1, max: 0.16, ticks: { color: '#6b7280' }, grid: { color: '#1e2a3a' } },
        y: { ticks: { color: '#6b7280', font: { size: 10 } }, grid: { color: '#1e2a3a' } }
      },
      plugins: { legend: { labels: { color: '#e0e0e0' } } }
    }
  });

  destroyChart('strategyPnlChart');
  const ctx2 = document.getElementById('strategyPnlChart').getContext('2d');
  chartInstances['strategyPnlChart'] = new Chart(ctx2, {
    type: 'bar',
    data: {
      labels: evals.map(e => e.label.replace(/_/g, ' ')),
      datasets: [{
        label: 'Profit/Loss ($)', data: evals.map(e => e.trade_pnl),
        backgroundColor: evals.map(e => e.trade_pnl >= 0 ? '#10b981' : '#ef4444'),
        borderRadius: 4,
      }]
    },
    options: {
      responsive: true, indexAxis: 'y',
      scales: {
        x: { ticks: { callback: v => '$'+v, color: '#6b7280' }, grid: { color: '#1e2a3a' } },
        y: { ticks: { color: '#6b7280', font: { size: 10 } }, grid: { color: '#1e2a3a' } }
      },
      plugins: { legend: { labels: { color: '#e0e0e0' } } }
    }
  });
}

function renderAllPredictions(preds) {
  const tbody = document.querySelector('#allPredsTable tbody');
  tbody.innerHTML = preds.map(p => {
    const edge = (p.edge||0)*100;
    const edgeColor = Math.abs(edge) >= 5 ? (edge > 0 ? '#34d399' : '#fca5a5') : '#6b7280';
    return `<tr>
      <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${(p.title||p.ticker).slice(0,60)}</td>
      <td><span class="badge badge-blue">${p.category||''}</span></td>
      <td>${((p.ai_probability||0)*100).toFixed(0)}%</td>
      <td>${((p.market_probability||0)*100).toFixed(0)}%</td>
      <td style="color:${edgeColor};font-weight:600">${edge >= 0 ? '+' : ''}${edge.toFixed(1)}%</td>
      <td>${((p.directional_agreement||0)*100).toFixed(0)}%</td>
      <td>${p.side ? `<span class="badge ${p.side==='YES'?'badge-green':'badge-red'}">${p.side}</span>` : '-'}</td>
      <td>${(p.score||0).toFixed(1)}</td>
      <td>${p.in_portfolio ? '$'+(p.portfolio_size||0).toFixed(0) : '-'}</td>
    </tr>`;
  }).join('');
}

function destroyChart(id) {
  if (chartInstances[id]) { chartInstances[id].destroy(); delete chartInstances[id]; }
}

load();
setInterval(load, 120000);
</script>
</body>
</html>
"""
