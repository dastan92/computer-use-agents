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
<title>Kalshi AI Trader Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0a0e17;
    color: #e0e0e0;
    min-height: 100vh;
  }
  .header {
    background: linear-gradient(135deg, #1a1f35 0%, #0d1321 100%);
    padding: 24px 32px;
    border-bottom: 1px solid #1e2a3a;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .header h1 {
    font-size: 24px;
    font-weight: 600;
    background: linear-gradient(90deg, #00d4ff, #7b61ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .header .subtitle { color: #6b7280; font-size: 14px; margin-top: 4px; }
  .header .live-badge {
    background: #064e3b; color: #34d399; padding: 4px 12px; border-radius: 9999px;
    font-size: 12px; font-weight: 600; animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.6; } }
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

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin-bottom: 24px;
  }
  .stat-card {
    background: #111827;
    border: 1px solid #1e2a3a;
    border-radius: 12px;
    padding: 16px;
  }
  .stat-card .label { color: #6b7280; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
  .stat-card .value {
    font-size: 28px;
    font-weight: 700;
    margin-top: 6px;
    background: linear-gradient(90deg, #00d4ff, #7b61ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .stat-card .value.green { background: linear-gradient(90deg, #10b981, #34d399); -webkit-background-clip: text; }
  .stat-card .value.red { background: linear-gradient(90deg, #ef4444, #f87171); -webkit-background-clip: text; }
  .stat-card .detail { color: #6b7280; font-size: 11px; margin-top: 4px; }

  .charts-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
    gap: 20px;
    margin-bottom: 24px;
  }
  .chart-card {
    background: #111827;
    border: 1px solid #1e2a3a;
    border-radius: 12px;
    padding: 20px;
  }
  .chart-card h3 { font-size: 15px; margin-bottom: 12px; color: #e0e0e0; }

  .table-card {
    background: #111827;
    border: 1px solid #1e2a3a;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    overflow-x: auto;
  }
  .table-card h3 { font-size: 15px; margin-bottom: 12px; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { text-align: left; padding: 8px 10px; color: #6b7280; border-bottom: 1px solid #1e2a3a;
       text-transform: uppercase; font-size: 10px; letter-spacing: 1px; }
  td { padding: 8px 10px; border-bottom: 1px solid #0d1321; }
  tr:hover td { background: #1a1f35; }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 9999px;
    font-size: 10px; font-weight: 600;
  }
  .badge-green { background: #064e3b; color: #34d399; }
  .badge-red { background: #7f1d1d; color: #fca5a5; }
  .badge-blue { background: #1e3a5f; color: #60a5fa; }
  .badge-yellow { background: #78350f; color: #fbbf24; }

  .portfolio-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
    gap: 12px;
  }
  .port-card {
    background: #0d1321;
    border: 1px solid #1e2a3a;
    border-radius: 8px;
    padding: 14px;
    position: relative;
    overflow: hidden;
  }
  .port-card .side-bar {
    position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
  }
  .port-card .side-bar.yes { background: #10b981; }
  .port-card .side-bar.no { background: #ef4444; }
  .port-card .title { font-size: 12px; font-weight: 500; margin-bottom: 6px; padding-left: 8px; }
  .port-card .metrics {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px;
    padding-left: 8px; font-size: 11px;
  }
  .port-card .metric-label { color: #6b7280; font-size: 10px; }
  .port-card .metric-value { font-weight: 600; }

  .pred-bar {
    height: 6px; border-radius: 3px; background: #1e2a3a; overflow: hidden; margin: 6px 0;
  }
  .pred-bar-fill { height: 100%; border-radius: 3px; transition: width 0.5s; }
  .pred-meta { display: flex; justify-content: space-between; font-size: 10px; color: #6b7280; }
</style>
</head>
<body>
<div class="header">
  <div>
    <h1>Kalshi AI Trader</h1>
    <div class="subtitle">Ensemble Alpha Engine v3 &mdash; Powered by Claude Sonnet</div>
  </div>
  <div class="live-badge" id="liveBadge">LOADING...</div>
</div>
<div class="container">
  <div class="tabs">
    <div class="tab active" onclick="switchTab('portfolio')">Portfolio</div>
    <div class="tab" onclick="switchTab('backtest')">Backtest History</div>
    <div class="tab" onclick="switchTab('strategies')">Strategy Analysis</div>
    <div class="tab" onclick="switchTab('all-predictions')">All Predictions</div>
  </div>

  <!-- PORTFOLIO TAB -->
  <div id="tab-portfolio" class="tab-content active">
    <div id="portfolioStats" class="stats-grid"></div>
    <div class="charts-grid">
      <div class="chart-card">
        <h3>Cluster Allocation</h3>
        <canvas id="clusterChart"></canvas>
      </div>
      <div class="chart-card">
        <h3>Category Distribution</h3>
        <canvas id="categoryChart"></canvas>
      </div>
    </div>
    <div class="table-card">
      <h3>Active Portfolio Positions</h3>
      <div id="portfolio" class="portfolio-grid"></div>
    </div>
  </div>

  <!-- BACKTEST TAB -->
  <div id="tab-backtest" class="tab-content">
    <div id="stats" class="stats-grid"></div>
    <div class="charts-grid">
      <div class="chart-card">
        <h3>Accuracy by Iteration</h3>
        <canvas id="accuracyChart"></canvas>
      </div>
      <div class="chart-card">
        <h3>Brier Score: AI vs Market</h3>
        <canvas id="brierChart"></canvas>
      </div>
    </div>
    <div class="charts-grid">
      <div class="chart-card">
        <h3>Trade Win Rate & P&L</h3>
        <canvas id="tradeChart"></canvas>
      </div>
      <div class="chart-card">
        <h3>Calibration (Latest)</h3>
        <canvas id="calibrationChart"></canvas>
      </div>
    </div>
    <div class="table-card">
      <h3>Iteration Comparison</h3>
      <table id="iterTable">
        <thead>
          <tr>
            <th>Iter</th><th>Accuracy</th><th>Brier (AI)</th><th>Brier (Mkt)</th>
            <th>AI vs Mkt</th><th>Trade WR</th><th>P&L</th><th>Cost</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <!-- STRATEGIES TAB -->
  <div id="tab-strategies" class="tab-content">
    <div class="charts-grid">
      <div class="chart-card">
        <h3>Strategy Backtest Comparison</h3>
        <canvas id="strategyCompareChart"></canvas>
      </div>
      <div class="chart-card">
        <h3>Strategy Trade P&L</h3>
        <canvas id="strategyPnlChart"></canvas>
      </div>
    </div>
    <div class="table-card">
      <h3>Ensemble Backtest Results (80 settled markets)</h3>
      <table id="ensembleTable">
        <thead>
          <tr>
            <th>Strategy</th><th>Accuracy</th><th>Brier</th><th>Mkt Brier</th>
            <th>Brier Edge</th><th>Trades</th><th>Win Rate</th><th>P&L</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <!-- ALL PREDICTIONS TAB -->
  <div id="tab-all-predictions" class="tab-content">
    <div class="table-card">
      <h3>All Market Predictions</h3>
      <table id="allPredsTable">
        <thead>
          <tr>
            <th>Market</th><th>Category</th><th>AI Prob</th><th>Market</th>
            <th>Edge</th><th>Agreement</th><th>Side</th><th>Score</th><th>Size</th>
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
  const [itersRes, predsRes, summaryRes, reportRes, ensembleRes] = await Promise.all([
    fetch(API + '/api/iterations').then(r => r.json()),
    fetch(API + '/api/predictions').then(r => r.json()),
    fetch(API + '/api/summary').then(r => r.json()),
    fetch(API + '/api/alpha-report').then(r => r.json()),
    fetch(API + '/api/ensemble-backtest').then(r => r.json()).catch(() => ({})),
  ]);

  document.getElementById('liveBadge').textContent =
    predsRes.length ? `${predsRes.length} MARKETS` : 'NO DATA';

  renderPortfolioStats(summaryRes, reportRes);
  renderPortfolio(predsRes);
  renderClusterChart(reportRes);
  renderCategoryChart(reportRes);
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
      <div class="label">Portfolio Positions</div>
      <div class="value">${report.portfolio_positions || 0}</div>
      <div class="detail">${report.tradeable_signals || 0} tradeable signals found</div>
    </div>
    <div class="stat-card">
      <div class="label">Total Allocation</div>
      <div class="value green">$${alloc.toFixed(0)}</div>
      <div class="detail">${(util*100).toFixed(0)}% of $2,000 portfolio</div>
    </div>
    <div class="stat-card">
      <div class="label">Avg Edge</div>
      <div class="value ${(report.avg_edge||0) > 0.05 ? 'green' : ''}">${((report.avg_edge||0)*100).toFixed(1)}%</div>
      <div class="detail">Across active positions</div>
    </div>
    <div class="stat-card">
      <div class="label">Avg Confidence</div>
      <div class="value">${((report.avg_confidence||0)*100).toFixed(0)}%</div>
      <div class="detail">Ensemble agreement-weighted</div>
    </div>
    <div class="stat-card">
      <div class="label">YES / NO Split</div>
      <div class="value">${report.yes_signals||0} / ${report.no_signals||0}</div>
      <div class="detail">Directional diversification</div>
    </div>
    <div class="stat-card">
      <div class="label">Clusters</div>
      <div class="value">${report.n_clusters||0}</div>
      <div class="detail">Correlation groups</div>
    </div>
  `;
}

function renderPortfolio(preds) {
  const el = document.getElementById('portfolio');
  const portfolio = preds.filter(p => p.in_portfolio);
  if (!portfolio.length) {
    el.innerHTML = '<p style="color:#6b7280;padding:16px;">No active positions. Run the alpha engine to generate trades.</p>';
    return;
  }
  el.innerHTML = portfolio.map(p => {
    const side = p.side || '?';
    const sideClass = side === 'YES' ? 'yes' : 'no';
    const edge = ((p.edge||0)*100).toFixed(1);
    const conf = ((p.confidence||0)*100).toFixed(0);
    const agr = ((p.directional_agreement||0)*100).toFixed(0);
    const mr = p.mr_agrees ? '<span class="badge badge-yellow">MR</span>' : '';
    return `<div class="port-card">
      <div class="side-bar ${sideClass}"></div>
      <div class="title">${(p.title||p.ticker).slice(0,70)}</div>
      <div class="pred-bar"><div class="pred-bar-fill" style="width:${(p.ai_probability||0.5)*100}%;background:${side==='YES'?'#10b981':'#ef4444'}"></div></div>
      <div class="metrics">
        <div><div class="metric-label">Side</div><div class="metric-value"><span class="badge ${side==='YES'?'badge-green':'badge-red'}">${side}</span></div></div>
        <div><div class="metric-label">Edge</div><div class="metric-value">${edge}%</div></div>
        <div><div class="metric-label">Size</div><div class="metric-value">$${(p.portfolio_size||0).toFixed(0)}</div></div>
        <div><div class="metric-label">Agreement</div><div class="metric-value">${agr}% ${mr}</div></div>
      </div>
      <div class="pred-meta" style="margin-top:6px;">
        <span>AI: ${((p.ai_probability||0)*100).toFixed(0)}% | Mkt: ${((p.market_probability||0)*100).toFixed(0)}%</span>
        <span>${p.category||''}</span>
        <span>${(p.days_to_close||0).toFixed(0)}d</span>
      </div>
    </div>`;
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
        label: 'Allocation ($)',
        data: sizes,
        backgroundColor: '#7b61ff',
        borderRadius: 4,
        yAxisID: 'y',
      }, {
        label: 'Positions',
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
      <div class="detail">Iteration #${summary.best_iteration}</div>
    </div>
    <div class="stat-card">
      <div class="label">AI Beats Market?</div>
      <div class="value ${summary.ai_beats_market ? 'green' : 'red'}">${summary.ai_beats_market ? 'YES' : 'NO'}</div>
      <div class="detail">Brier: ${(bm.avg_brier_ai||0).toFixed(4)} vs ${(bm.avg_brier_market||0).toFixed(4)}</div>
    </div>
    <div class="stat-card">
      <div class="label">Trade Win Rate</div>
      <div class="value ${(bm.trade_win_rate||0) > 0.5 ? 'green' : 'red'}">${((bm.trade_win_rate||0)*100).toFixed(1)}%</div>
      <div class="detail">${bm.n_trades||0} trades taken</div>
    </div>
    <div class="stat-card">
      <div class="label">Simulated P&L</div>
      <div class="value ${(bm.trade_pnl||0) >= 0 ? 'green' : 'red'}">$${(bm.trade_pnl||0).toFixed(2)}</div>
      <div class="detail">Edge > 5% trades only</div>
    </div>
    <div class="stat-card">
      <div class="label">Total API Cost</div>
      <div class="value">$${(summary.total_api_cost||0).toFixed(2)}</div>
      <div class="detail">${summary.total_iterations} iterations</div>
    </div>
  `;
}

function renderAccuracyChart(iters) {
  destroyChart('accuracyChart');
  const ctx = document.getElementById('accuracyChart').getContext('2d');
  chartInstances['accuracyChart'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: iters.map(i => '#' + i.iteration),
      datasets: [{
        label: 'Accuracy',
        data: iters.map(i => ((i.metrics||{}).win_rate||0)*100),
        borderColor: '#00d4ff',
        backgroundColor: 'rgba(0,212,255,0.1)',
        fill: true, tension: 0.3, pointRadius: 5, pointBackgroundColor: '#00d4ff',
      }, {
        label: '60% Target',
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
      labels: iters.map(i => '#' + i.iteration),
      datasets: [{
        label: 'AI Brier', data: iters.map(i => (i.metrics||{}).avg_brier_ai||0),
        backgroundColor: '#7b61ff', borderRadius: 4,
      }, {
        label: 'Market Brier', data: iters.map(i => (i.metrics||{}).avg_brier_market||0),
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
        label: 'Predicted', data: buckets.map(b => (cal[b].avg_predicted||0)*100),
        backgroundColor: '#7b61ff', borderRadius: 4,
      }, {
        label: 'Actual', data: buckets.map(b => (cal[b].actual_rate||0)*100),
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
      labels: iters.map(i => '#' + i.iteration),
      datasets: [{
        label: 'Win Rate %',
        data: iters.map(i => ((i.metrics||{}).trade_win_rate||0)*100),
        backgroundColor: iters.map(i => ((i.metrics||{}).trade_win_rate||0) >= 0.5 ? '#10b981' : '#374151'),
        borderRadius: 4, yAxisID: 'y',
      }, {
        label: 'P&L ($)', data: iters.map(i => (i.metrics||{}).trade_pnl||0),
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
      <td>#${i.iteration}</td>
      <td><span class="badge ${(m.win_rate||0) >= 0.75 ? 'badge-green' : 'badge-blue'}">${((m.win_rate||0)*100).toFixed(1)}%</span></td>
      <td>${(m.avg_brier_ai||0).toFixed(4)}</td>
      <td>${(m.avg_brier_market||0).toFixed(4)}</td>
      <td><span class="badge ${beats ? 'badge-green' : 'badge-red'}">${beats ? 'AI Wins' : 'Market'}</span></td>
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
      <td><strong>${e.label}</strong></td>
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

  // Brier comparison
  destroyChart('strategyCompareChart');
  const ctx1 = document.getElementById('strategyCompareChart').getContext('2d');
  chartInstances['strategyCompareChart'] = new Chart(ctx1, {
    type: 'bar',
    data: {
      labels: evals.map(e => e.label.replace('_', ' ')),
      datasets: [{
        label: 'AI Brier (lower=better)', data: evals.map(e => e.brier),
        backgroundColor: evals.map(e => e.brier_edge > 0 ? '#7b61ff' : '#374151'),
        borderRadius: 4,
      }, {
        label: 'Market Brier', data: evals.map(e => e.market_brier),
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

  // P&L comparison
  destroyChart('strategyPnlChart');
  const ctx2 = document.getElementById('strategyPnlChart').getContext('2d');
  chartInstances['strategyPnlChart'] = new Chart(ctx2, {
    type: 'bar',
    data: {
      labels: evals.map(e => e.label.replace('_', ' ')),
      datasets: [{
        label: 'Trade P&L ($)', data: evals.map(e => e.trade_pnl),
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
