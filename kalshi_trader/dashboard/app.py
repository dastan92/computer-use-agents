"""Trading dashboard API and web server."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Kalshi AI Trader Dashboard")

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


@app.get("/api/iterations")
def get_iterations():
    return load_iterations()


@app.get("/api/predictions")
def get_predictions():
    return load_predictions()


@app.get("/api/summary")
def get_summary():
    iterations = load_iterations()
    if not iterations:
        return {"error": "No backtest data yet"}

    best = max(iterations, key=lambda x: x.get("metrics", {}).get("win_rate", 0))
    latest = iterations[-1]

    return {
        "total_iterations": len(iterations),
        "best_iteration": best.get("iteration"),
        "best_accuracy": best.get("metrics", {}).get("win_rate"),
        "best_brier": best.get("metrics", {}).get("avg_brier_ai"),
        "latest_iteration": latest.get("iteration"),
        "latest_accuracy": latest.get("metrics", {}).get("win_rate"),
        "total_api_cost": sum(i.get("api_stats", {}).get("total_cost", 0) for i in iterations),
        "ai_beats_market": best.get("metrics", {}).get("brier_improvement", 0) > 0,
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
  }
  .header h1 {
    font-size: 24px;
    font-weight: 600;
    background: linear-gradient(90deg, #00d4ff, #7b61ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .header .subtitle { color: #6b7280; font-size: 14px; margin-top: 4px; }
  .container { max-width: 1400px; margin: 0 auto; padding: 24px; }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }
  .stat-card {
    background: #111827;
    border: 1px solid #1e2a3a;
    border-radius: 12px;
    padding: 20px;
  }
  .stat-card .label { color: #6b7280; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
  .stat-card .value {
    font-size: 32px;
    font-weight: 700;
    margin-top: 8px;
    background: linear-gradient(90deg, #00d4ff, #7b61ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .stat-card .value.green { background: linear-gradient(90deg, #10b981, #34d399); -webkit-background-clip: text; }
  .stat-card .value.red { background: linear-gradient(90deg, #ef4444, #f87171); -webkit-background-clip: text; }
  .stat-card .detail { color: #6b7280; font-size: 12px; margin-top: 4px; }

  .charts-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
    gap: 24px;
    margin-bottom: 24px;
  }
  .chart-card {
    background: #111827;
    border: 1px solid #1e2a3a;
    border-radius: 12px;
    padding: 24px;
  }
  .chart-card h3 { font-size: 16px; margin-bottom: 16px; color: #e0e0e0; }

  .table-card {
    background: #111827;
    border: 1px solid #1e2a3a;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
    overflow-x: auto;
  }
  .table-card h3 { font-size: 16px; margin-bottom: 16px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; padding: 10px 12px; color: #6b7280; border-bottom: 1px solid #1e2a3a;
       text-transform: uppercase; font-size: 11px; letter-spacing: 1px; }
  td { padding: 10px 12px; border-bottom: 1px solid #0d1321; }
  tr:hover td { background: #1a1f35; }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 9999px;
    font-size: 11px; font-weight: 600;
  }
  .badge-green { background: #064e3b; color: #34d399; }
  .badge-red { background: #7f1d1d; color: #fca5a5; }
  .badge-blue { background: #1e3a5f; color: #60a5fa; }

  .predictions-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
    gap: 12px;
  }
  .pred-card {
    background: #0d1321;
    border: 1px solid #1e2a3a;
    border-radius: 8px;
    padding: 16px;
  }
  .pred-card .title { font-size: 13px; font-weight: 500; margin-bottom: 8px; }
  .pred-bar {
    height: 8px; border-radius: 4px; background: #1e2a3a; overflow: hidden; margin: 8px 0;
  }
  .pred-bar-fill { height: 100%; border-radius: 4px; transition: width 0.5s; }
  .pred-meta { display: flex; justify-content: space-between; font-size: 11px; color: #6b7280; }
</style>
</head>
<body>
<div class="header">
  <h1>Kalshi AI Trader</h1>
  <div class="subtitle">Powered by Claude Sonnet &mdash; Prediction Market Intelligence</div>
</div>
<div class="container">
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
      <h3>Calibration (Iteration #9)</h3>
      <canvas id="calibrationChart"></canvas>
    </div>
    <div class="chart-card">
      <h3>Trade Win Rate & P&L</h3>
      <canvas id="tradeChart"></canvas>
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
  <div class="table-card">
    <h3>Live Predictions</h3>
    <div id="predictions" class="predictions-grid"></div>
  </div>
</div>
<script>
const API = window.location.origin;

async function load() {
  const [itersRes, predsRes, summaryRes] = await Promise.all([
    fetch(API + '/api/iterations').then(r => r.json()),
    fetch(API + '/api/predictions').then(r => r.json()),
    fetch(API + '/api/summary').then(r => r.json()),
  ]);

  renderStats(summaryRes, itersRes);
  renderAccuracyChart(itersRes);
  renderBrierChart(itersRes);
  renderCalibrationChart(itersRes);
  renderTradeChart(itersRes);
  renderTable(itersRes);
  renderPredictions(predsRes);
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
    <div class="stat-card">
      <div class="label">Iterations</div>
      <div class="value">${summary.total_iterations}</div>
      <div class="detail">Latest: #${summary.latest_iteration}</div>
    </div>
  `;
}

function renderAccuracyChart(iters) {
  const ctx = document.getElementById('accuracyChart').getContext('2d');
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: iters.map(i => 'Iter ' + i.iteration),
      datasets: [{
        label: 'Accuracy',
        data: iters.map(i => ((i.metrics||{}).win_rate||0)*100),
        borderColor: '#00d4ff',
        backgroundColor: 'rgba(0,212,255,0.1)',
        fill: true,
        tension: 0.3,
        pointRadius: 6,
        pointBackgroundColor: '#00d4ff',
      }, {
        label: '60% Target',
        data: iters.map(() => 60),
        borderColor: '#ef4444',
        borderDash: [5, 5],
        pointRadius: 0,
      }]
    },
    options: {
      responsive: true,
      scales: {
        y: { min: 40, max: 100, ticks: { callback: v => v + '%', color: '#6b7280' }, grid: { color: '#1e2a3a' } },
        x: { ticks: { color: '#6b7280' }, grid: { color: '#1e2a3a' } }
      },
      plugins: { legend: { labels: { color: '#e0e0e0' } } }
    }
  });
}

function renderBrierChart(iters) {
  const ctx = document.getElementById('brierChart').getContext('2d');
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: iters.map(i => 'Iter ' + i.iteration),
      datasets: [{
        label: 'AI Brier',
        data: iters.map(i => (i.metrics||{}).avg_brier_ai||0),
        backgroundColor: '#7b61ff',
        borderRadius: 4,
      }, {
        label: 'Market Brier',
        data: iters.map(i => (i.metrics||{}).avg_brier_market||0),
        backgroundColor: '#374151',
        borderRadius: 4,
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
  const best = iters[iters.length - 1]; // latest
  const cal = (best.metrics||{}).calibration||{};
  const buckets = Object.keys(cal);
  const ctx = document.getElementById('calibrationChart').getContext('2d');
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: buckets,
      datasets: [{
        label: 'Predicted',
        data: buckets.map(b => (cal[b].avg_predicted||0)*100),
        backgroundColor: '#7b61ff',
        borderRadius: 4,
      }, {
        label: 'Actual',
        data: buckets.map(b => (cal[b].actual_rate||0)*100),
        backgroundColor: '#10b981',
        borderRadius: 4,
      }]
    },
    options: {
      responsive: true,
      scales: {
        y: { min: 0, max: 100, ticks: { callback: v => v + '%', color: '#6b7280' }, grid: { color: '#1e2a3a' } },
        x: { ticks: { color: '#6b7280' }, grid: { color: '#1e2a3a' } }
      },
      plugins: { legend: { labels: { color: '#e0e0e0' } } }
    }
  });
}

function renderTradeChart(iters) {
  const ctx = document.getElementById('tradeChart').getContext('2d');
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: iters.map(i => 'Iter ' + i.iteration),
      datasets: [{
        label: 'Win Rate %',
        data: iters.map(i => ((i.metrics||{}).trade_win_rate||0)*100),
        backgroundColor: iters.map(i => ((i.metrics||{}).trade_win_rate||0) >= 0.5 ? '#10b981' : '#374151'),
        borderRadius: 4,
        yAxisID: 'y',
      }, {
        label: 'P&L ($)',
        data: iters.map(i => (i.metrics||{}).trade_pnl||0),
        type: 'line',
        borderColor: '#f59e0b',
        backgroundColor: 'rgba(245,158,11,0.1)',
        fill: true,
        tension: 0.3,
        pointRadius: 5,
        yAxisID: 'y1',
      }]
    },
    options: {
      responsive: true,
      scales: {
        y: { min: 0, max: 70, position: 'left', ticks: { callback: v => v + '%', color: '#6b7280' }, grid: { color: '#1e2a3a' } },
        y1: { position: 'right', ticks: { callback: v => '$' + v.toFixed(1), color: '#6b7280' }, grid: { display: false } },
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
      <td><span class="badge ${beats ? 'badge-green' : 'badge-red'}">${beats ? 'AI Wins' : 'Market Wins'}</span></td>
      <td>${((m.trade_win_rate||0)*100).toFixed(1)}%</td>
      <td class="${(m.trade_pnl||0) >= 0 ? '' : 'red'}">$${(m.trade_pnl||0).toFixed(2)}</td>
      <td>$${(i.api_stats||{}).total_cost?.toFixed(4)||'?'}</td>
    </tr>`;
  }).join('');
}

function renderPredictions(preds) {
  const el = document.getElementById('predictions');
  if (!preds.length) {
    el.innerHTML = '<p style="color:#6b7280;padding:16px;">No live predictions yet. Run the prediction service to see real-time forecasts.</p>';
    return;
  }
  el.innerHTML = preds.slice(0, 20).map(p => {
    const prob = (p.ai_probability||0.5)*100;
    const color = prob > 60 ? '#10b981' : prob < 40 ? '#ef4444' : '#f59e0b';
    return `<div class="pred-card">
      <div class="title">${(p.title||p.ticker||'').slice(0,80)}</div>
      <div class="pred-bar"><div class="pred-bar-fill" style="width:${prob}%;background:${color}"></div></div>
      <div class="pred-meta">
        <span>AI: ${prob.toFixed(0)}%</span>
        <span>Market: ${((p.market_probability||0.5)*100).toFixed(0)}%</span>
        <span>Edge: ${((p.edge||0)*100).toFixed(1)}%</span>
        <span>${p.category||''}</span>
      </div>
    </div>`;
  }).join('');
}

load();
setInterval(load, 60000); // refresh every minute
</script>
</body>
</html>
"""
