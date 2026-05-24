import { useEffect, useState } from 'react'
import './index.css'

interface Position {
  symbol: string;
  qty: number;
  avg_entry_price: number;
  current_price: number;
  market_value: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
}

interface Decision {
  symbol: string;
  score: number;
  confidence: number;
  action: string;
}

interface StatusData {
  updated_at: string;
  cycle_count: number;
  equity: number;
  cash: number;
  market_value: number;
  unrealized_pnl: number;
  kill_switch: boolean;
  positions: Position[];
  decisions: Decision[];
  error?: string;
}

interface WinRatePeriod {
  trades: number;
  wins: number;
  losses: number;
  win_rate: number | null;
  total_pnl: number;
}

interface WinRateData {
  '1d': WinRatePeriod;
  '1w': WinRatePeriod;
  '1m': WinRatePeriod;
  '3m': WinRatePeriod;
  '6m': WinRatePeriod;
  '1y': WinRatePeriod;
}

const PERIOD_LABELS: Record<string, string> = {
  '1d': '1 Day', '1w': '1 Week', '1m': '1 Month',
  '3m': '3 Months', '6m': '6 Months', '1y': '1 Year',
};

function App() {
  const [data, setData] = useState<StatusData | null>(null);
  const [winRate, setWinRate] = useState<WinRateData | null>(null);
  const [winRateError, setWinRateError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [apiOffline, setApiOffline] = useState(false);

  // Status poll — every 5 s
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch('/api/status');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        setApiOffline(false);
        setData(json);
      } catch (e) {
        console.error("Failed to fetch status:", e);
        setApiOffline(true);
      } finally {
        setLoading(false);
      }
    };
    fetchStatus();
    const id = setInterval(fetchStatus, 5000);
    return () => clearInterval(id);
  }, []);

  // Win-rate poll — independent, every 30 s
  useEffect(() => {
    const fetchWinRate = async () => {
      try {
        const res = await fetch('/api/winrate');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        setWinRate(json);
        setWinRateError(null);
      } catch (e: any) {
        console.error("Failed to fetch win rate:", e);
        setWinRateError(e?.message ?? "Unknown error");
      }
    };
    fetchWinRate();
    const id = setInterval(fetchWinRate, 30000);
    return () => clearInterval(id);
  }, []);

  const handleEmergencySell = async () => {
    if (!confirm("Are you sure you want to liquidate all positions?")) return;
    try {
      const res = await fetch('/api/emergency_sell', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason: "Emergency Sell Triggered via Node UI" })
      });
      const result = await res.json();
      alert(result.message || "Liquidation triggered.");
    } catch (e) {
      alert("Failed to send emergency sell command.");
    }
  };

  if (loading && !data) return <div className="loader">Loading...</div>;
  if (apiOffline) return (
    <div className="error-screen">
      <h2>Backend Offline</h2>
      <p>Cannot reach the API at <code>/api/status</code>. Start the FastAPI backend and refresh.</p>
    </div>
  );
  if (data?.error) return <div className="error-screen">{data.error}</div>;

  return (
    <div className="dashboard">
      <header className="header glass">
        <div className="logo">
          <h1>DayTradingBot <span>PRO</span></h1>
          <span className="live-indicator">LIVE</span>
        </div>
        <button className="btn-danger" onClick={handleEmergencySell}>
          🚨 EMERGENCY SELL ALL
        </button>
      </header>

      <div className="metrics-grid">
        <div className="metric-card glass">
          <div className="label">Total Equity</div>
          <div className="value">${data?.equity?.toLocaleString(undefined, {minimumFractionDigits: 2})}</div>
        </div>
        <div className="metric-card glass">
          <div className="label">Cash Available</div>
          <div className="value">${data?.cash?.toLocaleString(undefined, {minimumFractionDigits: 2})}</div>
        </div>
        <div className="metric-card glass">
          <div className="label">Invested</div>
          <div className="value">${data?.market_value?.toLocaleString(undefined, {minimumFractionDigits: 2})}</div>
        </div>
        <div className="metric-card glass">
          <div className="label">Open P&L</div>
          <div className={`value ${(data?.unrealized_pnl || 0) >= 0 ? 'pos' : 'neg'}`}>
            ${(data?.unrealized_pnl || 0) > 0 ? '+' : ''}{data?.unrealized_pnl?.toLocaleString(undefined, {minimumFractionDigits: 2})}
          </div>
        </div>
      </div>

      <div className="content-grid">
        <div className="panel glass">
          <h2>Open Positions</h2>
          {data?.positions?.length === 0 ? (
            <p className="muted">No open positions.</p>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Qty</th>
                  <th>Entry</th>
                  <th>Last</th>
                  <th>Value</th>
                  <th>P&L</th>
                </tr>
              </thead>
              <tbody>
                {data?.positions?.map(p => (
                  <tr key={p.symbol}>
                    <td><strong>{p.symbol}</strong></td>
                    <td>{p.qty}</td>
                    <td>${p.avg_entry_price.toFixed(2)}</td>
                    <td>${p.current_price.toFixed(2)}</td>
                    <td>${p.market_value.toLocaleString(undefined, {minimumFractionDigits:2})}</td>
                    <td className={p.unrealized_pnl >= 0 ? 'pos' : 'neg'}>
                      ${p.unrealized_pnl > 0 ? '+' : ''}{p.unrealized_pnl.toFixed(2)} ({p.unrealized_pnl_pct > 0 ? '+' : ''}{(p.unrealized_pnl_pct * 100).toFixed(2)}%)
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        <div className="panel glass">
          <h2>Live Signal Scores</h2>
          {data?.decisions?.length === 0 ? (
            <p className="muted">No signals evaluated.</p>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Score</th>
                  <th>Confidence</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {data?.decisions?.sort((a,b) => b.score - a.score).map(d => (
                  <tr key={d.symbol}>
                    <td><strong>{d.symbol}</strong></td>
                    <td>{d.score > 0 ? '+' : ''}{d.score.toFixed(3)}</td>
                    <td>{d.confidence.toFixed(2)}</td>
                    <td>
                      <span className={`badge ${d.action.toLowerCase()}`}>{d.action}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      <div className="panel glass" style={{margin: '0 0 24px 0'}}>
        <h2>Win Rate</h2>
        {winRateError ? (
          <p className="muted" style={{color:'#e74c3c'}}>⚠ Could not load win rate: {winRateError}</p>
        ) : !winRate ? (
          <p className="muted">Loading win rate…</p>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>Period</th>
                <th>Trades</th>
                <th>Wins</th>
                <th>Losses</th>
                <th>Win Rate</th>
                <th>Total P&L</th>
              </tr>
            </thead>
            <tbody>
              {(Object.keys(PERIOD_LABELS) as Array<keyof WinRateData>).map(key => {
                const p = winRate[key];
                return (
                  <tr key={key}>
                    <td><strong>{PERIOD_LABELS[key]}</strong></td>
                    <td>{p.trades}</td>
                    <td className={p.wins > 0 ? 'pos' : ''}>{p.wins}</td>
                    <td className={p.losses > 0 ? 'neg' : ''}>{p.losses}</td>
                    <td>
                      {p.win_rate === null
                        ? <span className="muted">No trades</span>
                        : <strong className={p.win_rate >= 0.5 ? 'pos' : 'neg'}>
                            {(p.win_rate * 100).toFixed(1)}%
                          </strong>
                      }
                    </td>
                    <td className={p.total_pnl > 0 ? 'pos' : p.total_pnl < 0 ? 'neg' : ''}>
                      {p.total_pnl > 0 ? '+' : ''}${p.total_pnl.toFixed(2)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

export default App
