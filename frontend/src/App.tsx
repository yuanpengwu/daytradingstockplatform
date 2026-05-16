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

function App() {
  const [data, setData] = useState<StatusData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch('/api/status');
        const json = await res.json();
        setData(json);
      } catch (e) {
        console.error("Failed to fetch status:", e);
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
    const interval = setInterval(fetchData, 5000); // Poll every 5s
    return () => clearInterval(interval);
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
    </div>
  )
}

export default App
