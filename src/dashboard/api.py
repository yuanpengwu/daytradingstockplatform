import json
import yaml
from pathlib import Path
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ..utils.trade_history import TradeHistory

# Resolve the project root regardless of the CWD at startup
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TRADES_PATH  = _PROJECT_ROOT / "trades.json"
_STATUS_PATH  = _PROJECT_ROOT / "status.json"

app = FastAPI(title="DayTradingBot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/winrate")
def get_winrate():
    return TradeHistory(path=str(_TRADES_PATH)).win_rate_summary()


@app.get("/api/status")
def get_status(response: Response):
    response.headers["Cache-Control"] = "no-store"
    if not _STATUS_PATH.exists():
        return {"error": "No status yet. Ensure main.py is running."}
    with open(_STATUS_PATH) as f:
        return json.load(f)

class SellRequest(BaseModel):
    reason: str = "Manual dashboard emergency sell"

@app.post("/api/emergency_sell")
def emergency_sell(req: SellRequest):
    p = Path("config.yaml")
    if not p.exists():
        p = Path("config.yaml.example")
    with open(p) as f:
        cfg = yaml.safe_load(f)
        
    from src.brokers import get_broker
    broker = get_broker(cfg["broker"]["name"], cfg["broker"])
    from src.execution.trader import Trader
    from src.risk.risk_manager import RiskManager
    
    risk = RiskManager(cfg.get("risk", {}))
    trader = Trader(broker, risk, notify_channels=cfg.get("notifications", {}).get("channels", ["console"]))
    trader.flatten_all(req.reason)
    return {"status": "success", "message": "Liquidation orders sent!"}

from fastapi.staticfiles import StaticFiles
frontend_path = Path("frontend/dist")
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
