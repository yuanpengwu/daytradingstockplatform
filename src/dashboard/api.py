import json
import yaml
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="DayTradingBot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/status")
def get_status():
    p = Path("status.json")
    if not p.exists():
        return {"error": "No status yet. Ensure main.py is running."}
    with open(p) as f:
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
