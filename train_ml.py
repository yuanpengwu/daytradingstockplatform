import yaml
import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from src.signals.ml_model import MLSignal
from src.utils.logger import get_logger
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

log = get_logger("train_ml")

def main():
    load_dotenv()
    log.info("Loading config...")
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    tickers = cfg["universe"]["tickers"]
    log.info(f"Target universe: {tickers}")
    
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    if not key or not secret:
        log.error("Alpaca keys not found in .env!")
        return
        
    client = StockHistoricalDataClient(key, secret)
    
    log.info("Downloading historical data using Alpaca (5m intervals, 30 days)...")
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=30)
    
    req = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        start=start_dt,
        end=end_dt,
        feed=DataFeed.IEX
    )
    
    try:
        bars = client.get_stock_bars(req).df
    except Exception as e:
        log.error(f"Failed to fetch data from Alpaca: {e}")
        return
        
    bars_per_ticker = {}
    for ticker in tickers:
        try:
            if ticker in bars.index.get_level_values(0):
                df = bars.loc[ticker].copy()
                # Alpaca columns are lowercase (open, high, low, close, volume)
                # MLSignal expects Capitalized columns (Open, High, Low, Close, Volume)
                df.rename(columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume"
                }, inplace=True)
                bars_per_ticker[ticker] = df
        except Exception as e:
            log.warning(f"Failed to extract data for {ticker}: {e}")
            
    log.info(f"Assembled data for {len(bars_per_ticker)} tickers.")
    
    if not bars_per_ticker:
        log.error("No data assembled. Aborting training.")
        return

    # Train the model
    log.info("Beginning model training on GPU...")
    MLSignal.train_from_history(
        tickers=list(bars_per_ticker.keys()),
        bars_per_ticker=bars_per_ticker,
        horizon_minutes=15,
        threshold=0.002,
        save_to="models/xgb_intraday.joblib"
    )
    
    log.info("Training complete. Model deployed to models/xgb_intraday.joblib.")

if __name__ == "__main__":
    main()
