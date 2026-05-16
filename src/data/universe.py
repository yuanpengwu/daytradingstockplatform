import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from ..utils.logger import get_logger

log = get_logger(__name__)

# Major Sector ETFs
SECTORS = {
    "XLK": "Technology",
    "XLV": "Health Care",
    "XLF": "Financials",
    "XLY": "Consumer Discretionary",
    "XLC": "Communication Services",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLB": "Materials"
}

# Predefined pool of high-liquidity stocks per sector
SECTOR_STOCKS = {
    "XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "ADBE", "CRM", "AMD", "QCOM", "INTC", "TXN"],
    "XLV": ["UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY"],
    "XLF": ["BRK-B", "JPM", "V", "MA", "BAC", "WFC", "MS", "GS", "C", "BLK"],
    "XLY": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "BKNG", "TJX", "TGT"],
    "XLC": ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "VZ", "T", "CHTR", "TMUS", "EA"],
    "XLI": ["HON", "UPS", "UNP", "BA", "RTX", "CAT", "LMT", "GE", "DE", "MMM"],
    "XLP": ["PG", "PEP", "KO", "WMT", "COST", "PM", "MO", "TGT", "DG", "KMB"],
    "XLE": ["XOM", "CVX", "COP", "SLB", "EOG", "PXD", "MPC", "PSX", "VLO", "OXY"],
    "XLU": ["NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "ED", "PEG"],
    "XLRE": ["AMT", "PLD", "CCI", "EQIX", "PSA", "O", "SPG", "WELL", "DLR", "AVB"],
    "XLB": ["LIN", "APD", "SHW", "NEM", "ECL", "FCX", "CTVA", "DOW", "NUE", "VMC"]
}

class DynamicUniverse:
    def __init__(self, cfg: dict):
        self.enabled = cfg.get("dynamic", False)
        self.max_tickers = cfg.get("max_tickers", 10)
        self.method = cfg.get("method", "sector_momentum")
        self.fallback = cfg.get("fallback_tickers", ["SPY", "QQQ"])
        self.lookback_days = 5

    def select_tickers(self) -> list[str]:
        if not self.enabled:
            return self.fallback
            
        log.info(f"Dynamic Universe enabled. Running '{self.method}' selection...")
        try:
            if self.method == "sector_momentum":
                return self._select_sector_momentum()
            else:
                log.warning(f"Unknown method {self.method}. Using fallback.")
                return self.fallback
        except Exception as e:
            log.error(f"Universe selection failed: {e}. Using fallback tickers.")
            return self.fallback

    def _select_sector_momentum(self) -> list[str]:
        import os
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from alpaca.data.enums import DataFeed

        key = os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_API_SECRET")
        if not key or not secret:
            raise RuntimeError("Alpaca keys missing, cannot fetch ETF data.")
            
        client = StockHistoricalDataClient(key, secret)

        """Rank sectors by 5-day return and pick top stocks from the best sector."""
        etfs = list(SECTORS.keys())
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=15) # Fetch more days to ensure we get 5 trading days
        
        log.info(f"Fetching recent data for {len(etfs)} Sector ETFs using Alpaca...")
        
        req = StockBarsRequest(
            symbol_or_symbols=etfs,
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=start_dt,
            end=end_dt,
            feed=DataFeed.IEX
        )
        
        bars = client.get_stock_bars(req).df
        
        if bars.empty:
            raise RuntimeError("Failed to fetch ETF data from Alpaca.")
            
        # Group by symbol and calculate return
        returns = {}
        for ticker in etfs:
            if ticker in bars.index.get_level_values(0):
                df = bars.loc[ticker]
                if len(df) >= 2:
                    period = min(self.lookback_days, len(df) - 1)
                    ret = (df['close'].iloc[-1] / df['close'].iloc[-1 - period]) - 1
                    returns[ticker] = ret
                    
        if not returns:
            raise RuntimeError("Not enough ETF data points.")
            
        import pandas as pd
        returns_s = pd.Series(returns).sort_values(ascending=False)
        best_etf = returns_s.index[0]
        best_return = returns_s.iloc[0]
        
        log.info(f"Top Sector identified: {best_etf} ({SECTORS[best_etf]}) with {best_return:.2%} return.")
        
        # Select stocks from this sector
        candidate_stocks = SECTOR_STOCKS.get(best_etf, [])
        if not candidate_stocks:
            return self.fallback
            
        selected = candidate_stocks[:self.max_tickers]
        
        # Ensure fallback tickers are always included
        combined = selected + [t for t in self.fallback if t not in selected]
        log.info(f"Selected {len(combined)} tickers from {best_etf} and fallback pool: {combined}")
        
        return combined
