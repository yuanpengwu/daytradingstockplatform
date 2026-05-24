"""Sector → constituent ticker mapping.

Used by RiskManager to enforce the max-1-position-per-sector rule,
preventing correlated blow-ups when an entire sector sells off.
"""
from __future__ import annotations

# Sector name → list of common tickers in that sector.
# Keep this list broad so the check catches the most common holdings.
SECTOR_MAP: dict[str, list[str]] = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "AMD", "INTC", "QCOM", "CRM", "TXN",
        "AVGO", "ORCL", "CSCO", "ADBE", "MU", "AMAT", "KLAC", "LRCX",
        "MRVL", "PANW", "SNPS", "CDNS", "FTNT", "ANSS",
    ],
    "Communication Services": [
        "META", "GOOGL", "GOOG", "NFLX", "DIS", "T", "VZ", "CMCSA",
        "ATVI", "EA", "TTWO", "RBLX", "SNAP", "PINS", "MTCH",
    ],
    "Financials": [
        "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA", "AXP",
        "BLK", "SCHW", "USB", "PNC", "TFC", "COF", "DFS", "BX",
        "KKR", "APO", "SPGI", "MCO", "ICE", "CME",
    ],
    "Energy": [
        "XOM", "CVX", "SLB", "OXY", "COP", "MPC", "PSX", "VLO",
        "EOG", "HAL", "BKR", "DVN", "HES", "MRO", "APA", "FANG",
        "PXD", "CTRA", "EQT",
    ],
    "Health Care": [
        "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT", "LLY",
        "BMY", "AMGN", "MDT", "ISRG", "SYK", "BSX", "ELV", "CI",
        "HUM", "CVS", "MCK", "CAH", "ABC",
    ],
    "Consumer Discretionary": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW",
        "BKNG", "GM", "F", "RIVN", "LCID", "RH", "ULTA", "BBY",
        "M", "KSS", "GPS", "ANF",
    ],
    "Consumer Staples": [
        "PG", "KO", "PEP", "WMT", "COST", "PM", "MO", "CL",
        "GIS", "K", "HSY", "MKC", "CAG", "CPB", "SJM",
    ],
    "Industrials": [
        "HON", "UPS", "RTX", "CAT", "DE", "BA", "MMM", "GE",
        "LMT", "NOC", "GD", "LHX", "TDG", "FDX", "XPO", "ODFL",
        "CSX", "NSC", "UNP", "URI", "PCAR",
    ],
    "Materials": [
        "LIN", "APD", "ECL", "NEM", "FCX", "DD", "NUE", "ALB",
        "CF", "MOS", "FMC", "EMN", "CE", "PPG", "SHW",
    ],
    "Real Estate": [
        "AMT", "PLD", "CCI", "EQIX", "SPG", "O", "VICI", "WELL",
        "PSA", "EXR", "AVB", "EQR", "VTR", "PEAK", "ARE",
    ],
    "Utilities": [
        "NEE", "DUK", "SO", "AEP", "EXC", "D", "PCG", "SRE",
        "ED", "WEC", "ES", "XEL", "CMS", "ETR",
    ],
}

# ETF ticker → sector name (for the sector-ETF universe picks).
ETF_SECTOR: dict[str, str] = {
    "XLK":  "Technology",
    "XLC":  "Communication Services",
    "XLF":  "Financials",
    "XLE":  "Energy",
    "XLV":  "Health Care",
    "XLY":  "Consumer Discretionary",
    "XLP":  "Consumer Staples",
    "XLI":  "Industrials",
    "XLB":  "Materials",
    "XLRE": "Real Estate",
    "XLU":  "Utilities",
}

# Reverse map: ticker → sector name (built once at import time).
TICKER_SECTOR: dict[str, str] = {
    ticker: sector
    for sector, tickers in SECTOR_MAP.items()
    for ticker in tickers
}
TICKER_SECTOR.update(ETF_SECTOR)

# Broad market ETFs — exempt from the sector concentration rule.
MARKET_ETFS: set[str] = {"SPY", "QQQ", "IWM", "DIA", "VTI"}


def get_sector(symbol: str) -> str:
    """Return the sector name for a symbol, or 'Unknown' if not mapped."""
    return TICKER_SECTOR.get(symbol.upper(), "Unknown")


def is_market_etf(symbol: str) -> bool:
    """Return True for broad-market ETFs that are exempt from sector limits."""
    return symbol.upper() in MARKET_ETFS
