"""
config.py — All scanner settings in one place.
Edit this file to change tickers, intervals, and indicator parameters.
"""

import os

CONFIG = {
    # ── Telegram ──────────────────────────────────────────────────────────────
    # Get these by:
    #   1. Message @BotFather on Telegram → /newbot → copy the token
    #   2. Message your bot, then visit:
    #      https://api.telegram.org/bot<TOKEN>/getUpdates → copy "id" from "chat"
    "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE"),
    "TELEGRAM_CHAT_ID":   os.getenv("TELEGRAM_CHAT_ID",   "YOUR_CHAT_ID_HERE"),

    # ── Watchlist ─────────────────────────────────────────────────────────────
    "tickers": [
        "AAPL", "NVDA", "TSLA", "MSFT", "AMD",
        "META", "GOOGL", "SPY", "QQQ",
    ],

    # ── Candle interval ───────────────────────────────────────────────────────
    # Options: "1m","2m","5m","15m","30m","60m","1h","90m","1d"
    # yfinance intraday data (< 1d) is available for last 60 days max.
    "interval": "30m",

    # ── Lookback period for yfinance ──────────────────────────────────────────
    # Must be enough bars to calculate all indicators.
    # For 30m interval → "1mo" gives ~300+ bars. Fine.
    "period": "1mo",

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    "bb_period": 20,
    "bb_std":    0.01,   # standard deviation multiplier

    # ── RSI ───────────────────────────────────────────────────────────────────
    "rsi_period":    14,
    "rsi_threshold": 1,   # signal fires when RSI > this (long) or < 100-this (short)

    # ── Zero Lag MACD (port of Pine Script Enhanced v1.2 by Albert Callisto) ──
    "macd": {
        "fast":           12,
        "slow":           26,
        "signal_period":  9,
        "macd_ema_period": 9,
        "use_ema":        True,   # False = SMA ("Glaz mode")
        "use_old_algo":   False,  # False = real zero-lag signal line
    },

    # ── Pre-market gap alerts ─────────────────────────────────────────────────
    # Fires between 8:45–9:30 AM ET for any ticker gapping ± this % vs prev close
    "gap_pct_threshold": 2.0,   # e.g. 2.0 = alert on gaps of 2% or more

    # ── Multi-timeframe confirmation ──────────────────────────────────────────
    # When True, a LONG signal also requires the daily trend to be bullish (price > 50-day SMA).
    # A SHORT signal requires the daily trend to be bearish.
    # Set False to fire signals regardless of higher timeframe.
    "require_htf_confirmation": False,
}
