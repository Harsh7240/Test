"""
Bollinger Band + RSI + Pre-market Gap Scanner
Sends Telegram alerts on confirmed candle close signals.
Runs 24/7.
"""

import os
import time
import logging
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from config import CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

# ── Deduplication: prevents repeat alerts for same candle ─────────────────────
_alerted: set = set()

# ── Telegram ──────────────────────────────────────────────────────────────────

def send_telegram(message: str):
    token   = CONFIG["TELEGRAM_BOT_TOKEN"]
    chat_id = CONFIG["TELEGRAM_CHAT_ID"]
    url     = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(
            url,
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=10
        )
        r.raise_for_status()
        log.info(f"Telegram sent: {message[:80]}...")
    except Exception as e:
        log.error(f"Telegram error: {e}")

# ── Data fetching ──────────────────────────────────────────────────────────────

def fetch_ohlcv(ticker: str, interval: str, period: str) -> pd.DataFrame:
    try:
        df = yf.download(
            ticker, interval=interval, period=period,
            progress=False, auto_adjust=True
        )
        if df.empty:
            log.warning(f"No data returned for {ticker}")
            return pd.DataFrame()
        df = df.rename(columns={
            "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume"
        })
        # Drop last candle — may be incomplete
        df = df.iloc[:-1]
        return df
    except Exception as e:
        log.error(f"Fetch error {ticker}: {e}")
        return pd.DataFrame()

# ── Indicators ─────────────────────────────────────────────────────────────────

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()

def calc_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    delta    = closes.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_bollinger(closes: pd.Series, period: int = 20, std_mult: float = 2.0):
    mid = closes.rolling(period).mean()
    std = closes.rolling(period).std(ddof=0)
    return mid + std_mult * std, mid, mid - std_mult * std  # upper, mid, lower

# ── Pre-market gap ─────────────────────────────────────────────────────────────

def calc_premarket_gap(ticker: str, gap_pct_threshold: float) -> dict | None:
    try:
        t          = yf.Ticker(ticker)
        info       = t.fast_info
        prev_close = info.previous_close
        pre_mkt    = info.pre_market_price
        if prev_close and pre_mkt:
            gap_pct = ((pre_mkt - prev_close) / prev_close) * 100
            if abs(gap_pct) >= gap_pct_threshold:
                return {
                    "direction": "UP" if gap_pct > 0 else "DOWN",
                    "gap_pct":   gap_pct,
                    "prev_close": prev_close,
                    "pre_mkt":   pre_mkt
                }
    except Exception as e:
        log.warning(f"Pre-market gap check failed for {ticker}: {e}")
    return None

# ── Multi-timeframe confirmation ───────────────────────────────────────────────

def check_higher_tf_trend(ticker: str, primary_interval: str) -> str:
    htf_map = {"30m": ("1d", "3mo"), "1h": ("1d", "3mo"), "15m": ("1h", "1mo")}
    htf_interval, htf_period = htf_map.get(primary_interval, ("1d", "3mo"))
    df = fetch_ohlcv(ticker, htf_interval, htf_period)
    if df.empty or len(df) < 20:
        return "neutral"
    close = df["close"].squeeze()
    ma50  = sma(close, min(50, len(close) // 2)).iloc[-1]
    price = close.iloc[-1]
    if price > ma50:   return "bullish"
    if price < ma50:   return "bearish"
    return "neutral"

# ── Signal evaluation ──────────────────────────────────────────────────────────

def evaluate_ticker(ticker: str, cfg: dict) -> list[dict]:
    interval      = cfg["interval"]
    period        = cfg["period"]
    bb_period     = cfg["bb_period"]
    bb_std        = cfg["bb_std"]
    rsi_period    = cfg["rsi_period"]
    rsi_threshold = cfg["rsi_threshold"]
    require_htf   = cfg["require_htf_confirmation"]

    df = fetch_ohlcv(ticker, interval, period)
    min_bars = max(bb_period, rsi_period) + 5
    if df.empty or len(df) < min_bars:
        log.warning(f"{ticker}: insufficient data ({len(df) if not df.empty else 0} bars, need {min_bars})")
        return []

    close  = df["close"].squeeze()
    volume = df["volume"].squeeze()

    rsi                        = calc_rsi(close, rsi_period)
    bb_upper, bb_mid, bb_lower = calc_bollinger(close, bb_period, bb_std)

    p = {
        "price":       float(close.iloc[-1]),
        "rsi":         float(rsi.iloc[-1]),
        "bb_upper":    float(bb_upper.iloc[-1]),
        "bb_mid":      float(bb_mid.iloc[-1]),
        "bb_lower":    float(bb_lower.iloc[-1]),
        "volume":      float(volume.iloc[-1]),
        "vol_avg":     float(volume.rolling(20).mean().iloc[-1]),
        "candle_time": str(df.index[-1]),
    }

    log.info(
        f"{ticker} | Price={p['price']:.2f}  BB_upper={p['bb_upper']:.2f}  "
        f"BB_lower={p['bb_lower']:.2f}  RSI={p['rsi']:.1f}  "
        f"above_BB={'YES' if p['price'] > p['bb_upper'] else 'no'}  "
        f"below_BB={'YES' if p['price'] < p['bb_lower'] else 'no'}"
    )

    signals = []

    # ── LONG: price above upper BB + RSI overbought ────────────────────────
    if p["price"] > p["bb_upper"] and p["rsi"] > rsi_threshold:
        alert_key = (ticker, "LONG", p["candle_time"])
        if alert_key not in _alerted:
            htf = check_higher_tf_trend(ticker, interval) if require_htf else "not checked"
            if (not require_htf) or htf == "bullish":
                _alerted.add(alert_key)
                signals.append({**p, "ticker": ticker, "direction": "LONG", "htf_trend": htf})
        else:
            log.info(f"{ticker} LONG already alerted for candle {p['candle_time']} — skipping")

    # ── SHORT: price below lower BB + RSI oversold ─────────────────────────
    rsi_lower = 100 - rsi_threshold   # e.g. 25 when threshold is 75
    if p["price"] < p["bb_lower"] and p["rsi"] < rsi_lower:
        alert_key = (ticker, "SHORT", p["candle_time"])
        if alert_key not in _alerted:
            htf = check_higher_tf_trend(ticker, interval) if require_htf else "not checked"
            if (not require_htf) or htf == "bearish":
                _alerted.add(alert_key)
                signals.append({**p, "ticker": ticker, "direction": "SHORT", "htf_trend": htf})
        else:
            log.info(f"{ticker} SHORT already alerted for candle {p['candle_time']} — skipping")

    return signals

# ── Pre-market scan ────────────────────────────────────────────────────────────

def run_premarket_scan(tickers: list[str], cfg: dict):
    log.info("Running pre-market gap scan...")
    found = []
    for ticker in tickers:
        gap = calc_premarket_gap(ticker, cfg["gap_pct_threshold"])
        if gap:
            found.append((ticker, gap))
    if not found:
        log.info("No pre-market gaps above threshold.")
        return
    lines = [f"🌅 <b>Pre-Market Gap Alert</b> — {datetime.now(ET).strftime('%Y-%m-%d')}\n"]
    for ticker, g in found:
        arrow = "⬆️" if g["direction"] == "UP" else "⬇️"
        lines.append(
            f"{arrow} <b>{ticker}</b> Gap {g['direction']}: {g['gap_pct']:+.2f}%\n"
            f"   Prev close: ${g['prev_close']:.2f} → Pre-mkt: ${g['pre_mkt']:.2f}"
        )
    send_telegram("\n".join(lines))

# ── Format alert message ───────────────────────────────────────────────────────

def format_signal_message(sig: dict, cfg: dict) -> str:
    direction = sig["direction"]
    emoji     = "🟢" if direction == "LONG" else "🔴"
    bb_key    = "bb_upper" if direction == "LONG" else "bb_lower"
    vol_ratio = sig["volume"] / sig["vol_avg"] if sig["vol_avg"] else 1
    htf_line  = f"\n📊 HTF Trend: {sig['htf_trend'].upper()}" if cfg["require_htf_confirmation"] else ""

    return (
        f"{emoji} <b>{sig['ticker']} — {direction} Signal</b>\n"
        f"⏰ Candle: {sig['candle_time']}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Price:      ${sig['price']:.2f}\n"
        f"📈 BB {'Upper' if direction == 'LONG' else 'Lower'}:  ${sig[bb_key]:.2f}\n"
        f"⚡ RSI ({cfg['rsi_period']}):   {sig['rsi']:.1f}\n"
        f"📊 Volume:     {vol_ratio:.1f}x avg"
        f"{htf_line}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"⚙️ {cfg['interval']} · BB({cfg['bb_period']},{cfg['bb_std']}) · RSI>{cfg['rsi_threshold']}"
    )

# ── Main scan ──────────────────────────────────────────────────────────────────

def run_signal_scan(tickers: list[str], cfg: dict):
    log.info(f"Scanning: {', '.join(tickers)}")
    total = 0
    for ticker in tickers:
        try:
            sigs = evaluate_ticker(ticker, cfg)
            for sig in sigs:
                send_telegram(format_signal_message(sig, cfg))
                total += 1
        except Exception as e:
            log.error(f"Error evaluating {ticker}: {e}")
    log.info(f"Scan complete — {total} signal(s) fired.")

def wait_for_next_candle_close(interval: str) -> int:
    interval_minutes = {
        "1m": 1, "2m": 2, "5m": 5, "15m": 15,
        "30m": 30, "60m": 60, "1h": 60, "90m": 90, "1d": 1440,
    }
    minutes = interval_minutes.get(interval, 30)
    now     = datetime.now(ET)
    current_minute   = now.hour * 60 + now.minute
    next_close_minute = (current_minute // minutes + 1) * minutes
    next_close = now.replace(
        hour=next_close_minute // 60,
        minute=next_close_minute % 60,
        second=30, microsecond=0
    )
    if next_close <= now:
        next_close += timedelta(minutes=minutes)
    sleep_secs = (next_close - now).total_seconds()
    log.info(f"Next {interval} candle close at {next_close.strftime('%H:%M:%S ET')} — sleeping {sleep_secs:.0f}s")
    return int(sleep_secs)

def is_market_hours(now: datetime) -> bool:
    """True if within regular US market hours Mon–Fri 9:30–16:00 ET."""
    if now.weekday() >= 5:
        return False
    open_time  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_time = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return open_time <= now <= close_time

def is_premarket(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    start = now.replace(hour=8, minute=45, second=0, microsecond=0)
    end   = now.replace(hour=9, minute=30, second=0, microsecond=0)
    return start <= now < end

# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    cfg     = CONFIG
    tickers = cfg["tickers"]

    log.info("=" * 55)
    log.info("  BB + RSI Signal Scanner  [24/7]")
    log.info(f"  Tickers:  {', '.join(tickers)}")
    log.info(f"  Interval: {cfg['interval']}  |  RSI > {cfg['rsi_threshold']}")
    log.info(f"  BB({cfg['bb_period']},{cfg['bb_std']})  |  Gap > {cfg['gap_pct_threshold']}%")
    log.info("=" * 55)

    send_telegram(
        f"🚀 <b>Scanner started [24/7]</b>\n"
        f"Watching: {', '.join(tickers)}\n"
        f"Interval: {cfg['interval']} · RSI > {cfg['rsi_threshold']} · BB({cfg['bb_period']},{cfg['bb_std']})\n"
        f"Conditions: Price outside BB + RSI threshold"
    )

    premarket_alerted_today = None

    while True:
        now   = datetime.now(ET)
        today = now.date()

        # Pre-market gap scan once per trading day
        if is_premarket(now) and premarket_alerted_today != today:
            run_premarket_scan(tickers, cfg)
            premarket_alerted_today = today
            time.sleep(60)
            continue

        # Signal scan only during market hours
        if is_market_hours(now):
            run_signal_scan(tickers, cfg)
            sleep_secs = wait_for_next_candle_close(cfg["interval"])
        else:
            log.info(f"Market closed ({now.strftime('%H:%M ET')}). Sleeping 5 minutes.")
            sleep_secs = 300

        time.sleep(sleep_secs)

        # Signal scan — runs 24/7
        #run_signal_scan(tickers, cfg)
        
        #sleep_secs = wait_for_next_candle_close(cfg["interval"])
        #time.sleep(sleep_secs)

if __name__ == "__main__":
    main()
