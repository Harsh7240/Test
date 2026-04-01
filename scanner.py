"""
Zero Lag MACD + Bollinger Band + RSI + Pre-market Gap Scanner
Sends Telegram alerts on confirmed candle close signals.
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

# ── Telegram ──────────────────────────────────────────────────────────────────

def send_telegram(message: str):
    token = CONFIG["TELEGRAM_BOT_TOKEN"]
    chat_id = CONFIG["TELEGRAM_CHAT_ID"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=10)
        r.raise_for_status()
        log.info(f"Telegram sent: {message[:60]}...")
    except Exception as e:
        log.error(f"Telegram error: {e}")

# ── Data fetching ──────────────────────────────────────────────────────────────

def fetch_ohlcv(ticker: str, interval: str, period: str) -> pd.DataFrame:
    """Fetch OHLCV data via yfinance. Returns DataFrame with OHLCV columns."""
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=True)
        if df.empty:
            log.warning(f"No data for {ticker}")
            return pd.DataFrame()
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                 "Close": "close", "Volume": "volume"})
        # Drop the last (potentially incomplete) candle
        df = df.iloc[:-1]
        return df
    except Exception as e:
        log.error(f"Fetch error {ticker}: {e}")
        return pd.DataFrame()

# ── Indicators ─────────────────────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()

def calc_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_bollinger(closes: pd.Series, period: int = 20, std_mult: float = 2.0):
    mid = closes.rolling(period).mean()
    std = closes.rolling(period).std(ddof=0)
    return mid + std_mult * std, mid, mid - std_mult * std  # upper, mid, lower

def calc_zero_lag_macd(
    closes: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
    macd_ema_period: int = 9,
    use_ema: bool = True,
    use_old_algo: bool = False,
) -> pd.DataFrame:
    """
    Port of the Pine Script Zero Lag MACD Enhanced v1.2 by Albert Callisto.
    Returns DataFrame with: macd, signal, hist, macd_ema, crossover (bull), crossunder (bear)
    """
    ma_fn = ema if use_ema else sma

    # Fast zero-lag line
    ma1 = ma_fn(closes, fast)
    ma2 = ma_fn(ma1, fast)
    zlema_fast = 2 * ma1 - ma2

    # Slow zero-lag line
    mas1 = ma_fn(closes, slow)
    mas2 = ma_fn(mas1, slow)
    zlema_slow = 2 * mas1 - mas2

    macd_line = zlema_fast - zlema_slow

    # Signal line
    if use_old_algo:
        sig = sma(macd_line, signal_period)
    else:
        emasig1 = ema(macd_line, signal_period)
        emasig2 = ema(emasig1, signal_period)
        sig = 2 * emasig1 - emasig2

    hist = macd_line - sig
    macd_ema_line = ema(macd_line, macd_ema_period)

    # Crossover detection (current bar crosses, previous bar did not)
    cross_bull = (macd_line > sig) & (macd_line.shift(1) <= sig.shift(1))
    cross_bear = (macd_line < sig) & (macd_line.shift(1) >= sig.shift(1))

    return pd.DataFrame({
        "macd": macd_line,
        "signal": sig,
        "hist": hist,
        "macd_ema": macd_ema_line,
        "cross_bull": cross_bull,
        "cross_bear": cross_bear,
    }, index=closes.index)

def calc_premarket_gap(ticker: str, gap_pct_threshold: float) -> dict | None:
    """
    Returns gap info if pre-market gap exceeds threshold.
    Uses previous close vs current pre-market price.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        prev_close = info.previous_close
        pre_mkt = info.pre_market_price
        if prev_close and pre_mkt:
            gap_pct = ((pre_mkt - prev_close) / prev_close) * 100
            if abs(gap_pct) >= gap_pct_threshold:
                direction = "UP" if gap_pct > 0 else "DOWN"
                return {"direction": direction, "gap_pct": gap_pct, "prev_close": prev_close, "pre_mkt": pre_mkt}
    except Exception as e:
        log.warning(f"Pre-market gap check failed for {ticker}: {e}")
    return None

# ── Multi-timeframe confirmation ───────────────────────────────────────────────

def check_higher_tf_trend(ticker: str, primary_interval: str) -> str:
    """
    Returns 'bullish', 'bearish', or 'neutral' based on a higher timeframe.
    Primary 30m → confirm on 1d. Primary 1h → confirm on 1wk.
    """
    htf_map = {"30m": ("1d", "3mo"), "1h": ("1d", "3mo"), "15m": ("1h", "1mo")}
    htf_interval, htf_period = htf_map.get(primary_interval, ("1d", "3mo"))
    df = fetch_ohlcv(ticker, htf_interval, htf_period)
    if df.empty or len(df) < 20:
        return "neutral"
    close = df["close"].squeeze()
    ma50 = sma(close, min(50, len(close) // 2)).iloc[-1]
    price = close.iloc[-1]
    if price > ma50:
        return "bullish"
    elif price < ma50:
        return "bearish"
    return "neutral"

# ── Signal evaluation ──────────────────────────────────────────────────────────

def evaluate_ticker(ticker: str, cfg: dict) -> list[dict]:
    """
    Returns list of signal dicts (may be empty) for a given ticker.
    """
    interval = cfg["interval"]
    period = cfg["period"]
    bb_period = cfg["bb_period"]
    bb_std = cfg["bb_std"]
    rsi_period = cfg["rsi_period"]
    rsi_threshold = cfg["rsi_threshold"]
    macd_cfg = cfg["macd"]
    gap_threshold = cfg["gap_pct_threshold"]
    require_htf = cfg["require_htf_confirmation"]

    df = fetch_ohlcv(ticker, interval, period)
    if df.empty or len(df) < max(bb_period, rsi_period, macd_cfg["slow"] + macd_cfg["signal_period"]) + 5:
        log.warning(f"{ticker}: insufficient data")
        return []

    close = df["close"].squeeze()
    volume = df["volume"].squeeze()

    # ── Core indicators
    rsi = calc_rsi(close, rsi_period)
    bb_upper, bb_mid, bb_lower = calc_bollinger(close, bb_period, bb_std)
    macd_df = calc_zero_lag_macd(
        close,
        fast=macd_cfg["fast"],
        slow=macd_cfg["slow"],
        signal_period=macd_cfg["signal_period"],
        macd_ema_period=macd_cfg["macd_ema_period"],
        use_ema=macd_cfg["use_ema"],
        use_old_algo=macd_cfg["use_old_algo"],
    )

    # Latest values
    latest = {
        "price":      float(close.iloc[-1]),
        "rsi":        float(rsi.iloc[-1]),
        "bb_upper":   float(bb_upper.iloc[-1]),
        "bb_mid":     float(bb_mid.iloc[-1]),
        "bb_lower":   float(bb_lower.iloc[-1]),
        "macd":       float(macd_df["macd"].iloc[-1]),
        "macd_sig":   float(macd_df["signal"].iloc[-1]),
        "macd_hist":  float(macd_df["hist"].iloc[-1]),
        "macd_ema":   float(macd_df["macd_ema"].iloc[-1]),
        "cross_bull": bool(macd_df["cross_bull"].iloc[-1]),
        "cross_bear": bool(macd_df["cross_bear"].iloc[-1]),
        "volume":     float(volume.iloc[-1]),
        "vol_avg":    float(volume.rolling(20).mean().iloc[-1]),
        "candle_time": str(df.index[-1]),
    }

    signals = []

    # ── LONG signal: price above upper BB + RSI overbought + MACD bull cross
    bb_breakout_long = latest["price"] > latest["bb_upper"]
    rsi_overbought   = latest["rsi"] > rsi_threshold
    macd_bull        = latest["cross_bull"]

    if bb_breakout_long and rsi_overbought and macd_bull:
        htf = check_higher_tf_trend(ticker, interval) if require_htf else "not checked"
        htf_ok = (htf == "bullish") or (not require_htf)
        if htf_ok or not require_htf:
            signals.append({
                "ticker": ticker,
                "direction": "LONG",
                "price": latest["price"],
                "rsi": latest["rsi"],
                "bb_upper": latest["bb_upper"],
                "macd": latest["macd"],
                "macd_sig": latest["macd_sig"],
                "htf_trend": htf,
                "volume": latest["volume"],
                "vol_avg": latest["vol_avg"],
                "candle_time": latest["candle_time"],
            })

    # ── SHORT signal: price below lower BB + RSI oversold + MACD bear cross
    rsi_lower = 100 - rsi_threshold   # e.g. 25 if threshold is 75
    bb_breakout_short = latest["price"] < latest["bb_lower"]
    rsi_oversold      = latest["rsi"] < rsi_lower
    macd_bear         = latest["cross_bear"]

    if bb_breakout_short and rsi_oversold and macd_bear:
        htf = check_higher_tf_trend(ticker, interval) if require_htf else "not checked"
        htf_ok = (htf == "bearish") or (not require_htf)
        if htf_ok or not require_htf:
            signals.append({
                "ticker": ticker,
                "direction": "SHORT",
                "price": latest["price"],
                "rsi": latest["rsi"],
                "bb_lower": latest["bb_lower"],
                "macd": latest["macd"],
                "macd_sig": latest["macd_sig"],
                "htf_trend": htf,
                "volume": latest["volume"],
                "vol_avg": latest["vol_avg"],
                "candle_time": latest["candle_time"],
            })

    return signals

# ── Pre-market gap scan (runs at ~9:00 AM ET) ──────────────────────────────────

def run_premarket_scan(tickers: list[str], cfg: dict):
    log.info("Running pre-market gap scan...")
    threshold = cfg["gap_pct_threshold"]
    found = []
    for ticker in tickers:
        gap = calc_premarket_gap(ticker, threshold)
        if gap:
            found.append((ticker, gap))

    if not found:
        log.info("No pre-market gaps found above threshold.")
        return

    lines = [f"🌅 <b>Pre-Market Gap Alert</b> — {datetime.now(ET).strftime('%Y-%m-%d')}\n"]
    for ticker, g in found:
        arrow = "⬆️" if g["direction"] == "UP" else "⬇️"
        lines.append(
            f"{arrow} <b>{ticker}</b> Gap {g['direction']}: {g['gap_pct']:+.2f}%\n"
            f"   Prev close: ${g['prev_close']:.2f} → Pre-mkt: ${g['pre_mkt']:.2f}"
        )
    send_telegram("\n".join(lines))

# ── Format and send signal alert ──────────────────────────────────────────────

def format_signal_message(sig: dict, cfg: dict) -> str:
    direction = sig["direction"]
    emoji = "🟢" if direction == "LONG" else "🔴"
    vol_ratio = sig["volume"] / sig["vol_avg"] if sig["vol_avg"] else 1
    bb_key = "bb_upper" if direction == "LONG" else "bb_lower"

    htf_line = ""
    if cfg["require_htf_confirmation"]:
        htf_line = f"\n📊 HTF Trend: {sig['htf_trend'].upper()}"

    return (
        f"{emoji} <b>{sig['ticker']} — {direction} Signal</b>\n"
        f"⏰ Candle close: {sig['candle_time']}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Price:       ${sig['price']:.2f}\n"
        f"📈 BB {'Upper' if direction=='LONG' else 'Lower'}:   ${sig[bb_key]:.2f}\n"
        f"⚡ RSI ({cfg['rsi_period']}):    {sig['rsi']:.1f}\n"
        f"🔀 MACD:        {sig['macd']:.4f}\n"
        f"🔀 Signal:      {sig['macd_sig']:.4f}\n"
        f"📊 Volume:      {vol_ratio:.1f}x avg"
        f"{htf_line}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"⚙️ {cfg['interval']} candle · BB({cfg['bb_period']},{cfg['bb_std']}) · RSI>{cfg['rsi_threshold']}"
    )

# ── Main scan loop ─────────────────────────────────────────────────────────────

def run_signal_scan(tickers: list[str], cfg: dict):
    log.info(f"Running signal scan for: {', '.join(tickers)}")
    total_signals = 0
    for ticker in tickers:
        try:
            sigs = evaluate_ticker(ticker, cfg)
            for sig in sigs:
                msg = format_signal_message(sig, cfg)
                send_telegram(msg)
                total_signals += 1
        except Exception as e:
            log.error(f"Error evaluating {ticker}: {e}")
    if total_signals == 0:
        log.info("Scan complete — no signals triggered.")
    else:
        log.info(f"Scan complete — {total_signals} signal(s) sent.")

def wait_for_next_candle_close(interval: str) -> int:
    """
    Returns seconds to sleep until the next candle close for the given interval.
    Adds a 30-second buffer after close to allow yfinance data to propagate.
    """
    interval_minutes = {
        "1m": 1, "2m": 2, "5m": 5, "15m": 15,
        "30m": 30, "60m": 60, "1h": 60, "90m": 90,
        "1d": 1440,
    }
    minutes = interval_minutes.get(interval, 30)
    now = datetime.now(ET)
    current_minute = now.hour * 60 + now.minute
    candles_elapsed = current_minute // minutes
    next_close_minute = (candles_elapsed + 1) * minutes
    next_close = now.replace(hour=next_close_minute // 60, minute=next_close_minute % 60, second=30, microsecond=0)
    if next_close <= now:
        next_close += timedelta(minutes=minutes)
    sleep_secs = (next_close - now).total_seconds()
    log.info(f"Next {interval} candle close at {next_close.strftime('%H:%M:%S ET')} (sleeping {sleep_secs:.0f}s)")
    return int(sleep_secs)

def is_market_hours(now: datetime) -> bool:
    """True if within regular US market hours Mon–Fri 9:30–16:00 ET."""
    if now.weekday() >= 5:
        return False
    open_time  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_time = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return open_time <= now <= close_time

def is_premarket(now: datetime) -> bool:
    """True if 8:45–9:30 AM ET on a weekday."""
    if now.weekday() >= 5:
        return False
    start = now.replace(hour=8, minute=45, second=0, microsecond=0)
    end   = now.replace(hour=9, minute=30, second=0, microsecond=0)
    return start <= now < end

def main():
    cfg     = CONFIG
    tickers = cfg["tickers"]

    log.info("=" * 55)
    log.info("  Zero Lag MACD + BB + RSI Signal Scanner")
    log.info(f"  Tickers:  {', '.join(tickers)}")
    log.info(f"  Interval: {cfg['interval']}  |  RSI > {cfg['rsi_threshold']}")
    log.info(f"  BB({cfg['bb_period']},{cfg['bb_std']})  |  Gap > {cfg['gap_pct_threshold']}%")
    log.info("=" * 55)

    send_telegram(
        f"🚀 <b>Scanner started</b>\n"
        f"Watching: {', '.join(tickers)}\n"
        f"Interval: {cfg['interval']} · RSI > {cfg['rsi_threshold']} · BB({cfg['bb_period']},{cfg['bb_std']})"
    )

    premarket_alerted_today = None

    while True:
        now = datetime.now(ET)
        today = now.date()

        # Pre-market gap scan once per day
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

if __name__ == "__main__":
    main()
