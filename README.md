# Zero Lag MACD + BB + RSI Signal Scanner

Free, real-time (on candle close) stock signal alerts to Telegram.
Runs 24/7 on Railway or Render at zero cost.

---

## What it does

Scans your watchlist every 30 minutes (or whatever interval you set) and sends a
Telegram message when ALL of these conditions are true on a **closed candle**:

**LONG signal:**
- Price closes **above upper Bollinger Band** (default BB 20, 2σ)
- RSI(14) **above 75**
- Zero Lag MACD (Enhanced v1.2 — Albert Callisto) **bullish crossover**
- (Optional) Daily trend **bullish** (price above 50-day SMA)

**SHORT signal:**
- Price closes **below lower Bollinger Band**
- RSI(14) **below 25**
- Zero Lag MACD **bearish crossover**
- (Optional) Daily trend **bearish**

**Pre-market gap alert (8:45–9:30 AM ET):**
- Any ticker gapping up or down ≥ your threshold % vs previous close

---

## Setup (takes about 10 minutes)

### Step 1 — Create your Telegram bot

1. Open Telegram and search for **@BotFather**
2. Send `/newbot` and follow the prompts
3. Copy the **bot token** (looks like `123456789:ABCdef...`)
4. Search for your new bot in Telegram and send it any message
5. Visit this URL in your browser (replace `TOKEN`):
   ```
   https://api.telegram.org/botTOKEN/getUpdates
   ```
6. Find `"chat":{"id":XXXXXXXXX}` — that number is your **Chat ID**

### Step 2 — Configure your scanner

Open `config.py` and edit:

```python
"TELEGRAM_BOT_TOKEN": "YOUR_BOT_TOKEN_HERE",
"TELEGRAM_CHAT_ID":   "YOUR_CHAT_ID_HERE",
"tickers": ["AAPL", "NVDA", "TSLA"],   # your watchlist
"interval": "30m",                      # candle size
"rsi_threshold": 75,                    # RSI overbought level
"gap_pct_threshold": 2.0,              # pre-market gap %
"require_htf_confirmation": True,       # daily trend filter
```

All MACD parameters are also in `config.py` under the `"macd"` key.

### Step 3 — Deploy to Railway (free, always-on)

1. Install the Railway CLI:
   ```bash
   npm install -g @railway/cli
   ```
2. Log in:
   ```bash
   railway login
   ```
3. From this folder:
   ```bash
   railway init
   railway up
   ```
4. Set your secrets as environment variables (never put tokens in code):
   ```bash
   railway variables set TELEGRAM_BOT_TOKEN=your_token_here
   railway variables set TELEGRAM_CHAT_ID=your_chat_id_here
   ```
5. Done. Your scanner is live.

**Alternative — Render:**
1. Push this folder to a GitHub repo
2. Go to render.com → New → Background Worker
3. Connect your repo, set runtime to Docker
4. Add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` as environment variables
5. Deploy

### Step 4 — Run locally (optional, for testing)

```bash
pip install -r requirements.txt
export TELEGRAM_BOT_TOKEN=your_token
export TELEGRAM_CHAT_ID=your_chat_id
python scanner.py
```

---

## Customizing the interval

Change `"interval"` in `config.py`. Supported values:

| Value | Candle size | Notes |
|-------|-------------|-------|
| `"5m"` | 5 minutes | Very active, many signals |
| `"15m"` | 15 minutes | Good for day trading |
| `"30m"` | 30 minutes | **Default — recommended** |
| `"1h"` | 1 hour | Fewer, higher-quality signals |
| `"1d"` | Daily | Swing trading |

---

## Notes on data accuracy

- **yfinance** is free and requires no API key
- Intraday data (< 1d) has a ~15-minute delay from Yahoo Finance
- However, signals only fire on **confirmed closed candles** — so the delay does
  not affect signal accuracy, only the time you receive the alert
- For zero-latency live tick data, you would need a paid feed (Polygon.io, etc.)

---

## Files

| File | Purpose |
|------|---------|
| `scanner.py` | Main script — indicators, signal logic, scheduler |
| `config.py` | All settings — edit this to customize |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container definition for cloud deployment |
| `railway.toml` | Railway deployment config |
