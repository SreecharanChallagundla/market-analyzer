# Bharat Market Analyzer

- `market_agent.py` - scores 28 NSE stocks, writes `docs/data.json` (and Telegram alerts if configured).
- `docs/index.html` - dashboard (GitHub Pages). Reads `data.json`; falls back to simulated data if missing.
- `.github/workflows/agent.yml` - runs the agent at 9:30, 11:30, 13:30, 15:30 IST on weekdays.
