# Bharat Market Analyzer

- `market_agent.py` - scores 28 NSE stocks, writes `docs/data.json` (and Telegram alerts if configured).
- `docs/index.html` - dashboard (GitHub Pages). Reads `data.json`; falls back to simulated data if missing.
- `.github/workflows/agent.yml` - runs the agent at 9:30, 11:30, 13:30, 15:30 IST on weekdays.

## Setup
1. Create a public GitHub repo and upload everything in this folder (keep the `.github` folder).
2. Settings > Pages > Source: `main` branch, folder `/docs`. Your site: `https://<user>.github.io/<repo>/`
3. Actions tab > market-agent > Run workflow (first run). Refresh the site in ~1-2 minutes.
4. Optional Telegram: Settings > Secrets and variables > Actions > add `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`.

To change stocks, edit `WATCHLIST` in `market_agent.py`. Educational only, not investment advice.
