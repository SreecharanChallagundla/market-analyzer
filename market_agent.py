#!/usr/bin/env python3
"""
Market Signal Agent (NSE/BSE) - runs every 2 hours during market hours and
publishes a BUY / AVOID list with entry, stop-loss, target and sell rules.

Setup:   pip install yfinance pandas numpy requests
Run:     python market_agent.py            # loops forever (Mon-Fri, 09:30-15:30 IST)
         python market_agent.py --once     # single run now
         python market_agent.py --demo --once   # synthetic data, no internet
Optional Telegram alerts: set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID env vars.

Educational tool. Rule-based signals are NOT investment advice.
"""
import os, sys, time, json, argparse, datetime as dt
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore", module="yfinance")      # hide harmless pandas deprecation noise from yfinance
warnings.filterwarnings("ignore", message=".*inplace.*")

IST = ZoneInfo("Asia/Kolkata")
INTERVAL_HOURS = 2
WATCHLIST = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "SBIN", "BHARTIARTL",
             "ITC", "LT", "HINDUNILVR", "SUNPHARMA", "MARUTI", "M&M", "AXISBANK", "WIPRO",
             "NTPC", "ONGC", "TITAN", "ASIANPAINT", "ADANIPORTS", "KOTAKBANK", "BAJFINANCE",
             "TATASTEEL", "POWERGRID", "COALINDIA", "HCLTECH", "ULTRACEMCO", "NESTLEIND"]
CAPITAL, RISK_PCT = 100000, 1.0       # rupees; % of capital risked per trade (position sizing)
MIN_BUY_SCORE, TOP_N = 4, 5

# Settings can be overridden from config.json (edit it directly on GitHub)
try:
    _cfg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")))
except Exception:
    _cfg = {}
WATCHLIST = [s.strip().upper().replace(".NS", "") for s in _cfg.get("watchlist", WATCHLIST) if str(s).strip()]
CAPITAL = _cfg.get("capital", CAPITAL); RISK_PCT = _cfg.get("risk_pct", RISK_PCT)
MIN_BUY_SCORE = _cfg.get("min_buy_score", MIN_BUY_SCORE); TOP_N = _cfg.get("top_n", TOP_N)
OUT_DIR = "reports"
DOCS_DIR = os.path.dirname(os.path.abspath(__file__))   # data files go in the repo root, next to index.html
INDEX_TICKERS = {"NIFTY 50": "^NSEI", "NIFTY BANK": "^NSEBANK", "NIFTY IT": "^CNXIT",
                 "NIFTY NEXT 50": "^NSMIDCP", "NIFTY MIDCAP 100": "NIFTY_MIDCAP_100.NS", "NIFTY FMCG": "^CNXFMCG",
                 "SENSEX": "^BSESN", "BSE 500": "BSE-500.BO", "BSE MIDCAP": "BSE-MIDCAP.BO",
                 "BSE SMALLCAP": "BSE-SMLCAP.BO"}   # any that fail are skipped; dashboard shows simulated values for them

# ---------- data ----------
def load(symbols, demo=False):
    if demo:
        out = {}
        for s in symbols:
            r = np.random.default_rng(abs(hash(s)) % 2**32)
            c = 1000 * np.exp(np.cumsum(r.normal(0.0004, 0.014, 300)))
            out[s] = pd.DataFrame({"Open": np.r_[c[0], c[:-1]], "Close": c, "High": c * 1.01, "Low": c * 0.99,
                                   "Volume": r.integers(1e6, 5e6, 300)})
        return out
    import yfinance as yf
    tick = [s + ".NS" for s in symbols]
    raw = yf.download(tick, period="1y", interval="1d", group_by="ticker",
                      auto_adjust=True, progress=False, threads=True)
    out = {}
    for s, t in zip(symbols, tick):
        try:
            d = raw[t].dropna()
            if len(d) > 210: out[s] = d
        except Exception:
            pass
    return out

# ---------- indicators ----------
def analyse(df):
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    sma20, sma50, sma200 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    d = c.diff(); g = d.clip(lower=0).rolling(14).mean(); ls = (-d.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + g / ls.replace(0, np.nan))
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean(); sig = macd.ewm(span=9).mean(); hist = macd - sig
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    p = c.iloc[-1]
    score, why = 0, []
    def add(cond, pts, txt):
        nonlocal score
        if cond: score += pts; why.append(("+" if pts > 0 else "-") + txt)
    add(p > sma200.iloc[-1], 1, "above 200-DMA"); add(p < sma200.iloc[-1], -1, "below 200-DMA")
    add(sma20.iloc[-1] > sma50.iloc[-1], 1, "20>50 DMA"); add(sma20.iloc[-1] < sma50.iloc[-1], -1, "20<50 DMA")
    add(45 <= rsi.iloc[-1] <= 68, 1, f"RSI {rsi.iloc[-1]:.0f} healthy")
    add(rsi.iloc[-1] > 75, -1, f"RSI {rsi.iloc[-1]:.0f} overbought"); add(rsi.iloc[-1] < 35, -1, f"RSI {rsi.iloc[-1]:.0f} weak")
    add(hist.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2], 1, "MACD rising"); add(hist.iloc[-1] < 0, -1, "MACD negative")
    add(p >= c.iloc[-21:-1].max(), 1, "20-day breakout")
    add(v.iloc[-1] > 1.3 * v.rolling(20).mean().iloc[-1], 1, "volume surge")
    a = atr.iloc[-1]
    stop, t1 = p - 1.5 * a, p + 3 * a                       # 1:2 risk-reward
    qty = int((CAPITAL * RISK_PCT / 100) / max(p - stop, 0.01))
    return dict(price=p, score=score, why=", ".join(why), stop=stop, target=t1, qty=qty,
                trail=f"exit if close < 20-DMA ({sma20.iloc[-1]:.2f}) or RSI > 75 or MACD turns negative",
                ret1m=(p / c.iloc[-22] - 1) * 100)

# ---------- export for the dashboard ----------
def load_indices():
    import yfinance as yf
    out = {}
    for name, t in INDEX_TICKERS.items():
        try:
            hdf = yf.Ticker(t).history(period="1y", interval="1d").dropna(subset=["Close"]); c = hdf["Close"]
            if len(c) > 60:
                out[name] = [round(float(x), 2) for x in c.tail(250)]; IDX_OHLC[name] = _ohlc(hdf, 120)
        except Exception:
            pass
    return out

FUND = {"mcap": "marketCap", "pe": "trailingPE", "fpe": "forwardPE", "pb": "priceToBook", "eps": "trailingEps",
        "roe": "returnOnEquity", "margin": "profitMargins", "de": "debtToEquity", "target": "targetMeanPrice",
        "rec": "recommendationKey", "n": "numberOfAnalystOpinions", "sector": "sector", "industry": "industry"}

def _news(t):
    out = []
    try:
        for n in (t.news or [])[:8]:
            c = n.get("content", n)
            url = (c.get("canonicalUrl") or c.get("clickThroughUrl") or {}).get("url") or n.get("link") or ""
            title = c.get("title") or n.get("title")
            pub = (c.get("provider") or {}).get("displayName") or n.get("publisher") or ""
            when = c.get("pubDate") or n.get("providerPublishTime") or ""
            if isinstance(when, (int, float)): when = dt.datetime.fromtimestamp(when, IST).date().isoformat()
            if title and str(url).startswith("http"):
                out.append({"t": title, "u": url, "p": pub, "d": str(when)[:10]})
    except Exception:
        pass
    return out[:5]

def build_reports(rows, demo):
    """Per-stock report: agent signal + (live runs only) fundamentals, analyst view and news headlines."""
    out, t0 = {}, time.time()
    for s, r in rows.items():
        rep = {"signal": {k: (round(float(v), 2) if isinstance(v, (int, float, np.number)) else v) for k, v in r.items()}}
        if not demo and time.time() - t0 < 300:            # time budget so the job never hangs
            try:
                import yfinance as yf
                t = yf.Ticker(s + ".NS")
                try:
                    info = t.info or {}
                    rep["fund"] = {k: info[v] for k, v in FUND.items() if info.get(v) is not None}
                except Exception:
                    pass
                rep["news"] = _news(t)
            except Exception:
                pass
        out[s] = rep
    return out

def export_json(data, buys, avoid, demo, rows=None):
    clean = lambda r: {k: (round(float(v), 2) if isinstance(v, (int, float, np.floating)) else v) for k, v in r.items()}
    out = {"updated": dt.datetime.now(IST).isoformat(timespec="minutes"), "demo": demo,
           "stocks": {s: [round(float(x), 2) for x in df["Close"].tail(250)] for s, df in data.items()},
           "indices": {} if demo else load_indices(),
           "ohlc": {**{s: _ohlc(df, 120) for s, df in data.items()}, **IDX_OHLC},
           "reports": build_reports(rows or {}, demo),
           "buys": [{"s": s, **clean(r)} for s, r in buys], "avoid": [{"s": s, **clean(r)} for s, r in avoid]}
    os.makedirs(DOCS_DIR, exist_ok=True)
    with open(os.path.join(DOCS_DIR, "data.json"), "w") as f: json.dump(out, f, separators=(",", ":"), default=str)

# ---------- OHLC, live (intraday) and PCR ----------
IDX_OHLC = {}

def _ohlc(df, n):
    return [[round(float(r.Open), 2), round(float(r.High), 2), round(float(r.Low), 2), round(float(r.Close), 2)]
            for r in df.tail(n).itertuples()]

def run_live():
    """Fast job: 15-minute bars for the last 2 days + quotes, written to docs/live.json."""
    import yfinance as yf
    tick = {**{s: s + ".NS" for s in WATCHLIST}, **INDEX_TICKERS}
    raw = yf.download(list(tick.values()), period="2d", interval="15m", group_by="ticker",
                      auto_adjust=True, progress=False, threads=True)
    now = dt.datetime.now(IST); intr, quotes = {}, {}
    for k, t in tick.items():
        try:
            d = raw[t].dropna(subset=["Close"])
            if d.empty: continue
            intr[k] = [[int(i.timestamp()), round(float(r.Open), 2), round(float(r.High), 2),
                        round(float(r.Low), 2), round(float(r.Close), 2)] for i, r in zip(d.index, d.itertuples())]
            days = list(d.index.tz_convert(IST).date) if d.index.tz is not None else list(d.index.date)
            prev = [c for c, dd in zip(d["Close"], days) if dd != days[-1]]
            p = float(d["Close"].iloc[-1])
            quotes[k] = {"p": round(p, 2), "c": round((p / float(prev[-1]) - 1) * 100, 2) if prev else 0.0}
        except Exception:
            pass
    os.makedirs(DOCS_DIR, exist_ok=True)
    with open(os.path.join(DOCS_DIR, "live.json"), "w") as f:
        json.dump({"updated": now.isoformat(timespec="minutes"), "open": market_open(now),
                   "intraday": intr, "quotes": quotes}, f, separators=(",", ":"))
    print(f"live.json written: {len(intr)} symbols")

def fetch_pcr(sym):
    """Put-Call Ratio (open interest) from NSE's option chain. NSE may block cloud IPs; callers handle failure."""
    import requests
    h = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
         "Accept": "application/json,text/plain,*/*", "Accept-Language": "en-US,en;q=0.9",
         "Referer": "https://www.nseindia.com/option-chain"}
    s = requests.Session(); s.get("https://www.nseindia.com/option-chain", headers=h, timeout=15)
    rec = s.get(f"https://www.nseindia.com/api/option-chain-indices?symbol={sym}", headers=h, timeout=15).json()["records"]
    tot = lambda rows, k: sum((x.get(k) or {}).get("openInterest", 0) for x in rows)
    exp = rec["expiryDates"][0]; near = [x for x in rec["data"] if x.get("expiryDate") == exp]
    return {"all": round(tot(rec["data"], "PE") / max(tot(rec["data"], "CE"), 1), 3),
            "near": round(tot(near, "PE") / max(tot(near, "CE"), 1), 3), "exp": exp}

def update_pcr():
    path = os.path.join(DOCS_DIR, "pcr.json")
    try: hist = json.load(open(path))
    except Exception: hist = {}
    today, ok = dt.datetime.now(IST).date().isoformat(), False
    for sym in ("NIFTY", "BANKNIFTY"):
        try:
            v = fetch_pcr(sym); v["d"] = today
            hist[sym] = ([x for x in hist.get(sym, []) if x["d"] != today] + [v])[-250:]; ok = True
        except Exception as e:
            print(f"PCR {sym} unavailable: {e}")
    if ok:
        os.makedirs(DOCS_DIR, exist_ok=True)
        with open(path, "w") as f: json.dump(hist, f, separators=(",", ":"))

# ---------- report ----------
def build_report(demo=False):
    data = load(WATCHLIST, demo)
    rows = {s: analyse(df) for s, df in data.items()}
    ranked = sorted(rows.items(), key=lambda x: (-x[1]["score"], -x[1]["ret1m"]))
    buys = [(s, r) for s, r in ranked if r["score"] >= MIN_BUY_SCORE][:TOP_N]
    avoid = [(s, r) for s, r in ranked[::-1] if r["score"] <= -2][:TOP_N]
    export_json(data, buys, avoid, demo, rows)
    if not demo: update_pcr()
    now = dt.datetime.now(IST).strftime("%d %b %Y, %I:%M %p IST")
    L = [f"MARKET SIGNALS - {now}" + ("  [DEMO DATA]" if demo else ""), ""]
    L.append("BUY WATCHLIST (ranked)")
    if not buys: L.append("  No stock meets the entry criteria right now. Staying in cash is a valid call.")
    for i, (s, r) in enumerate(buys, 1):
        L += [f"{i}. {s}  score {r['score']}  |  Buy ~Rs {r['price']:.2f}",
              f"   Stop-loss Rs {r['stop']:.2f} | Target Rs {r['target']:.2f} | Qty {r['qty']} (risk {RISK_PCT}% of Rs {CAPITAL:,})",
              f"   WHEN TO SELL: target hit, stop hit, or {r['trail']}",
              f"   Why: {r['why']}"]
    L += ["", "AVOID / EXIT LIST"]
    for s, r in avoid: L.append(f"- {s}  score {r['score']}  ({r['why']})")
    if not avoid: L.append("- none")
    L += ["", "Rule-based signals for education only; not investment advice. Do your own research."]
    return "\n".join(L)

def publish(text):
    print("\n" + text + "\n")
    os.makedirs(OUT_DIR, exist_ok=True)
    fn = os.path.join(OUT_DIR, dt.datetime.now(IST).strftime("%Y-%m-%d_%H%M") + ".txt")
    open(fn, "w", encoding="utf-8").write(text)
    tok, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if tok and chat:
        import requests
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      data={"chat_id": chat, "text": text[:4000]}, timeout=15)

def market_open(n):
    return n.weekday() < 5 and dt.time(9, 15) <= n.time() <= dt.time(15, 45)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--once", action="store_true"); ap.add_argument("--demo", action="store_true"); ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    if a.live: return run_live()
    if a.once: return publish(build_report(a.demo))
    print("Agent started. Reports every", INTERVAL_HOURS, "hours during NSE/BSE hours. Ctrl+C to stop.")
    last = None
    while True:
        n = dt.datetime.now(IST)
        slot = (n.date(), (n.hour - 9) // INTERVAL_HOURS)
        if market_open(n) and n.hour >= 9 and slot != last and n.time() >= dt.time(9, 30):
            try: publish(build_report(a.demo)); last = slot
            except Exception as e: print("Run failed:", e)
        time.sleep(60)

if __name__ == "__main__":
    main()
