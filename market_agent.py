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

def _arg_market():
    a = sys.argv
    if "--market" in a and a.index("--market") + 1 < len(a): return a[a.index("--market") + 1].upper()
    return os.getenv("MARKET", "IN").upper()
MARKET = "US" if _arg_market() == "US" else "IN"      # python market_agent.py --market US
US = MARKET == "US"
IST = ZoneInfo("America/New_York" if US else "Asia/Kolkata")     # local market time zone (name kept for the code below)
TZN, SUF, CUR = ("ET", "", "$") if US else ("IST", ".NS", "Rs ")
def _clean(o):
    """Make values JSON-safe: NaN / Infinity become null (browsers cannot parse NaN)."""
    if isinstance(o, (float, np.floating)): return float(o) if np.isfinite(o) else None
    if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
    return o
def _jd(o, f): json.dump(_clean(o), f, separators=(",", ":"), default=str, allow_nan=False)
FN = lambda n: ("us_" if US else "") + n               # US data files are prefixed us_
WATCHLIST_US = ["JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "V", "MA", "BRK-B", "UNH", "LLY", "JNJ", "PFE", "MRK", "ABBV", "TMO", "ABT", "NVO", "XOM", "CVX", "COP", "PG", "KO", "PM", "MCD", "NKE", "HD", "LOW", "DIS", "T", "VZ", "CAT", "GE", "RTX", "UNP", "DE", "IBM", "ORCL", "CRM", "TSM", "UBER", "NEE", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "COST", "NFLX", "ADBE", "AMD", "PEP", "CSCO", "TMUS", "INTC", "QCOM", "TXN", "AMGN", "INTU", "AMAT", "ISRG", "BKNG", "HON", "SBUX", "GILD", "ADP", "MU", "LRCX", "PANW", "MDLZ", "REGN", "KLAC", "CRWD", "ADI", "SNPS", "CDNS", "PYPL", "ABNB", "MRVL", "WMT"]
INTERVAL_HOURS = 2
WATCHLIST = ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "SBIN", "BHARTIARTL", "ITC", "LT", "HINDUNILVR", "SUNPHARMA", "MARUTI", "M&M", "AXISBANK", "WIPRO", "NTPC", "ONGC", "TITAN", "ASIANPAINT", "ADANIPORTS", "KOTAKBANK", "BAJFINANCE", "TATASTEEL", "POWERGRID", "COALINDIA", "HCLTECH", "ULTRACEMCO", "NESTLEIND", "PERSISTENT", "COFORGE", "MPHASIS", "TRENT", "POLYCAB", "DIXON", "ASTRAL", "PIIND", "FEDERALBNK", "IDFCFIRSTB", "AUBANK", "OBEROIRLTY", "GODREJPROP", "PRESTIGE", "JINDALSTEL", "SAIL", "NMDC", "MAXHEALTH", "SUPREMEIND", "VOLTAS", "CUMMINSIND", "BHARATFORG", "ASHOKLEY", "TVSMOTOR", "BALKRISIND", "CROMPTON", "ABCAPITAL", "LICHSGFIN", "IRCTC", "IRFC", "RVNL", "NHPC", "PFC", "RECLTD", "HAL", "BEL", "BHEL"]
CAPITAL, RISK_PCT = 100000, 1.0       # rupees; % of capital risked per trade (position sizing)
MIN_BUY_SCORE, TOP_N = 4, 5

# Settings can be overridden from config.json (edit it directly on GitHub)
try:
    _cfg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")))
except Exception:
    _cfg = {}
WATCHLIST = [s.strip().upper().replace(".NS", "") for s in _cfg.get("watchlist_us" if US else "watchlist", WATCHLIST_US if US else WATCHLIST) if str(s).strip()]
CAPITAL = _cfg.get("capital_us", 10000) if US else _cfg.get("capital", CAPITAL); RISK_PCT = _cfg.get("risk_pct", RISK_PCT)
MIN_BUY_SCORE = _cfg.get("min_buy_score", MIN_BUY_SCORE); TOP_N = _cfg.get("top_n", TOP_N)
OUT_DIR = "reports"
DOCS_DIR = os.path.dirname(os.path.abspath(__file__))   # data files go in the repo root, next to index.html
INDEX_TICKERS = {"NIFTY 50": "^NSEI", "NIFTY BANK": "^NSEBANK", "NIFTY IT": "^CNXIT", "NIFTY NEXT 50": "^NSMIDCP",
                 "NIFTY 500": "^CRSLDX", "NIFTY FMCG": "^CNXFMCG", "NIFTY AUTO": "^CNXAUTO", "NIFTY PHARMA": "^CNXPHARMA",
                 "NIFTY METAL": "^CNXMETAL", "NIFTY ENERGY": "^CNXENERGY", "NIFTY FIN SERVICE": "^CNXFIN", "INDIA VIX": "^INDIAVIX",
                 "NIFTY MIDCAP 100": "NIFTY_MIDCAP_100.NS", "NIFTY SMALLCAP 100": "^CNXSC", "NIFTY PSU BANK": "^CNXPSUBANK", "NIFTY REALTY": "^CNXREALTY",
                 "SENSEX": "^BSESN", "BSE 500": "BSE-500.BO", "BSE MIDCAP": "BSE-MIDCAP.BO", "BSE SMALLCAP": "BSE-SMLCAP.BO"}   # any that fail are skipped; dashboard shows simulated values for them
if US: INDEX_TICKERS = {"NYSE COMPOSITE": "^NYA", "NYSE US 100": "^NY", "DOW JONES": "^DJI", "NASDAQ": "^IXIC", "NASDAQ 100": "^NDX", "SEMICONDUCTOR": "^SOX",
                 "S&P 500": "^GSPC", "RUSSELL 2000": "^RUT", "S&P MIDCAP 400": "^MID", "VIX": "^VIX"}

# ---------- data ----------
def load(symbols, demo=False):
    if demo:
        out = {}
        for s in symbols:
            r = np.random.default_rng(abs(hash(s)) % 2**32)
            c = 1000 * np.exp(np.cumsum(r.normal(0.0004, 0.014, 300)))
            out[s] = pd.DataFrame({"Open": np.r_[c[0], c[:-1]], "Close": c, "High": c * 1.01, "Low": c * 0.99,
                                   "Volume": r.integers(1e6, 5e6, 300)}, index=pd.bdate_range(end=dt.date.today(), periods=301)[-300:])
        return out
    import yfinance as yf
    tick = [s + SUF for s in symbols]
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
                out[name] = [round(float(x), 2) for x in c.tail(250)]; IDX_OHLC[name] = _ohlc(hdf, 120); IDX_HIST[name] = _hist(hdf)
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
                t = yf.Ticker(s + SUF)
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
    with open(os.path.join(DOCS_DIR, FN("data.json")), "w") as f: _jd(out, f)
    hist = {"updated": out["updated"], "demo": demo, "stocks": {**{s: _hist(df) for s, df in data.items()}, **IDX_HIST}}
    with open(os.path.join(DOCS_DIR, FN("history.json")), "w") as f:
        _jd(hist, f)

# ---------- OHLC, live (intraday) ----------
IDX_OHLC = {}
IDX_HIST = {}

def _hist(df, n=250):
    """Columnar daily history with dates, for the dashboard's price-history table."""
    d = df.tail(n).ffill().bfill()
    col = lambda c: [round(float(x), 2) for x in d[c]]
    return {"d": [i.strftime("%Y-%m-%d") for i in d.index], "o": col("Open"), "h": col("High"),
            "l": col("Low"), "c": col("Close"),
            "v": [int(x) for x in d["Volume"].fillna(0)] if "Volume" in d else []}

def _ohlc(df, n):
    return [[round(float(r.Open), 2), round(float(r.High), 2), round(float(r.Low), 2), round(float(r.Close), 2)]
            for r in df.tail(n).itertuples()]

def run_live():
    """Fast job: 15-minute bars for the last 2 days + quotes, written to docs/live.json."""
    import yfinance as yf
    tick = {**{s: s + SUF for s in WATCHLIST}, **INDEX_TICKERS}
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
    with open(os.path.join(DOCS_DIR, FN("live.json")), "w") as f:
        _jd({"updated": now.isoformat(timespec="minutes"), "open": market_open(now),
                   "intraday": intr, "quotes": quotes}, f)
    print(f"live.json written: {len(intr)} symbols")

FUNDS = ["Parag Parikh Flexi Cap", "HDFC Mid Cap", "Nippon India Small Cap", "UTI Nifty 50 Index", "SBI Large Cap",
         "Kotak Emerging Equity", "Mirae Asset Large Cap", "HDFC Flexi Cap"]
FUNDS = _cfg.get("funds", FUNDS)          # names (matched against AMFI scheme names) or numeric AMFI scheme codes
METALS = {"GOLD_USD": ("Gold", "GC=F", "USD / oz"), "SILVER_USD": ("Silver", "SI=F", "USD / oz"), "PLATINUM_USD": ("Platinum", "PL=F", "USD / oz"),
          "GOLDBEES": ("Gold ETF (GOLDBEES)", "GOLDBEES.NS", "Rs / unit"),
          "SILVERBEES": ("Silver ETF (SILVERBEES)", "SILVERBEES.NS", "Rs / unit")}
if US: METALS = {"GOLD_USD": ("Gold", "GC=F", "USD / oz"), "SILVER_USD": ("Silver", "SI=F", "USD / oz"), "PLATINUM_USD": ("Platinum", "PL=F", "USD / oz"),
                 "GLD": ("Gold ETF (GLD)", "GLD", "$ / share"), "SLV": ("Silver ETF (SLV)", "SLV", "$ / share")}

FX = {"USD": ("US Dollar", 1), "EUR": ("Euro", 1), "GBP": ("British Pound", 1), "JPY": ("Japanese Yen", 100),
      "AED": ("UAE Dirham", 1), "SGD": ("Singapore Dollar", 1), "AUD": ("Australian Dollar", 1),
      "CAD": ("Canadian Dollar", 1), "CHF": ("Swiss Franc", 1), "CNY": ("Chinese Yuan", 1), "SAR": ("Saudi Riyal", 1)}
CURRENCIES = _cfg.get("currencies", list(FX))       # 3-letter codes; each is fetched as <CODE>INR=X

FX_US = {"EUR": ("EURUSD=X", "Euro", "$ per 1 EUR"), "GBP": ("GBPUSD=X", "British Pound", "$ per 1 GBP"),
         "AUD": ("AUDUSD=X", "Australian Dollar", "$ per 1 AUD"), "JPY": ("USDJPY=X", "Japanese Yen", "JPY per $1"),
         "CAD": ("USDCAD=X", "Canadian Dollar", "CAD per $1"), "CHF": ("USDCHF=X", "Swiss Franc", "CHF per $1"),
         "CNY": ("USDCNY=X", "Chinese Yuan", "CNY per $1"), "INR": ("USDINR=X", "Indian Rupee", "INR per $1"),
         "MXN": ("USDMXN=X", "Mexican Peso", "MXN per $1")}
FUNDS_US = _cfg.get("funds_us", ["FXAIX", "VFIAX", "VTSAX", "SWPPX", "FCNTX", "VIMAX", "VSMAX", "VBTLX"])   # Yahoo mutual-fund tickers

def fetch_fx_us():
    import yfinance as yf
    out = {}
    for code in _cfg.get("currencies_us", list(FX_US)):
        t, name, unit = FX_US.get(code, (f"USD{code}=X", code, f"{code} per $1"))
        try:
            tk = yf.Ticker(t); df = tk.history(period="1y", interval="1d").dropna(subset=["Close"])
            if len(df) < 2: print(f"Currency {code} has no data"); continue
            out[code] = {"name": f"{code} \u00b7 {name}", "unit": unit, "news": _news(tk), **_hist(df)}
        except Exception as e:
            print(f"Currency {code} unavailable: {e}")
    return out

def fetch_funds_us():
    import yfinance as yf
    out = []
    for t in FUNDS_US:
        try:
            tk = yf.Ticker(t); df = tk.history(period="5y", interval="1d").dropna(subset=["Close"])
            rows = [(i.date(), float(v)) for i, v in df["Close"].items()]
            try: info = tk.info or {}
            except Exception: info = {}
            last, r = rows[-1][1], {}
            def at(days):
                x = rows[-1][0] - dt.timedelta(days=days); p = [v for d, v in rows if d <= x]
                return p[-1] if p else None
            for k, dd in (("1M", 30), ("3M", 91), ("6M", 182), ("1Y", 365)):
                v = at(dd); r[k] = round((last / v - 1) * 100, 2) if v else None
            for k, y in (("3Y", 3), ("5Y", 5)):
                v = at(365 * y); r[k] = round(((last / v) ** (1 / y) - 1) * 100, 2) if v else None
            tail = rows[-250:]
            num = lambda v: v if isinstance(v, (int, float)) else None
            out.append({"code": t, "exp": num(info.get("netExpenseRatio")), "aum": num(info.get("totalAssets")), "news": _news(tk), "name": info.get("longName") or t, "house": info.get("fundFamily") or "", "cat": info.get("category") or "",
                        "nav": round(last, 4), "date": rows[-1][0].isoformat(), "chg": round((last / rows[-2][1] - 1) * 100, 2) if len(rows) > 1 else 0.0,
                        "r": r, "d": [d.isoformat() for d, _ in tail], "c": [round(v, 4) for _, v in tail], **(_risk_pack(rows, _bench_for(info.get("category") or "", info.get("longName") or t)) or {})})
        except Exception as e:
            print(f"Fund {t} failed: {e}")
    return out

def fetch_fx():
    if US: return fetch_fx_us()
    import yfinance as yf
    out = {}
    for code in CURRENCIES:
        name, mult = FX.get(code, (code, 1))
        try:
            tk = yf.Ticker(f"{code}INR=X"); df = tk.history(period="1y", interval="1d").dropna(subset=["Close"])
            if len(df) < 2: print(f"Currency {code} has no data"); continue
            if mult != 1: df[["Open", "High", "Low", "Close"]] = df[["Open", "High", "Low", "Close"]] * mult
            out[code] = {"name": f"{code} \u00b7 {name}", "unit": f"\u20b9 per {mult if mult != 1 else 1} {code}", "news": _news(tk), **_hist(df)}
        except Exception as e:
            print(f"Currency {code} unavailable: {e}")
    return out

def _derive(base, fxmap, factor):
    """USD/oz series -> INR series using the day's USD/INR close (indicative: no duty, GST or premium)."""
    out, last = {k: [] for k in "ohlc"}, next(iter(fxmap.values()))
    for i, d in enumerate(base["d"]):
        last = fxmap.get(d, last)
        for k in "ohlc": out[k].append(round(base[k][i] * last * factor, 2))
    return {"d": base["d"], **out, "v": []}

def fetch_metals():
    import yfinance as yf
    out, raw = {}, {}
    for key, (name, t, unit) in METALS.items():
        try:
            tk = yf.Ticker(t); df = tk.history(period="1y", interval="1d").dropna(subset=["Close"])
            if len(df) < 2: print(f"Metal {key} has no data"); continue
            raw[key] = _hist(df); out[key] = {"name": name, "unit": unit.replace("Rs", "\u20b9"), "news": _news(tk), **({"g": 31.1035, "cur": "$"} if "oz" in unit else {}), **raw[key]}
        except Exception as e:
            print(f"Metal {key} unavailable: {e}")
    if US: return out
    tkf = yf.Ticker("USDINR=X"); fxdf = tkf.history(period="1y", interval="1d").dropna(subset=["Close"])
    if len(fxdf) < 2: return out
    out["USDINR"] = {"name": "USD / INR", "unit": "\u20b9 per $", "news": _news(tkf), **_hist(fxdf)}
    fxmap = {i.strftime("%Y-%m-%d"): float(v) for i, v in fxdf["Close"].items()}
    for src, key, name, unit, f in (("GOLD_USD", "GOLD_INR", "Gold (India, indicative)", "\u20b9 / 10 g", 10 / 31.1035),
                                    ("SILVER_USD", "SILVER_INR", "Silver (India, indicative)", "\u20b9 / kg", 1000 / 31.1035),
                                    ("PLATINUM_USD", "PLATINUM_INR", "Platinum (India, indicative)", "\u20b9 / 10 g", 10 / 31.1035)):
        if src in raw: out[key] = {"name": name, "unit": unit, "g": 10 if "10 g" in unit else 1000, "cur": "\u20b9", "news": out.get(src, {}).get("news", []), **_derive(raw[src], fxmap, f)}
    return out

# ---------- fund risk, rating and capture ratios ----------
RF = (_cfg.get("risk_free_us", 4.0) if US else _cfg.get("risk_free", 6.5)) / 100       # annual risk-free rate used for Sharpe, Sortino and alpha
BENCH_NAMES = {"^NSEI": "Nifty 50", "^CRSMID": "Nifty Midcap 100", "NIFTY_MIDCAP_100.NS": "Nifty Midcap 100", "^CNXSC": "Nifty Smallcap 100",
               "NIFTYSMLCAP100.NS": "Nifty Smallcap 100", "^GSPC": "S&P 500", "^RUT": "Russell 2000", "^MID": "S&P MidCap 400", "AGG": "US Aggregate Bond (AGG)"}
_BENCH = {}

def _bench_for(cat, name):
    t = f"{cat} {name}".lower()
    if US:
        return ["AGG"] if "bond" in t else ["^RUT", "^GSPC"] if "small" in t else ["^MID", "^GSPC"] if "mid" in t else ["^GSPC"]
    return ["^CNXSC", "NIFTYSMLCAP100.NS", "^NSEI"] if "small" in t else ["^CRSMID", "NIFTY_MIDCAP_100.NS", "^NSEI"] if "mid" in t else ["^NSEI"]

def _bench_series(cands):
    import yfinance as yf
    for t in cands:
        if t not in _BENCH:
            try:
                df = yf.Ticker(t).history(period="6y", interval="1d").dropna(subset=["Close"])
                _BENCH[t] = pd.Series(df["Close"].values, index=pd.to_datetime([i.date() for i in df.index])) if len(df) > 500 else None
            except Exception:
                _BENCH[t] = None
        if _BENCH[t] is not None: return BENCH_NAMES.get(t, t), _BENCH[t]
    return None, None

def _risk_pack(rows, cands):
    """Risk metrics, capture ratios and a transparent 1-5 model rating from NAV history vs a benchmark. Returns {} if data is short."""
    try:
        rows = rows[-1300:]
        f = pd.Series([v for _, v in rows], index=pd.to_datetime([d for d, _ in rows])); f = f[~f.index.duplicated()].sort_index()
        f = f[f.index >= f.index[-1] - pd.Timedelta(days=1095)]
        if len(f) < 240: return {}
        span = (f.index[-1] - f.index[0]).days; basis = "3Y" if span >= 1000 else f"{span / 365:.1f}Y"
        d, r = RF / 252, f.pct_change().dropna(); sd = r.std()
        ex = r - d; ddev = float(np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(252))
        vol = float(sd * np.sqrt(252)); mdd = float((f / f.cummax() - 1).min())
        sharpe = float(ex.mean() * 252 / vol) if vol > 0 else None; sortino = float(ex.mean() * 252 / ddev) if ddev > 0 else None
        beta = alpha = up = down = None; label, b = _bench_series(cands)
        if b is not None:
            j = pd.concat([f, b], axis=1, join="inner").dropna(); j.columns = ["f", "b"]
            if len(j) >= 240:
                jr = j.pct_change().dropna(); var = jr["b"].var()
                if var > 0:
                    beta = float(jr["f"].cov(jr["b"]) / var)
                    alpha = float(((jr["f"] - d).mean() - beta * (jr["b"] - d).mean()) * 252)
                m = j.groupby(j.index.to_period("M")).last().pct_change().dropna()
                bu, bd = m[m["b"] > 0], m[m["b"] < 0]
                if len(m) >= 12 and len(bu) >= 3 and len(bd) >= 3:
                    g = lambda x: float(np.prod(1 + x) ** (1 / len(x)) - 1)
                    up, down = g(bu["f"]) / g(bu["b"]) * 100, g(bd["f"]) / g(bd["b"]) * 100
        level = "Low" if vol < .06 else "Moderate" if vol < .12 else "High" if vol < .18 else "Very high"
        cl = lambda x: min(max(x, 0.0), 1.0)
        comp = [(20, cl(1 - abs(mdd) / 0.40))]
        if sharpe is not None: comp.append((35, cl(sharpe / 1.5)))
        if up is not None and down and down > 0: comp.append((25, cl((up / down - 0.8) / 0.6)))
        if alpha is not None: comp.append((20, cl((alpha * 100 + 2) / 8)))
        score = round(sum(w * s for w, s in comp) / sum(w for w, _ in comp) * 100)
        stars = 5 if score >= 80 else 4 if score >= 65 else 3 if score >= 50 else 2 if score >= 35 else 1
        rd = lambda x, n=2: None if x is None else round(float(x), n)
        return {"risk": {"vol": rd(vol * 100), "mdd": rd(mdd * 100), "beta": rd(beta), "sharpe": rd(sharpe), "sortino": rd(sortino),
                         "alpha": rd(None if alpha is None else alpha * 100), "up": rd(up, 1), "down": rd(down, 1),
                         "ratio": rd(up / down) if up is not None and down and down > 0 else None,
                         "bench": label, "basis": basis, "level": level, "rf": rd(RF * 100)},
                "rating": {"stars": stars, "score": score}}
    except Exception as e:
        print(f"Risk metrics failed: {e}"); return {}

def _nw(s):
    import re
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", str(s).lower()).split())

def _mf_resolve(entry):
    import requests
    if str(entry).strip().isdigit(): return int(entry)
    words = [w for w in _nw(entry).split() if w not in ("direct", "growth", "plan", "fund")]
    res = requests.get("https://api.mfapi.in/mf/search", params={"q": " ".join(words[:2])}, timeout=30).json()
    cand = [x for x in res if all(w in _nw(x["schemeName"]) for w in words)]
    good = lambda n: "direct" in _nw(n) and "growth" in _nw(n) and not any(b in _nw(n) for b in ("idcw", "dividend", "bonus", "payout", "reinvest"))
    pref = [x for x in cand if good(x["schemeName"])] or cand
    return int(min(pref, key=lambda x: len(x["schemeName"]))["schemeCode"]) if pref else None

def _mf_fetch(code):
    import requests
    j = requests.get(f"https://api.mfapi.in/mf/{code}", timeout=60).json()
    rows = sorted((dt.datetime.strptime(x["date"], "%d-%m-%Y").date(), float(x["nav"]))
                  for x in j["data"] if x.get("nav") not in (None, "", "N.A."))
    def at(days):
        t = rows[-1][0] - dt.timedelta(days=days); prev = [v for d, v in rows if d <= t]
        return prev[-1] if prev else None
    last, r = rows[-1][1], {}
    for k, dd in (("1M", 30), ("3M", 91), ("6M", 182), ("1Y", 365)):
        v = at(dd); r[k] = round((last / v - 1) * 100, 2) if v else None
    for k, y in (("3Y", 3), ("5Y", 5)):
        v = at(365 * y); r[k] = round(((last / v) ** (1 / y) - 1) * 100, 2) if v else None
    tail, m = rows[-250:], j.get("meta", {})
    return {"code": str(code), "name": m.get("scheme_name", ""), "house": m.get("fund_house", ""), "cat": m.get("scheme_category", ""), "type": m.get("scheme_type", ""),
            "nav": last, "date": rows[-1][0].isoformat(), "chg": round((last / rows[-2][1] - 1) * 100, 2) if len(rows) > 1 else 0.0,
            "r": r, "d": [d.isoformat() for d, _ in tail], "c": [round(v, 4) for _, v in tail], **(_risk_pack(rows, _bench_for(m.get("scheme_category", ""), m.get("scheme_name", ""))) or {})}

CRYPTO = _cfg.get("crypto", ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT", "LINK", "LTC", "TRX", "SHIB", "BCH", "XLM", "ATOM", "NEAR"])
CRYPTO_NAMES = {"BTC": "Bitcoin", "ETH": "Ethereum", "BNB": "BNB", "SOL": "Solana", "XRP": "XRP", "ADA": "Cardano", "DOGE": "Dogecoin", "AVAX": "Avalanche",
                "DOT": "Polkadot", "LINK": "Chainlink", "LTC": "Litecoin", "TRX": "TRON", "SHIB": "Shiba Inu", "BCH": "Bitcoin Cash", "XLM": "Stellar",
                "ATOM": "Cosmos", "NEAR": "NEAR Protocol"}

def fetch_crypto():
    """Daily prices for the main coins in US dollars (the market trades every day, so rows are calendar days)."""
    import yfinance as yf
    out = {}
    for code in CRYPTO:
        try:
            tk = yf.Ticker(f"{code}-USD"); df = tk.history(period="2y", interval="1d").dropna(subset=["Close"])
            if len(df) < 30: print(f"Crypto {code} has no data"); continue
            try: cap = (tk.info or {}).get("marketCap")
            except Exception: cap = None
            out[code] = {"name": CRYPTO_NAMES.get(code, code), "unit": "$ / coin", "cal": True, "cap": cap if isinstance(cap, (int, float)) else None,
                         "news": _news(tk), **_hist(df, 370)}
        except Exception as e:
            print(f"Crypto {code} unavailable: {e}")
    return out

def fetch_evi():
    """Equity Valuation Index (proxy): the index price divided by its own 60-month average, times 100.
    India uses the BSE Sensex (long history), the US uses the S&P 500."""
    import yfinance as yf
    t, label = ("^GSPC", "S&P 500") if US else ("^BSESN", "BSE Sensex")
    df = yf.Ticker(t).history(period="max", interval="1mo").dropna(subset=["Close"])
    c = pd.Series(df["Close"].values, index=pd.to_datetime([i.date() for i in df.index]))
    c = c[~c.index.duplicated(keep="last")].sort_index()
    if len(c) < 120: raise ValueError("not enough monthly history")
    v = (c / c.rolling(60).mean() * 100).dropna().tail(241)
    return {"name": "Equity Valuation Index", "bench": label, "method": "index price / its 5-year average x 100",
            "d": [i.strftime("%Y-%m") for i in v.index], "v": [round(float(x), 1) for x in v.values]}

def update_extra():
    """Metals (gold, silver) and mutual funds -> extra.json. Failures are logged and never stop the run."""
    path = os.path.join(DOCS_DIR, FN("extra.json"))
    try: old = json.load(open(path))
    except Exception: old = {}
    new, errs = {}, {}
    try: new["metals"] = fetch_metals()
    except Exception as e: errs["metals"] = str(e)[:120]; print(f"Metals unavailable: {e}")
    try:
        fx = fetch_fx()
        if fx: new["fx"] = fx
        else: errs["fx"] = "Yahoo Finance returned no data for any currency pair"
    except Exception as e: errs["fx"] = str(e)[:120]; print(f"Currencies unavailable: {e}")
    try:
        cr = fetch_crypto()
        if cr: new["crypto"] = cr
        else: errs["crypto"] = "Yahoo Finance returned no data for any coin"
    except Exception as e: errs["crypto"] = str(e)[:120]; print(f"Crypto unavailable: {e}")
    try: new["evi"] = fetch_evi()
    except Exception as e: errs["evi"] = str(e)[:120]; print(f"Valuation index unavailable: {e}")
    funds = fetch_funds_us() if US else []
    for entry in ([] if US else FUNDS):
        try:
            code = _mf_resolve(entry)
            if code is None: print(f"Fund not found: {entry}"); continue
            funds.append(_mf_fetch(code))
        except Exception as e: print(f"Fund '{entry}' failed: {e}")
    if funds: new["funds"] = funds
    else: errs["funds"] = ("no fund could be loaded from Yahoo Finance" if US else "no fund could be loaded from mfapi.in (check names or use scheme codes in config.json)")
    if new or errs:
        old.update(new); old["errors"] = errs; old["updated"] = dt.datetime.now(IST).isoformat(timespec="minutes")
        os.makedirs(DOCS_DIR, exist_ok=True)
        with open(path, "w") as f: _jd(old, f)
        print(f"extra.json written: {len(new.get('metals', {}))} metal series, {len(funds)} funds")

# ---------- report ----------
def build_report(demo=False):
    data = load(WATCHLIST, demo)
    rows = {s: analyse(df) for s, df in data.items()}
    ranked = sorted(rows.items(), key=lambda x: (-x[1]["score"], -x[1]["ret1m"]))
    buys = [(s, r) for s, r in ranked if r["score"] >= MIN_BUY_SCORE][:TOP_N]
    avoid = [(s, r) for s, r in ranked[::-1] if r["score"] <= -2][:TOP_N]
    export_json(data, buys, avoid, demo, rows)
    if not demo: update_extra()
    now = dt.datetime.now(IST).strftime(f"%d %b %Y, %I:%M %p {TZN}")
    L = [f"MARKET SIGNALS - {now}" + ("  [DEMO DATA]" if demo else ""), ""]
    L.append("BUY WATCHLIST (ranked)")
    if not buys: L.append("  No stock meets the entry criteria right now. Staying in cash is a valid call.")
    for i, (s, r) in enumerate(buys, 1):
        L += [f"{i}. {s}  score {r['score']}  |  Buy ~{CUR}{r['price']:.2f}",
              f"   Stop-loss {CUR}{r['stop']:.2f} | Target {CUR}{r['target']:.2f} | Qty {r['qty']} (risk {RISK_PCT}% of {CUR}{CAPITAL:,})",
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
    return n.weekday() < 5 and (dt.time(9, 30) <= n.time() <= dt.time(16, 15) if US else dt.time(9, 15) <= n.time() <= dt.time(15, 45))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--once", action="store_true"); ap.add_argument("--demo", action="store_true"); ap.add_argument("--live", action="store_true"); ap.add_argument("--market", default="IN")
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
