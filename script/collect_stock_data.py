"""
ONE-SHOT 5,552 STOCK DATA COLLECTOR
===================================

Run once from GitHub Actions:

    python scripts/collect_stock_data.py

Input:
    latest_price_mapping_5552.csv

Collects in one pipeline:
- Current/previous OHLC, 52W high/low
- 1D/1W/1M/3M/6M/1Y returns
- 1Y historical OHLCV
- Average/median volume
- ADTV / median daily traded value
- volume trend, volatility, gap frequency, ATR
- SMA/EMA, RSI, MACD, ADX, Bollinger, Stochastic, ROC, momentum
- yfinance fundamentals where supplied
- NSE official daily bhavcopy fields when available
- NSE delivery %, number of trades, traded value
- NSE price band / surveillance / volatility / 52W data when available
- AMFI Large/Mid/Small classification
- NSE listing date/age from the supplied NSE master
- NSE/BSE mapping from the supplied master CSV

Important:
Missing source data is stored as blank/NA. Nothing is guessed.
Bid/ask is not stored as a fake historical value; it requires live market
depth/broker data and is marked unavailable.

Outputs:
    data/master_stock_data.csv
    data/collection_errors.csv
    data/ohlcv_1y/<ticker>.csv
    data/raw_sources/*.csv
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "latest_price_mapping_5552.csv"
OUT = ROOT / "data"
RAW = OUT / "raw_sources"
HISTORY = OUT / "ohlcv_1y"

OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)
HISTORY.mkdir(parents=True, exist_ok=True)

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/151.0 Safari/537.36"
)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
})

NSE_ARCHIVE = "https://nsearchives.nseindia.com"
NSE_HOME = "https://www.nseindia.com"


def num(x):
    try:
        if x is None or pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def first(*values):
    for x in values:
        if x is not None:
            try:
                if not pd.isna(x):
                    return x
            except Exception:
                return x
    return np.nan


def clean_columns(df):
    df = df.copy()
    df.columns = [
        re.sub(r"\s+", " ", str(c).strip()).upper()
        for c in df.columns
    ]
    return df


def find_col(df, *names):
    cols = {str(c).strip().upper(): c for c in df.columns}
    for name in names:
        if name.upper() in cols:
            return cols[name.upper()]
    for c in df.columns:
        u = str(c).upper()
        if any(name.upper() in u for name in names):
            return c
    return None


def get_nse_archive(url, timeout=30):
    r = SESSION.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


def trading_date():
    # NSE reports are EOD. On weekends/holidays, walk backwards.
    d = datetime.now().date()
    for _ in range(10):
        if d.weekday() < 5:
            return d
        d -= timedelta(days=1)
    return d


def nse_bhavcopy(date):
    ds = date.strftime("%d%m%Y")
    urls = [
        f"{NSE_ARCHIVE}/products/content/sec_bhavdata_full_{ds}.csv",
        f"{NSE_ARCHIVE}/content/historical/EQUITIES/{date.strftime('%Y')}/"
        f"CM_{date.strftime('%d%b%Y').upper()}_1.csv",
    ]
    for url in urls:
        try:
            raw = get_nse_archive(url)
            df = pd.read_csv(io.BytesIO(raw))
            df = clean_columns(df)
            if "SYMBOL" in df.columns:
                return df
        except Exception:
            pass
    return pd.DataFrame()


def nse_report_from_all_reports(date, keyword):
    """
    Best-effort discovery from NSE's public All Reports page.
    Used for reports whose archive filename changes.
    """
    try:
        r = SESSION.get(
            f"{NSE_HOME}/all-reports",
            params={"type": "equities"},
            timeout=30,
        )
        r.raise_for_status()
        html = r.text
    except Exception:
        return pd.DataFrame()

    matches = []
    for href, text in re.findall(
        r'href=["\']([^"\']+)["\'][^>]*>(.*?)</',
        html,
        flags=re.I | re.S,
    ):
        label = re.sub(r"<.*?>", " ", text)
        label = re.sub(r"\s+", " ", label).strip()
        if keyword.lower() in label.lower() or keyword.lower() in href.lower():
            matches.append(urljoin(NSE_HOME, href))

    for url in matches[:10]:
        try:
            raw = get_nse_archive(url)
            if url.lower().endswith(".zip"):
                z = zipfile.ZipFile(io.BytesIO(raw))
                for member in z.namelist():
                    if member.lower().endswith((".csv", ".dat")):
                        data = z.read(member)
                        df = pd.read_csv(io.BytesIO(data), sep=None, engine="python")
                        return clean_columns(df)
            else:
                return clean_columns(
                    pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
                )
        except Exception:
            continue

    return pd.DataFrame()


def normalize_nse_daily(df):
    if df.empty or "SYMBOL" not in df.columns:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["NSE_SYMBOL"] = df["SYMBOL"].astype(str).str.strip().str.upper()

    mapping = {
        "OPEN": ["OPEN"],
        "HIGH": ["HIGH"],
        "LOW": ["LOW"],
        "CLOSE": ["CLOSE"],
        "PREV_CLOSE": ["PREV_CLOSE", "PREVCLOSE", "PREV CLOSE"],
        "VOLUME": ["TTL_TRD_QNTY", "TOTTRDQTY", "TOTAL TRADED QUANTITY"],
        "TRADED_VALUE": ["TURNOVER_LACS", "TOTTRDVAL", "TOTAL TRADED VALUE"],
        "DELIVERY_QTY": ["DELIV_QTY", "DELIVERY QTY"],
        "DELIVERY_PCT": ["DELIV_PER", "DELIVERY PERCENTAGE"],
        "NO_OF_TRADES": ["NO_OF_TRADES", "NO OF TRADES"],
        "SERIES": ["SERIES"],
    }

    for target, names in mapping.items():
        c = find_col(df, *names)
        out[target] = df[c] if c else np.nan

    for c in [
        "OPEN", "HIGH", "LOW", "CLOSE", "PREV_CLOSE", "VOLUME",
        "TRADED_VALUE", "DELIVERY_QTY", "DELIVERY_PCT", "NO_OF_TRADES"
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out.drop_duplicates("NSE_SYMBOL")


def load_amfi_classification():
    """
    Finds the 2026 Jan-Jun Excel link from AMFI's official classification page.
    """
    url = "https://www.amfiindia.com/otherdata/categorisation-of-stocks"
    try:
        r = SESSION.get(url, timeout=30)
        r.raise_for_status()
        html = r.text
    except Exception:
        return pd.DataFrame()

    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.I)

    candidates = []
    for href in hrefs:
        h = href.lower()
        if ("2026" in h or "2026" in r.text.lower()) and (
            ".xls" in h or ".xlsx" in h
        ):
            candidates.append(urljoin(url, href))

    # Prefer files around the Jan-Jun 2026 section.
    for href in candidates:
        try:
            raw = SESSION.get(href, timeout=30).content
            df = pd.read_excel(io.BytesIO(raw))
            df = clean_columns(df)

            text = " ".join(map(str, df.columns))
            if any(x in text for x in ["ISIN", "COMPANY", "NAME"]):
                return df
        except Exception:
            continue

    return pd.DataFrame()


def technicals(df):
    if df.empty:
        return {}

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    prev = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev).abs(),
            (low - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr14 = tr.rolling(14).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1 / 14, adjust=False).mean()
    al = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    rsi14 = 100 - 100 / (1 + rs)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()

    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0),
        index=df.index,
    )
    atr14_for_adx = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / atr14_for_adx
    minus_di = 100 * minus_dm.rolling(14).mean() / atr14_for_adx
    dx = 100 * (plus_di - minus_di).abs() / (
        plus_di + minus_di
    ).replace(0, np.nan)
    adx14 = dx.rolling(14).mean()

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    st_low = low.rolling(14).min()
    st_high = high.rolling(14).max()
    stoch = 100 * (close - st_low) / (st_high - st_low).replace(0, np.nan)

    daily_ret = close.pct_change()
    gaps = (df["Open"] / prev - 1).abs() * 100

    traded_value = close * volume
    avg_vol20 = volume.tail(20).mean()
    med_vol252 = volume.tail(252).median()
    adtv20 = traded_value.tail(20).mean()
    med_dtv252 = traded_value.tail(252).median()

    high52 = close.tail(252).max()
    low52 = close.tail(252).min()

    def ret(n):
        if len(close) <= n:
            return np.nan
        old = close.iloc[-n - 1]
        return (close.iloc[-1] / old - 1) * 100 if old else np.nan

    return {
        "price": close.iloc[-1],
        "previous_close": prev.iloc[-1],
        "open": df["Open"].iloc[-1],
        "day_high": high.iloc[-1],
        "day_low": low.iloc[-1],
        "52w_high": high52,
        "52w_low": low52,
        "return_1d_pct": ret(1),
        "return_1w_pct": ret(5),
        "return_1m_pct": ret(21),
        "return_3m_pct": ret(63),
        "return_6m_pct": ret(126),
        "return_1y_pct": ret(252),
        "average_volume_20": avg_vol20,
        "median_volume_252": med_vol252,
        "adtv_20_inr": adtv20,
        "median_daily_traded_value_252_inr": med_dtv252,
        "volume_ratio_20": (
            volume.iloc[-1] / avg_vol20 if avg_vol20 else np.nan
        ),
        "volume_trend_20_vs_50": (
            volume.tail(20).mean() / volume.tail(50).mean()
            if volume.tail(50).mean()
            else np.nan
        ),
        "annualized_volatility_20_pct": (
            daily_ret.tail(20).std() * np.sqrt(252) * 100
        ),
        "gap_frequency_gt_8pct_90d": (
            (gaps.tail(90) > 8).mean() * 100
        ),
        "atr_14": atr14.iloc[-1],
        "atr_14_pct": (
            atr14.iloc[-1] / close.iloc[-1] * 100
            if close.iloc[-1]
            else np.nan
        ),
        "sma_20": close.rolling(20).mean().iloc[-1],
        "sma_50": close.rolling(50).mean().iloc[-1],
        "sma_100": close.rolling(100).mean().iloc[-1],
        "sma_200": close.rolling(200).mean().iloc[-1],
        "ema_20": close.ewm(span=20, adjust=False).mean().iloc[-1],
        "ema_50": close.ewm(span=50, adjust=False).mean().iloc[-1],
        "ema_200": close.ewm(span=200, adjust=False).mean().iloc[-1],
        "rsi_14": rsi14.iloc[-1],
        "macd": macd.iloc[-1],
        "macd_signal": macd_signal.iloc[-1],
        "adx_14": adx14.iloc[-1],
        "bollinger_middle_20": bb_mid.iloc[-1],
        "bollinger_upper_20": bb_upper.iloc[-1],
        "bollinger_lower_20": bb_lower.iloc[-1],
        "stochastic_14": stoch.iloc[-1],
        "roc_20_pct": (
            (close.iloc[-1] / close.iloc[-21] - 1) * 100
            if len(close) > 20
            else np.nan
        ),
        "momentum_20": (
            close.iloc[-1] - close.iloc[-21]
            if len(close) > 20
            else np.nan
        ),
        "52w_breakout": bool(close.iloc[-1] >= high52),
        "distance_from_52w_high_pct": (
            (close.iloc[-1] / high52 - 1) * 100
            if high52
            else np.nan
        ),
        "trend_regime": (
            "BULLISH"
            if close.iloc[-1] > close.ewm(span=50, adjust=False).mean().iloc[-1]
            > close.ewm(span=200, adjust=False).mean().iloc[-1]
            else "BEARISH"
            if close.iloc[-1] < close.ewm(span=50, adjust=False).mean().iloc[-1]
            < close.ewm(span=200, adjust=False).mean().iloc[-1]
            else "MIXED"
        ),
        "volume_breakout": bool(
            avg_vol20 and volume.iloc[-1] >= avg_vol20 * 2
        ),
        "latest_volume": volume.iloc[-1],
        "history_date": str(df.index[-1].date()),
    }


def yf_fundamentals(symbol):
    t = yf.Ticker(symbol)
    try:
        info = t.info or {}
    except Exception:
        info = {}

    try:
        fast = dict(t.fast_info)
    except Exception:
        fast = {}

    keys = {
        "market_cap": "marketCap",
        "enterprise_value": "enterpriseValue",
        "revenue": "totalRevenue",
        "revenue_growth": "revenueGrowth",
        "ebitda": "ebitda",
        "ebitda_margin": "ebitdaMargins",
        "ebit": "ebit",
        "net_profit": "netIncomeToCommon",
        "eps": "trailingEps",
        "eps_growth": "earningsGrowth",
        "pe_ratio": "trailingPE",
        "forward_pe": "forwardPE",
        "pb_ratio": "priceToBook",
        "ps_ratio": "priceToSalesTrailing12Months",
        "roe": "returnOnEquity",
        "roa": "returnOnAssets",
        "debt": "totalDebt",
        "debt_to_equity": "debtToEquity",
        "current_ratio": "currentRatio",
        "free_cash_flow": "freeCashflow",
        "operating_cash_flow": "operatingCashflow",
        "dividend_yield": "dividendYield",
        "book_value": "bookValue",
        "shares_outstanding": "sharesOutstanding",
        "float_shares": "floatShares",
        "promoter_holding_proxy": "heldPercentInsiders",
        "institutional_holding_proxy": "heldPercentInstitutions",
    }

    out = {k: info.get(v, np.nan) for k, v in keys.items()}

    out["price"] = first(
        fast.get("last_price"),
        info.get("currentPrice"),
        info.get("regularMarketPrice"),
    )
    out["previous_close"] = first(
        fast.get("previous_close"),
        info.get("previousClose"),
    )
    out["sector"] = info.get("sector", np.nan)
    out["industry"] = info.get("industry", np.nan)

    # These are intentionally NOT renamed as actual FII/DII/promoter values.
    # Yahoo's institutional/insider fields are only proxies.
    return out


def fetch_one(row):
    row = dict(row)
    ticker = str(row.get("PRICE_TICKER", "")).strip()

    if not ticker:
        row["fetch_status"] = "ERROR"
        row["error"] = "missing ticker"
        return row, None

    try:
        hist = yf.download(
            ticker,
            period="1y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        hist = hist[["Open", "High", "Low", "Close", "Volume"]].dropna(
            subset=["Close"]
        )

        if hist.empty:
            raise ValueError("no OHLCV returned")

        result = dict(row)
        result.update(technicals(hist))
        result.update(yf_fundamentals(ticker))

        # Official-source fields populated after daily NSE merge.
        result["fetch_status"] = "OK"
        result["error"] = ""

        return result, hist

    except Exception as exc:
        row["fetch_status"] = "ERROR"
        row["error"] = str(exc)
        return row, None


def merge_nse_daily(master, daily):
    if daily.empty:
        return master

    master = master.copy()
    daily = daily.copy()

    # Only NSE-listed rows can be matched by NSE symbol.
    if "NSE_SYMBOL" not in master.columns:
        return master

    merged = master.merge(
        daily,
        on="NSE_SYMBOL",
        how="left",
        suffixes=("", "_NSE"),
    )

    for c in [
        "OPEN", "HIGH", "LOW", "CLOSE", "PREV_CLOSE", "VOLUME",
        "TRADED_VALUE", "DELIVERY_QTY", "DELIVERY_PCT", "NO_OF_TRADES",
        "SERIES"
    ]:
        src = c
        if src in merged.columns:
            target = {
                "OPEN": "nse_open",
                "HIGH": "nse_high",
                "LOW": "nse_low",
                "CLOSE": "nse_close",
                "PREV_CLOSE": "nse_previous_close",
                "VOLUME": "nse_traded_quantity",
                "TRADED_VALUE": "nse_traded_value",
                "DELIVERY_QTY": "nse_delivery_quantity",
                "DELIVERY_PCT": "delivery_pct",
                "NO_OF_TRADES": "number_of_trades",
                "SERIES": "nse_series",
            }[c]
            merged[target] = merged[src]

    # Daily traded value in some NSE reports is in lakh rupees.
    if "nse_traded_value" in merged.columns:
        # Keep source value and provide an explicit INR estimate.
        merged["nse_traded_value_inr"] = pd.to_numeric(
            merged["nse_traded_value"], errors="coerce"
        )
        merged["nse_traded_value_inr"] = merged["nse_traded_value_inr"] * 100000

    return merged.drop(columns=[
        c for c in [
            "OPEN", "HIGH", "LOW", "CLOSE", "PREV_CLOSE", "VOLUME",
            "TRADED_VALUE", "DELIVERY_QTY", "DELIVERY_PCT",
            "NO_OF_TRADES", "SERIES"
        ] if c in merged.columns])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(INPUT))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    universe = pd.read_csv(args.input, dtype=str).fillna("")
    if args.limit:
        universe = universe.head(args.limit)

    print(f"Universe: {len(universe)}")

    # 1) One NSE daily official fetch for all NSE symbols.
    date = trading_date()
    print(f"Fetching NSE official daily data for {date}...")
    daily = normalize_nse_daily(nse_bhavcopy(date))

    if not daily.empty:
        daily.to_csv(RAW / f"nse_bhavcopy_{date:%Y%m%d}.csv", index=False)
        print(f"NSE daily rows: {len(daily)}")
    else:
        print("WARNING: NSE daily bhavcopy unavailable.")

    # 2) AMFI current classification.
    print("Fetching AMFI 2026 Jan-Jun classification...")
    amfi = load_amfi_classification()

    if not amfi.empty:
        amfi.to_csv(RAW / "amfi_2026_jan_jun.csv", index=False)
        print(f"AMFI rows: {len(amfi)}")
    else:
        print("WARNING: AMFI classification unavailable.")

    # 3) One 1Y OHLCV + fundamental fetch per mapped ticker.
    rows = universe.to_dict("records")
    results = []
    errors = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, r): r for r in rows}

        for i, future in enumerate(as_completed(futures), 1):
            result, hist = future.result()
            results.append(result)

            if hist is not None:
                ticker = result.get("PRICE_TICKER", "unknown")
                safe = re.sub(r"[^A-Za-z0-9_.-]", "_", ticker)
                hist.to_csv(HISTORY / f"{safe}.csv")

            if result.get("fetch_status") == "ERROR":
                errors.append({
                    "ISIN": result.get("ISIN", ""),
                    "PRICE_TICKER": result.get("PRICE_TICKER", ""),
                    "error": result.get("error", ""),
                })

            if i % 25 == 0 or i == len(rows):
                print(
                    f"Progress {i}/{len(rows)} | errors={len(errors)}"
                )

    master = pd.DataFrame(results)

    # 4) Merge official NSE daily data.
    master = merge_nse_daily(master, daily)

    # 5) AMFI classification by ISIN where possible.
    if not amfi.empty and "ISIN" in master.columns:
        amfi = clean_columns(amfi)

        isin_col = find_col(amfi, "ISIN", "ISIN CODE")
        class_col = find_col(
            amfi,
            "CLASSIFICATION",
            "CATEGORY",
            "CAP CATEGORY",
            "TYPE",
        )

        if isin_col and class_col:
            ac = amfi[[isin_col, class_col]].copy()
            ac.columns = ["ISIN", "large_mid_small_cap"]
            ac["ISIN"] = ac["ISIN"].astype(str).str.strip().str.upper()
            ac = ac.drop_duplicates("ISIN")
            master["ISIN"] = master["ISIN"].astype(str).str.strip().str.upper()
            master = master.merge(ac, on="ISIN", how="left", suffixes=("", "_AMFI"))

    # 6) Listing age from existing exchange master dates.
    today = pd.Timestamp.today().normalize()
    listing_col = None
    for c in ["NSE_LISTING_DATE", "BSE_LISTING_DATE"]:
        if c in master.columns:
            listing_col = c
            break

    if listing_col:
        dt = pd.to_datetime(master[listing_col], errors="coerce")
        master["listing_age_years"] = (
            (today - dt).dt.days / 365.25
        )

    # 7) Required fields that cannot safely be obtained historically
    # from yfinance are explicitly marked NA.
    unavailable = [
        "bid",
        "ask",
        "promoter_pledge",
        "fii_holding",
        "dii_holding",
        "upper_circuit",
        "lower_circuit",
        "price_band",
        "asm_status",
        "gsm_status",
        "surveillance_indicator",
        "roce",
    ]
    for c in unavailable:
        if c not in master.columns:
            master[c] = np.nan

    # Fallback price fields.
    if "price" not in master:
        master["price"] = np.nan

    if "latest_close" not in master and "latest_close" not in master.columns:
        master["latest_close"] = master.get("price", np.nan)

    # Save master.
    master.to_csv(OUT / "master_stock_data.csv", index=False)

    pd.DataFrame(errors).to_csv(
        OUT / "collection_errors.csv",
        index=False,
    )

    # Small machine-readable run summary.
    summary = {
        "run_date": str(date),
        "universe": len(universe),
        "rows_output": len(master),
        "successful_ohlcv": int((master["fetch_status"] == "OK").sum()),
        "failed_ohlcv": int((master["fetch_status"] == "ERROR").sum()),
        "nse_daily_rows": len(daily),
        "amfi_rows": len(amfi),
    }
    pd.DataFrame([summary]).to_csv(
        OUT / "collection_summary.csv",
        index=False,
    )

    print("\nDONE")
    print(f"Master: {OUT / 'master_stock_data.csv'}")
    print(f"Errors: {OUT / 'collection_errors.csv'}")
    print(f"Summary: {OUT / 'collection_summary.csv'}")
    print(f"OHLCV: {HISTORY}")


if __name__ == "__main__":
    main()
