"""
MASTER STOCK DATA COLLECTOR
===========================

Collects market, fundamental, liquidity, classification and technical
data for the 5,552-stock NSE+BSE universe.

Input:
    latest_price_mapping_5552.csv

Outputs:
    data/stock_master_data.csv
    data/stock_ohlcv/*.csv
    data/collection_errors.csv

Primary sources:
    - yfinance for quotes, fundamentals and historical OHLCV
    - NSE/BSE master mapping already present in input CSV
    - Derived metrics calculated locally from OHLCV

Notes:
    NSE/BSE-specific delivery, trades, ASM/GSM, circuit and surveillance
    fields are left as columns for later official-source wiring if not
    available from yfinance. The collector does not invent missing values.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "latest_price_mapping_5552.csv"
DATA_DIR = ROOT / "data"
HISTORY_DIR = DATA_DIR / "stock_ohlcv"

HISTORY_PERIOD = "1y"
HISTORY_INTERVAL = "1d"
BATCH_SIZE = 50
MAX_WORKERS = 6
RETRIES = 3


def clean_number(value):
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def first_valid(*values):
    for value in values:
        if value is not None:
            try:
                if not pd.isna(value):
                    return value
            except (TypeError, ValueError):
                return value
    return np.nan


def safe_info(ticker: yf.Ticker) -> dict:
    try:
        return ticker.info or {}
    except Exception:
        return {}


def safe_fast_info(ticker: yf.Ticker) -> dict:
    try:
        return dict(ticker.fast_info)
    except Exception:
        return {}


def fetch_history(symbol: str) -> pd.DataFrame:
    last_error = None

    for attempt in range(RETRIES):
        try:
            df = yf.download(
                symbol,
                period=HISTORY_PERIOD,
                interval=HISTORY_INTERVAL,
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            if df is None or df.empty:
                raise ValueError("No OHLCV data returned")

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            required = ["Open", "High", "Low", "Close", "Volume"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"Missing OHLCV columns: {missing}")

            df = df[required].copy()
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df = df.dropna(subset=["Close"])

            return df

        except Exception as exc:
            last_error = exc
            time.sleep(2 ** attempt)

    raise RuntimeError(f"History failed for {symbol}: {last_error}")


def pct_return(close: pd.Series, days: int):
    if len(close) <= days:
        return np.nan
    old = clean_number(close.iloc[-days - 1])
    new = clean_number(close.iloc[-1])
    if pd.isna(old) or old == 0 or pd.isna(new):
        return np.nan
    return (new / old - 1.0) * 100.0


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    return pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).rolling(period).mean()


def macd(close: pd.Series):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    return line, signal


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    tr = true_range(df)
    atr_val = tr.rolling(period).mean()

    plus_di = 100 * plus_dm.rolling(period).mean() / atr_val
    minus_di = 100 * minus_dm.rolling(period).mean() / atr_val

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    return dx.rolling(period).mean()


def bollinger(close: pd.Series, period: int = 20, std_mult: float = 2):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return mid, mid + std_mult * std, mid - std_mult * std


def stochastic(df: pd.DataFrame, period: int = 14) -> pd.Series:
    low = df["Low"].rolling(period).min()
    high = df["High"].rolling(period).max()
    return 100 * (df["Close"] - low) / (high - low).replace(0, np.nan)


def technical_metrics(df: pd.DataFrame) -> dict:
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma100 = close.rolling(100).mean()
    sma200 = close.rolling(200).mean()

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    rsi14 = rsi(close)
    atr14 = atr(df)
    adx14 = adx(df)
    macd_line, macd_signal = macd(close)
    bb_mid, bb_upper, bb_lower = bollinger(close)
    stoch14 = stochastic(df)

    returns = close.pct_change()
    daily_vol = returns.rolling(20).std() * np.sqrt(252) * 100

    gaps = (df["Open"] / df["Close"].shift(1) - 1).abs() * 100
    gap_frequency_8 = (gaps.tail(90) > 8).mean() * 100

    median_volume = volume.tail(252).median()
    avg_volume = volume.tail(20).mean()

    traded_value = close * volume
    adtv = traded_value.tail(20).mean()
    median_dtv = traded_value.tail(252).median()

    vol_ratio = volume.iloc[-1] / avg_volume if avg_volume else np.nan

    high_52 = close.tail(252).max()
    low_52 = close.tail(252).min()

    latest = df.iloc[-1]

    return {
        "average_volume_20": avg_volume,
        "median_volume_252": median_volume,
        "adtv_20_inr": adtv,
        "median_daily_traded_value_252_inr": median_dtv,
        "volume_ratio_20": vol_ratio,
        "volume_trend_20_vs_50": (
            volume.tail(20).mean() / volume.tail(50).mean()
            if volume.tail(50).mean()
            else np.nan
        ),
        "annualized_volatility_20_pct": daily_vol.iloc[-1],
        "gap_frequency_gt_8pct_90d": gap_frequency_8,
        "atr_14": atr14.iloc[-1],
        "atr_14_pct": (
            atr14.iloc[-1] / close.iloc[-1] * 100
            if close.iloc[-1]
            else np.nan
        ),
        "sma_20": sma20.iloc[-1],
        "sma_50": sma50.iloc[-1],
        "sma_100": sma100.iloc[-1],
        "sma_200": sma200.iloc[-1],
        "ema_20": ema20.iloc[-1],
        "ema_50": ema50.iloc[-1],
        "ema_200": ema200.iloc[-1],
        "rsi_14": rsi14.iloc[-1],
        "macd": macd_line.iloc[-1],
        "macd_signal": macd_signal.iloc[-1],
        "adx_14": adx14.iloc[-1],
        "bb_middle_20": bb_mid.iloc[-1],
        "bb_upper_20": bb_upper.iloc[-1],
        "bb_lower_20": bb_lower.iloc[-1],
        "stochastic_14": stoch14.iloc[-1],
        "roc_20_pct": (
            (close.iloc[-1] / close.iloc[-21] - 1) * 100
            if len(close) > 20 and close.iloc[-21] != 0
            else np.nan
        ),
        "momentum_20": (
            close.iloc[-1] - close.iloc[-21]
            if len(close) > 20
            else np.nan
        ),
        "high_52w": high_52,
        "low_52w": low_52,
        "distance_from_52w_high_pct": (
            (close.iloc[-1] / high_52 - 1) * 100
            if high_52
            else np.nan
        ),
        "breakout_52w": bool(
            len(close) > 20 and close.iloc[-1] >= high_52
        ),
        "trend_regime": (
            "BULLISH"
            if close.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1]
            else "BEARISH"
            if close.iloc[-1] < ema50.iloc[-1] < ema200.iloc[-1]
            else "MIXED"
        ),
        "volume_breakout": bool(
            avg_volume > 0 and volume.iloc[-1] >= avg_volume * 2
        ),
        "latest_ohlcv_date": str(df.index[-1].date()),
        "latest_open": latest["Open"],
        "latest_high": latest["High"],
        "latest_low": latest["Low"],
        "latest_close": latest["Close"],
        "latest_volume": latest["Volume"],
        "return_1d_pct": pct_return(close, 1),
        "return_1w_pct": pct_return(close, 5),
        "return_1m_pct": pct_return(close, 21),
        "return_3m_pct": pct_return(close, 63),
        "return_6m_pct": pct_return(close, 126),
        "return_1y_pct": pct_return(close, 252),
    }


def fundamental_metrics(info: dict, fast: dict) -> dict:
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
        "held_percent_insiders": "heldPercentInsiders",
        "held_percent_institutions": "heldPercentInstitutions",
    }

    result = {out: info.get(key, np.nan) for out, key in keys.items()}

    result["promoter_holding"] = result["held_percent_insiders"]
    result["fii_dii_holding"] = result["held_percent_institutions"]

    result["price"] = first_valid(
        fast.get("last_price"),
        info.get("currentPrice"),
        info.get("regularMarketPrice"),
    )
    result["previous_close"] = first_valid(
        fast.get("previous_close"),
        info.get("previousClose"),
        info.get("regularMarketPreviousClose"),
    )
    result["day_high"] = first_valid(
        fast.get("day_high"),
        info.get("dayHigh"),
    )
    result["day_low"] = first_valid(
        fast.get("day_low"),
        info.get("dayLow"),
    )
    result["52w_high"] = first_valid(
        fast.get("year_high"),
        info.get("fiftyTwoWeekHigh"),
    )
    result["52w_low"] = first_valid(
        fast.get("year_low"),
        info.get("fiftyTwoWeekLow"),
    )

    return result


def collect_one(row: dict) -> tuple[dict, pd.DataFrame | None, str | None]:
    ticker_symbol = str(row.get("PRICE_TICKER", "")).strip()
    if not ticker_symbol:
        return row, None, "Missing PRICE_TICKER"

    try:
        ticker = yf.Ticker(ticker_symbol)
        info = safe_info(ticker)
        fast = safe_fast_info(ticker)
        history = fetch_history(ticker_symbol)

        metrics = {}
        metrics.update(fundamental_metrics(info, fast))
        metrics.update(technical_metrics(history))

        result = dict(row)

        # Remove old blank price fields before filling actual values.
        result.update(metrics)

        # yfinance identity fields where available.
        result["sector"] = info.get("sector", np.nan)
        result["industry"] = info.get("industry", np.nan)
        result["business_summary"] = info.get("longBusinessSummary", np.nan)

        # Explicit placeholders for data that needs official NSE/BSE sources.
        result["total_traded_value"] = np.nan
        result["number_of_trades"] = np.nan
        result["delivery_pct"] = np.nan
        result["free_float"] = first_valid(
            info.get("floatShares"),
            info.get("sharesOutstanding"),
        )
        result["bid"] = np.nan
        result["ask"] = np.nan
        result["upper_circuit"] = np.nan
        result["lower_circuit"] = np.nan
        result["price_band"] = np.nan
        result["asm_status"] = np.nan
        result["gsm_status"] = np.nan
        result["surveillance_indicator"] = np.nan
        result["promoter_pledge"] = np.nan
        result["dii_holding"] = np.nan
        result["fii_holding"] = np.nan
        result["large_mid_small_cap"] = np.nan
        result["listing_age_years"] = np.nan

        result["fetch_status"] = "OK"

        return result, history, None

    except Exception as exc:
        result = dict(row)
        result["fetch_status"] = "ERROR"
        return result, None, str(exc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    universe = pd.read_csv(input_path, dtype=str).fillna("")

    if args.limit > 0:
        universe = universe.head(args.limit)

    rows = universe.to_dict("records")
    results = []
    errors = []

    print(f"Universe: {len(rows)}")
    print(f"Workers: {args.workers}")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(collect_one, row): row for row in rows
        }

        for index, future in enumerate(as_completed(futures), start=1):
            result, history, error = future.result()
            results.append(result)

            ticker = result.get("PRICE_TICKER", "")
            if history is not None and ticker:
                safe_name = ticker.replace("/", "_").replace("\\", "_")
                history.to_csv(HISTORY_DIR / f"{safe_name}.csv")

            if error:
                errors.append(
                    {
                        "ISIN": result.get("ISIN", ""),
                        "PRICE_TICKER": ticker,
                        "error": error,
                    }
                )

            if index % 25 == 0 or index == len(rows):
                print(
                    f"Progress: {index}/{len(rows)} | "
                    f"Errors: {len(errors)}"
                )

    master = pd.DataFrame(results)

    preferred_order = [
        "ISIN",
        "COMPANY",
        "NSE_SYMBOL",
        "BSE_SYMBOL",
        "BSE_CODE",
        "EXCHANGE",
        "PRICE_TICKER",
        "fetch_status",
        "price",
        "previous_close",
        "latest_open",
        "latest_high",
        "latest_low",
        "latest_close",
        "52w_high",
        "52w_low",
        "return_1d_pct",
        "return_1w_pct",
        "return_1m_pct",
        "return_3m_pct",
        "return_6m_pct",
        "return_1y_pct",
        "average_volume_20",
        "median_volume_252",
        "adtv_20_inr",
        "median_daily_traded_value_252_inr",
        "volume_ratio_20",
        "volume_trend_20_vs_50",
        "annualized_volatility_20_pct",
        "gap_frequency_gt_8pct_90d",
        "atr_14",
        "atr_14_pct",
        "total_traded_value",
        "number_of_trades",
        "delivery_pct",
        "free_float",
        "bid",
        "ask",
        "upper_circuit",
        "lower_circuit",
        "price_band",
        "asm_status",
        "gsm_status",
        "surveillance_indicator",
        "market_cap",
        "enterprise_value",
        "revenue",
        "revenue_growth",
        "ebitda",
        "ebitda_margin",
        "ebit",
        "net_profit",
        "eps",
        "eps_growth",
        "pe_ratio",
        "forward_pe",
        "pb_ratio",
        "ps_ratio",
        "roe",
        "roa",
        "roce",
        "debt",
        "debt_to_equity",
        "current_ratio",
        "free_cash_flow",
        "operating_cash_flow",
        "dividend_yield",
        "book_value",
        "promoter_holding",
        "promoter_pledge",
        "fii_holding",
        "dii_holding",
        "sector",
        "industry",
        "large_mid_small_cap",
        "NSE_LISTING_DATE",
        "BSE_LISTING_DATE",
        "listing_age_years",
        "NSE_SYMBOL",
        "BSE_SYMBOL",
        "BSE_CODE",
        "SME_FLAG",
        "BSE_EXCLUSIVE",
        "business_summary",
    ]

    # Add ROCE explicitly if it was not available from yfinance.
    if "roce" not in master.columns:
        master["roce"] = np.nan

    # Only keep columns that exist, preserving useful extras at the end.
    ordered = [c for c in preferred_order if c in master.columns]
    extras = [c for c in master.columns if c not in ordered]
    master = master[ordered + extras]

    master.to_csv(DATA_DIR / "stock_master_data.csv", index=False)

    if errors:
        pd.DataFrame(errors).to_csv(
            DATA_DIR / "collection_errors.csv",
            index=False,
        )
    else:
        pd.DataFrame(
            columns=["ISIN", "PRICE_TICKER", "error"]
        ).to_csv(DATA_DIR / "collection_errors.csv", index=False)

    print()
    print("Collection complete.")
    print(f"Master: {DATA_DIR / 'stock_master_data.csv'}")
    print(f"History: {HISTORY_DIR}")
    print(f"Errors: {DATA_DIR / 'collection_errors.csv'}")
    print(f"Rows collected: {len(master)}")
    print(f"Successful: {(master['fetch_status'] == 'OK').sum()}")
    print(f"Failed: {(master['fetch_status'] == 'ERROR').sum()}")


if __name__ == "__main__":
    main()
