"""
PRICE + 1Y OHLCV COLLECTOR
Input: latest_price_mapping_5552.csv
Output:
  data/price_ohlcv.csv
  data/ohlcv_1y/*.csv
  data/price_collection_errors.csv

Uses Yahoo in batches. Invalid BSE Yahoo symbols are recorded as unavailable;
they are NOT treated as delisted and do not crash the run.
"""

from __future__ import annotations

import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "latest_price_mapping_5552.csv"
OUT = ROOT / "data"
HISTORY = OUT / "ohlcv_1y"
OUT.mkdir(parents=True, exist_ok=True)
HISTORY.mkdir(parents=True, exist_ok=True)

# Conservative batch size to reduce Yahoo rate limiting.
BATCH_SIZE = 25
RETRIES = 3
BACKOFF = 15


def download_batch(tickers):
    for attempt in range(RETRIES):
        try:
            data = yf.download(
                tickers=tickers,
                period="1y",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
                group_by="column",
            )

            if data is None or data.empty:
                return {}

            result = {}

            if isinstance(data.columns, pd.MultiIndex):
                levels = [list(data.columns.get_level_values(i)) for i in range(2)]

                for ticker in tickers:
                    try:
                        if ticker in levels[1]:
                            df = data.xs(ticker, axis=1, level=1).copy()
                        elif ticker in levels[0]:
                            df = data.xs(ticker, axis=1, level=0).copy()
                        else:
                            continue

                        required = ["Open", "High", "Low", "Close", "Volume"]
                        if not all(c in df.columns for c in required):
                            continue

                        df = df[required].dropna(subset=["Close"])
                        if not df.empty:
                            result[ticker] = df
                    except Exception:
                        continue
            else:
                ticker = tickers[0]
                required = ["Open", "High", "Low", "Close", "Volume"]
                if all(c in data.columns for c in required):
                    df = data[required].dropna(subset=["Close"])
                    if not df.empty:
                        result[ticker] = df

            return result

        except Exception as exc:
            if attempt == RETRIES - 1:
                print(f"Batch failed after retries: {exc}")
                return {}
            time.sleep(BACKOFF * (attempt + 1))

    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(INPUT))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    universe = pd.read_csv(args.input, dtype=str).fillna("")
    universe["PRICE_TICKER"] = universe["PRICE_TICKER"].astype(str).str.strip()
    universe = universe[universe["PRICE_TICKER"] != ""].copy()

    # Keep first mapping row for each ticker.
    universe = universe.drop_duplicates("PRICE_TICKER").reset_index(drop=True)

    tickers = universe["PRICE_TICKER"].tolist()
    print(f"Stocks/tickers: {len(tickers)}")

    rows = []
    errors = []

    for start in range(0, len(tickers), args.batch_size):
        batch = tickers[start:start + args.batch_size]
        batch_no = start // args.batch_size + 1
        total_batches = (len(tickers) + args.batch_size - 1) // args.batch_size

        print(
            f"[{batch_no}/{total_batches}] "
            f"Downloading {start + 1}-{min(start + len(batch), len(tickers))}"
        )

        histories = download_batch(batch)

        for ticker in batch:
            matches = universe[universe["PRICE_TICKER"] == ticker]
            df = histories.get(ticker)

            if df is None:
                for _, source in matches.iterrows():
                    errors.append({
                        "ISIN": source.get("ISIN", ""),
                        "COMPANY": source.get("COMPANY", ""),
                        "NSE_SYMBOL": source.get("NSE_SYMBOL", ""),
                        "BSE_SYMBOL": source.get("BSE_SYMBOL", ""),
                        "BSE_CODE": source.get("BSE_CODE", ""),
                        "PRICE_TICKER": ticker,
                        "status": "NO_YAHOO_DATA",
                        "reason": "Yahoo returned no historical OHLCV",
                    })
                continue

            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", ticker)
            df.to_csv(HISTORY / f"{safe}.csv")

            last = df.iloc[-1]
            close = df["Close"]

            row = matches.iloc[0].to_dict()
            row.update({
                "status": "OK",
                "price_date": str(df.index[-1].date()),
                "latest_price": float(last["Close"]),
                "previous_close": (
                    float(df["Close"].iloc[-2])
                    if len(df) >= 2 else None
                ),
                "open": float(last["Open"]),
                "day_high": float(last["High"]),
                "day_low": float(last["Low"]),
                "volume": float(last["Volume"]),
                "52w_high": float(close.tail(252).max()),
                "52w_low": float(close.tail(252).min()),
                "return_1d_pct": (
                    (close.iloc[-1] / close.iloc[-2] - 1) * 100
                    if len(close) >= 2 else None
                ),
                "return_1w_pct": (
                    (close.iloc[-1] / close.iloc[-6] - 1) * 100
                    if len(close) >= 6 else None
                ),
                "return_1m_pct": (
                    (close.iloc[-1] / close.iloc[-22] - 1) * 100
                    if len(close) >= 22 else None
                ),
                "return_3m_pct": (
                    (close.iloc[-1] / close.iloc[-64] - 1) * 100
                    if len(close) >= 64 else None
                ),
                "return_6m_pct": (
                    (close.iloc[-1] / close.iloc[-127] - 1) * 100
                    if len(close) >= 127 else None
                ),
                "return_1y_pct": (
                    (close.iloc[-1] / close.iloc[0] - 1) * 100
                    if len(close) > 1 else None
                ),
            })
            rows.append(row)

        # Checkpoint after every batch.
        pd.DataFrame(rows).to_csv(
            OUT / "price_ohlcv_checkpoint.csv", index=False
        )
        pd.DataFrame(errors).to_csv(
            OUT / "price_collection_errors.csv", index=False
        )

    final = pd.DataFrame(rows)
    final.to_csv(OUT / "price_ohlcv.csv", index=False)

    pd.DataFrame(errors).to_csv(
        OUT / "price_collection_errors.csv", index=False
    )

    print("\nDONE")
    print(f"Successful: {len(final)}")
    print(f"Unavailable: {len(errors)}")
    print(f"Output: {OUT / 'price_ohlcv.csv'}")


if __name__ == "__main__":
    main()
