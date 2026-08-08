name: Collect Price and OHLCV

on:
  workflow_dispatch:

jobs:
  collect:
    runs-on: ubuntu-latest
    timeout-minutes: 360

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pandas yfinance

      - name: Collect price and 1Y OHLCV
        run: |
          python scripts/collect_price_ohlcv.py

      - name: Upload price data
        uses: actions/upload-artifact@v4
        with:
          name: price-ohlcv-data
          path: |
            data/price_ohlcv.csv
            data/price_ohlcv_checkpoint.csv
            data/price_collection_errors.csv
            data/ohlcv_1y/
          retention-days: 7
