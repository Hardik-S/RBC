from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd
import yfinance as yf


def fetch_intraday_data(
    ticker: str,
    period: str = "5d",
    interval: str = "5m",
) -> pd.DataFrame:
    """
    Pull recent intraday OHLCV data from Yahoo Finance.

    Notes:
    - Yahoo limits history for small intervals; defaults are intentionally recent.
    - Index is converted to America/New_York for session-level logic.
    """
    raw = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
        prepost=False,
        threads=False,
    )
    if raw.empty:
        raise ValueError(f"No data returned for ticker={ticker} period={period} interval={interval}.")

    frame = raw.rename(columns=lambda c: c.lower().replace(" ", "_")).copy()
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Ticker {ticker} is missing required columns: {sorted(missing)}")

    frame = frame.loc[:, ["open", "high", "low", "close", "volume"]]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()

    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    frame.index = frame.index.tz_convert("America/New_York")

    frame = frame.dropna(subset=["open", "high", "low", "close"])
    frame["volume"] = frame["volume"].fillna(0.0).astype(float)
    frame["ticker"] = ticker
    return frame


def fetch_universe_data(
    tickers: Iterable[str],
    period: str = "5d",
    interval: str = "5m",
) -> Dict[str, pd.DataFrame]:
    """Pull intraday data for a list of tickers."""
    output: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        output[ticker] = fetch_intraday_data(ticker=ticker, period=period, interval=interval)
    return output
