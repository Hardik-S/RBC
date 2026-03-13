from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def infer_bars_per_day(index: pd.DatetimeIndex, fallback: int = 78) -> int:
    """Infer median intraday bar count per day from a datetime index."""
    if len(index) < 2:
        return fallback
    ny_index = index.tz_convert("America/New_York")
    counts = pd.Series(ny_index.date).value_counts()
    if counts.empty:
        return fallback
    return int(max(1, counts.median()))


def add_microstructure_features(
    df: pd.DataFrame,
    rv_window: int = 26,
    liquidity_window: int = 26,
    profile_lookback_days: int = 5,
) -> pd.DataFrame:
    """
    Add execution-relevant features:
    - Realized volatility
    - Rolling volume profile
    - Liquidity proxies
    """
    enriched = df.copy()
    bars_per_day = infer_bars_per_day(enriched.index)
    annualizer = np.sqrt(252 * bars_per_day)

    enriched["log_return"] = np.log(enriched["close"]).diff()
    enriched["realized_vol"] = (
        enriched["log_return"]
        .rolling(rv_window, min_periods=max(5, rv_window // 2))
        .std()
        .mul(annualizer)
    )

    enriched["dollar_volume"] = enriched["close"] * enriched["volume"]
    enriched["rolling_volume"] = enriched["volume"].rolling(liquidity_window, min_periods=3).mean()
    enriched["rolling_dollar_volume"] = (
        enriched["dollar_volume"].rolling(liquidity_window, min_periods=3).mean()
    )

    # Simple spread proxy from bar range.
    enriched["spread_proxy"] = (
        (enriched["high"] - enriched["low"]) / enriched["close"].replace(0.0, np.nan)
    )

    # Amihud-style intraday illiquidity proxy.
    enriched["amihud_illiq"] = (
        enriched["log_return"].abs() / enriched["dollar_volume"].replace(0.0, np.nan)
    ).rolling(liquidity_window, min_periods=3).mean()

    # Impact proxy: price movement per square-root volume.
    enriched["impact_proxy"] = (
        enriched["log_return"].abs() / np.sqrt(enriched["volume"].replace(0.0, np.nan))
    ).rolling(liquidity_window, min_periods=3).mean()

    enriched["relative_volume"] = enriched["volume"] / enriched["rolling_volume"].replace(0.0, np.nan)

    ny_index = enriched.index.tz_convert("America/New_York")
    enriched["date_key"] = ny_index.date
    enriched["time_bucket"] = ny_index.strftime("%H:%M")

    # Rolling estimate of expected volume at each intraday bucket.
    enriched["volume_profile_est"] = enriched.groupby("time_bucket")["volume"].transform(
        lambda s: s.shift(1).rolling(profile_lookback_days, min_periods=2).mean()
    )
    fallback_profile = enriched.groupby("time_bucket")["volume"].transform("mean")
    enriched["volume_profile_est"] = (
        enriched["volume_profile_est"]
        .fillna(fallback_profile)
        .fillna(enriched["volume"].median())
    )

    enriched["profile_day_total"] = (
        enriched.groupby("date_key")["volume_profile_est"].transform("sum").replace(0.0, np.nan)
    )
    bars_in_day = enriched.groupby("date_key")["volume"].transform("count").replace(0.0, np.nan)
    enriched["expected_volume_share"] = (
        enriched["volume_profile_est"] / enriched["profile_day_total"]
    ).fillna(1.0 / bars_in_day)
    enriched["expected_volume_share"] = enriched["expected_volume_share"].fillna(0.0)

    day_total_volume = enriched.groupby("date_key")["volume"].transform("sum").replace(0.0, np.nan)
    enriched["realized_volume_share"] = (enriched["volume"] / day_total_volume).fillna(0.0)

    return enriched


def build_feature_panel(universe_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Add features for each ticker in a universe dictionary."""
    return {ticker: add_microstructure_features(df) for ticker, df in universe_data.items()}
