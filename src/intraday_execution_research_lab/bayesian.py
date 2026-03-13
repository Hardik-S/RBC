from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class BayesianVolatilityRegimeUpdater:
    """
    Two-state Bayesian volatility model:
    - Low-vol regime
    - High-vol regime
    """

    low_mean: float
    low_std: float
    high_mean: float
    high_std: float
    p_stay_low: float = 0.92
    p_stay_high: float = 0.90
    prior_high: float = 0.30

    def _transition_prior(self, p_high_prev: float) -> float:
        return p_high_prev * self.p_stay_high + (1.0 - p_high_prev) * (1.0 - self.p_stay_low)

    def update(self, observation: float, p_high_prev: float) -> float:
        if not np.isfinite(observation):
            return float(np.clip(p_high_prev, 0.01, 0.99))

        prior_high = self._transition_prior(p_high_prev)
        high_like = norm.pdf(observation, loc=self.high_mean, scale=max(self.high_std, 1e-6))
        low_like = norm.pdf(observation, loc=self.low_mean, scale=max(self.low_std, 1e-6))

        numerator = prior_high * high_like
        denominator = numerator + (1.0 - prior_high) * low_like
        if denominator <= 0:
            return float(np.clip(prior_high, 0.01, 0.99))

        posterior_high = numerator / denominator
        return float(np.clip(posterior_high, 0.01, 0.99))


def calibrate_regime_model(realized_vol: pd.Series) -> BayesianVolatilityRegimeUpdater:
    """
    Calibrate simple regime distributions by splitting around median realized vol.
    """
    clean = realized_vol.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty or clean.shape[0] < 12:
        return BayesianVolatilityRegimeUpdater(
            low_mean=0.14,
            low_std=0.05,
            high_mean=0.32,
            high_std=0.10,
            prior_high=0.35,
        )

    median = clean.median()
    low = clean[clean <= median]
    high = clean[clean > median]
    if low.empty or high.empty:
        return BayesianVolatilityRegimeUpdater(
            low_mean=float(clean.quantile(0.35)),
            low_std=float(clean.std(ddof=0) + 1e-6),
            high_mean=float(clean.quantile(0.65)),
            high_std=float(clean.std(ddof=0) + 1e-6),
            prior_high=0.50,
        )

    return BayesianVolatilityRegimeUpdater(
        low_mean=float(low.mean()),
        low_std=float(low.std(ddof=0) + 1e-6),
        high_mean=float(high.mean()),
        high_std=float(high.std(ddof=0) + 1e-6),
        prior_high=float((clean > median).mean()),
    )


def posterior_high_vol_probability(
    realized_vol: pd.Series,
    model: BayesianVolatilityRegimeUpdater | None = None,
) -> pd.Series:
    """
    Run forward Bayesian updates and return p(high-vol regime) time series.
    """
    if model is None:
        model = calibrate_regime_model(realized_vol)

    aligned = realized_vol.replace([np.inf, -np.inf], np.nan).ffill()
    p = model.prior_high
    out = []
    for value in aligned:
        p = model.update(float(value) if np.isfinite(value) else np.nan, p_high_prev=p)
        out.append(p)
    return pd.Series(out, index=realized_vol.index, name="p_high_vol")


def compute_execution_urgency(
    p_high_vol: pd.Series,
    spread_proxy: pd.Series,
    base_urgency: float = 0.30,
) -> pd.Series:
    """
    Convert regime probability + liquidity stress into [0, 1] urgency.

    Higher urgency implies stronger front-loading of execution.
    """
    spread_rank = spread_proxy.fillna(spread_proxy.median()).rank(pct=True).fillna(0.50)
    urgency = base_urgency + 0.45 * p_high_vol.fillna(0.50) + 0.20 * spread_rank
    urgency = np.clip(urgency, 0.05, 0.99)
    return pd.Series(urgency, index=p_high_vol.index, name="urgency")
