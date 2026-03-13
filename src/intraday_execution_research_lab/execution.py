from __future__ import annotations

import numpy as np


def _normalized_weights(weights: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    if clipped.size == 0:
        raise ValueError("weights must not be empty")
    total = clipped.sum()
    if total <= 0:
        return np.full(clipped.shape[0], 1.0 / clipped.shape[0], dtype=float)
    return clipped / total


def _shares_from_weights(weights: np.ndarray, total_qty: int) -> np.ndarray:
    if total_qty <= 0:
        raise ValueError("total_qty must be positive")
    normalized = _normalized_weights(weights)
    raw = normalized * total_qty
    shares = np.floor(raw).astype(int)
    remainder = total_qty - shares.sum()
    if remainder > 0:
        fractional = raw - shares
        bump_idx = np.argsort(fractional)[-remainder:]
        shares[bump_idx] += 1
    return shares


def twap_schedule(num_bars: int, total_qty: int) -> np.ndarray:
    """Equal-sized execution schedule."""
    if num_bars <= 0:
        raise ValueError("num_bars must be positive")
    return _shares_from_weights(np.ones(num_bars), total_qty)


def vwap_schedule(volume_profile: np.ndarray, total_qty: int) -> np.ndarray:
    """Schedule proportional to expected intraday volume curve."""
    return _shares_from_weights(volume_profile, total_qty)


def adaptive_schedule(
    total_qty: int,
    urgency: np.ndarray,
    volume_profile: np.ndarray,
) -> np.ndarray:
    """
    Adaptive schedule using urgency from Bayesian regime probabilities.

    Mechanics:
    - Start from a blend of neutral remaining-quantity pace and VWAP target.
    - Scale each bar with urgency to front-load in stressed regimes.
    """
    urg = np.clip(np.asarray(urgency, dtype=float), 0.01, 0.99)
    if urg.size == 0:
        raise ValueError("urgency must not be empty")
    vol_w = _normalized_weights(np.asarray(volume_profile, dtype=float))

    num_bars = urg.size
    schedule = np.zeros(num_bars, dtype=int)
    remaining = int(total_qty)

    for i in range(num_bars):
        bars_left = num_bars - i
        if bars_left == 1:
            schedule[i] = remaining
            break

        neutral_pace = remaining / bars_left
        vwap_hint = total_qty * vol_w[i]
        urgency_multiplier = 0.60 + 1.40 * urg[i]  # approximately [0.61, 1.99]

        target = (0.55 * neutral_pace + 0.45 * vwap_hint) * urgency_multiplier
        max_now = max(0, remaining - (bars_left - 1))  # keep at least 1 share for each remaining bar
        qty = int(np.clip(np.floor(target), 0, max_now))
        schedule[i] = qty
        remaining -= qty

    if remaining > 0:
        schedule[-1] += remaining
    return schedule
