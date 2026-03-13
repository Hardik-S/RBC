from __future__ import annotations

import math
from typing import Dict, Tuple

from scipy.optimize import brentq
from scipy.stats import norm


def _validate(option_type: str) -> str:
    normalized = option_type.lower()
    if normalized not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    return normalized


def d1_d2(spot: float, strike: float, rate: float, vol: float, time_to_expiry: float) -> Tuple[float, float]:
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if vol <= 0 or time_to_expiry <= 0:
        raise ValueError("vol and time_to_expiry must be positive")
    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * time_to_expiry) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    return d1, d2


def black_scholes_price(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    time_to_expiry: float,
    option_type: str = "call",
) -> float:
    option_type = _validate(option_type)
    d1, d2 = d1_d2(spot=spot, strike=strike, rate=rate, vol=vol, time_to_expiry=time_to_expiry)
    discounted_strike = strike * math.exp(-rate * time_to_expiry)
    if option_type == "call":
        return spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2)
    return discounted_strike * norm.cdf(-d2) - spot * norm.cdf(-d1)


def delta(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    time_to_expiry: float,
    option_type: str = "call",
) -> float:
    option_type = _validate(option_type)
    d1, _ = d1_d2(spot=spot, strike=strike, rate=rate, vol=vol, time_to_expiry=time_to_expiry)
    if option_type == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1.0


def gamma(spot: float, strike: float, rate: float, vol: float, time_to_expiry: float) -> float:
    d1, _ = d1_d2(spot=spot, strike=strike, rate=rate, vol=vol, time_to_expiry=time_to_expiry)
    return norm.pdf(d1) / (spot * vol * math.sqrt(time_to_expiry))


def vega(spot: float, strike: float, rate: float, vol: float, time_to_expiry: float) -> float:
    d1, _ = d1_d2(spot=spot, strike=strike, rate=rate, vol=vol, time_to_expiry=time_to_expiry)
    return spot * norm.pdf(d1) * math.sqrt(time_to_expiry)


def theta(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    time_to_expiry: float,
    option_type: str = "call",
) -> float:
    option_type = _validate(option_type)
    d1, d2 = d1_d2(spot=spot, strike=strike, rate=rate, vol=vol, time_to_expiry=time_to_expiry)
    discounted_strike = strike * math.exp(-rate * time_to_expiry)
    first_term = -(spot * norm.pdf(d1) * vol) / (2.0 * math.sqrt(time_to_expiry))
    if option_type == "call":
        second_term = -rate * discounted_strike * norm.cdf(d2)
    else:
        second_term = rate * discounted_strike * norm.cdf(-d2)
    return first_term + second_term


def rho(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    time_to_expiry: float,
    option_type: str = "call",
) -> float:
    option_type = _validate(option_type)
    _, d2 = d1_d2(spot=spot, strike=strike, rate=rate, vol=vol, time_to_expiry=time_to_expiry)
    discounted_strike = strike * time_to_expiry * math.exp(-rate * time_to_expiry)
    if option_type == "call":
        return discounted_strike * norm.cdf(d2)
    return -discounted_strike * norm.cdf(-d2)


def greeks(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    time_to_expiry: float,
    option_type: str = "call",
) -> Dict[str, float]:
    return {
        "delta": delta(spot, strike, rate, vol, time_to_expiry, option_type=option_type),
        "gamma": gamma(spot, strike, rate, vol, time_to_expiry),
        "vega": vega(spot, strike, rate, vol, time_to_expiry),
        "theta": theta(spot, strike, rate, vol, time_to_expiry, option_type=option_type),
        "rho": rho(spot, strike, rate, vol, time_to_expiry, option_type=option_type),
    }


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
    option_type: str = "call",
    vol_lower: float = 1e-4,
    vol_upper: float = 5.0,
    max_iter: int = 200,
) -> float:
    """Solve for implied volatility with Brent's root-finder."""
    option_type = _validate(option_type)
    if market_price <= 0:
        raise ValueError("market_price must be positive")

    def objective(vol: float) -> float:
        return (
            black_scholes_price(
                spot=spot,
                strike=strike,
                rate=rate,
                vol=vol,
                time_to_expiry=time_to_expiry,
                option_type=option_type,
            )
            - market_price
        )

    return float(brentq(objective, vol_lower, vol_upper, maxiter=max_iter))
