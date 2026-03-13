from __future__ import annotations

import math
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intraday_execution_research_lab.black_scholes import (  # noqa: E402
    black_scholes_price,
    implied_volatility,
)


class BlackScholesTests(unittest.TestCase):
    def test_call_put_parity(self) -> None:
        spot = 100.0
        strike = 100.0
        rate = 0.03
        vol = 0.20
        t = 1.0
        call = black_scholes_price(spot, strike, rate, vol, t, "call")
        put = black_scholes_price(spot, strike, rate, vol, t, "put")

        lhs = call - put
        rhs = spot - strike * math.exp(-rate * t)
        self.assertAlmostEqual(lhs, rhs, places=8)

    def test_implied_vol_recovery(self) -> None:
        spot = 250.0
        strike = 255.0
        rate = 0.02
        true_vol = 0.27
        t = 0.5

        market_price = black_scholes_price(spot, strike, rate, true_vol, t, "call")
        solved = implied_volatility(
            market_price=market_price,
            spot=spot,
            strike=strike,
            rate=rate,
            time_to_expiry=t,
            option_type="call",
        )
        self.assertAlmostEqual(solved, true_vol, places=6)


if __name__ == "__main__":
    unittest.main()
