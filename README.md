# intraday-execution-research-lab

Python research project that models intraday execution for SPY and QQQ, tests TWAP/VWAP/adaptive strategies, and evaluates slippage under realistic microstructure assumptions.

Built as a portfolio-quality sample for an RBC Global Equities Quantitative Trading Analyst application.

## What This Project Covers
- Pulls recent intraday data for `SPY` and `QQQ` using `yfinance`
- Computes:
  - Realized volatility (annualized from intraday returns)
  - Rolling volume profile and expected volume share by time bucket
  - Liquidity proxies (spread proxy, Amihud-style illiquidity, impact proxy)
- Implements execution algorithms:
  - `TWAP`
  - `VWAP`
  - `Adaptive` strategy with Bayesian volatility-regime urgency
- Simulates parent-order execution and compares slippage vs:
  - Arrival price
  - Session VWAP
- Generates clean charts and CSV/text outputs
- Includes a compact Black-Scholes module with Greeks and implied volatility

## Repo Structure
```text
intraday-execution-research-lab/
├─ src/intraday_execution_research_lab/
│  ├─ data.py
│  ├─ features.py
│  ├─ bayesian.py
│  ├─ execution.py
│  ├─ backtest.py
│  ├─ visualization.py
│  └─ black_scholes.py
├─ scripts/
│  └─ run_research.py
├─ tests/
│  └─ test_black_scholes.py
├─ intraday_output/
│  ├─ charts/
│  └─ reports/
├─ requirements.txt
└─ README.md
```

## Quick Start
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
python scripts/run_research.py --tickers SPY QQQ --period 5d --interval 5m
```

## Output Artifacts
Running the script creates:
- `intraday_output/charts/`
  - `spy_regime_urgency.png`
  - `qqq_regime_urgency.png`
  - `spy_execution_completion.png`
  - `qqq_execution_completion.png`
  - `slippage_comparison.png`
- `intraday_output/reports/`
  - `strategy_slippage_summary.csv`
  - `{ticker}_{strategy}_fills.csv`
  - `summary.txt`

## Key Modeling Assumptions
- Intraday costs are estimated from bar-based proxies (no full order book).
- Participation cap limits aggressiveness except for final forced completion.
- Bayesian regime updater uses two volatility states and simple transition probabilities.
- Parent order size is set to ~8% of session volume (minimum 1,000 shares) for comparability.

## Why This Is Relevant For Equities Quant Trading
- Demonstrates practical execution benchmarking and implementation shortfall analysis.
- Connects volatility/liquidity state detection to dynamic execution urgency.
- Shows production-style research structure: modular code, reproducible outputs, and clear documentation.
