from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intraday_execution_research_lab.backtest import (  # noqa: E402
    ParentOrder,
    compile_metrics,
    run_execution_experiment,
)
from intraday_execution_research_lab.bayesian import (  # noqa: E402
    compute_execution_urgency,
    posterior_high_vol_probability,
)
from intraday_execution_research_lab.data import fetch_universe_data  # noqa: E402
from intraday_execution_research_lab.features import add_microstructure_features  # noqa: E402
from intraday_execution_research_lab.visualization import (  # noqa: E402
    plot_cumulative_execution,
    plot_price_vol_urgency,
    plot_slippage_summary,
)


def select_latest_session(df: pd.DataFrame, min_bars: int = 40) -> pd.DataFrame:
    """Select latest day with at least min_bars observations."""
    if df.empty:
        raise ValueError("DataFrame is empty.")
    date_key = pd.Series(df.index.date, index=df.index)
    counts = date_key.value_counts()
    eligible = counts[counts >= min_bars]
    chosen_date = eligible.index.max() if not eligible.empty else counts.index.max()
    session = df.loc[date_key == chosen_date].copy()
    return session


def write_summary_report(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Write concise text report for quick recruiter-friendly review."""
    lines = []
    lines.append("Intraday Execution Research Lab - Summary")
    lines.append("=" * 44)
    lines.append("")

    grouped = metrics_df.groupby("strategy", as_index=False)[
        ["slippage_arrival_bps", "slippage_vwap_bps"]
    ].mean()
    grouped = grouped.sort_values("slippage_arrival_bps")
    lines.append("Average slippage across SPY/QQQ (buy parent orders):")
    for _, row in grouped.iterrows():
        lines.append(
            f"- {row['strategy']}: arrival={row['slippage_arrival_bps']:.2f} bps, "
            f"vwap={row['slippage_vwap_bps']:.2f} bps"
        )

    lines.append("")
    lines.append("Lower slippage is better for a buy order in this simulation.")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run intraday execution research workflow.")
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ"])
    parser.add_argument("--period", default="5d")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--participation-cap", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="intraday_output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    charts_dir = output_dir / "charts"
    reports_dir = output_dir / "reports"
    charts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    universe = fetch_universe_data(tickers=args.tickers, period=args.period, interval=args.interval)

    all_metrics = []
    for ticker, raw_df in universe.items():
        features = add_microstructure_features(raw_df)
        p_high = posterior_high_vol_probability(features["realized_vol"])
        urgency = compute_execution_urgency(p_high, features["spread_proxy"])
        features["p_high_vol"] = p_high
        features["urgency"] = urgency

        session = select_latest_session(features, min_bars=40)
        session_urgency = session["urgency"].fillna(session["urgency"].median()).fillna(0.50)

        session_volume = float(session["volume"].sum())
        parent_qty = max(1_000, int(0.08 * session_volume))
        order = ParentOrder(ticker=ticker, side="buy", quantity=parent_qty)

        results = run_execution_experiment(
            session_df=session,
            order=order,
            urgency=session_urgency,
            participation_cap=args.participation_cap,
            seed=args.seed,
        )
        metrics = compile_metrics(results)
        metrics.insert(0, "ticker", ticker)
        metrics.insert(1, "order_qty", parent_qty)
        all_metrics.append(metrics)

        # Save per-strategy fill logs.
        for strategy, result in results.items():
            safe = strategy.lower()
            result.fills.to_csv(reports_dir / f"{ticker.lower()}_{safe}_fills.csv", index=True)

        plot_price_vol_urgency(
            session_df=session,
            ticker=ticker,
            output_path=charts_dir / f"{ticker.lower()}_regime_urgency.png",
        )
        plot_cumulative_execution(
            results=results,
            ticker=ticker,
            output_path=charts_dir / f"{ticker.lower()}_execution_completion.png",
        )

    metrics_df = pd.concat(all_metrics, ignore_index=True).sort_values(["ticker", "strategy"])
    metrics_df.to_csv(reports_dir / "strategy_slippage_summary.csv", index=False)
    write_summary_report(metrics_df, reports_dir / "summary.txt")

    plot_slippage_summary(metrics_df, charts_dir / "slippage_comparison.png")

    print("Completed intraday execution research run.")
    print(f"Charts:  {charts_dir.resolve()}")
    print(f"Reports: {reports_dir.resolve()}")


if __name__ == "__main__":
    main()
