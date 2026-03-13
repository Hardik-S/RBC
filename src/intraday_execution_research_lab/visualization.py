from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
import pandas as pd

from .backtest import ExecutionResult

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_price_vol_urgency(session_df: pd.DataFrame, ticker: str, output_path: Path) -> None:
    """Chart price, realized vol, regime probability, and urgency for the execution day."""
    _ensure_parent(output_path)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(session_df.index, session_df["close"], color="black", linewidth=1.4, label="Price")
    axes[0].set_title(f"{ticker} Intraday Price")
    axes[0].set_ylabel("Price")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper left")

    axes[1].plot(
        session_df.index,
        session_df["realized_vol"],
        color="#0f766e",
        linewidth=1.4,
        label="Realized Vol (ann.)",
    )
    ax2 = axes[1].twinx()
    ax2.plot(
        session_df.index,
        session_df["p_high_vol"],
        color="#c2410c",
        linewidth=1.2,
        label="P(High Vol)",
    )
    ax2.plot(session_df.index, session_df["urgency"], color="#1d4ed8", linewidth=1.2, label="Urgency")

    axes[1].set_title(f"{ticker} Vol Regime and Urgency")
    axes[1].set_ylabel("Realized Vol")
    ax2.set_ylabel("Probability / Urgency")
    axes[1].grid(alpha=0.25)

    lines_1, labels_1 = axes[1].get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_cumulative_execution(
    results: Dict[str, ExecutionResult],
    ticker: str,
    output_path: Path,
) -> None:
    """Chart cumulative completion profile by execution strategy."""
    _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(11, 5))

    first_result = next(iter(results.values()))
    time_index = first_result.fills.index
    for strategy, result in results.items():
        cumulative = result.executed_schedule.cumsum()
        completion = cumulative / max(cumulative[-1], 1)
        ax.plot(time_index, completion, linewidth=1.8, label=strategy)

    ax.set_title(f"{ticker} Cumulative Parent Completion by Strategy")
    ax.set_ylabel("Completion Ratio")
    ax.set_xlabel("Timestamp")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_slippage_summary(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Compare slippage vs arrival and VWAP across ticker/strategy combinations."""
    _ensure_parent(output_path)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    labels = metrics_df["ticker"] + "-" + metrics_df["strategy"]
    x = range(len(metrics_df))

    axes[0].bar(x, metrics_df["slippage_arrival_bps"], color="#1d4ed8")
    axes[0].axhline(0.0, color="black", linewidth=1.0)
    axes[0].set_title("Slippage vs Arrival (bps)")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, metrics_df["slippage_vwap_bps"], color="#0f766e")
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_title("Slippage vs Session VWAP (bps)")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
