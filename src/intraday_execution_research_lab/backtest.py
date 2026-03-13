from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .execution import adaptive_schedule, twap_schedule, vwap_schedule


@dataclass
class ParentOrder:
    ticker: str
    side: str
    quantity: int

    @property
    def direction(self) -> int:
        side = self.side.lower()
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")
        return 1 if side == "buy" else -1


@dataclass
class ExecutionResult:
    strategy: str
    fills: pd.DataFrame
    planned_schedule: np.ndarray
    executed_schedule: np.ndarray
    metrics: Dict[str, float]


def simulate_schedule(
    df: pd.DataFrame,
    order: ParentOrder,
    strategy_name: str,
    planned_schedule: np.ndarray,
    participation_cap: float = 0.20,
    impact_coeff: float = 0.0015,
    spread_coeff: float = 0.35,
    seed: int = 42,
) -> ExecutionResult:
    """
    Simulate bar-by-bar execution with simple microstructure cost model.

    Assumptions:
    - Execution constrained by max participation per bar (except final bar forced completion).
    - Slippage increases with spread proxy and square-root participation.
    """
    if len(planned_schedule) != len(df):
        raise ValueError("planned_schedule length must match dataframe length")
    if order.quantity <= 0:
        raise ValueError("order quantity must be positive")

    rng = np.random.default_rng(seed)
    planned = np.asarray(planned_schedule, dtype=int)
    executed = np.zeros(len(df), dtype=int)
    fill_price = np.full(len(df), np.nan, dtype=float)

    carry = 0
    for i, (_, row) in enumerate(df.iterrows()):
        target_qty = int(planned[i] + carry)
        if target_qty < 0:
            target_qty = 0

        if i < len(df) - 1:
            cap_qty = int(np.floor(participation_cap * max(float(row["volume"]), 0.0)))
            exec_qty = min(target_qty, max(cap_qty, 0))
            carry = target_qty - exec_qty
        else:
            # Force parent completion at the end of window.
            exec_qty = target_qty
            carry = 0

        if exec_qty <= 0:
            continue

        mid = float(row["close"])
        spread = float(np.nan_to_num(row.get("spread_proxy", 0.0005), nan=0.0005, posinf=0.0005))
        illiq = float(np.nan_to_num(row.get("amihud_illiq", 0.0), nan=0.0, posinf=0.0))
        volume = max(float(row["volume"]), 1.0)
        participation = exec_qty / volume

        # Impact scales with sqrt(participation) and intraday illiquidity.
        impact_component = impact_coeff * np.sqrt(max(participation, 0.0)) * (
            1.0 + np.clip(illiq * 1_000_000.0, 0.0, 2.0)
        )
        spread_component = spread_coeff * spread
        noise = rng.normal(0.0, 0.00004)

        signed_cost = order.direction * (spread_component + impact_component + noise)
        fill_price[i] = mid * (1.0 + signed_cost)
        executed[i] = exec_qty

    fills = df.loc[:, ["close", "volume"]].copy()
    fills["planned_qty"] = planned
    fills["executed_qty"] = executed
    fills["fill_price"] = fill_price
    fills["notional"] = fills["executed_qty"] * fills["fill_price"].fillna(0.0)

    filled_qty = int(executed.sum())
    if filled_qty == 0:
        raise ValueError(f"No shares were executed for strategy={strategy_name}.")

    avg_fill_price = float(np.nansum(fill_price * executed) / filled_qty)
    arrival_price = float(df["close"].iloc[0])
    benchmark_vwap = float(np.average(df["close"], weights=np.clip(df["volume"], 1.0, None)))

    metrics = {
        "strategy": strategy_name,
        "filled_qty": float(filled_qty),
        "avg_fill_price": avg_fill_price,
        "arrival_price": arrival_price,
        "benchmark_vwap": benchmark_vwap,
        "slippage_arrival_bps": float(order.direction * (avg_fill_price / arrival_price - 1.0) * 1e4),
        "slippage_vwap_bps": float(order.direction * (avg_fill_price / benchmark_vwap - 1.0) * 1e4),
        "participation_rate": float(filled_qty / max(df["volume"].sum(), 1.0)),
    }

    return ExecutionResult(
        strategy=strategy_name,
        fills=fills,
        planned_schedule=planned,
        executed_schedule=executed,
        metrics=metrics,
    )


def run_execution_experiment(
    session_df: pd.DataFrame,
    order: ParentOrder,
    urgency: pd.Series,
    participation_cap: float = 0.20,
    seed: int = 42,
) -> Dict[str, ExecutionResult]:
    """Run TWAP, VWAP, and adaptive strategy for one parent order."""
    num_bars = len(session_df)
    if num_bars == 0:
        raise ValueError("session_df is empty")

    volume_profile = session_df["volume_profile_est"].fillna(session_df["volume"]).to_numpy(dtype=float)
    aligned_urgency = (
        urgency.reindex(session_df.index)
        .ffill()
        .bfill()
        .fillna(0.50)
        .to_numpy(dtype=float)
    )

    schedules = {
        "TWAP": twap_schedule(num_bars=num_bars, total_qty=order.quantity),
        "VWAP": vwap_schedule(volume_profile=volume_profile, total_qty=order.quantity),
        "Adaptive": adaptive_schedule(
            total_qty=order.quantity,
            urgency=aligned_urgency,
            volume_profile=volume_profile,
        ),
    }

    results: Dict[str, ExecutionResult] = {}
    for i, (name, schedule) in enumerate(schedules.items()):
        results[name] = simulate_schedule(
            df=session_df,
            order=order,
            strategy_name=name,
            planned_schedule=schedule,
            participation_cap=participation_cap,
            seed=seed + 13 * i,
        )
    return results


def compile_metrics(results: Dict[str, ExecutionResult]) -> pd.DataFrame:
    """Convert strategy metrics dictionary into a tidy DataFrame."""
    return pd.DataFrame([res.metrics for res in results.values()]).sort_values("strategy")
