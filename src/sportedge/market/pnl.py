"""Historical market/model alignment and simple P&L simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sportedge.betting.strategy import kelly_stake


@dataclass(frozen=True)
class Trade:
    timestamp: int
    price: float
    model_p: float
    edge: float
    stake: float
    shares: float
    pnl: float


def align_model_prices(
    states: pd.DataFrame,
    prices: pd.DataFrame,
    tolerance_seconds: int = 600,
) -> pd.DataFrame:
    """As-of join model states to the most recent market price."""
    required_state_cols = {"timestamp", "model_p"}
    required_price_cols = {"timestamp", "price"}
    if not required_state_cols.issubset(states.columns):
        raise ValueError(f"states must include {sorted(required_state_cols)}")
    if not required_price_cols.issubset(prices.columns):
        raise ValueError(f"prices must include {sorted(required_price_cols)}")

    left = states.sort_values("timestamp").copy()
    right = prices.sort_values("timestamp")[["timestamp", "price"]].copy()
    aligned = pd.merge_asof(
        left,
        right,
        on="timestamp",
        direction="backward",
        tolerance=tolerance_seconds,
    )
    aligned = aligned.dropna(subset=["price"]).copy()
    aligned["edge"] = aligned["model_p"] - aligned["price"]
    return aligned


def simulate_buy_and_hold(
    aligned: pd.DataFrame,
    token_won: bool,
    min_edge: float = 0.04,
    bankroll: float = 100.0,
    kelly_fraction: float = 0.25,
    max_stake: float = 5.0,
    cooldown_seconds: int = 600,
) -> list[Trade]:
    """Buy when edge clears threshold, then value each entry at resolution."""
    trades: list[Trade] = []
    next_allowed_ts = -np.inf
    settlement = 1.0 if token_won else 0.0

    for row in aligned.sort_values("timestamp").itertuples(index=False):
        timestamp = int(row.timestamp)
        price = float(row.price)
        model_p = float(row.model_p)
        edge = float(row.edge)
        if timestamp < next_allowed_ts or edge < min_edge:
            continue
        stake = kelly_stake(model_p, price, bankroll, kelly_fraction, max_stake)
        if stake <= 0.0:
            continue
        shares = stake / price
        pnl = shares * settlement - stake
        trades.append(
            Trade(
                timestamp=timestamp,
                price=price,
                model_p=model_p,
                edge=edge,
                stake=stake,
                shares=shares,
                pnl=pnl,
            )
        )
        next_allowed_ts = timestamp + cooldown_seconds
    return trades


def trades_frame(trades: list[Trade]) -> pd.DataFrame:
    return pd.DataFrame([trade.__dict__ for trade in trades])
