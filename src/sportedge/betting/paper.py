"""Persistent paper-trading ledger and P&L helpers."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from sportedge.betting.executor import Fill

LEDGER_COLUMNS = [
    "ts",
    "side",
    "size",
    "price",
    "model_p",
    "edge",
    "mode",
    "token_id",
    "event_id",
    "sport",
    "league",
    "home_team",
    "away_team",
    "selected_team",
]


class PaperLedger:
    """Append-only paper fill store backed by parquet."""

    def __init__(self, path: str = "data/cache/paper_ledger.parquet") -> None:
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=LEDGER_COLUMNS)
        df = pd.read_parquet(self.path)
        for col in LEDGER_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[LEDGER_COLUMNS].copy()

    def append(self, fill: Fill) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        current = self.load()
        row = pd.DataFrame([asdict(fill)], columns=LEDGER_COLUMNS)
        pd.concat([current, row], ignore_index=True).to_parquet(self.path, index=False)

    def summary(
        self,
        marks: dict[str, float] | None = None,
        settlements: dict[str, float] | None = None,
    ) -> dict[str, float | int]:
        return summarize_fills(self.load(), marks=marks, settlements=settlements)

    def report(
        self,
        marks: dict[str, float] | None = None,
        settlements: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        return annotate_fills(self.load(), marks=marks, settlements=settlements)


def annotate_fills(
    fills: pd.DataFrame,
    marks: dict[str, float] | None = None,
    settlements: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Return fill-level paper P&L with mark/settlement status."""
    marks = marks or {}
    settlements = settlements or {}
    df = fills.copy()
    for col in LEDGER_COLUMNS:
        if col not in df.columns:
            df[col] = None
    if df.empty:
        return pd.DataFrame(
            columns=[
                *LEDGER_COLUMNS,
                "shares",
                "mark",
                "settlement",
                "is_settled",
                "pnl",
            ]
        )
    df = df[LEDGER_COLUMNS].copy()
    df["size"] = pd.to_numeric(df["size"], errors="coerce").fillna(0.0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["shares"] = df.apply(
        lambda row: float(row["size"]) / float(row["price"]) if float(row["price"]) > 0 else 0.0,
        axis=1,
    )
    df["mark"] = df["token_id"].map(lambda token: marks.get(str(token or "")))
    df["settlement"] = df["token_id"].map(lambda token: settlements.get(str(token or "")))
    df["is_settled"] = df["settlement"].notna()

    def _pnl(row) -> float:
        value = row["settlement"] if pd.notna(row["settlement"]) else row["mark"]
        if pd.isna(value):
            return 0.0
        return float(row["shares"]) * float(value) - float(row["size"])

    df["pnl"] = df.apply(_pnl, axis=1)
    return df


def summarize_fills(
    fills: pd.DataFrame,
    marks: dict[str, float] | None = None,
    settlements: dict[str, float] | None = None,
) -> dict[str, float | int]:
    """Summarize YES buys using marks for open P&L and settlements for resolved P&L."""
    marks = marks or {}
    settlements = settlements or {}
    empty_summary = {
        "fills": 0,
        "settled_fills": 0,
        "open_positions": 0,
        "staked": 0.0,
        "settled_staked": 0.0,
        "open_exposure": 0.0,
        "realized_pnl": 0.0,
        "realized_roi": 0.0,
        "unrealized_pnl": 0.0,
        "total_pnl": 0.0,
    }
    if fills.empty:
        return empty_summary

    df = annotate_fills(fills, marks=marks, settlements=settlements)
    df = df[df["price"] > 0.0].copy()
    if df.empty:
        return {
            **empty_summary,
        }

    settled = df[df["is_settled"]]
    open_df = df[~df["is_settled"]]
    realized = float(settled["pnl"].sum())
    settled_staked = float(settled["size"].sum())
    realized_roi = realized / settled_staked if settled_staked else 0.0
    unrealized = float(open_df[open_df["mark"].notna()]["pnl"].sum())
    open_exposure = float(open_df["size"].sum())
    open_positions = int(len(open_df))

    return {
        "fills": int(len(df)),
        "settled_fills": int(len(settled)),
        "open_positions": int(open_positions),
        "staked": float(df["size"].sum()),
        "settled_staked": float(settled_staked),
        "open_exposure": float(open_exposure),
        "realized_pnl": float(realized),
        "realized_roi": float(realized_roi),
        "unrealized_pnl": float(unrealized),
        "total_pnl": float(realized + unrealized),
    }


def collect_replay_settlements(
    fills: pd.DataFrame,
    training_path: str = "data/cache/training.parquet",
) -> dict[str, float]:
    """Infer settlements for offline replay fills from cached game outcomes.

    Replay rows use ``selected_team`` as ``home`` or ``away`` and ``event_id`` as
    the cached game id. That is enough to settle historical paper fills without
    an ESPN lookup or a live market quote.
    """
    required = {"token_id", "event_id", "selected_team"}
    if fills.empty or not required.issubset(fills.columns):
        return {}
    path = Path(training_path)
    if not path.exists():
        return {}
    training = pd.read_parquet(path, columns=["game_id", "home_win"])
    if training.empty:
        return {}
    outcomes = (
        training.dropna(subset=["game_id", "home_win"])
        .drop_duplicates(subset=["game_id"])
        .assign(game_id=lambda df: df["game_id"].astype(str), home_win=lambda df: pd.to_numeric(df["home_win"]))
        .set_index("game_id")["home_win"]
        .to_dict()
    )

    settlements: dict[str, float] = {}
    for row in fills.itertuples(index=False):
        token_id = str(getattr(row, "token_id", "") or "")
        event_id = str(getattr(row, "event_id", "") or "")
        selected = str(getattr(row, "selected_team", "") or "").strip().lower()
        if not token_id or event_id not in outcomes or selected not in {"home", "away"}:
            continue
        home_win = float(outcomes[event_id])
        settlements[token_id] = home_win if selected == "home" else 1.0 - home_win
    return settlements
