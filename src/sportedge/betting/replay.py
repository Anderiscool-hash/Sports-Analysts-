"""Offline paper replay over aligned model/market rows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sportedge.betting.executor import PaperExecutor
from sportedge.betting.strategy import Strategy
from sportedge.config import Config
from sportedge.market.edge import BottomDetector


@dataclass(frozen=True)
class ReplaySummary:
    rows_seen: int
    fills: int
    staked: float


@dataclass(frozen=True)
class ReplayFileSummary:
    path: str
    rows_seen: int = 0
    fills: int = 0
    staked: float = 0.0
    skipped: bool = False
    reason: str = ""


@dataclass(frozen=True)
class ReplayEvaluationSummary:
    rows_seen: int
    fills: int
    staked: float
    pnl: float
    wins: int

    @property
    def roi(self) -> float:
        return self.pnl / self.staked if self.staked else 0.0


def replay_aligned_rows(
    aligned: pd.DataFrame,
    cfg: Config,
    ledger_path: str,
    token_id: str,
    selected_team: str = "",
    event_id: str = "",
    sport: str = "basketball",
    league: str = "nba",
    home_team: str = "",
    away_team: str = "",
) -> ReplaySummary:
    """Replay aligned rows through the same paper strategy used live.

    Required columns: ``timestamp``, ``model_p``, ``price``. If ``edge`` is missing
    it is computed as ``model_p - price``.
    """
    required = {"timestamp", "model_p", "price"}
    if not required.issubset(aligned.columns):
        raise ValueError(f"aligned rows must include {sorted(required)}")
    rows = aligned.sort_values("timestamp").copy()
    if "edge" not in rows.columns:
        rows["edge"] = rows["model_p"] - rows["price"]

    detector = BottomDetector(cfg.edge.dip_threshold, cfg.edge.min_edge, cfg.edge.rebound_ticks)
    strategy = Strategy(
        cfg.edge.min_edge,
        cfg.kelly_fraction,
        cfg.max_stake,
        cfg.bankroll,
        cfg.edge.min_price,
        cfg.edge.max_price,
    )
    executor = PaperExecutor(ledger_path=ledger_path)
    exposure = 0.0
    metadata = {
        "event_id": event_id,
        "sport": sport,
        "league": league,
        "home_team": home_team,
        "away_team": away_team,
        "selected_team": selected_team,
    }

    for row in rows.itertuples(index=False):
        price = float(row.price)
        model_p = float(row.model_p)
        signal = detector.update(price, model_p)
        order = strategy.decide(signal)
        if order is None:
            continue
        if exposure + order.size > cfg.bankroll:
            continue
        fill = executor.place(order, token_id, metadata=metadata)
        exposure += fill.size

    return ReplaySummary(rows_seen=int(len(rows)), fills=len(executor.fills), staked=executor.staked)


def evaluate_aligned_rows(aligned: pd.DataFrame, cfg: Config) -> ReplayEvaluationSummary:
    """Evaluate aligned rows in memory without writing to the paper ledger."""
    required = {"timestamp", "model_p", "price", "token_won"}
    if not required.issubset(aligned.columns):
        raise ValueError(f"aligned rows must include {sorted(required)}")
    rows = aligned.sort_values("timestamp").copy()
    if "edge" not in rows.columns:
        rows["edge"] = rows["model_p"] - rows["price"]

    detector = BottomDetector(cfg.edge.dip_threshold, cfg.edge.min_edge, cfg.edge.rebound_ticks)
    strategy = Strategy(
        cfg.edge.min_edge,
        cfg.kelly_fraction,
        cfg.max_stake,
        cfg.bankroll,
        cfg.edge.min_price,
        cfg.edge.max_price,
    )
    fills = 0
    staked = 0.0
    pnl = 0.0
    wins = 0
    exposure = 0.0
    token_won = float(pd.to_numeric(rows["token_won"], errors="coerce").dropna().iloc[-1])

    for row in rows.itertuples(index=False):
        signal = detector.update(float(row.price), float(row.model_p))
        order = strategy.decide(signal)
        if order is None:
            continue
        if exposure + order.size > cfg.bankroll:
            continue
        fills += 1
        staked += order.size
        exposure += order.size
        pnl += (order.size / order.price) * token_won - order.size
        wins += 1 if token_won == 1.0 else 0
    return ReplayEvaluationSummary(
        rows_seen=int(len(rows)),
        fills=fills,
        staked=float(staked),
        pnl=float(pnl),
        wins=wins,
    )


def evaluate_file(path: str | Path, cfg: Config) -> ReplayFileSummary:
    """Evaluate one aligned parquet without mutating a ledger."""
    p = Path(path)
    try:
        aligned = pd.read_parquet(p)
    except Exception as exc:  # noqa: BLE001
        return ReplayFileSummary(str(p), skipped=True, reason=f"read failed: {exc}")
    required = {"timestamp", "model_p", "price", "token_won"}
    if not required.issubset(aligned.columns):
        return ReplayFileSummary(str(p), skipped=True, reason="missing evaluation columns")
    try:
        summary = evaluate_aligned_rows(aligned, cfg)
    except Exception as exc:  # noqa: BLE001
        return ReplayFileSummary(str(p), skipped=True, reason=f"evaluation failed: {exc}")
    return ReplayFileSummary(
        str(p),
        rows_seen=summary.rows_seen,
        fills=summary.fills,
        staked=summary.staked,
    )


def evaluate_directory(
    directory: str | Path,
    cfg: Config,
    pattern: str = "aligned*.parquet",
) -> ReplayEvaluationSummary:
    """Evaluate every matching aligned parquet in memory."""
    rows_seen = 0
    fills = 0
    staked = 0.0
    pnl = 0.0
    wins = 0
    for path in sorted(Path(directory).glob(pattern)):
        try:
            summary = evaluate_aligned_rows(pd.read_parquet(path), cfg)
        except Exception:  # noqa: BLE001 - incompatible files are ignored like replay_directory
            continue
        rows_seen += summary.rows_seen
        fills += summary.fills
        staked += summary.staked
        pnl += summary.pnl
        wins += summary.wins
    return ReplayEvaluationSummary(rows_seen, fills, staked, pnl, wins)


def replay_file(
    path: str | Path,
    cfg: Config,
    ledger_path: str,
    token_id: str = "",
) -> ReplayFileSummary:
    """Replay one parquet if it has the aligned replay schema; otherwise skip."""
    p = Path(path)
    try:
        aligned = pd.read_parquet(p)
    except Exception as exc:  # noqa: BLE001
        return ReplayFileSummary(str(p), skipped=True, reason=f"read failed: {exc}")

    required = {"timestamp", "model_p", "price"}
    if not required.issubset(aligned.columns):
        return ReplayFileSummary(str(p), skipped=True, reason="missing aligned columns")
    inferred_source = aligned.get("token_id", aligned.get("token_outcome", pd.Series(["REPLAY"])))
    inferred_token = token_id or str(inferred_source.iloc[0])
    if not inferred_token or inferred_token == "nan":
        inferred_token = p.stem
    event_id = str(aligned.get("game_id", pd.Series([""])).iloc[0] or "")
    selected_team = str(aligned.get("token_outcome", pd.Series([""])).iloc[0] or "")
    summary = replay_aligned_rows(
        aligned,
        cfg,
        ledger_path=ledger_path,
        token_id=inferred_token,
        selected_team=selected_team,
        event_id=event_id,
    )
    return ReplayFileSummary(
        str(p),
        rows_seen=summary.rows_seen,
        fills=summary.fills,
        staked=summary.staked,
    )


def replay_directory(
    directory: str | Path,
    cfg: Config,
    ledger_path: str,
    pattern: str = "aligned*.parquet",
) -> list[ReplayFileSummary]:
    """Replay every matching aligned parquet in a directory."""
    root = Path(directory)
    return [replay_file(path, cfg, ledger_path) for path in sorted(root.glob(pattern))]
