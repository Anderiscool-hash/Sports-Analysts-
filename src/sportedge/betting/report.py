"""Paper ledger mark-to-market and settlement reporting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from sportedge.betting.paper import PaperLedger, collect_replay_settlements
from sportedge.data.espn_live import get_game_detail
from sportedge.market.kalshi import KalshiClient
from sportedge.types import LiveDetail


@dataclass(frozen=True)
class PaperReport:
    summary: dict[str, float | int]
    fills: pd.DataFrame
    marks: dict[str, float]
    settlements: dict[str, float]


def settlement_for_selection(detail: LiveDetail, selected_team: str) -> float | None:
    """Return binary settlement for a selected team/outcome once the game is final."""
    if detail.status != "post":
        return None
    selected = (selected_team or "").strip().lower()
    if not selected:
        return None
    if detail.sport == "soccer" and selected == "draw":
        return 1.0 if detail.home_score == detail.away_score else 0.0
    if selected == detail.home_team.lower():
        return 1.0 if detail.home_score > detail.away_score else 0.0
    if selected == detail.away_team.lower():
        return 1.0 if detail.away_score > detail.home_score else 0.0
    return None


def collect_marks(fills: pd.DataFrame, client: KalshiClient) -> dict[str, float]:
    marks: dict[str, float] = {}
    if fills.empty or "token_id" not in fills.columns:
        return marks
    for token_id in sorted({str(t) for t in fills["token_id"].dropna() if str(t)}):
        price = client.get_price(token_id, "BUY")
        if price is not None:
            marks[token_id] = price
    return marks


def collect_espn_settlements(fills: pd.DataFrame) -> dict[str, float]:
    settlements: dict[str, float] = {}
    required = {"token_id", "event_id", "sport", "league", "selected_team"}
    if fills.empty or not required.issubset(fills.columns):
        return settlements

    details: dict[tuple[str, str, str], LiveDetail | None] = {}
    for row in fills.itertuples(index=False):
        token_id = str(getattr(row, "token_id", "") or "")
        event_id = str(getattr(row, "event_id", "") or "")
        sport = str(getattr(row, "sport", "") or "")
        league = str(getattr(row, "league", "") or "")
        selected_team = str(getattr(row, "selected_team", "") or "")
        if not token_id or not event_id or not sport or not league:
            continue
        key = (sport, league, event_id)
        if key not in details:
            details[key] = get_game_detail(sport, league, event_id)
        detail = details[key]
        if detail is None:
            continue
        settlement = settlement_for_selection(detail, selected_team)
        if settlement is not None:
            settlements[token_id] = settlement
    return settlements


def collect_all_settlements(
    fills: pd.DataFrame,
    replay_training_path: str = "data/cache/training.parquet",
    settle_from_espn: bool = True,
) -> dict[str, float]:
    """Collect every settlement source used by paper reports and the live gate."""
    settlements = collect_replay_settlements(fills, replay_training_path)
    if settle_from_espn:
        settlements.update(collect_espn_settlements(fills))
    return settlements


def build_paper_report(
    ledger_path: str = "data/cache/paper_ledger.parquet",
    client: KalshiClient | None = None,
    settle_from_espn: bool = True,
    replay_training_path: str = "data/cache/training.parquet",
) -> PaperReport:
    ledger = PaperLedger(ledger_path)
    fills = ledger.load()
    client = client or KalshiClient()
    marks = collect_marks(fills, client)
    settlements = collect_all_settlements(
        fills,
        replay_training_path=replay_training_path,
        settle_from_espn=settle_from_espn,
    )
    return PaperReport(
        summary=ledger.summary(marks=marks, settlements=settlements),
        fills=ledger.report(marks=marks, settlements=settlements),
        marks=marks,
        settlements=settlements,
    )
