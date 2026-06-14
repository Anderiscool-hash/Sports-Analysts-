import pandas as pd

from sportedge.betting.executor import Fill
from sportedge.betting.paper import PaperLedger
from sportedge.betting.report import (
    build_paper_report,
    collect_all_settlements,
    collect_espn_settlements,
    collect_marks,
    settlement_for_selection,
)
from sportedge.types import LiveDetail


def test_settlement_for_selection_basketball():
    detail = LiveDetail(
        sport="basketball",
        league="nba",
        home_team="Lakers",
        away_team="Celtics",
        home_score=101,
        away_score=99,
        status="post",
    )

    assert settlement_for_selection(detail, "Lakers") == 1.0
    assert settlement_for_selection(detail, "Celtics") == 0.0


def test_collect_marks_reads_current_prices():
    fills = pd.DataFrame({"token_id": ["tok1", "tok2", "tok1"]})

    class _Client:
        def get_price(self, token_id, side):  # noqa: ARG002
            return {"tok1": 0.55, "tok2": None}[token_id]

    assert collect_marks(fills, _Client()) == {"tok1": 0.55}


def test_collect_espn_settlements_from_final_game(monkeypatch):
    fills = pd.DataFrame(
        [
            {
                "token_id": "tok",
                "event_id": "401",
                "sport": "basketball",
                "league": "nba",
                "selected_team": "Lakers",
            }
        ]
    )
    detail = LiveDetail(
        sport="basketball",
        league="nba",
        home_team="Lakers",
        away_team="Celtics",
        home_score=101,
        away_score=99,
        status="post",
    )
    monkeypatch.setattr("sportedge.betting.report.get_game_detail", lambda *args: detail)

    assert collect_espn_settlements(fills) == {"tok": 1.0}


def test_collect_all_settlements_merges_replay_and_espn(tmp_path, monkeypatch):
    training = tmp_path / "training.parquet"
    pd.DataFrame({"game_id": ["g1"], "home_win": [1]}).to_parquet(training, index=False)
    fills = pd.DataFrame(
        [
            {"token_id": "replay", "event_id": "g1", "selected_team": "home"},
            {
                "token_id": "espn",
                "event_id": "401",
                "sport": "basketball",
                "league": "nba",
                "selected_team": "Lakers",
            },
        ]
    )
    detail = LiveDetail(
        sport="basketball",
        league="nba",
        home_team="Lakers",
        away_team="Celtics",
        home_score=101,
        away_score=99,
        status="post",
    )
    monkeypatch.setattr("sportedge.betting.report.get_game_detail", lambda *args: detail)

    settlements = collect_all_settlements(fills, replay_training_path=str(training))

    assert settlements == {"replay": 1.0, "espn": 1.0}


def test_build_paper_report_summarizes_ledger(tmp_path, monkeypatch):
    path = tmp_path / "paper.parquet"
    ledger = PaperLedger(str(path))
    ledger.append(
        Fill(
            ts=1.0,
            side="BUY",
            size=5.0,
            price=0.50,
            model_p=0.70,
            edge=0.20,
            mode="paper",
            token_id="tok",
            event_id="401",
            sport="basketball",
            league="nba",
            home_team="Lakers",
            away_team="Celtics",
            selected_team="Lakers",
        )
    )
    detail = LiveDetail(
        sport="basketball",
        league="nba",
        home_team="Lakers",
        away_team="Celtics",
        home_score=101,
        away_score=99,
        status="post",
    )
    monkeypatch.setattr("sportedge.betting.report.get_game_detail", lambda *args: detail)

    class _Client:
        def get_price(self, token_id, side):  # noqa: ARG002
            return 0.60

    report = build_paper_report(str(path), client=_Client())

    assert report.summary["realized_pnl"] == 5.0
    assert report.summary["open_positions"] == 0
    assert report.settlements == {"tok": 1.0}
