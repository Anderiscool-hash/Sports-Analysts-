import pandas as pd

from sportedge.data import espn_scraper
from sportedge.data.nba_scraper import TRAINING_COLUMNS


def _event() -> dict:
    return {
        "id": "401656359",
        "status": {"type": {"completed": True}},
        "competitions": [
            {
                "competitors": [
                    {"homeAway": "home", "winner": True},
                    {"homeAway": "away", "winner": False},
                ]
            }
        ],
    }


def test_espn_event_training_rows(monkeypatch):
    def fake_summary(event_id: str) -> dict:
        assert event_id == "401656359"
        return {
            "plays": [
                {
                    "homeScore": 0,
                    "awayScore": 0,
                    "period": {"number": 1},
                    "clock": {"displayValue": "12:00"},
                },
                {
                    "homeScore": 107,
                    "awayScore": 89,
                    "period": {"number": 4},
                    "clock": {"displayValue": "0.0"},
                },
            ]
        }

    monkeypatch.setattr(espn_scraper, "_summary", fake_summary)
    rows = espn_scraper.event_training_rows(_event())

    assert isinstance(rows, pd.DataFrame)
    assert list(rows.columns) == TRAINING_COLUMNS
    assert len(rows) == 2
    assert rows.iloc[0]["game_id"] == "401656359"
    assert rows.iloc[0]["seconds_remaining"] == 2880.0
    assert rows.iloc[1]["seconds_remaining"] == 0.0
    assert rows.iloc[1]["home_win"] == 1


def test_espn_event_training_rows_skips_incomplete():
    event = _event()
    event["status"] = {"type": {"completed": False, "state": "pre"}}

    rows = espn_scraper.event_training_rows(event)

    assert rows.empty
    assert list(rows.columns) == TRAINING_COLUMNS
