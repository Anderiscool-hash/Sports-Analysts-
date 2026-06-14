import pandas as pd

from sportedge.betting.executor import Fill, PaperExecutor
from sportedge.betting.paper import PaperLedger, annotate_fills, collect_replay_settlements, summarize_fills
from sportedge.betting.strategy import Order


def test_paper_executor_persists_fill(tmp_path):
    path = tmp_path / "paper.parquet"
    executor = PaperExecutor(ledger_path=str(path))

    fill = executor.place(Order("BUY", 5.0, 0.50, 0.70, 0.20), "NBA-LAL-WIN")

    df = pd.read_parquet(path)
    assert len(df) == 1
    assert df.iloc[0]["token_id"] == "NBA-LAL-WIN"
    assert df.iloc[0]["size"] == fill.size
    assert "event_id" in df.columns


def test_paper_ledger_summary_marks_open_pnl(tmp_path):
    path = tmp_path / "paper.parquet"
    ledger = PaperLedger(str(path))
    ledger.append(Fill(1.0, "BUY", 5.0, 0.50, 0.70, 0.20, "paper", "tok"))

    summary = ledger.summary(marks={"tok": 0.60})

    assert summary["fills"] == 1
    assert summary["open_exposure"] == 5.0
    assert summary["unrealized_pnl"] == 1.0
    assert summary["total_pnl"] == 1.0


def test_summarize_fills_settles_resolved_positions():
    fills = pd.DataFrame(
        [
            {
                "ts": 1.0,
                "side": "BUY",
                "size": 5.0,
                "price": 0.50,
                "model_p": 0.70,
                "edge": 0.20,
                "mode": "paper",
                "token_id": "tok",
            }
        ]
    )

    summary = summarize_fills(fills, settlements={"tok": 1.0})

    assert summary["settled_fills"] == 1
    assert summary["settled_staked"] == 5.0
    assert summary["open_positions"] == 0
    assert summary["realized_pnl"] == 5.0
    assert summary["realized_roi"] == 1.0
    assert summary["total_pnl"] == 5.0


def test_annotate_fills_marks_and_settles_rows():
    fills = pd.DataFrame(
        [
            {
                "ts": 1.0,
                "side": "BUY",
                "size": 5.0,
                "price": 0.50,
                "model_p": 0.70,
                "edge": 0.20,
                "mode": "paper",
                "token_id": "open",
            },
            {
                "ts": 2.0,
                "side": "BUY",
                "size": 5.0,
                "price": 0.50,
                "model_p": 0.70,
                "edge": 0.20,
                "mode": "paper",
                "token_id": "settled",
            },
        ]
    )

    report = annotate_fills(fills, marks={"open": 0.60}, settlements={"settled": 0.0})

    assert report.loc[report["token_id"] == "open", "pnl"].iloc[0] == 1.0
    assert report.loc[report["token_id"] == "settled", "pnl"].iloc[0] == -5.0
    assert bool(report.loc[report["token_id"] == "settled", "is_settled"].iloc[0]) is True


def test_collect_replay_settlements_from_training_cache(tmp_path):
    training = tmp_path / "training.parquet"
    pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g2"],
            "home_win": [1, 1, 0],
        }
    ).to_parquet(training, index=False)
    fills = pd.DataFrame(
        [
            {"token_id": "home-g1", "event_id": "g1", "selected_team": "home"},
            {"token_id": "away-g1", "event_id": "g1", "selected_team": "away"},
            {"token_id": "away-g2", "event_id": "g2", "selected_team": "away"},
        ]
    )

    settlements = collect_replay_settlements(fills, str(training))

    assert settlements == {"home-g1": 1.0, "away-g1": 0.0, "away-g2": 1.0}
