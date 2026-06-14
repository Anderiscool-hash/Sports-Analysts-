import pandas as pd

from sportedge.betting.replay import (
    evaluate_aligned_rows,
    evaluate_directory,
    replay_aligned_rows,
    replay_directory,
    replay_file,
)
from sportedge.config import Config


def test_replay_aligned_rows_writes_paper_fills(tmp_path):
    aligned = pd.DataFrame(
        {
            "timestamp": [100, 200, 300],
            "model_p": [0.70, 0.70, 0.70],
            "price": [0.60, 0.50, 0.52],
        }
    )
    cfg = Config()
    cfg.edge.min_edge = 0.04
    cfg.edge.dip_threshold = 0.05
    cfg.edge.rebound_ticks = 1
    cfg.bankroll = 100
    cfg.max_stake = 5
    path = tmp_path / "paper.parquet"

    summary = replay_aligned_rows(
        aligned,
        cfg,
        ledger_path=str(path),
        token_id="REPLAY-LAL",
        selected_team="Lakers",
        event_id="401",
        home_team="Lakers",
        away_team="Celtics",
    )

    assert summary.rows_seen == 3
    assert summary.fills == 1
    ledger = pd.read_parquet(path)
    assert len(ledger) == 1
    assert ledger.iloc[0]["token_id"] == "REPLAY-LAL"
    assert ledger.iloc[0]["selected_team"] == "Lakers"


def test_replay_requires_aligned_columns(tmp_path):
    cfg = Config()
    try:
        replay_aligned_rows(pd.DataFrame({"price": [0.5]}), cfg, str(tmp_path / "x.parquet"), "tok")
    except ValueError as exc:
        assert "timestamp" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_replay_file_skips_incompatible_parquet(tmp_path):
    path = tmp_path / "raw_prices.parquet"
    pd.DataFrame({"timestamp": [1], "price": [0.5]}).to_parquet(path, index=False)

    summary = replay_file(path, Config(), str(tmp_path / "paper.parquet"))

    assert summary.skipped is True
    assert "missing aligned columns" in summary.reason


def test_replay_directory_processes_matching_aligned_files(tmp_path):
    aligned = pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g1"],
            "timestamp": [100, 200, 300],
            "model_p": [0.70, 0.70, 0.70],
            "price": [0.60, 0.50, 0.52],
            "token_outcome": ["Lakers", "Lakers", "Lakers"],
        }
    )
    aligned.to_parquet(tmp_path / "aligned_one.parquet", index=False)
    pd.DataFrame({"timestamp": [1], "price": [0.5]}).to_parquet(
        tmp_path / "aligned_bad.parquet",
        index=False,
    )
    cfg = Config()
    cfg.edge.min_edge = 0.04
    cfg.edge.dip_threshold = 0.05
    cfg.edge.rebound_ticks = 1

    summaries = replay_directory(tmp_path, cfg, str(tmp_path / "paper.parquet"))

    assert len(summaries) == 2
    assert sum(item.fills for item in summaries) == 1
    assert sum(item.skipped for item in summaries) == 1


def test_replay_file_prefers_token_id_column(tmp_path):
    aligned = pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g1"],
            "timestamp": [100, 200, 300],
            "model_p": [0.70, 0.70, 0.70],
            "price": [0.60, 0.50, 0.52],
            "token_outcome": ["home", "home", "home"],
            "token_id": ["real-token", "real-token", "real-token"],
        }
    )
    path = tmp_path / "aligned_one.parquet"
    ledger = tmp_path / "paper.parquet"
    aligned.to_parquet(path, index=False)
    cfg = Config()
    cfg.edge.min_edge = 0.04
    cfg.edge.dip_threshold = 0.05
    cfg.edge.rebound_ticks = 1

    summary = replay_file(path, cfg, str(ledger))

    assert summary.fills == 1
    fills = pd.read_parquet(ledger)
    assert fills.iloc[0]["token_id"] == "real-token"


def test_evaluate_aligned_rows_reports_pnl_without_ledger(tmp_path):
    aligned = pd.DataFrame(
        {
            "timestamp": [100, 200, 300],
            "model_p": [0.70, 0.70, 0.70],
            "price": [0.60, 0.50, 0.52],
            "token_won": [1, 1, 1],
        }
    )
    cfg = Config()
    cfg.edge.min_edge = 0.04
    cfg.edge.dip_threshold = 0.05
    cfg.edge.rebound_ticks = 1
    cfg.bankroll = 100
    cfg.max_stake = 5

    summary = evaluate_aligned_rows(aligned, cfg)

    assert summary.rows_seen == 3
    assert summary.fills == 1
    assert summary.staked == 5
    assert round(summary.pnl, 4) == round(5 / 0.52 - 5, 4)
    assert summary.wins == 1
    assert not (tmp_path / "paper.parquet").exists()


def test_evaluate_directory_ignores_incompatible_files(tmp_path):
    aligned = pd.DataFrame(
        {
            "timestamp": [100, 200, 300],
            "model_p": [0.70, 0.70, 0.70],
            "price": [0.60, 0.50, 0.52],
            "token_won": [1, 1, 1],
        }
    )
    aligned.to_parquet(tmp_path / "aligned_good.parquet", index=False)
    pd.DataFrame({"timestamp": [1], "price": [0.5]}).to_parquet(
        tmp_path / "aligned_bad.parquet",
        index=False,
    )
    cfg = Config()
    cfg.edge.min_edge = 0.04
    cfg.edge.dip_threshold = 0.05
    cfg.edge.rebound_ticks = 1

    summary = evaluate_directory(tmp_path, cfg)

    assert summary.rows_seen == 3
    assert summary.fills == 1
    assert round(summary.pnl, 4) == round(5 / 0.52 - 5, 4)
