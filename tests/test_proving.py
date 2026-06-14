import pandas as pd

from sportedge.betting.report import PaperReport
from sportedge.config import Config
from sportedge.market.kalshi import KalshiDiscoveryResult
from sportedge.market.scanner import GameMarketCoverage
from sportedge.proving import (
    MarketReadiness,
    PaperReadiness,
    ProvingGroundStatus,
    TrainingReadiness,
    inspect_training_readiness,
    next_action,
    summarize_market_readiness,
    summarize_paper_readiness,
)
from sportedge.types import GameCandidate


def test_summarize_market_readiness_counts_ready_games():
    game = GameCandidate("basketball", "nba", "1", "Lakers", "Celtics", "in", "Q1")
    coverage = [
        GameMarketCoverage(
            game,
            KalshiDiscoveryResult("tok", "Will Lakers win?", "Lakers", 1.0),
            None,
        )
    ]

    summary = summarize_market_readiness(coverage)

    assert summary.games_scanned == 1
    assert summary.ready_games == 1
    assert summary.ready is True


def test_inspect_training_readiness_missing_cache():
    status = inspect_training_readiness("missing.parquet", min_games=3)

    assert status.exists is False
    assert status.ready is False
    assert status.reason == "training cache missing"


def test_inspect_training_readiness_counts_games(tmp_path):
    path = tmp_path / "training.parquet"
    pd.DataFrame({"game_id": ["g1", "g1", "g2"], "home_win": [1, 1, 0]}).to_parquet(
        path,
        index=False,
    )

    status = inspect_training_readiness(str(path), min_games=2)

    assert status.exists is True
    assert status.ready is True
    assert status.games == 2


def test_summarize_paper_readiness_uses_report_summary():
    report = PaperReport(
        summary={
            "fills": 5,
            "settled_fills": 3,
            "open_positions": 2,
            "open_exposure": 10.0,
            "realized_pnl": 1.0,
            "realized_roi": 0.10,
            "total_pnl": 1.25,
        },
        fills=pd.DataFrame(),
        marks={},
        settlements={},
    )

    status = summarize_paper_readiness(report, gate_ok=True, gate_reason="passed")

    assert status.fills == 5
    assert status.settled_fills == 3
    assert status.open_positions == 2
    assert status.open_exposure == 10.0
    assert status.realized_pnl == 1.0
    assert status.realized_roi == 0.10
    assert status.total_pnl == 1.25
    assert status.gate_ok is True


def _status(markets_ready=False, fills=3, pnl=8.69, training_ready=True):
    return ProvingGroundStatus(
        markets=MarketReadiness(games_scanned=1, ready_games=1 if markets_ready else 0),
        paper=PaperReadiness(
            fills=fills,
            settled_fills=fills,
            open_positions=0,
            open_exposure=0.0,
            realized_pnl=pnl,
            realized_roi=0.1 if pnl >= 0 else -0.1,
            total_pnl=pnl,
            gate_ok=False,
            gate_reason="needs fills",
        ),
        training=TrainingReadiness(
            path="training.parquet",
            rows=100,
            games=10,
            min_games=3,
            exists=True,
            ready=training_ready,
            reason="ready",
        ),
        coverage=[],
        report=PaperReport(summary={}, fills=pd.DataFrame(), marks={}, settlements={}),
    )


def test_next_action_prioritizes_more_paper_fills_when_market_missing():
    cfg = Config()
    cfg.paper_gate.min_fills = 25

    action = next_action(_status(markets_ready=False, fills=3, pnl=8.69), cfg)

    assert "need 22 more paper fills" in action
    assert "scan_kalshi_games.py --sport all --debug-rejections" in action


def test_next_action_uses_ready_market_when_available():
    cfg = Config()
    cfg.paper_gate.min_fills = 25

    action = next_action(_status(markets_ready=True, fills=3, pnl=8.69), cfg)

    assert "collect 22 more paper fills from ready markets" in action
    assert "--auto-pick-ready" in action
