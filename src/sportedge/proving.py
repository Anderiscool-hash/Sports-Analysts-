"""Status helpers for the paper-trading proving ground."""

from __future__ import annotations

from dataclasses import dataclass

from sportedge.betting.executor import paper_gate_status
from sportedge.betting.report import PaperReport, build_paper_report
from sportedge.config import Config
from sportedge.market.scanner import GameMarketCoverage, scan_live_game_markets
from sportedge.model.captured import CapturedTrainingSummary, inspect_captured_training


@dataclass(frozen=True)
class MarketReadiness:
    games_scanned: int
    ready_games: int

    @property
    def ready(self) -> bool:
        return self.ready_games > 0


@dataclass(frozen=True)
class TrainingReadiness:
    path: str
    rows: int
    games: int
    min_games: int
    exists: bool
    ready: bool
    reason: str


@dataclass(frozen=True)
class PaperReadiness:
    fills: int
    settled_fills: int
    open_positions: int
    open_exposure: float
    realized_pnl: float
    realized_roi: float
    total_pnl: float
    gate_ok: bool
    gate_reason: str

    @property
    def profitable(self) -> bool:
        return self.total_pnl >= 0.0


@dataclass(frozen=True)
class ProvingGroundStatus:
    markets: MarketReadiness
    paper: PaperReadiness
    training: TrainingReadiness
    coverage: list[GameMarketCoverage]
    report: PaperReport

    @property
    def live_ready(self) -> bool:
        return self.markets.ready and self.paper.gate_ok and self.training.ready


def summarize_market_readiness(coverage: list[GameMarketCoverage]) -> MarketReadiness:
    return MarketReadiness(
        games_scanned=len(coverage),
        ready_games=sum(1 for row in coverage if row.has_market),
    )


def inspect_training_readiness(
    data_path: str,
    min_games: int = 3,
) -> TrainingReadiness:
    try:
        summary: CapturedTrainingSummary = inspect_captured_training(data_path)
    except FileNotFoundError:
        return TrainingReadiness(
            path=data_path,
            rows=0,
            games=0,
            min_games=min_games,
            exists=False,
            ready=False,
            reason="training cache missing",
        )
    ready = summary.games >= min_games
    reason = (
        f"ready: {summary.games} games / {summary.rows} rows"
        if ready
        else f"needs {min_games} games; found {summary.games}"
    )
    return TrainingReadiness(
        path=summary.path,
        rows=summary.rows,
        games=summary.games,
        min_games=min_games,
        exists=True,
        ready=ready,
        reason=reason,
    )


def summarize_paper_readiness(
    report: PaperReport,
    gate_ok: bool,
    gate_reason: str,
) -> PaperReadiness:
    summary = report.summary
    return PaperReadiness(
        fills=int(summary.get("fills", 0)),
        settled_fills=int(summary.get("settled_fills", 0)),
        open_positions=int(summary.get("open_positions", 0)),
        open_exposure=float(summary.get("open_exposure", 0.0)),
        realized_pnl=float(summary.get("realized_pnl", 0.0)),
        realized_roi=float(summary.get("realized_roi", 0.0)),
        total_pnl=float(summary.get("total_pnl", 0.0)),
        gate_ok=gate_ok,
        gate_reason=gate_reason,
    )


def build_proving_ground_status(
    cfg: Config,
    ledger_path: str = "data/cache/paper_ledger.parquet",
    training_cache: str = "data/cache/training.parquet",
    min_training_games: int = 3,
    scan_markets: bool = True,
    sports: set[str] | None = None,
    statuses: set[str] | None = None,
) -> ProvingGroundStatus:
    coverage = scan_live_game_markets(sports=sports, statuses=statuses) if scan_markets else []
    report = build_paper_report(ledger_path)
    gate_ok, gate_reason = paper_gate_status(cfg, ledger_path)
    return ProvingGroundStatus(
        markets=summarize_market_readiness(coverage),
        paper=summarize_paper_readiness(report, gate_ok, gate_reason),
        training=inspect_training_readiness(training_cache, min_training_games),
        coverage=coverage,
        report=report,
    )


def next_action(status: ProvingGroundStatus, cfg: Config) -> str:
    """Operator-facing next command for the proving-ground bottleneck."""
    if not status.training.ready:
        return "watch completed NBA games in the dashboard, then run: python scripts/train_evaluate_captured.py"
    min_settled_fills = (
        cfg.paper_gate.min_settled_fills
        if cfg.paper_gate.min_settled_fills is not None
        else cfg.paper_gate.min_fills
    )
    if status.paper.fills < cfg.paper_gate.min_fills:
        remaining = cfg.paper_gate.min_fills - status.paper.fills
        if status.markets.ready:
            return (
                f"collect {remaining} more paper fills from ready markets: "
                "python -m sportedge.live.dashboard --auto-pick-ready"
            )
        return (
            f"need {remaining} more paper fills; no ready live market right now. "
            "Run: python scripts/scan_kalshi_games.py --sport all --debug-rejections "
            "or wait with: python -m sportedge.live.dashboard --wait-ready"
        )
    if status.paper.settled_fills < min_settled_fills:
        remaining = min_settled_fills - status.paper.settled_fills
        return f"need {remaining} more settled paper fills before live can unlock"
    if not status.paper.profitable:
        return "paper sample is large enough but not profitable; tune/evaluate before enabling live"
    if not status.markets.ready:
        return "paper gate is satisfied; wait for a valid live market: python -m sportedge.live.dashboard --wait-ready"
    return "ready for continued paper trading; live still requires mode=live, confirm_live=true, and keys"
