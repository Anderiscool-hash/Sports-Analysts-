"""Model-quality backtest for the 3-way (1X2) soccer win-probability model.

Replays historical matches minute-by-minute through the model and scores each forecast
against the actual result with the metrics that suit ordered Home/Draw/Away outcomes:

* **RPS** (Ranked Probability Score) — the standard 1X2 metric; penalises forecasts by
  how far off they are on the ordinal Home -> Draw -> Away scale. Lower is better.
* **multiclass log-loss** and **multiclass Brier** — proper scoring rules.
* a per-outcome **reliability** table (predicted vs. empirical frequency).

This answers "is the model any good?" — the calibration gate before live trading. It
needs only match results + goal timings (no market data).

CLI:
    python -m sportedge.model.soccer_backtest matches.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from sportedge.model.soccer_winprob import SoccerWinProbModel
from sportedge.types import REGULATION_MINUTES, SoccerGameState, WinProb3

# Ordered 1X2 outcomes. Order matters for RPS (ordinal Home -> Draw -> Away scale).
OUTCOMES = ("home", "draw", "away")
_INDEX = {o: i for i, o in enumerate(OUTCOMES)}


def _probs_tuple(p: WinProb3) -> tuple[float, float, float]:
    return (p.home, p.draw, p.away)


def _onehot(outcome: str) -> tuple[float, float, float]:
    vec = [0.0, 0.0, 0.0]
    vec[_INDEX[outcome]] = 1.0
    return tuple(vec)  # type: ignore[return-value]


def rps(probs: WinProb3, outcome: str) -> float:
    """Ranked Probability Score for one forecast. 0 = perfect, ~0.5 worst-ish."""
    p = _probs_tuple(probs)
    o = _onehot(outcome)
    cum_p = 0.0
    cum_o = 0.0
    total = 0.0
    for i in range(len(OUTCOMES) - 1):  # r-1 cumulative steps
        cum_p += p[i]
        cum_o += o[i]
        total += (cum_p - cum_o) ** 2
    return total / (len(OUTCOMES) - 1)


def log_loss(probs: WinProb3, outcome: str) -> float:
    import math

    p_actual = _probs_tuple(probs)[_INDEX[outcome]]
    return -math.log(min(max(p_actual, 1e-12), 1.0))


def brier(probs: WinProb3, outcome: str) -> float:
    p = _probs_tuple(probs)
    o = _onehot(outcome)
    return sum((p[i] - o[i]) ** 2 for i in range(len(OUTCOMES)))


def outcome_from_score(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home"
    if home_goals < away_goals:
        return "away"
    return "draw"


def reconstruct_states(
    home_team: str,
    away_team: str,
    goals: list[tuple[float, str]],
    lambda_home: float,
    lambda_away: float,
    sample_minutes: list[int] | None = None,
) -> list[SoccerGameState]:
    """Build minute-by-minute states from goal events.

    ``goals`` is a list of ``(minute, "home"|"away")``. A state is emitted at each
    minute in ``sample_minutes`` (default every 5') with the score as of that minute.
    """
    if sample_minutes is None:
        sample_minutes = list(range(0, REGULATION_MINUTES + 1, 5))
    events = sorted(goals, key=lambda g: g[0])
    states: list[SoccerGameState] = []
    for minute in sample_minutes:
        hg = sum(1 for m, side in events if side == "home" and m <= minute)
        ag = sum(1 for m, side in events if side == "away" and m <= minute)
        states.append(
            SoccerGameState(
                home_team=home_team,
                away_team=away_team,
                home_goals=hg,
                away_goals=ag,
                minute=float(minute),
                lambda_home=lambda_home,
                lambda_away=lambda_away,
            )
        )
    return states


@dataclass
class Match:
    home_team: str
    away_team: str
    outcome: str
    goals: list[tuple[float, str]]
    lambda_home: float = 1.45
    lambda_away: float = 1.15

    @classmethod
    def from_dict(cls, d: dict) -> "Match":
        goals = [(float(m), str(side)) for m, side in d.get("goals", [])]
        outcome = d.get("outcome")
        if outcome is None:
            hg = sum(1 for _, s in goals if s == "home")
            ag = sum(1 for _, s in goals if s == "away")
            outcome = outcome_from_score(hg, ag)
        return cls(
            home_team=str(d["home_team"]),
            away_team=str(d["away_team"]),
            outcome=str(outcome),
            goals=goals,
            lambda_home=float(d.get("lambda_home", 1.45)),
            lambda_away=float(d.get("lambda_away", 1.15)),
        )


@dataclass
class BacktestResult:
    n_forecasts: int = 0
    mean_rps: float = 0.0
    mean_log_loss: float = 0.0
    mean_brier: float = 0.0
    rps_by_bucket: dict[str, float] = field(default_factory=dict)
    reliability: dict[str, list[tuple[float, float, int]]] = field(default_factory=dict)


def _bucket(minute: float) -> str:
    lo = int(minute // 15) * 15
    return f"{lo:02d}-{lo + 15:02d}"


def backtest(
    model: SoccerWinProbModel,
    matches: list[Match],
    sample_minutes: list[int] | None = None,
    n_reliability_bins: int = 5,
) -> BacktestResult:
    """Score every (state, final-outcome) forecast across all matches."""
    rps_sum = ll_sum = br_sum = 0.0
    n = 0
    bucket_sum: dict[str, float] = {}
    bucket_n: dict[str, int] = {}
    # reliability accumulators per outcome: bin -> [sum_pred, sum_actual, count]
    rel: dict[str, list[list[float]]] = {
        o: [[0.0, 0.0, 0.0] for _ in range(n_reliability_bins)] for o in OUTCOMES
    }

    for match in matches:
        states = reconstruct_states(
            match.home_team,
            match.away_team,
            match.goals,
            match.lambda_home,
            match.lambda_away,
            sample_minutes,
        )
        for state in states:
            probs = model.predict(state)
            r = rps(probs, match.outcome)
            rps_sum += r
            ll_sum += log_loss(probs, match.outcome)
            br_sum += brier(probs, match.outcome)
            n += 1

            b = _bucket(state.minute)
            bucket_sum[b] = bucket_sum.get(b, 0.0) + r
            bucket_n[b] = bucket_n.get(b, 0) + 1

            p = _probs_tuple(probs)
            o = _onehot(match.outcome)
            for oi, name in enumerate(OUTCOMES):
                bin_idx = min(n_reliability_bins - 1, int(p[oi] * n_reliability_bins))
                rel[name][bin_idx][0] += p[oi]
                rel[name][bin_idx][1] += o[oi]
                rel[name][bin_idx][2] += 1

    if n == 0:
        return BacktestResult()

    reliability = {
        name: [
            (cell[0] / cell[2], cell[1] / cell[2], int(cell[2]))
            for cell in bins
            if cell[2] > 0
        ]
        for name, bins in rel.items()
    }
    return BacktestResult(
        n_forecasts=n,
        mean_rps=rps_sum / n,
        mean_log_loss=ll_sum / n,
        mean_brier=br_sum / n,
        rps_by_bucket={b: bucket_sum[b] / bucket_n[b] for b in sorted(bucket_sum)},
        reliability=reliability,
    )


def load_matches(path: str) -> list[Match]:
    data = json.loads(open(path, encoding="utf-8").read())
    return [Match.from_dict(d) for d in data]


def main() -> None:
    import argparse

    from rich.console import Console
    from rich.table import Table

    ap = argparse.ArgumentParser(description="Soccer 3-way model-quality backtest")
    ap.add_argument("matches", help="JSON list of matches (home_team,away_team,goals,outcome)")
    ap.add_argument("--model", default="models/soccer_winprob.joblib")
    args = ap.parse_args()

    console = Console()
    model = SoccerWinProbModel.load(args.model)
    result = backtest(model, load_matches(args.matches))

    console.rule(
        f"Soccer backtest - {result.n_forecasts} forecasts - "
        f"model={'trained' if model.is_trained else 'poisson-fallback'}"
    )
    summary = Table(show_header=False, box=None)
    summary.add_row("mean RPS", f"{result.mean_rps:.4f}  (lower better)")
    summary.add_row("mean log-loss", f"{result.mean_log_loss:.4f}")
    summary.add_row("mean Brier", f"{result.mean_brier:.4f}")
    console.print(summary)

    if result.rps_by_bucket:
        bt = Table(title="RPS by match minute")
        bt.add_column("minute")
        bt.add_column("RPS")
        for bucket, value in result.rps_by_bucket.items():
            bt.add_row(bucket, f"{value:.4f}")
        console.print(bt)


if __name__ == "__main__":
    main()
