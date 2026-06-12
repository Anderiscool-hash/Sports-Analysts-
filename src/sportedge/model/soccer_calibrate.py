"""Pre-match calibration: fit team attack/defense strengths and home advantage from
historical match results, producing the full-match expected-goals (lambda) priors that
seed the in-game Poisson model.

Model (independent-Poisson / Dixon-Coles base):

    log(lambda_home) = mu + home_adv + attack[home] - defense[away]
    log(lambda_away) = mu            + attack[away] - defense[home]

Fit by maximum-likelihood (gradient descent on the Poisson NLL) over a list of
``MatchResult``. Attack/defense are mean-centered for identifiability. The fitted
``TeamRatings`` then yields ``(lambda_home, lambda_away)`` for any matchup — exactly the
two numbers ``SoccerGameState`` needs.

CLI:
    python -m sportedge.model.soccer_calibrate results.csv --home Brazil --away Croatia
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# League-average goals per team when a match has no strength info — the global prior
# the ratings are expressed relative to. ~1.35 fits World Cup scoring.
_DEFAULT_BASE_GOALS = 1.35


@dataclass(frozen=True)
class MatchResult:
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int


@dataclass
class TeamRatings:
    """Fitted strengths. ``attack``/``defense`` are log-scale, mean-centered."""

    attack: dict[str, float] = field(default_factory=dict)
    defense: dict[str, float] = field(default_factory=dict)
    mu: float = float(np.log(_DEFAULT_BASE_GOALS))
    home_adv: float = 0.0

    def lambdas(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Full-match expected goals (home, away). Unknown teams use league-average
        (attack/defense 0), so the function is always defined."""
        ah = self.attack.get(home_team, 0.0)
        aa = self.attack.get(away_team, 0.0)
        dh = self.defense.get(home_team, 0.0)
        da = self.defense.get(away_team, 0.0)
        lam_home = float(np.exp(self.mu + self.home_adv + ah - da))
        lam_away = float(np.exp(self.mu + aa - dh))
        return lam_home, lam_away

    def to_json(self, path: str) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "attack": self.attack,
                    "defense": self.defense,
                    "mu": self.mu,
                    "home_adv": self.home_adv,
                },
                indent=2,
            )
        )

    @classmethod
    def from_json(cls, path: str) -> "TeamRatings":
        data = json.loads(Path(path).read_text())
        return cls(
            attack=dict(data.get("attack", {})),
            defense=dict(data.get("defense", {})),
            mu=float(data.get("mu", np.log(_DEFAULT_BASE_GOALS))),
            home_adv=float(data.get("home_adv", 0.0)),
        )


def fit_ratings(
    results: list[MatchResult],
    iterations: int = 4000,
    lr: float = 0.02,
) -> TeamRatings:
    """Maximum-likelihood fit of attack/defense/home-advantage via gradient descent
    on the Poisson negative log-likelihood. Deterministic (zero init)."""
    if not results:
        return TeamRatings()

    teams = sorted({r.home_team for r in results} | {r.away_team for r in results})
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    h = np.array([idx[r.home_team] for r in results])
    a = np.array([idx[r.away_team] for r in results])
    kh = np.array([r.home_goals for r in results], dtype=float)
    ka = np.array([r.away_goals for r in results], dtype=float)

    attack = np.zeros(n)
    defense = np.zeros(n)
    base = (kh.mean() + ka.mean()) / 2 or _DEFAULT_BASE_GOALS
    mu = float(np.log(max(_DEFAULT_BASE_GOALS, base)))
    home_adv = 0.0
    m = len(results)

    for _ in range(iterations):
        lam_h = np.exp(mu + home_adv + attack[h] - defense[a])
        lam_a = np.exp(mu + attack[a] - defense[h])
        rh = lam_h - kh  # dNLL/d(eta_home)
        ra = lam_a - ka  # dNLL/d(eta_away)

        g_mu = (rh.sum() + ra.sum()) / m
        g_home = rh.sum() / m
        g_attack = np.zeros(n)
        g_defense = np.zeros(n)
        np.add.at(g_attack, h, rh)
        np.add.at(g_attack, a, ra)
        np.add.at(g_defense, a, -rh)
        np.add.at(g_defense, h, -ra)
        g_attack /= m
        g_defense /= m

        mu -= lr * g_mu
        home_adv -= lr * g_home
        attack -= lr * g_attack
        defense -= lr * g_defense

        # Re-center for identifiability, folding the shift back into mu so lambdas
        # are unchanged: +attack appears in both etas, -defense in both etas.
        a_mean = attack.mean()
        d_mean = defense.mean()
        attack -= a_mean
        defense -= d_mean
        mu += a_mean - d_mean

    return TeamRatings(
        attack={t: float(attack[idx[t]]) for t in teams},
        defense={t: float(defense[idx[t]]) for t in teams},
        mu=float(mu),
        home_adv=float(home_adv),
    )


def load_results_csv(path: str) -> list[MatchResult]:
    """Load results from a CSV with columns:
    ``home_team,away_team,home_goals,away_goals``."""
    import pandas as pd

    df = pd.read_csv(path)
    return [
        MatchResult(
            home_team=str(row.home_team),
            away_team=str(row.away_team),
            home_goals=int(row.home_goals),
            away_goals=int(row.away_goals),
        )
        for row in df.itertuples(index=False)
    ]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Fit soccer team ratings -> lambda priors")
    ap.add_argument("csv", help="results CSV: home_team,away_team,home_goals,away_goals")
    ap.add_argument("--home", help="home team to report lambdas for")
    ap.add_argument("--away", help="away team to report lambdas for")
    ap.add_argument("--out", default="models/soccer_ratings.json")
    args = ap.parse_args()

    ratings = fit_ratings(load_results_csv(args.csv))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ratings.to_json(args.out)
    print(f"Saved ratings for {len(ratings.attack)} teams -> {args.out}")
    if args.home and args.away:
        lam_h, lam_a = ratings.lambdas(args.home, args.away)
        print(f"{args.home} vs {args.away}: lambda_home={lam_h:.3f} lambda_away={lam_a:.3f}")


if __name__ == "__main__":
    main()
