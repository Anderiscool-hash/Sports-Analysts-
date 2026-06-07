"""Score the win-probability model against historical play-by-play.

    python scripts/run_backtest.py --finals-game-ids 0042400401 0042400402

Reuses data/cache/training.parquet if present; otherwise pulls --seasons first.
Reports Brier / log-loss / accuracy / AUC + a calibration table, overall and for
the Finals subset. No market data; model-quality only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# allow running as a plain script without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from sportedge.data.nba_scraper import build_training_set  # noqa: E402
from sportedge.data.storage import load_parquet, save_parquet  # noqa: E402
from sportedge.model.backtest import (  # noqa: E402
    evaluate_overall_and_subset,
    split_by_game,
)
from sportedge.model.live_winprob import WinProbModel  # noqa: E402

console = Console()


def _load_data(args):
    df = load_parquet(args.cache)
    if df is not None and not df.empty:
        return df
    console.print(
        f"[yellow]No cache '{args.cache}'; pulling {args.season_type} {args.seasons}…[/]"
    )
    df = build_training_set(args.seasons, args.season_type)
    if not df.empty:
        save_parquet(df, args.cache)
    return df


def _metrics_table(report):
    t = Table(
        title=f"[bold]{report.label}[/]  "
        f"({report.n_games} games, {report.n_states} states)"
    )
    t.add_column("metric")
    t.add_column("value", justify="right")
    t.add_row("Brier", f"{report.brier:.4f}")
    t.add_row("log-loss", f"{report.log_loss:.4f}")
    t.add_row("accuracy", f"{report.accuracy:.4f}")
    t.add_row("AUC", f"{report.auc:.4f}")
    return t


def _calibration_table(report):
    t = Table(title=f"[bold]{report.label} — reliability[/]")
    t.add_column("bin")
    t.add_column("n", justify="right")
    t.add_column("pred", justify="right")
    t.add_column("observed", justify="right")
    for b in report.calibration:
        if b.count == 0:
            continue
        t.add_row(
            f"{b.lo:.1f}-{b.hi:.1f}",
            str(b.count),
            f"{b.mean_pred:.3f}",
            f"{b.observed_freq:.3f}",
        )
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest the win-prob model")
    ap.add_argument("--cache", default="training")
    ap.add_argument("--seasons", nargs="+", default=["2023-24", "2022-23", "2021-22"])
    ap.add_argument(
        "--season-type", default="Playoffs", choices=["Playoffs", "Regular Season"]
    )
    ap.add_argument("--finals-game-ids", nargs="*", default=[])
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-path", default="models/winprob.joblib")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = _load_data(args)
    if df is None or df.empty:
        console.print("[red]No data to backtest. Check connectivity / nba_api.[/]")
        sys.exit(1)

    _, test_df = split_by_game(df, args.finals_game_ids, args.test_frac, args.seed)
    if test_df.empty:
        console.print("[red]Empty test split — need more games or a larger --test-frac.[/]")
        sys.exit(1)

    model = WinProbModel.load(args.model_path)
    console.rule(
        f"Backtest — model={'trained' if model.is_trained else 'logistic-fallback'}"
    )
    reports = evaluate_overall_and_subset(model, test_df, args.finals_game_ids)

    for key in ("overall", "finals"):
        if key in reports:
            console.print(_metrics_table(reports[key]))
            console.print(_calibration_table(reports[key]))
    if "finals" not in reports:
        console.print(
            "[yellow]No Finals subset (pass --finals-game-ids present in the data).[/]"
        )

    if args.out:
        payload = {
            k: {
                "n_games": r.n_games,
                "n_states": r.n_states,
                "brier": r.brier,
                "log_loss": r.log_loss,
                "accuracy": r.accuracy,
                "auc": r.auc,
            }
            for k, r in reports.items()
        }
        Path(args.out).write_text(json.dumps(payload, indent=2))
        console.print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
