# Model-quality Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable, offline-testable harness that scores the win-probability model (logistic fallback now, xgboost later) against historical NBA play-by-play with Brier / log-loss / accuracy / AUC and a calibration curve, reported overall and for a Finals subset.

**Architecture:** Pure-metrics module (numpy only, TDD'd like `market/edge.py`) + a thin runner that splits rows by game and feeds each test state through `model.predict` + a CLI script that loads cached data and prints a Rich report. No market data; model-quality only.

**Tech Stack:** Python, numpy, pandas, rich, pytest. Reuses `data/storage.py` (parquet cache), `data/nba_scraper.py` (`build_training_set`), `model/live_winprob.py` (`WinProbModel`), `types.py` (`GameState`).

**Run tests with:** `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest <path> -v`

---

### Task 1: Pure scoring metrics

**Files:**
- Create: `src/sportedge/model/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_metrics.py`:

```python
import math

import numpy as np

from sportedge.model.metrics import (
    CalibrationBin,
    accuracy,
    auc,
    brier_score,
    calibration_bins,
    log_loss,
)


def test_brier_perfect():
    probs = np.array([1.0, 0.0, 1.0, 0.0])
    labels = np.array([1, 0, 1, 0])
    assert brier_score(probs, labels) == 0.0


def test_brier_all_half():
    probs = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([1, 0, 1, 0])
    assert brier_score(probs, labels) == 0.25


def test_log_loss_perfect_near_zero():
    probs = np.array([1.0, 0.0, 1.0, 0.0])
    labels = np.array([1, 0, 1, 0])
    assert log_loss(probs, labels) < 1e-10


def test_accuracy_half():
    probs = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 0, 1, 0])  # 2nd and 3rd predictions wrong
    assert accuracy(probs, labels) == 0.5


def test_auc_separable():
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert auc(probs, labels) == 1.0


def test_auc_single_class_is_nan():
    probs = np.array([0.3, 0.6, 0.9])
    labels = np.array([1, 1, 1])
    assert math.isnan(auc(probs, labels))


def test_auc_ties_half():
    probs = np.array([0.5, 0.5])
    labels = np.array([0, 1])
    assert auc(probs, labels) == 0.5


def test_calibration_bins_counts_and_freq():
    probs = np.array([0.05, 0.05, 0.95, 0.95])
    labels = np.array([0, 1, 1, 1])
    bins = calibration_bins(probs, labels, n_bins=10)
    assert isinstance(bins[0], CalibrationBin)
    assert bins[0].count == 2
    assert bins[0].mean_pred == 0.05
    assert bins[0].observed_freq == 0.5
    assert bins[9].count == 2
    assert bins[9].observed_freq == 1.0
    assert bins[5].count == 0
    assert math.isnan(bins[5].mean_pred)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sportedge.model.metrics'`

- [ ] **Step 3: Write the implementation**

Create `src/sportedge/model/metrics.py`:

```python
"""Pure scoring metrics for probabilistic forecasts.

Dependency-light (numpy only) and fully unit-tested, mirroring market/edge.py.
Each function takes parallel 1-D arrays: probs (model P in [0, 1]) and labels (0/1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    return float(np.mean((p - y) ** 2))


def log_loss(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
    y = np.asarray(labels, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def accuracy(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    preds = (p >= threshold).astype(float)
    return float(np.mean(preds == y))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks with ties resolved to their average (for rank-based AUC)."""
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(len(values), dtype=float)
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # ranks are 1-based
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def auc(probs: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U). nan if only one class is present."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    n_pos = float(np.sum(y == 1))
    n_neg = float(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(p)
    sum_pos = float(np.sum(ranks[y == 1]))
    return (sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)


@dataclass(frozen=True)
class CalibrationBin:
    lo: float
    hi: float
    count: int
    mean_pred: float
    observed_freq: float


def calibration_bins(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> list[CalibrationBin]:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[CalibrationBin] = []
    for k in range(n_bins):
        lo, hi = float(edges[k]), float(edges[k + 1])
        if k == n_bins - 1:  # include the right edge in the final bin
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        count = int(np.sum(mask))
        if count == 0:
            bins.append(CalibrationBin(lo, hi, 0, float("nan"), float("nan")))
        else:
            bins.append(
                CalibrationBin(
                    lo, hi, count, float(np.mean(p[mask])), float(np.mean(y[mask]))
                )
            )
    return bins
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_metrics.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/sportedge/model/metrics.py tests/test_metrics.py
git commit -m "feat: pure scoring metrics for win-prob backtest"
```

---

### Task 2: Game-level split and evaluation runner

**Files:**
- Create: `src/sportedge/model/backtest.py`
- Test: `tests/test_backtest.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_backtest.py`:

```python
import pandas as pd

from sportedge.model.backtest import (
    evaluate,
    evaluate_overall_and_subset,
    split_by_game,
)

COLUMNS = [
    "game_id",
    "home_score",
    "away_score",
    "period",
    "seconds_remaining",
    "pre_game_home_prob",
    "home_win",
]


def _rows(game_id: str, home_win: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [game_id, game_id, game_id],
            "home_score": [10, 20, 30],
            "away_score": [8, 15, 25],
            "period": [1, 2, 4],
            "seconds_remaining": [1800.0, 1200.0, 5.0],
            "pre_game_home_prob": [0.6, 0.6, 0.6],
            "home_win": [home_win, home_win, home_win],
        },
        columns=COLUMNS,
    )


def _frame(n_games: int = 10) -> pd.DataFrame:
    return pd.concat(
        [_rows(f"g{i}", i % 2) for i in range(n_games)], ignore_index=True
    )


class _StubModel:
    is_trained = False

    def predict(self, state) -> float:
        return 0.5


def test_split_disjoint_games():
    df = _frame(10)
    train, test = split_by_game(df, test_frac=0.3, seed=1)
    assert set(train["game_id"]).isdisjoint(set(test["game_id"]))


def test_split_finals_always_in_test():
    df = _frame(10)
    train, test = split_by_game(df, finals_game_ids=["g0", "g1"], test_frac=0.3, seed=1)
    assert {"g0", "g1"}.issubset(set(test["game_id"]))
    assert "g0" not in set(train["game_id"])


def test_split_test_frac_honored():
    df = _frame(10)
    _, test = split_by_game(df, test_frac=0.3, seed=1)
    assert test["game_id"].nunique() == 3


def test_evaluate_stub_model_deterministic():
    df = _frame(4)
    report = evaluate(_StubModel(), df, "overall")
    assert report.n_games == 4
    assert report.n_states == 12
    assert report.brier == 0.25  # all predictions 0.5


def test_evaluate_overall_and_subset_finals_key():
    df = _frame(10)
    reports = evaluate_overall_and_subset(_StubModel(), df, finals_game_ids=["g0"])
    assert "overall" in reports
    assert "finals" in reports
    assert reports["finals"].n_games == 1


def test_evaluate_overall_and_subset_no_finals():
    df = _frame(10)
    reports = evaluate_overall_and_subset(_StubModel(), df, finals_game_ids=[])
    assert "overall" in reports
    assert "finals" not in reports
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_backtest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sportedge.model.backtest'`

- [ ] **Step 3: Write the implementation**

Create `src/sportedge/model/backtest.py`:

```python
"""Game-level split and model evaluation for the win-prob backtest.

Splits historical rows by game (never by row — every state in a game shares one
final label, so a row-level split would leak the answer), runs each test state
through model.predict, and scores with model/metrics.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sportedge.model.metrics import (
    CalibrationBin,
    accuracy,
    auc,
    brier_score,
    calibration_bins,
    log_loss,
)
from sportedge.types import GameState


@dataclass(frozen=True)
class BacktestReport:
    label: str
    n_games: int
    n_states: int
    brier: float
    log_loss: float
    accuracy: float
    auc: float
    calibration: list[CalibrationBin]


def split_by_game(
    df: pd.DataFrame,
    finals_game_ids: list[str] = (),
    test_frac: float = 0.3,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into (train, test) by game id. Finals ids are forced into test;
    the rest of the test set is a random sample sized to ~test_frac of all games."""
    game_ids = list(pd.unique(df["game_id"]))
    id_set = set(game_ids)
    finals = [g for g in finals_game_ids if g in id_set]
    finals_set = set(finals)
    others = [g for g in game_ids if g not in finals_set]

    rng = np.random.default_rng(seed)
    others_arr = np.array(others, dtype=object)
    rng.shuffle(others_arr)

    n_test_total = max(1, round(test_frac * len(game_ids)))
    n_from_others = max(0, n_test_total - len(finals))
    test_ids = finals_set | set(others_arr[:n_from_others].tolist())
    train_ids = id_set - test_ids

    assert train_ids.isdisjoint(test_ids), "game leaked across split"

    train_df = df[df["game_id"].isin(train_ids)].reset_index(drop=True)
    test_df = df[df["game_id"].isin(test_ids)].reset_index(drop=True)
    return train_df, test_df


def evaluate(model, rows: pd.DataFrame, label: str) -> BacktestReport:
    """Score `model` over `rows` (TRAINING_COLUMNS schema) and return a report."""
    if len(rows) == 0:
        return BacktestReport(
            label, 0, 0, float("nan"), float("nan"), float("nan"), float("nan"), []
        )
    labels = rows["home_win"].to_numpy(dtype=float)
    probs = np.empty(len(rows), dtype=float)
    for i, (_, r) in enumerate(rows.iterrows()):
        # Team identity is irrelevant: model.predict consumes only score_diff,
        # seconds_remaining, period, pre_game_home_prob (see model/features.py).
        state = GameState(
            home_team="HOME",
            away_team="AWAY",
            home_score=int(r["home_score"]),
            away_score=int(r["away_score"]),
            period=int(r["period"]),
            seconds_remaining=float(r["seconds_remaining"]),
            pre_game_home_prob=float(r["pre_game_home_prob"]),
        )
        probs[i] = model.predict(state)
    return BacktestReport(
        label=label,
        n_games=int(rows["game_id"].nunique()),
        n_states=int(len(rows)),
        brier=brier_score(probs, labels),
        log_loss=log_loss(probs, labels),
        accuracy=accuracy(probs, labels),
        auc=auc(probs, labels),
        calibration=calibration_bins(probs, labels),
    )


def evaluate_overall_and_subset(
    model, test_df: pd.DataFrame, finals_game_ids: list[str] = ()
) -> dict[str, BacktestReport]:
    """Report on the whole test set, plus the Finals subset if any of its ids are
    present in `test_df`."""
    reports = {"overall": evaluate(model, test_df, "overall")}
    finals_rows = test_df[test_df["game_id"].isin(set(finals_game_ids))]
    if len(finals_rows) > 0:
        reports["finals"] = evaluate(model, finals_rows, "finals")
    return reports
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_backtest.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/sportedge/model/backtest.py tests/test_backtest.py
git commit -m "feat: game-level split and evaluation runner for backtest"
```

---

### Task 3: CLI runner with Rich report

**Files:**
- Create: `scripts/run_backtest.py`

This is a wiring/IO script (no unit test, matching `scripts/fetch_historical.py`). It is verified by an offline smoke run against a synthetic parquet cache.

- [ ] **Step 1: Write the script**

Create `scripts/run_backtest.py`:

```python
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
```

- [ ] **Step 2: Build a synthetic cache for an offline smoke test**

Run (creates `data/cache/smoke.parquet` with 6 fake games, no network):

```bash
PYTHONPATH=src ./.venv/Scripts/python.exe -c "
import pandas as pd
from sportedge.data.storage import save_parquet
rows = []
for i in range(6):
    won = i % 2
    for (hs, av, per, sec) in [(5,3,1,2000.0),(20,18,2,1000.0),(30,29,4,10.0)]:
        rows.append(dict(game_id=f'g{i}', home_score=hs, away_score=av, period=per,
                         seconds_remaining=sec, pre_game_home_prob=0.6, home_win=won))
save_parquet(pd.DataFrame(rows), 'smoke')
print('wrote smoke cache')
"
```

Expected: `wrote smoke cache`

- [ ] **Step 3: Run the script against the synthetic cache**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe scripts/run_backtest.py --cache smoke --finals-game-ids g0 --test-frac 0.5`
Expected: prints a `logistic-fallback` rule, an `overall` metrics table + reliability table, and a `finals` metrics table for game `g0`. No traceback.

- [ ] **Step 4: Remove the synthetic cache**

Run: `rm data/cache/smoke.parquet`

- [ ] **Step 5: Run the full test suite**

Run: `PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest -q`
Expected: PASS (all prior tests + 14 new).

- [ ] **Step 6: Commit**

```bash
git add scripts/run_backtest.py
git commit -m "feat: run_backtest CLI with Rich metrics and reliability report"
```

---

## Notes for the implementer

- The default `--model-path models/winprob.joblib` will not exist until `train.py` is run; `WinProbModel.load` falls back to the logistic model and the report header says `logistic-fallback`. That is the expected first run.
- A real (non-smoke) run needs `data/cache/training.parquet`, produced by `python scripts/fetch_historical.py` (network, slow). The harness itself stays offline-testable via the synthetic cache.
- `--finals-game-ids` are explicit NBA game ids because `LeagueGameFinder` has no reliable "Finals round" label.
