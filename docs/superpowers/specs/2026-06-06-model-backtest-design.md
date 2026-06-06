# Spec — Model-quality Backtest

**Date:** 2026-06-06
**Status:** Approved design, pending spec review
**Scope:** Measure whether the live win-probability model actually predicts NBA game
outcomes. This is the model-quality backtest only. The trading P&L / bet-win-rate
backtest is explicitly out of scope (see "Out of scope").

## Problem

We want to know our "win rate and stuff like that." That phrase splits into two
separate measurements:

1. **Model quality** — does `model_p` (P(home wins)) track reality? Measurable now
   from historical play-by-play, no market data, non-circular.
2. **Trading P&L** — does the long-only "snipe the bottom" strategy actually profit?
   Requires a historical Polymarket *price stream* we do not currently store.

This spec covers (1). It is the honest gate the project design already calls for
("calibrate on Finals games; only then consider live") and a prerequisite for any
meaningful trading claim: if `model_p` does not track reality, the edge the strategy
trades on is fiction.

## Goals

- Score whatever model `WinProbModel.load` returns (logistic fallback today, trained
  xgboost later) against real, completed games.
- Report standard probabilistic-forecast metrics: Brier score, log-loss, accuracy,
  AUC, and a calibration (reliability) curve.
- Report an overall held-out set (statistically meaningful) AND the NBA Finals games
  as a separately called-out subset (the games we actually intend to bet).
- Be reusable, pure-tested, and runnable offline against cached data.

## Out of scope (YAGNI / honesty)

- No P&L, bet win rate, ROI, or drawdown — that is the trading backtest, blocked on
  historical price data.
- No training — `train.py` is unchanged; the harness scores whatever model loads.
- No market/Polymarket calls.

## Trading model assumptions (documented context, not built here)

Recorded so the future trading backtest replays the real strategy. The live path is:
`model_p` -> market price (home-team YES token) -> `edge = model_p - price` ->
`BottomDetector` (dip + rebound + edge) -> fractional-Kelly sizing -> executor.

Key properties the trading backtest must honor later:

- **Long-only:** only ever BUY the home-team YES share; never the away side, never sell.
- **Trigger is not raw edge:** a bottom fires only after a price dip of >= `dip_threshold`
  that has begun rebounding for `rebound_ticks`, while edge >= `min_edge`.
- **Hold to settlement:** no take-profit / stop-loss. A position resolves at game end
  (YES pays 1 if home wins, else 0). "Win rate" = fraction of placed bets where home
  wins; ROI also depends on entry price.
- **Idealized fills:** price is read as implied probability, ignoring spread and fees.

None of this is implemented in this spec; it is here to keep the later spec honest.

## Architecture

Approach A (pure-metrics module + thin runner + script), matching the existing
flat-module-per-concern style and the pure, TDD'd ethos of `market/edge.py`.

### New files

- `src/sportedge/model/metrics.py` — pure metric functions (numpy only, no network).
- `src/sportedge/model/backtest.py` — game-level split + evaluation runner.
- `scripts/run_backtest.py` — CLI: load cached data (or pull), load model, evaluate,
  print a Rich report, optionally write metrics JSON.
- `tests/test_metrics.py` — pure metric unit tests.
- `tests/test_backtest.py` — split-by-game and evaluate-with-stub-model tests.

### `metrics.py` (pure)

All take `probs: np.ndarray` and `labels: np.ndarray` (0/1), both 1-D, equal length.

- `brier_score(probs, labels) -> float` — mean of `(prob - label)**2`.
- `log_loss(probs, labels, eps=1e-15) -> float` — clip probs to `[eps, 1-eps]`.
- `accuracy(probs, labels, threshold=0.5) -> float`.
- `auc(probs, labels) -> float` — rank-based (Mann-Whitney U). Returns `nan` if only
  one class is present.
- `calibration_bins(probs, labels, n_bins=10) -> list[CalibrationBin]` where
  `CalibrationBin` is a frozen dataclass `(lo, hi, count, mean_pred, observed_freq)`.
  Empty bins are returned with `count=0` and `nan` summaries.

### `backtest.py`

- `@dataclass(frozen=True) BacktestReport`:
  `label: str, n_games: int, n_states: int, brier: float, log_loss: float,
  accuracy: float, auc: float, calibration: list[CalibrationBin]`.
- `split_by_game(df, finals_game_ids=(), test_frac=0.3, seed=0) -> tuple[DataFrame, DataFrame]`
  - All `finals_game_ids` present in `df` go to the test set.
  - A random sample of the remaining game ids fills the test set up to `test_frac` of
    all games.
  - The rest form the train set (only relevant once xgboost training is added).
  - **Leakage guard:** asserts the train and test `game_id` sets are disjoint.
- `evaluate(model, rows, label) -> BacktestReport`
  - For each row, reconstruct `GameState(home_team="HOME", away_team="AWAY",
    home_score, away_score, period, seconds_remaining, pre_game_home_prob)`. Team
    identity is irrelevant: `model.predict` consumes only `score_diff`,
    `seconds_remaining`, `period`, `pre_game_home_prob` (see `model/features.py`).
  - Collect `prob = model.predict(state)` and `label = row.home_win`.
  - Compute all metrics; `n_games = rows["game_id"].nunique()`, `n_states = len(rows)`.
- `evaluate_overall_and_subset(model, test_df, finals_game_ids=()) -> dict[str, BacktestReport]`
  - Returns `{"overall": <report on all test rows>}` and, if any finals ids are
    present in `test_df`, `{"finals": <report on just those rows>}`.

### `scripts/run_backtest.py`

Mirrors `scripts/fetch_historical.py` (adds `src/` to `sys.path`, argparse CLI).

Arguments:
- `--cache` (default `training`) — parquet cache name under `data/cache/`. Reused if
  present so we don't re-hit the NBA API.
- `--seasons` (default same as fetch script) and
  `--season-type` (default `Playoffs`) — only used if the cache is missing, to pull
  via `build_training_set` then `save_parquet`.
- `--finals-game-ids` (nargs `*`, default empty) — explicit NBA game-id list for the
  Finals subset. Explicit ids are used because `LeagueGameFinder` has no reliable
  "Finals round" label. If empty, the Finals subset is skipped with a printed note.
- `--test-frac` (default `0.3`), `--seed` (default `0`).
- `--model-path` (default from project model path; falls back to logistic if absent).
- `--out` (optional) — write metrics as JSON.

Behavior: load/-pull data -> `split_by_game` -> `evaluate_overall_and_subset` ->
Rich report (overall metrics, Finals metrics if present, and a text reliability table)
-> optional JSON. The report header states whether the model is trained or the
logistic fallback (via `model.is_trained`).

## Data flow

```
data/cache/training.parquet            (from fetch_historical.py; or pulled here)
  -> split_by_game(finals_game_ids, test_frac, seed)
  -> test rows (TRAINING_COLUMNS)
  -> per row: reconstruct GameState -> model.predict -> (prob, label)
  -> metrics.* over (probs, labels)
  -> BacktestReport (overall) + BacktestReport (finals subset)
  -> Rich console report (+ optional metrics JSON)
```

`TRAINING_COLUMNS` = `game_id, home_score, away_score, period, seconds_remaining,
pre_game_home_prob, home_win` (produced by `data/nba_scraper.py`).

## Error handling

- Empty/missing cache and no network rows -> print a clear message and exit non-zero.
- Empty test split (e.g. too few games) -> clear message, exit non-zero.
- `auc` on a single-class test set -> `nan`, reported as such (not a crash).
- Per-game pull failures are already swallowed in `build_training_set`.

## Testing (TDD, no network)

`metrics.py`:
- Perfect predictions -> Brier `0.0`, log-loss approximately `0.0`.
- All-`0.5` predictions -> Brier `0.25`.
- Separable scores -> `auc == 1.0`; single-class input -> `auc` is `nan`.
- Hand-built set -> `calibration_bins` counts/means/observed match by hand.

`backtest.py`:
- `split_by_game`: train/test `game_id` sets disjoint; all finals ids land in test;
  `test_frac` honored within rounding.
- `evaluate`: a stub model whose `predict` returns a fixed value -> deterministic
  `BacktestReport` (metrics computed by hand).

## Build order

1. `metrics.py` + `tests/test_metrics.py` (pure, fast).
2. `backtest.py` + `tests/test_backtest.py` (stub model, no network).
3. `scripts/run_backtest.py` (wiring, Rich report).
4. Run on the logistic fallback against cached/pulled data to get a baseline.
