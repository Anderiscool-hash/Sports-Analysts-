# Design — SportEdge

## Objective

For NBA Finals Games 3–7: scrape data, model **live in-game win probability**,
compare to live Kalshi prices, and enter positions when the market price dips
below model fair value ("snipe the bottom") to capture value at the best odds.
Real-money execution is opt-in; everything runs in paper mode by default.

## Why live win-probability is the core asset

Pre-game predictions are a commodity — books and markets price them well. The edge
in *live* betting comes from reacting faster and more rationally than the crowd to
in-game events. A momentum run moves the market more than it moves the true win
probability. So the central model is a **time-and-score win-probability model**:

```
P(home wins | score_diff, seconds_remaining, period, possession, pre_game_prior)
```

This is trainable from historical play-by-play (every NBA game is thousands of
labeled states → final outcome). Finals Games 3–7 are then *applications* of and
calibration checks on that model, not the whole training set (5 games is far too
little to train on alone).

## Components

### 1. Data (`data/`)
- `nba_scraper.py`
  - **Historical**: `nba_api` `leaguegamefinder` + `playbyplayv2` to pull many
    seasons of play-by-play → (state, final_outcome) training rows. Basketball-
    Reference (`bs4`) as a fallback source.
  - **Live**: `nba_api.live.nba.endpoints.scoreboard` + `playbyplay` for the
    current game state, polled during the loop.
- `storage.py` — parquet cache under `data/` so we don't re-hit the API; thin
  SQLite option for bet/trade logging.

### 2. Model (`model/`)
- `features.py` — turn a raw game state into the model feature vector. Pure
  functions, unit-testable, shared by training and live inference (no train/serve
  skew).
- `train.py` — fit a gradient-boosted classifier (xgboost) on historical states;
  **calibrate** probabilities (reliability matters more than raw accuracy when the
  output is compared to market prices). Save to `models/`.
- `live_winprob.py` — load the model, expose `predict(state) -> P(win)`. Ships
  with a transparent **logistic fallback** (closed-form on score diff & time) so
  the pipeline runs before a model is trained.

### 3. Market (`market/`)
- `kalshi.py`
  - **Read** (`/markets/{ticker}`) — live prices (yes_bid/ask → [0,1] probability)
    and market lookup, no auth.
  - **Orders** (RSA-PSS signed `/portfolio/orders`) — only constructed when live
    mode is enabled and Kalshi keys are present.
- `edge.py` — **pure logic, TDD'd**:
  - `implied_prob_from_price`, `edge = model_p - price`
  - `BottomDetector` — tracks a rolling price series, flags a local "bottom":
    price fell ≥ `dip_threshold`, is now ticking back up, while model edge is
    still ≥ `min_edge`. This is the "snipe the bottom" trigger.

### 4. Betting (`betting/`)
- `strategy.py` — position sizing. Fractional **Kelly** from (model_p, price),
  capped by `max_stake` and `bankroll`. Refuses to size when edge < `min_edge`.
- `executor.py`
  - `PaperExecutor` — logs intended fills at the observed price; tracks P&L.
  - `KalshiLiveExecutor` — places real signed Kalshi orders. Constructed **only**
    if `mode == "live"` AND `confirm_live: true` in config AND Kalshi keys present.

### 5. Live loop (`live/loop.py`)
Poll game state → live P(win) → fetch market price → edge + bottom check →
strategy size → executor. Rich console table each tick. `--mode paper|live`.

## Risk / safety rules (non-negotiable in code)
1. `paper` is the default mode everywhere.
2. Live requires three independent switches: `--mode live`, `confirm_live: true`,
   and real keys in `.env`.
3. Per-trade `max_stake` and global `max_daily_loss` kill-switch.
4. All intended/real trades are logged before any order is sent.

## Honest status (what's real vs stub)
| Piece | State |
|-------|-------|
| Edge / bottom math + strategy sizing | **real, unit-tested** |
| Logistic fallback win-prob | **real** |
| Config + paper executor + loop wiring | **real** |
| nba_api historical/live calls | **real calls**, need network + light validation |
| xgboost trained model | **stub until** `fetch_historical` + `train` are run |
| Kalshi read/orders | **real client**, read paths first; live orders gated |

## Build order
1. ✅ scaffold + config
2. data layer (scraper + cache)
3. features + fallback + train
4. market client + edge (TDD) + strategy
5. paper executor + live loop
6. calibrate on Finals games; only then consider live
