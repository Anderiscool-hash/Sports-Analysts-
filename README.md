# SportEdge — live win-prob models (NBA + World Cup) + Kalshi edge engine

Analyze scraped sports data, build models that predict game outcomes **and live
in-game win probability**, then compare those models against live Kalshi prices
to find value and "snipe the bottom" — entering when the market temporarily
over-reacts and odds are cheapest relative to the model.

> ⚠️ **Money & legal disclaimer.** This software can place real trades on
> Kalshi. Real-money execution is **disabled by default** and gated behind
> explicit config flags. Prediction-market trading carries real financial risk.
> You are responsible for complying with the laws of your jurisdiction. Nothing
> here is financial advice. Start in `paper` mode.

## What "snipe the bottom" means here

During a live game the market price for a team swings hard on momentum (a 8–0 run,
a missed clutch shot). Often those swings overshoot what actually changed about the
win probability. The "bottom" is the lowest price point of an overshoot. If our
live model still rates the team materially higher than the dipped market price, that
dip is a value-buy entry. The engine watches the order book, detects these dislocations,
and (in paper mode) records or (in live mode) places the entry.

## Pipeline

```
scrape NBA data ──► features ──► train pre-game + live win-prob models
                                          │
live game state ──► live model ──► P(win)─┤
                                          ▼
Kalshi (read prices) ──► market price ──► edge = P(win) − price
                                          ▼
                          bottom detector / strategy
                                          ▼
                    executor  (paper  |  live[gated])
```

## Layout

| Path | Purpose |
|------|---------|
| `src/sportedge/data/` | NBA scraping (historical + live) and local cache |
| `src/sportedge/model/` | feature engineering + pre-game & live win-prob models |
| `src/sportedge/market/` | Kalshi client + edge / bottom detection |
| `src/sportedge/betting/` | strategy sizing + paper/live executor |
| `src/sportedge/live/` | orchestration loop |
| `scripts/` | one-off jobs (fetch history, train) |
| `tests/` | unit tests (edge logic is fully covered) |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env             # add keys only when going live
cp config/config.example.yaml config/config.yaml

# 1. Pull historical Finals data and train
python scripts/fetch_historical.py
python -m sportedge.model.train

# 2. Run the live loop in PAPER mode (safe, no orders)
python -m sportedge.live.loop --mode paper

# 3. Run the richer live dashboard
python -m sportedge.live.dashboard

# Or let the dashboard auto-pick the first game with a valid Kalshi winner market
python -m sportedge.live.dashboard --auto-pick-ready

# Or keep scanning until a valid Kalshi winner market appears, then start
python -m sportedge.live.dashboard --wait-ready

# Optional: scan live/upcoming ESPN games first and see which ones have direct,
# quoted Kalshi team-winner markets ready for paper trading.
python scripts/scan_kalshi_games.py

# Add rejection details when a game shows "NO MARKET".
python scripts/scan_kalshi_games.py --debug-rejections

# Check the whole proving-ground state: market coverage, paper gate, training data.
python scripts/proving_status.py

# The dashboard writes paper fills and captures completed NBA games by default:
# - paper fills: data/cache/paper_ledger.parquet
# - training rows: data/cache/training.parquet
# If no NBA Kalshi ticker is configured, it tries to auto-discover a direct,
# quoted home-team winner market and ignores combo/prop markets.

# 4. Train + evaluate from the captured game cache
python scripts/train_evaluate_captured.py --data data/cache/training.parquet

# 5. Report paper P&L, marking open trades and settling finished ESPN games
python scripts/paper_report.py --ledger data/cache/paper_ledger.parquet

# If no live Kalshi game is tradable right now, build aligned replay rows from
# cached raw Polymarket prices, then replay them through the same paper ledger.
python scripts/build_aligned_replays.py --directory data/cache --pattern "polymarket_*_10m.parquet"
python scripts/replay_paper.py --directory data/cache --pattern "aligned*.parquet"

# Evaluate replay files without writing to the paper ledger. Use --grid to compare
# edge/dip/rebound settings before changing config or replaying into the ledger.
python scripts/evaluate_replay.py --directory data/cache --pattern "aligned_generated_*.parquet"
python scripts/evaluate_replay.py --directory data/cache --pattern "aligned_generated_*.parquet" --grid
```

Dashboard options:

- `--training-cache PATH` writes captured final-labeled NBA snapshots somewhere else.
- `--no-record-training` turns off live training capture.
- `--paper-ledger PATH` writes paper fills somewhere else.
- `--no-paper-trading` turns off dashboard paper signals.

Live orders remain blocked even when `mode: live`, `confirm_live: true`, and
Kalshi keys are present until `paper_gate` passes. By default that means at least
25 paper fills, 25 settled paper fills, non-negative realized paper P&L, and
non-negative realized ROI and total paper P&L in the ledger. Open paper positions
do not count as settled proof and cannot offset a losing realized sample.

Replay fills generated from cached games are settled from `data/cache/training.parquet`
when the ledger `event_id` matches `game_id` and `selected_team` is `home` or
`away`. Paper fills from watched ESPN games are settled from ESPN final scores
when their `event_id`, `sport`, `league`, and `selected_team` metadata are present.
Paper reports and the live gate use the same settlement logic, so final watched
games count toward settled proof. The dashboard and lower-level NBA/soccer loops
write that metadata with paper fills. For lower-level loops, set
`market.espn_event_id` or `soccer.espn_event_id`; the dashboard fills this from
the picked ESPN game automatically.

Training and replay alignment automatically ignore corrupted late-game `0-0`
placeholder states. Strategy entries also require prices inside the configured
band (`edge.min_price` to `edge.max_price`) so dust-price lottery tickets do not
pollute paper results.

The lower-level paper loops also persist fills:

```bash
python -m sportedge.live.loop --mode paper --paper-ledger data/cache/paper_ledger.parquet
python -m sportedge.live.soccer_loop --mode paper --paper-ledger data/cache/paper_ledger.parquet
```

## Status

Scaffold + core logic in place. See `docs/DESIGN.md` for the build plan and the
honest list of what is real vs. stubbed.
