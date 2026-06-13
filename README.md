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
```

## Status

Scaffold + core logic in place. See `docs/DESIGN.md` for the build plan and the
honest list of what is real vs. stubbed.
