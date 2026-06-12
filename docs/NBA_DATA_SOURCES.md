# NBA Data Sources

Working notes from source discovery for SportEdge.

## Current Best Training Source

### Kaggle: eoinamoore/historical-nba-data-and-player-box-scores

Status: integrated.

Why it matters: includes `PlayByPlay.parquet` with score, period, clock, and final
game labels from `Games.csv`. This is the best source we have tested for
state-level win-probability training.

Local importer:

```powershell
.\.venv\Scripts\python.exe scripts\import_kaggle_nba.py `
  --root "C:\path\to\kaggle\folder" `
  --start-date 2021-10-19 `
  --end-date 2024-06-17 `
  --game-types "Regular Season" Playoffs `
  --cache training_kaggle_2021_24_regular_plus_playoffs
```

Latest model result from this source:

```text
Finals-only Brier:    0.1604
Finals-only log-loss: 0.4657
Finals-only accuracy: 0.7208
Finals-only AUC:      0.8398
```

## Already Integrated Supporting Sources

### nba_api

Status: integrated.

Useful for official NBA live state and historical fallback pulls. The V3
play-by-play endpoint works; deprecated V2 should not be used for training.

### ESPN hidden API

Status: integrated.

Useful for date-range play-by-play snapshots through scoreboard plus summary
endpoints. Good fallback source when official NBA API behavior changes.

### iSports API

Status: integrated for live fallback.

Useful for live basketball scores when official NBA live lookup has no matching
game. Reads `ISPORTS_API_KEY` from the environment.

### SportsBlaze

Status: integrated for schedule/boxscore access.

Useful for schedule, boxscores, roster stats, and team stat enrichment. The
documented NBA endpoints do not expose play-by-play snapshots, so this is not a
direct replacement for Kaggle/ESPN/NBA PBP training data.

### API-SPORTS

Status: integrated for probing/current-window access.

Useful across multiple sports once the plan supports historical data. The tested
free basketball plan was limited to a tiny current-date window and 100 requests
per day.

## New Leads From awesome-nba-data

Source list: https://github.com/JovaniPink/awesome-nba-data

### pbpstats

Priority: high for feature engineering.

The Python package parses NBA/WNBA/G-League play-by-play. It is not needed to
fetch raw rows now that Kaggle PBP is working, but it is a strong candidate for
deriving possession-level and lineup-context features from play-by-play.

Potential next features:

- possession number
- possession team
- home possession flag
- lineup/on-court context if available
- score differential by possession instead of by event
- garbage-time filtering

### Inpredictable

Priority: high for benchmarking.

Inpredictable has an NBA win-probability calculator and public win-probability
tools. This is useful as an external sanity check for model shape, especially
clock/score probability curves.

Potential use:

- compare SportEdge predictions against known score/time states
- build a small benchmark table for representative NBA states
- check late-game calibration against an external model

### balldontlie

Priority: medium; depends on plan/API key.

Advertises live play-by-play, odds, injuries, props, lineups, historical data,
and BDL Lab backtesting. This becomes interesting if we need historical odds or
market-style enrichment, but it is probably redundant for raw NBA play-by-play
now that Kaggle is available.

Potential use:

- historical odds if plan allows it
- injury/lineup features
- player props and market-derived features

### Basketball-Reference / Stathead

Priority: medium.

Useful for historical team/player quality features and validation. Scraping or
paid Stathead access may be needed for repeatable bulk use.

### NocturneBear/NBA-Data-2010-2024

Priority: medium-high for feature engineering.

Status: team totals importer added.

This repo provides regular-season and playoff team/player box-score data from
2010-2024. It does not provide in-game play-by-play, so it should not replace the
Kaggle PBP training source. It does provide compact team-game totals with perfect
game-id overlap against the current Kaggle 2021-24 training cache.

Local importer:

```powershell
.\.venv\Scripts\python.exe scripts\import_nocturne_nba.py
```

Latest import:

```text
35,678 team-game rows
17,828 games
3,877 / 3,877 overlap with training_kaggle_2021_24_regular_plus_playoffs
```

Potential use:

- rolling team net rating
- rolling offensive/defensive scoring profile
- recent form entering a game
- regular-season to playoff team-strength priors
- validation joins against Kaggle game IDs

## Recommended Next Step

Add possession/context features from the Kaggle play-by-play before chasing more
raw data. The current model already improved sharply with Kaggle PBP; the next
gains should come from better features, especially possession, period-clock
shape, timeouts/fouls if reliable, and lineup/team-strength context. The
NocturneBear team totals cache is a good immediate source for rolling team
strength.
