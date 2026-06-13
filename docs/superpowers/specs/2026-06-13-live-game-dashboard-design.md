# Live Game Dashboard — Design

**Date:** 2026-06-13
**Status:** Approved (pending spec review)

## Goal

A live-updating dashboard that runs **in the terminal/command prompt** (not a
separate windowed GUI app) showing rich, real-time detail for one in-progress
game. Works for both **NBA basketball** and **World Cup / soccer**, surfacing the
sport-appropriate details:

- Score and clock/period (or match minute + stoppage)
- Basketball: team **fouls**, possession, **free-throw** indicator
- Soccer: **yellow/red cards**, possession, **set-piece** indicator (free kick,
  penalty, corner)
- The model **win-probability**, the **Kalshi** market price, and the computed
  **edge** (display only — see Non-Goals)

## Non-Goals

- **No trading.** The dashboard is display-only. It shows model prob, Kalshi
  price, edge, and when a "snipe the bottom" signal would fire, but it never
  constructs an executor and never places an order. Actual trading stays in the
  existing `loop.py` / `soccer_loop.py`.
- No separate windowed/native GUI. Terminal only (`rich`).
- No new market venue. Kalshi reads only (no auth needed for prices).

## Approach

**Unified ESPN `summary` endpoint as the single live source for both sports.**

ESPN exposes `site.api.espn.com/apis/site/v2/sports/{sport}/{league}/summary?event={id}`
which carries the situational detail (team fouls, cards, possession, and a
`lastPlay` text block). One code path covers NBA and soccer, it is free and needs
no key, and it uses plain `requests` — so the dashboard does **not** depend on
`nba_api` or `tenacity` (the latter is currently uninstalled in the environment).

Rejected alternative: keep `nba_api` for NBA + ESPN for soccer. Two data paths,
reintroduces the `tenacity` requirement, more surface area, little benefit.

## Components

### 1. `src/sportedge/data/espn_live.py` (new)

Unified ESPN live client built on plain `requests` (mirrors `espn_soccer.py`'s
retry/degrade style — small backoff, return `None`/empty on failure).

- `list_live_games() -> list[GameCandidate]`
  - Scans the NBA scoreboard (`basketball/nba`) and the configured soccer
    leagues' scoreboards (default `fifa.world`, plus any extras).
  - Returns one `GameCandidate` per event with: `sport` ("basketball"|"soccer"),
    `league`, `event_id`, `home_team`, `away_team`, `status` (pre|in|post),
    `short_detail` (e.g. "Q3 4:21" or "67'").
- `get_game_detail(sport, league, event_id) -> LiveDetail | None`
  - Pulls the `summary` endpoint and parses it into a `LiveDetail`.
  - Returns `None` on any network/parse failure (caller keeps last good frame).

Endpoints:
- Scoreboard: `.../sports/{sport}/{league}/scoreboard`
- Summary: `.../sports/{sport}/{league}/summary?event={event_id}`

### 2. `LiveDetail` + `GameCandidate` dataclasses (in `src/sportedge/types.py`)

`GameCandidate` — minimal record for the picker (fields listed above).

`LiveDetail` — sport-agnostic live snapshot, from the home team's perspective:

Common:
- `sport: str`, `league: str`
- `home_team: str`, `away_team: str`
- `home_score: int`, `away_score: int`
- `status: str` (pre|in|post)
- `clock: str` (raw display clock, e.g. "4:21" or "67'")
- `period: int` (basketball quarter; 0 for soccer)
- `minute: float` (soccer match minute; 0.0 for basketball)
- `possession: str` ("home"|"away"|"" if unknown)
- `last_play_text: str` (verbatim from ESPN; always shown — source of truth)

Basketball:
- `home_fouls: int`, `away_fouls: int`
- `free_throw_active: bool` (best-effort from `lastPlay` text / bonus state)

Soccer:
- `home_yellow: int`, `away_yellow: int`
- `home_red: int`, `away_red: int`
- `set_piece: str` ("" | "free kick" | "penalty" | "corner") — best-effort from
  `lastPlay` text

Detection helpers (pure functions, unit-tested):
- `detect_free_throw(last_play_text) -> bool`
- `detect_set_piece(last_play_text) -> str`

These scan lowercased `lastPlay` text for keywords. Explicitly best-effort: they
catch many but not all situations and lag by a few seconds. `last_play_text` is
always rendered so the user sees the raw event regardless.

### 3. `src/sportedge/live/dashboard.py` (new)

Terminal TUI using `rich.live.Live` + `rich.layout.Layout`.

Flow:
1. **Picker** — call `list_live_games()`, print a numbered list of in-progress
   (and optionally upcoming) games, read the user's choice from stdin. If none
   are live, say so and exit cleanly.
2. **Live loop** — every `cfg.loop.poll_seconds`:
   - `detail = get_game_detail(...)`; if `None`, keep the last frame and mark
     the footer "stale".
   - Compute model probability: `WinProbModel` (basketball) or
     `SoccerWinProbModel` (soccer), loaded from `cfg.model.path` with the
     existing fallbacks.
   - Read Kalshi price(s) via `KalshiClient.get_price` (read-only) when a ticker
     is configured (`cfg.market.kalshi_ticker` for NBA; the three
     `cfg.soccer.kalshi_*_ticker` for soccer). Missing ticker → price panel shows
     "—".
   - `edge = model_p - price` via `sportedge.market.edge.edge`.
   - `render(detail, model_view, price_view)` builds the renderable; `Live.update`.
   - Stop when `status == "post"`.

Panels:
- **Header** — matchup + status/clock.
- **Score** — large, both teams.
- **Events** (sport-specific) — basketball: fouls per team, possession, free-throw
  flag. soccer: yellow/red per team, possession, set-piece flag. Both: the raw
  `last_play_text`.
- **Model & edge** — model win-prob, Kalshi price, edge (and a note if a snipe
  signal would fire). Display only.
- **Footer** — last-update timestamp, "stale" indicator on fetch failure.

### 4. Entry point

`python -m sportedge.live.dashboard` (a `main()` with `argparse`, optional
`--config`). No new script file required, but a thin `scripts/dashboard.py`
wrapper may be added for convenience.

## Data Flow (per tick)

```
get_game_detail() -> LiveDetail
     |                       \
     v                        v
model.predict()          KalshiClient.get_price()  (read-only, optional)
     \                       /
      \                     v
       ----> edge() ---> render(LiveDetail, probs, price) -> Live.update()
```

## Error Handling

- Network/parse failure in `espn_live` returns `None`/empty; the dashboard keeps
  the last good frame and flags "stale" in the footer. No crash.
- Missing Kalshi ticker → price/edge panel shows "—", game detail still renders.
- Missing/untrained model → existing logistic/Poisson fallback (already built in).
- Set-piece / free-throw flags are best-effort; raw `last_play_text` is always
  shown as the authoritative event.

## Testing

- **Pure render functions** (`LiveDetail` -> renderable): smoke-test that they
  build without error for basketball and soccer details.
- **Parsing**: `summary`-JSON fixtures -> `LiveDetail`, asserting score, fouls,
  cards, possession, last play for both sports.
- **Detection heuristics**: `detect_free_throw` / `detect_set_piece` against a
  table of representative `lastPlay` strings (positive and negative cases).
- **Picker filtering**: only `in`-progress games are selectable; empty-list path
  handled.
- The live `Live`-loop itself is not unit-tested (I/O + timing); logic is pushed
  into the testable pure functions above.

## Dependencies

- Reuses existing: `rich`, `requests`, `pydantic`, the win-prob models, `edge`,
  `KalshiClient`, `config`.
- No new third-party dependency. Notably avoids `nba_api`/`tenacity`.
