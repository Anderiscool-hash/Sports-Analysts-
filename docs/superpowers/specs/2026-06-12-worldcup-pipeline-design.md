# Design — SportEdge World Cup (3-way live win-prob + Polymarket)

**Date:** 2026-06-12
**Status:** Approved (goal-locked)

## Objective

Trade live 2026 FIFA World Cup match-result markets on Polymarket by being faster
and more rational than the crowd. For a live WC match: model in-game 3-way win
probability, compare to live Polymarket prices for each outcome token, and enter
when the market dips below model fair value ("snipe the bottom"). Paper mode is the
default; real-money execution is opt-in behind three independent switches.

### Why there's an edge
Pre-match odds are a commodity — books and markets price them well. The edge is in
*live* markets, where a momentum swing (a goal, a sending-off) moves the **price**
more than it moves the **true** win probability. A disciplined time-and-score model
that doesn't panic can systematically buy that overreaction at the best odds.

## Guiding principle — isolation

The existing NBA pipeline stays **untouched**. Soccer ships as parallel modules that
reuse the sport-agnostic spine:

- `market/` — Polymarket client, `edge`, `BottomDetector` ("snipe the bottom")
- `betting/` — fractional Kelly `Strategy`, paper/live `Executor`, safety gating

Only the **sport-specific** pieces get a soccer twin: game state, win-prob model,
data sources, live loop, and config.

## Key modeling decision — the win-prob model

A soccer scoreline evolves as two near-independent **Poisson goal processes**. This
yields a closed-form in-game 1X2 distribution with **no in-game training data
required**:

- Each team scores at rate λ over the match. Given current goals `(h, a)` and
  `minutes_remaining t`, remaining goals are `Poisson(λ_home · t/90)` and
  `Poisson(λ_away · t/90)`.
- The final result distribution is the difference (Skellam) of the two remaining-goal
  Poissons added to the current goal margin → **P(home) / P(draw) / P(away)** directly.
- **Historical calibration** (final-results fit): a Dixon–Coles / Poisson regression
  on thousands of past match results,
  `goals ~ attack_strength(team) + defense_strength(opp) + home_advantage`, produces
  the pre-match `λ_home` / `λ_away` for any matchup. The in-game part stays pure
  Poisson over remaining minutes.
- **v1 adjustments:** red cards lower the carded team's λ; a small "trailing teams
  attack more" tweak. The pre-match prior fades naturally as `t → 0`, exactly like the
  NBA logistic fallback.

### Scope assumption (explicit)
The model predicts the **90-minute regulation 1X2** result — the standard Polymarket
"Match result" market, where draws resolve to the Draw token. Knockout
"to advance / win incl. extra-time & penalties" markets are **out of scope for v1**.

## Components (new files, parallel to NBA)

### Types
- `types.py` → add `SoccerGameState` (home/away goals, `minute`, `minutes_remaining`,
  `home_red_cards`, `away_red_cards`, pre-match `lambda_home` / `lambda_away` priors)
  and a `WinProb3(home, draw, away)` result type. NBA `GameState` unchanged.

### Model (`model/`)
- `soccer_winprob.py` — `SoccerWinProbModel.predict(state) -> WinProb3`. Poisson
  closed-form as the always-available path; optional fitted estimator swap-in later.
- `soccer_features.py` — pure state→feature functions shared by train and live
  inference (no train/serve skew), parallel to `features.py`.
- `soccer_calibrate.py` — fit attack/defense/home-advantage params from historical
  results → λ priors. Parallel to `train.py`.

### Data (`data/`)
- `soccer_provider.py` — live `SoccerGameState` from a football provider
  (api-sports / isports / sportsblaze), **chained** to a fallback.
- `espn_soccer.py` — ESPN soccer scraper fallback (free, no key).
- `soccer_results.py` — historical results importer for calibration.

Live state resolution chains provider → ESPN → model-only display, mirroring the
existing NBA `_live_state` fallback.

### Market (`market/`)
- `polymarket.py` — small extension: resolve **three** outcome tokens
  (Home / Draw / Away) for a match market instead of one. Edge math and
  `BottomDetector` are reused unchanged, per token.

### Betting (`betting/`)
- Reused unchanged. `Strategy` (fractional Kelly) and `Executor` (paper/live) size and
  place per token.

### Live loop (`live/`)
- `soccer_loop.py` — poll `SoccerGameState` → 3-way probs → fetch 3 prices →
  **3 edges, 3 `BottomDetector`s** (one per token) → `Strategy` sizes each →
  `Executor` places. Rich table shows all three outcomes each tick. `--mode paper|live`.

### Config
- `config.py` / `config/config.yaml` — soccer market block (teams, three outcome
  labels, stage), alongside the existing NBA block.

## Data flow

```
SoccerGameState (provider → ESPN fallback)
  → SoccerWinProbModel.predict → (p_home, p_draw, p_away)
  → per token: edge = model_p − price
  → BottomDetector.update → Strategy.decide → Executor.place
```

Rich console table renders score, minute, all three model probabilities, all three
market prices, and per-token edge each tick.

## Error handling & safety (unchanged guarantees)

1. `paper` is the default mode everywhere.
2. Live requires three independent switches: `--mode live`, `confirm_live: true`, and
   real keys in `.env`.
3. Per-trade `max_stake` and a global daily-loss kill-switch — applied per token and
   in aggregate.
4. All intended/real trades are logged before any order is sent.
5. Provider failure → ESPN fallback → model-only display; never a crash.

## Testing

- **Pure / TDD:** Poisson 1X2 math against known cases (0-0 at minute 90 ≈ high draw
  probability; 1-0 with 1 minute left ≈ home near-certain), red-card λ adjustment,
  calibration fit on a fixture, and 3-token Home/Draw/Away mapping.
- Existing `edge` / `Strategy` / `Executor` tests apply unchanged (per token).

## Build order

1. `SoccerGameState` + `WinProb3` + Poisson `soccer_winprob` (TDD) — works immediately.
2. `soccer_features` + `soccer_calibrate` on historical results.
3. Polymarket 3-token resolution.
4. Data providers (soccer provider + ESPN fallback, chained).
5. `soccer_loop` wiring + config.
6. Paper-trade a live WC match; calibrate; only then consider live.

## Honest status (what will be real vs stub at v1)

| Piece | Planned state |
|-------|---------------|
| Poisson 1X2 in-game math + 3-token edge/bottom | **real, unit-tested** |
| Soccer live provider + ESPN fallback | **real calls**, need network validation |
| Dixon–Coles calibration on historical results | **real**, once results imported |
| Polymarket 3-token resolution | **real client extension** |
| Config + paper executor + soccer loop wiring | **real** |
| Fitted (non-Poisson) in-game classifier | **out of scope v1** (swap-in path exists) |
| Knockout ET/penalty advancement markets | **out of scope v1** |
