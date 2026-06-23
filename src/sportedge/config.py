"""Typed configuration loaded from config/config.yaml + secrets from .env."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel


class EdgeConfig(BaseModel):
    min_edge: float = 0.04
    dip_threshold: float = 0.05
    rebound_ticks: int = 1
    min_price: float = 0.05
    max_price: float = 0.95


class LoopConfig(BaseModel):
    poll_seconds: int = 5


class ModelConfig(BaseModel):
    path: str = "models/winprob.joblib"


class PaperGateConfig(BaseModel):
    """Minimum paper-trading proof required before live execution can be enabled."""

    enabled: bool = True
    min_fills: int = 25
    min_settled_fills: int | None = None
    min_realized_pnl: float = 0.0
    min_realized_roi: float = 0.0
    min_total_pnl: float = 0.0


class ExecutionConfig(BaseModel):
    """How live orders are priced and how long they may rest before auto-cancel.

    Borrowed from Krypt-Trader's order mechanics: in a score-lag snipe the edge is
    gone in seconds, so an unfilled order must cancel itself rather than sit and
    fill *after* the market has already corrected.
    """

    # "market" -> cross immediately; "limit_cross" -> sit at the best opposing
    # quote; "limit_mid" -> rest at the book midpoint (best price, slowest fill).
    order_style: str = "limit_cross"
    # Cancel a resting order this many seconds after placement (0 disables).
    order_expiration_sec: int = 8
    # If the live order book can't be read, fall back to the signal price nudged by
    # this many cents in the aggressive direction so a BUY still clears.
    cross_spread_fallback_offset_cents: int = 2
    # How often (seconds) to poll a resting order for fills before expiry.
    fill_poll_seconds: float = 1.0


class RiskConfig(BaseModel):
    """Runtime exposure caps and circuit breakers checked before every live order.

    All fractions are of current bankroll. These gate *new entries only*; they never
    touch existing positions. Paper mode ignores them.
    """

    enabled: bool = True
    # Stop opening new positions once total staked exposure reaches this fraction.
    max_total_exposure_fraction: float = 0.50
    # Always keep at least this fraction of bankroll in reserve (never staked).
    min_cash_reserve_fraction: float = 0.10
    # Hard ceiling on the number of simultaneously open positions.
    max_open_positions: int = 5
    # At most this many open positions per ESPN event (1 = one bet per game).
    max_positions_per_event: int = 1
    # Daily realized-PnL circuit breakers (USD). Loss is negative.
    daily_loss_cap: float = -20.0
    daily_take_profit: float = 0.0  # 0 disables the take-profit halt
    # Trading-hours window (local clock). Disabled by default for live sports.
    trading_hours_enabled: bool = False
    trading_hours_start: str = "00:00"
    trading_hours_end: str = "23:59"
    trading_timezone_offset_min: int = 0


class FlowConfig(BaseModel):
    """Optional Kalshi trade-feed confirmation layered on top of the model signal.

    Borrowed from Krypt-Trader's whale / momentum scanning, but used only to
    *confirm* a model-edge entry, never to originate one. ``mode="off"`` (default)
    leaves behavior unchanged; ``mode="confirm"`` requires the trade feed to show a
    whale trade or a contrarian sell-off before a model signal is allowed to fire.
    """

    mode: str = "off"  # "off" | "confirm"
    # Window of recent trades to consider.
    lookback_sec: int = 120
    # A single trade worth >= this many USD counts as a whale (count * price).
    whale_min_notional: float = 250.0
    # Contrarian: a YES-price drop of at least this (prob units) over the window
    # signals an overreaction worth fading (i.e. confirms buying the bottom).
    price_move_threshold: float = 0.05
    # A trade cluster needs at least this many trades and this much total USD.
    cluster_min_trades: int = 3
    cluster_min_notional: float = 100.0


class MarketConfig(BaseModel):
    """An NBA game market on Kalshi: a single YES contract for the home team to win."""

    market_slug: str = ""
    # ESPN event id used later to settle paper fills from final scores.
    espn_event_id: str = ""
    home_team: str = ""
    away_team: str = ""
    # Kalshi contract ticker for "home team wins" (read prices / place orders).
    kalshi_ticker: str = ""
    # Optional Kalshi contract ticker for "away team wins"; useful when discovery
    # finds only the other side of the NBA moneyline.
    kalshi_away_ticker: str = ""


class SoccerMarketConfig(BaseModel):
    """A World Cup 1X2 match market: three Kalshi outcome contracts plus the
    pre-match expected-goals priors that seed the Poisson model."""

    market_slug: str = ""
    # ESPN event id used later to settle paper fills from final scores.
    espn_event_id: str = ""
    league: str = "fifa.world"  # ESPN soccer league slug
    home_team: str = ""
    away_team: str = ""
    # Pre-match full-match expected goals (calibration output; sensible WC defaults).
    lambda_home: float = 1.45
    lambda_away: float = 1.15
    # Optional fitted soccer estimator. Blank uses the built-in live Poisson model.
    # This must not point at the NBA estimator in ``model.path``.
    model_path: str = ""
    # Kalshi 1X2 contract tickers; one per outcome.
    kalshi_home_ticker: str = ""
    kalshi_draw_ticker: str = ""
    kalshi_away_ticker: str = ""


class Config(BaseModel):
    mode: str = "paper"  # "paper" | "live"
    confirm_live: bool = False
    bankroll: float = 100.0
    max_stake: float = 5.0
    max_daily_loss: float = 20.0
    kelly_fraction: float = 0.25
    edge: EdgeConfig = EdgeConfig()
    loop: LoopConfig = LoopConfig()
    model: ModelConfig = ModelConfig()
    paper_gate: PaperGateConfig = PaperGateConfig()
    execution: ExecutionConfig = ExecutionConfig()
    risk: RiskConfig = RiskConfig()
    flow: FlowConfig = FlowConfig()
    market: MarketConfig = MarketConfig()
    soccer: SoccerMarketConfig = SoccerMarketConfig()

    @property
    def live_enabled(self) -> bool:
        """Live trading requires BOTH switches. The executor adds a third check
        (real keys present) before any order can be sent."""
        return self.mode == "live" and self.confirm_live


class Secrets(BaseModel):
    # Kalshi
    kalshi_api_key_id: str | None = None
    kalshi_private_key_pem: str | None = None
    kalshi_host: str = "https://api.elections.kalshi.com/trade-api/v2"

    @property
    def kalshi_complete(self) -> bool:
        return bool(self.kalshi_api_key_id and self.kalshi_private_key_pem)


def load_config(path: str = "config/config.yaml") -> Config:
    load_dotenv()
    p = Path(path)
    data = yaml.safe_load(p.read_text()) if p.exists() else {}
    return Config(**(data or {}))


def _read_kalshi_private_key() -> str | None:
    """Kalshi key may be supplied inline (PEM) or via a file path."""
    inline = os.getenv("KALSHI_PRIVATE_KEY_PEM")
    if inline:
        return inline
    path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    if path and Path(path).exists():
        return Path(path).read_text()
    return None


def load_secrets() -> Secrets:
    load_dotenv()
    return Secrets(
        kalshi_api_key_id=os.getenv("KALSHI_API_KEY_ID") or None,
        kalshi_private_key_pem=_read_kalshi_private_key(),
        kalshi_host=os.getenv("KALSHI_HOST") or "https://api.elections.kalshi.com/trade-api/v2",
    )
