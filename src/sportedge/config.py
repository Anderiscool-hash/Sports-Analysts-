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


class LoopConfig(BaseModel):
    poll_seconds: int = 5


class ModelConfig(BaseModel):
    path: str = "models/winprob.joblib"


class MarketConfig(BaseModel):
    """An NBA game market on Kalshi: a single YES contract for the home team to win."""

    market_slug: str = ""
    home_team: str = ""
    away_team: str = ""
    # Kalshi contract ticker for "home team wins" (read prices / place orders).
    kalshi_ticker: str = ""


class SoccerMarketConfig(BaseModel):
    """A World Cup 1X2 match market: three Kalshi outcome contracts plus the
    pre-match expected-goals priors that seed the Poisson model."""

    market_slug: str = ""
    league: str = "fifa.world"  # ESPN soccer league slug
    home_team: str = ""
    away_team: str = ""
    # Pre-match full-match expected goals (calibration output; sensible WC defaults).
    lambda_home: float = 1.45
    lambda_away: float = 1.15
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
