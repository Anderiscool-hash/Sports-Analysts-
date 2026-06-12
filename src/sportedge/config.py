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
    gamma_host: str = "https://gamma-api.polymarket.com"
    market_slug: str = ""
    home_team: str = ""
    away_team: str = ""


class SoccerMarketConfig(BaseModel):
    """A World Cup 1X2 match market: three outcome tokens plus the pre-match
    expected-goals priors that seed the Poisson model."""

    gamma_host: str = "https://gamma-api.polymarket.com"
    market_slug: str = ""
    league: str = "fifa.world"  # ESPN soccer league slug
    home_team: str = ""
    away_team: str = ""
    # Outcome labels as they appear on the Polymarket market (used to map tokens).
    home_outcome: str = ""  # e.g. "Brazil"
    draw_outcome: str = "Draw"
    away_outcome: str = ""  # e.g. "Croatia"
    # Pre-match full-match expected goals (calibration output; sensible WC defaults).
    lambda_home: float = 1.45
    lambda_away: float = 1.15


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
    private_key: str | None = None
    funder_address: str | None = None
    api_key: str | None = None
    api_secret: str | None = None
    api_passphrase: str | None = None
    clob_host: str = "https://clob.polymarket.com"
    chain_id: int = 137

    @property
    def complete(self) -> bool:
        return bool(self.private_key and self.funder_address)


def load_config(path: str = "config/config.yaml") -> Config:
    load_dotenv()
    p = Path(path)
    data = yaml.safe_load(p.read_text()) if p.exists() else {}
    return Config(**(data or {}))


def load_secrets() -> Secrets:
    load_dotenv()
    return Secrets(
        private_key=os.getenv("POLYMARKET_PRIVATE_KEY") or None,
        funder_address=os.getenv("POLYMARKET_FUNDER_ADDRESS") or None,
        api_key=os.getenv("POLYMARKET_API_KEY") or None,
        api_secret=os.getenv("POLYMARKET_API_SECRET") or None,
        api_passphrase=os.getenv("POLYMARKET_API_PASSPHRASE") or None,
        clob_host=os.getenv("POLYMARKET_CLOB_HOST") or "https://clob.polymarket.com",
        chain_id=int(os.getenv("POLYMARKET_CHAIN_ID") or 137),
    )
