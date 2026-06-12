"""Polymarket access: Gamma (market discovery, read-only) + CLOB (prices / orders).

Read paths (find market, get price) need no auth. Order placement is implemented
in betting/executor.py and only invoked when live mode is fully enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
import requests

from sportedge.config import Secrets


@dataclass
class MarketRef:
    slug: str
    question: str
    token_ids: list[str]          # CLOB token ids (usually [YES, NO])
    outcomes: list[str]           # parallel labels, e.g. ["Yes", "No"] or team names
    id: str = ""
    condition_id: str = ""
    end_date: str = ""
    closed: bool = False


@dataclass(frozen=True)
class PricePoint:
    timestamp: int
    price: float


def unix_ts(value: str | int | float | None) -> int | None:
    """Parse unix seconds or an ISO-ish datetime string into unix seconds."""
    if value is None or value == "":
        return None
    if isinstance(value, int | float):
        return int(value)
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


class PolymarketClient:
    def __init__(
        self,
        gamma_host: str = "https://gamma-api.polymarket.com",
        clob_host: str = "https://clob.polymarket.com",
        chain_id: int = 137,
        secrets: Secrets | None = None,
    ):
        self.gamma_host = gamma_host.rstrip("/")
        self.clob_host = clob_host.rstrip("/")
        self.chain_id = chain_id
        self.secrets = secrets
        self._clob = None

    # ----- Gamma: discovery (no auth) -----
    def _gamma_get(self, path: str, params: dict[str, object] | None = None):
        resp = requests.get(f"{self.gamma_host}{path}", params=params or {}, timeout=15)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _json_list(value) -> list:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        import json as _json

        return list(_json.loads(value))

    @classmethod
    def _market_ref(cls, market: dict) -> MarketRef:
        return MarketRef(
            slug=market.get("slug", ""),
            question=market.get("question", ""),
            token_ids=[str(token) for token in cls._json_list(market.get("clobTokenIds"))],
            outcomes=[str(outcome) for outcome in cls._json_list(market.get("outcomes"))],
            id=str(market.get("id", "")),
            condition_id=str(market.get("conditionId", "")),
            end_date=market.get("endDate") or market.get("endDateIso") or "",
            closed=bool(market.get("closed", False)),
        )

    def list_markets(
        self,
        *,
        slug: str = "",
        search: str = "",
        active: bool | None = None,
        closed: bool | None = False,
        limit: int = 20,
    ) -> list[dict]:
        params: dict[str, object] = {"limit": limit}
        if slug:
            params["slug"] = slug
        if search:
            params["search"] = search
        if active is not None:
            params["active"] = str(active).lower()
        if closed is not None:
            params["closed"] = str(closed).lower()
        return self._gamma_get("/markets", params)

    def list_events(
        self,
        *,
        search: str = "",
        active: bool | None = None,
        closed: bool | None = False,
        limit: int = 20,
    ) -> list[dict]:
        params: dict[str, object] = {"limit": limit}
        if search:
            params["search"] = search
        if active is not None:
            params["active"] = str(active).lower()
        if closed is not None:
            params["closed"] = str(closed).lower()
        return self._gamma_get("/events", params)

    def public_search(self, query: str, limit: int = 20) -> dict:
        """Search public events, markets, and profiles via Gamma public-search."""
        return self._gamma_get("/public-search", {"q": query, "limit": limit})

    def find_market(
        self,
        slug: str = "",
        query: str = "",
        closed: bool | None = False,
        active: bool | None = None,
    ) -> MarketRef | None:
        markets = self.list_markets(
            slug=slug,
            search=query,
            active=active,
            closed=closed,
            limit=20,
        )
        if not markets:
            return None
        if query and not slug:
            q = query.lower()
            markets = [m for m in markets if q in (m.get("question", "").lower())] or markets
        return self._market_ref(markets[0])

    # ----- CLOB: live price (no auth needed for reads) -----
    def _clob_client(self):
        if self._clob is None:
            from py_clob_client.client import ClobClient  # lazy: heavy import

            self._clob = ClobClient(self.clob_host, chain_id=self.chain_id)
        return self._clob

    def get_midpoint(self, token_id: str) -> float:
        """Mid price (≈ implied probability) for a CLOB token."""
        resp = self._clob_client().get_midpoint(token_id)
        return float(resp["mid"]) if isinstance(resp, dict) else float(resp)

    def get_price(self, token_id: str, side: str = "BUY") -> float:
        """Best price for a side. Falls back to midpoint on any issue."""
        try:
            resp = self._clob_client().get_price(token_id, side)
            return float(resp["price"]) if isinstance(resp, dict) else float(resp)
        except Exception:
            return self.get_midpoint(token_id)

    def get_prices_history(
        self,
        token_id: str,
        start_ts: int | None = None,
        end_ts: int | None = None,
        interval: str = "all",
        fidelity: int = 1,
    ) -> list[PricePoint]:
        """Historical token prices from CLOB /prices-history."""
        params: dict[str, object] = {
            "market": token_id,
            "interval": interval,
            "fidelity": fidelity,
        }
        if start_ts is not None:
            params["startTs"] = start_ts
        if end_ts is not None:
            params["endTs"] = end_ts
        resp = requests.get(f"{self.clob_host}/prices-history", params=params, timeout=30)
        resp.raise_for_status()
        history = resp.json().get("history", [])
        return [
            PricePoint(timestamp=int(point["t"]), price=float(point["p"]))
            for point in history
            if "t" in point and "p" in point
        ]

    def prices_history_frame(
        self,
        token_id: str,
        start_ts: int | None = None,
        end_ts: int | None = None,
        interval: str = "all",
        fidelity: int = 1,
    ) -> pd.DataFrame:
        """Historical token prices as a DataFrame ready for parquet caching."""
        rows = [
            {
                "token_id": token_id,
                "timestamp": point.timestamp,
                "price": point.price,
            }
            for point in self.get_prices_history(token_id, start_ts, end_ts, interval, fidelity)
        ]
        return pd.DataFrame(rows, columns=["token_id", "timestamp", "price"])
