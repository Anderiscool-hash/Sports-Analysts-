"""Polymarket access: Gamma (market discovery, read-only) + CLOB (prices / orders).

Read paths (find market, get price) need no auth. Order placement is implemented
in betting/executor.py and only invoked when live mode is fully enabled.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

from sportedge.config import Secrets


@dataclass
class MarketRef:
    slug: str
    question: str
    token_ids: list[str]          # CLOB token ids (usually [YES, NO])
    outcomes: list[str]           # parallel labels, e.g. ["Yes", "No"] or team names


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
    def find_market(self, slug: str = "", query: str = "") -> MarketRef | None:
        params: dict[str, object] = {"closed": "false", "limit": 20}
        if slug:
            params["slug"] = slug
        resp = requests.get(f"{self.gamma_host}/markets", params=params, timeout=15)
        resp.raise_for_status()
        markets = resp.json()
        if not markets:
            return None
        if query and not slug:
            q = query.lower()
            markets = [m for m in markets if q in (m.get("question", "").lower())] or markets
        m = markets[0]
        import json as _json

        token_ids = m.get("clobTokenIds") or "[]"
        if isinstance(token_ids, str):
            token_ids = _json.loads(token_ids)
        outcomes = m.get("outcomes") or "[]"
        if isinstance(outcomes, str):
            outcomes = _json.loads(outcomes)
        return MarketRef(
            slug=m.get("slug", ""),
            question=m.get("question", ""),
            token_ids=list(token_ids),
            outcomes=list(outcomes),
        )

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
