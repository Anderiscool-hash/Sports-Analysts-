"""Kalshi access: market discovery + live prices (read) and signed order placement.

Kalshi is a US-regulated exchange. Its 1X2 World Cup market is three yes/no contracts
(home / draw / away), each identified by a *ticker*. Prices are quoted in cents (1..99),
which this client converts to a [0, 1] probability so it feeds the same
``edge`` / ``BottomDetector`` / ``Strategy`` spine the loops are built on.

Read paths (market lookup, price) need no auth. Order placement uses Kalshi's RSA-PSS
request signing and is only ever invoked from a fully-gated live executor.

Docs: https://trading-api.readme.io/  (trade-api/v2)
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass

import requests

from sportedge.config import Secrets

DEFAULT_HOST = "https://api.elections.kalshi.com/trade-api/v2"


@dataclass
class KalshiMarketRef:
    """Market reference for the loop's use: ``token_ids`` are Kalshi tickers,
    ``outcomes`` their labels."""

    slug: str
    question: str
    token_ids: list[str]
    outcomes: list[str]


def cents_to_prob(cents: float) -> float:
    """Kalshi quotes 1..99 cents; the probability is cents / 100, clamped to (0, 1)."""
    return min(0.999, max(0.001, float(cents) / 100.0))


class KalshiClient:
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        secrets: Secrets | None = None,
    ):
        self.host = host.rstrip("/")
        self.secrets = secrets
        self._private_key = None

    # ----- discovery / price (no auth) -----
    def _get(self, path: str, params: dict | None = None) -> dict:
        resp = requests.get(f"{self.host}{path}", params=params or {}, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_market(self, ticker: str) -> dict:
        return self._get(f"/markets/{ticker}").get("market", {})

    def get_price(self, ticker: str, side: str = "BUY") -> float | None:
        """Best price to take a YES position, as a [0, 1] probability.

        BUY uses the yes-ask; SELL the yes-bid. Falls back to last price / midpoint.
        Returns ``None`` if the market quotes nothing usable.
        """
        try:
            m = self.get_market(ticker)
        except Exception:  # noqa: BLE001 - let the loop degrade
            return None
        yes_bid = m.get("yes_bid")
        yes_ask = m.get("yes_ask")
        last = m.get("last_price")
        cents: float | None
        if side.upper() == "BUY":
            cents = yes_ask if yes_ask else (last if last else yes_bid)
        else:
            cents = yes_bid if yes_bid else (last if last else yes_ask)
        if not cents:
            if yes_bid and yes_ask:
                cents = (yes_bid + yes_ask) / 2
            else:
                return None
        return cents_to_prob(cents)

    def find_market(self, slug: str = "", query: str = "") -> KalshiMarketRef | None:
        """Best-effort single-market lookup by ticker (``slug``) or text search."""
        if slug:
            m = self.get_market(slug)
            if m:
                return KalshiMarketRef(
                    slug=slug,
                    question=m.get("title", ""),
                    token_ids=[m.get("ticker", slug)],
                    outcomes=[m.get("yes_sub_title") or m.get("title", "")],
                )
        if query:
            data = self._get("/markets", {"status": "open", "limit": 20})
            for m in data.get("markets", []):
                if query.lower() in str(m.get("title", "")).lower():
                    return KalshiMarketRef(
                        slug=m.get("ticker", ""),
                        question=m.get("title", ""),
                        token_ids=[m.get("ticker", "")],
                        outcomes=[m.get("yes_sub_title") or m.get("title", "")],
                    )
        return None

    # ----- signing / orders (auth) -----
    def _load_private_key(self):
        if self._private_key is None:
            from cryptography.hazmat.primitives.serialization import load_pem_private_key

            if not self.secrets or not self.secrets.kalshi_private_key_pem:
                raise ValueError("Kalshi order placement requires a private key")
            self._private_key = load_pem_private_key(
                self.secrets.kalshi_private_key_pem.encode(), password=None
            )
        return self._private_key

    def _sign(self, message: str) -> str:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        signature = self._load_private_key().sign(
            message.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode()

    def auth_headers(self, method: str, path: str, timestamp_ms: int | None = None) -> dict:
        """Kalshi RSA-PSS request headers. ``path`` is the request path (no host)."""
        if not self.secrets or not self.secrets.kalshi_api_key_id:
            raise ValueError("Kalshi order placement requires an API key id")
        ts = str(timestamp_ms if timestamp_ms is not None else int(time.time() * 1000))
        message = ts + method.upper() + path
        return {
            "KALSHI-ACCESS-KEY": self.secrets.kalshi_api_key_id,
            "KALSHI-ACCESS-SIGNATURE": self._sign(message),
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "Content-Type": "application/json",
        }

    @staticmethod
    def build_order_payload(ticker: str, stake_usd: float, price_prob: float) -> dict:
        """A limit BUY of YES shares. ``count`` = whole shares affordable at this price;
        ``yes_price`` is in cents."""
        yes_price_cents = max(1, min(99, round(price_prob * 100)))
        count = max(1, int(stake_usd / max(price_prob, 0.01)))
        return {
            "ticker": ticker,
            "action": "buy",
            "side": "yes",
            "type": "limit",
            "count": count,
            "yes_price": yes_price_cents,
        }

    def place_order(self, ticker: str, stake_usd: float, price_prob: float) -> dict:
        """Submit a signed limit order. Caller is responsible for all safety gating."""
        path = "/portfolio/orders"
        payload = self.build_order_payload(ticker, stake_usd, price_prob)
        headers = self.auth_headers("POST", path)
        resp = requests.post(f"{self.host}{path}", json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()
