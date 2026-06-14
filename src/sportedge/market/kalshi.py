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
import re
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


@dataclass(frozen=True)
class KalshiMarketSnapshot:
    """Display-friendly subset of Kalshi market metadata and quotes."""

    ticker: str
    title: str = ""
    status: str = ""
    yes_bid: float | None = None
    yes_ask: float | None = None
    last_price: float | None = None
    volume: int | None = None
    volume_24h: int | None = None
    liquidity: int | None = None
    open_interest: int | None = None
    close_time: str = ""


@dataclass(frozen=True)
class KalshiDiscoveryResult:
    """A candidate market discovered for a team/game."""

    ticker: str
    title: str
    team: str
    score: float
    yes_bid: float | None = None
    yes_ask: float | None = None
    last_price: float | None = None
    volume: int | None = None
    liquidity: int | None = None
    reason: str = ""


@dataclass(frozen=True)
class KalshiMarketRejection:
    """Why a market search hit was not usable for direct team-win trading."""

    ticker: str
    title: str
    reason: str
    yes_bid: float | None = None
    yes_ask: float | None = None
    last_price: float | None = None


def cents_to_prob(cents: float) -> float:
    """Kalshi quotes 1..99 cents; the probability is cents / 100, clamped to (0, 1)."""
    return min(0.999, max(0.001, float(cents) / 100.0))


def candle_close_prob(candle: dict) -> float | None:
    """Best-effort close price (as a [0, 1] probability) from one candlestick.

    Kalshi candlesticks vary in shape; the price may sit under ``price.{close|mean}``
    or under ``yes_bid``/``yes_ask`` (either a number or a nested OHLC dict). Returns
    ``None`` if no usable price is present."""
    price = candle.get("price")
    if isinstance(price, dict):
        for key in ("close", "mean", "open"):
            value = price.get(key)
            if value:
                return cents_to_prob(value)
    for side in ("yes_ask", "yes_bid"):
        value = candle.get(side)
        if isinstance(value, dict):
            inner = value.get("close") or value.get("mean")
            if inner:
                return cents_to_prob(inner)
        elif value:
            return cents_to_prob(value)
    return None


def _optional_prob(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return cents_to_prob(float(value))
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _market_snapshot_from_dict(market: dict, fallback_ticker: str = "") -> KalshiMarketSnapshot:
    return KalshiMarketSnapshot(
        ticker=str(market.get("ticker") or fallback_ticker),
        title=str(market.get("title") or market.get("event_title") or ""),
        status=str(market.get("status") or ""),
        yes_bid=_optional_prob(market.get("yes_bid")),
        yes_ask=_optional_prob(market.get("yes_ask")),
        last_price=_optional_prob(market.get("last_price")),
        volume=_optional_int(market.get("volume")),
        volume_24h=_optional_int(market.get("volume_24h")),
        liquidity=_optional_int(market.get("liquidity")),
        open_interest=_optional_int(market.get("open_interest")),
        close_time=str(market.get("close_time") or ""),
    )


def _text_blob(market: dict) -> str:
    keys = (
        "ticker",
        "title",
        "event_title",
        "subtitle",
        "yes_sub_title",
        "no_sub_title",
        "category",
    )
    return " ".join(str(market.get(key) or "") for key in keys)


def _tokens(text: str) -> set[str]:
    return {part for part in re.split(r"[^a-z0-9]+", text.lower()) if len(part) >= 3}


def _meaningful_team_tokens(team: str) -> set[str]:
    stop = {"the", "team", "club", "fc", "sc", "city", "united"}
    return {token for token in _tokens(team) if token not in stop}


def _team_tokens_match(team: str, text_tokens: set[str]) -> bool:
    stop = {"the", "team", "club", "fc", "sc", "city", "united"}
    tokens = [
        part
        for part in re.split(r"[^a-z0-9]+", team.lower())
        if len(part) >= 3 and part not in stop
    ]
    if not tokens:
        return False
    if set(tokens).issubset(text_tokens):
        return True
    # Kalshi sports text often uses city names ("San Antonio") rather than full
    # team names ("San Antonio Spurs"). Only allow this for 3+ token names; two
    # token names like "Boston Celtics" still need the full phrase/nickname.
    city_tokens = set(tokens[:-1])
    return len(tokens) >= 3 and bool(city_tokens) and city_tokens.issubset(text_tokens)


def _has_quote(snapshot: KalshiMarketSnapshot) -> bool:
    return any(
        value is not None
        for value in (snapshot.yes_bid, snapshot.yes_ask, snapshot.last_price)
    )


def _looks_like_direct_winner_market(text: str) -> bool:
    lower = text.lower()
    if lower.count(",") >= 1:
        return False
    blocked = (
        " parlay",
        "multi",
        "same game",
        "cross category",
        " over ",
        " under ",
        "points",
        "runs",
        "goals",
        "rebounds",
        "assists",
        "strikeouts",
        "touchdowns",
        "wins by",
        "by over",
        "spread",
        "total",
        "+",
    )
    if any(word in lower for word in blocked):
        return False
    winner_words = (" win", " wins", "winner", "moneyline")
    return any(word in lower for word in winner_words)


def score_market_for_team(market: dict, team: str, opponent: str = "") -> KalshiDiscoveryResult | None:
    """Score whether a Kalshi market looks like a direct YES contract for ``team``.

    The filter is intentionally conservative: a wrong market is worse than no market
    for paper-trading evaluation.
    """
    snapshot = _market_snapshot_from_dict(market)
    text = _text_blob(market)
    rejection = rejection_reason_for_team_market(market, team, opponent)
    if rejection:
        return None
    text_tokens = _tokens(text)
    opponent_bonus = 0.0
    opponent_tokens = _meaningful_team_tokens(opponent)
    if opponent_tokens and opponent_tokens.issubset(text_tokens):
        opponent_bonus = 1.0
    quote_bonus = sum(
        0.5 for value in (snapshot.yes_bid, snapshot.yes_ask, snapshot.last_price) if value is not None
    )
    liquidity_bonus = min(float(snapshot.liquidity or 0) / 10_000.0, 2.0)
    volume_bonus = min(float(snapshot.volume or 0) / 1_000.0, 2.0)
    score = 5.0 + opponent_bonus + quote_bonus + liquidity_bonus + volume_bonus
    return KalshiDiscoveryResult(
        ticker=snapshot.ticker,
        title=snapshot.title,
        team=team,
        score=score,
        yes_bid=snapshot.yes_bid,
        yes_ask=snapshot.yes_ask,
        last_price=snapshot.last_price,
        volume=snapshot.volume,
        liquidity=snapshot.liquidity,
        reason="direct quoted winner market",
    )


def rejection_reason_for_team_market(market: dict, team: str, opponent: str = "") -> str:
    """Return a human-readable rejection reason, or ``""`` when accepted."""
    snapshot = _market_snapshot_from_dict(market)
    text = _text_blob(market)
    text_tokens = _tokens(text)
    if str(market.get("status") or "").lower() not in ("", "active", "open"):
        return f"status is {market.get('status')}"
    if not _has_quote(snapshot):
        return "no usable quote"
    if not _looks_like_direct_winner_market(text):
        return "not a direct team-winner market"
    if not _team_tokens_match(team, text_tokens):
        return "team name not found in market text"
    return ""


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

    def list_open_markets(
        self,
        query: str = "",
        limit: int = 200,
        max_pages: int = 3,
    ) -> list[dict]:
        """Fetch open markets, following cursors when Kalshi supplies them."""
        markets: list[dict] = []
        cursor = ""
        for _ in range(max_pages):
            params: dict[str, object] = {"status": "open", "limit": limit}
            if query:
                params["query"] = query
            if cursor:
                params["cursor"] = cursor
            data = self._get("/markets", params)
            markets.extend(data.get("markets", []) or [])
            cursor = str(data.get("cursor") or "")
            if not cursor:
                break
        return markets

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

    def get_market_snapshot(self, ticker: str) -> KalshiMarketSnapshot | None:
        """Best-effort quote + metadata snapshot for richer CLI display."""
        if not ticker:
            return None
        try:
            m = self.get_market(ticker)
        except Exception:  # noqa: BLE001 - display detail is optional
            return None
        if not m:
            return None
        return _market_snapshot_from_dict(m, ticker)

    def discover_team_win_market(
        self,
        team: str,
        opponent: str = "",
        limit: int = 200,
        max_pages: int = 3,
    ) -> KalshiDiscoveryResult | None:
        """Find the best direct, quoted YES market for a team to win.

        Returns ``None`` when only combo/prop/unquoted markets are found.
        """
        queries = [team]
        if opponent:
            queries.append(f"{team} {opponent}")
        seen: set[str] = set()
        candidates: list[KalshiDiscoveryResult] = []
        for query in queries:
            try:
                markets = self.list_open_markets(query=query, limit=limit, max_pages=max_pages)
            except Exception:  # noqa: BLE001 - discovery is best-effort
                markets = []
            for market in markets:
                ticker = str(market.get("ticker") or "")
                if ticker in seen:
                    continue
                seen.add(ticker)
                result = score_market_for_team(market, team, opponent)
                if result is not None:
                    candidates.append(result)
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[0] if candidates else None

    def explain_team_win_market_search(
        self,
        team: str,
        opponent: str = "",
        limit: int = 20,
        max_pages: int = 1,
    ) -> list[KalshiMarketRejection]:
        """Return rejected search hits with reasons for CLI diagnostics."""
        rejections: list[KalshiMarketRejection] = []
        seen: set[str] = set()
        queries = [team]
        if opponent:
            queries.append(f"{team} {opponent}")
        for query in queries:
            try:
                markets = self.list_open_markets(query=query, limit=limit, max_pages=max_pages)
            except Exception as exc:  # noqa: BLE001
                return [KalshiMarketRejection("", "", f"search failed: {exc}")]
            for market in markets:
                ticker = str(market.get("ticker") or "")
                if ticker in seen:
                    continue
                seen.add(ticker)
                reason = rejection_reason_for_team_market(market, team, opponent)
                if not reason:
                    continue
                snapshot = _market_snapshot_from_dict(market)
                rejections.append(
                    KalshiMarketRejection(
                        ticker=snapshot.ticker,
                        title=snapshot.title or _text_blob(market),
                        reason=reason,
                        yes_bid=snapshot.yes_bid,
                        yes_ask=snapshot.yes_ask,
                        last_price=snapshot.last_price,
                    )
                )
        return rejections

    def get_candlesticks(
        self,
        series_ticker: str,
        ticker: str,
        lookback_hours: int = 24,
        period_interval: int = 60,
    ) -> list[float]:
        """Historical close prices (as [0, 1] probabilities) for a contract.

        Hits the signed ``/series/{series}/markets/{ticker}/candlesticks`` endpoint
        (auth required, so Kalshi keys must be present). Returns ``[]`` on missing
        keys, network error, or an illiquid contract with no trade history."""
        path = f"/trade-api/v2/series/{series_ticker}/markets/{ticker}/candlesticks"
        # The signed message uses the full path incl. /trade-api/v2, so request
        # against the bare host (self.host already carries the /trade-api/v2 suffix).
        base = self.host.rsplit("/trade-api/", 1)[0]
        end_ts = int(time.time())
        start_ts = end_ts - lookback_hours * 3600
        try:
            headers = self.auth_headers("GET", path)
            resp = requests.get(
                base + path,
                params={"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_interval},
                headers=headers,
                timeout=20,
            )
            resp.raise_for_status()
            candles = resp.json().get("candlesticks", [])
        except Exception:  # noqa: BLE001 - chart data is optional; degrade to empty
            return []
        prices = [candle_close_prob(c) for c in candles]
        return [p for p in prices if p is not None]

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
