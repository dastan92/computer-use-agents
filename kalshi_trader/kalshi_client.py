"""Kalshi API client for market data, trading, and historical data."""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, utils

from .config import KalshiConfig
from .models import Event, Market, MarketHistory, OrderAction, Side, Trade

logger = logging.getLogger(__name__)


class KalshiClient:
    """Client for the Kalshi REST API v2.

    Supports both production and demo environments.
    Uses RSA signature authentication per Kalshi's API spec.
    """

    def __init__(self, config: KalshiConfig):
        self.config = config
        self.base_url = config.base_url
        self.session = requests.Session()
        self._private_key = None

        if config.private_key_path and Path(config.private_key_path).exists():
            self._load_private_key()

    def _load_private_key(self):
        """Load RSA private key for API authentication."""
        key_path = Path(self.config.private_key_path)
        with open(key_path, "rb") as f:
            self._private_key = serialization.load_pem_private_key(f.read(), password=None)
        logger.info("Loaded private key from %s", key_path)

    def _sign_request(self, method: str, path: str, timestamp_ms: int) -> str:
        """Create RSA signature for Kalshi API authentication.

        Signs: timestamp_ms + method + path
        """
        if not self._private_key:
            raise ValueError(
                "Private key not loaded. Set KALSHI_PRIVATE_KEY_PATH in .env"
            )

        message = f"{timestamp_ms}{method}{path}".encode()
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            utils.Prehashed(hashes.SHA256()),
        )
        import base64

        return base64.b64encode(signature).decode()

    def _get_headers(self, method: str, path: str) -> dict:
        """Build authenticated request headers."""
        timestamp_ms = int(time.time() * 1000)

        headers = {"Content-Type": "application/json"}

        if self._private_key and self.config.api_key_id:
            signature = self._sign_request(method, path, timestamp_ms)
            headers.update(
                {
                    "KALSHI-ACCESS-KEY": self.config.api_key_id,
                    "KALSHI-ACCESS-SIGNATURE": signature,
                    "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
                }
            )

        return headers

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        json_body: Optional[dict] = None,
    ) -> dict:
        """Make an authenticated request to the Kalshi API."""
        url = f"{self.base_url}{path}"
        headers = self._get_headers(method.upper(), path)

        response = self.session.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=json_body,
            timeout=30,
        )

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 5))
            logger.warning("Rate limited. Waiting %ds...", retry_after)
            time.sleep(retry_after)
            return self._request(method, path, params, json_body)

        response.raise_for_status()
        return response.json()

    # --- Market Data ---

    def get_markets(
        self,
        limit: int = 100,
        cursor: Optional[str] = None,
        event_ticker: Optional[str] = None,
        status: Optional[str] = None,
        series_ticker: Optional[str] = None,
        min_volume: Optional[int] = None,
    ) -> tuple[list[Market], Optional[str]]:
        """Fetch markets with optional filters.

        Returns (markets, next_cursor) for pagination.
        """
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if event_ticker:
            params["event_ticker"] = event_ticker
        if status:
            params["status"] = status
        if series_ticker:
            params["series_ticker"] = series_ticker

        data = self._request("GET", "/markets", params=params)
        markets = []

        for m in data.get("markets", []):
            markets.append(
                Market(
                    ticker=m.get("ticker", ""),
                    event_ticker=m.get("event_ticker", ""),
                    title=m.get("title", ""),
                    subtitle=m.get("subtitle", ""),
                    status=m.get("status", "unknown"),
                    yes_bid=m.get("yes_bid", 0),
                    yes_ask=m.get("yes_ask", 0),
                    no_bid=m.get("no_bid", 0),
                    no_ask=m.get("no_ask", 0),
                    last_price=m.get("last_price", 0),
                    volume=m.get("volume", 0),
                    volume_24h=m.get("volume_24h", 0),
                    open_interest=m.get("open_interest", 0),
                    close_time=m.get("close_time"),
                    expiration_time=m.get("expiration_time"),
                    result=m.get("result"),
                    category=m.get("category", ""),
                )
            )

        next_cursor = data.get("cursor")
        if min_volume:
            markets = [mk for mk in markets if mk.volume >= min_volume]

        return markets, next_cursor

    def get_market(self, ticker: str) -> Market:
        """Get a single market by ticker."""
        data = self._request("GET", f"/markets/{ticker}")
        m = data.get("market", data)
        return Market(
            ticker=m.get("ticker", ticker),
            event_ticker=m.get("event_ticker", ""),
            title=m.get("title", ""),
            subtitle=m.get("subtitle", ""),
            status=m.get("status", "unknown"),
            yes_bid=m.get("yes_bid", 0),
            yes_ask=m.get("yes_ask", 0),
            no_bid=m.get("no_bid", 0),
            no_ask=m.get("no_ask", 0),
            last_price=m.get("last_price", 0),
            volume=m.get("volume", 0),
            volume_24h=m.get("volume_24h", 0),
            open_interest=m.get("open_interest", 0),
            close_time=m.get("close_time"),
            expiration_time=m.get("expiration_time"),
            result=m.get("result"),
            category=m.get("category", ""),
        )

    def get_market_history(
        self,
        ticker: str,
        limit: int = 1000,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
    ) -> list[MarketHistory]:
        """Fetch historical candlestick/snapshot data for a market."""
        params = {"limit": limit}
        if min_ts:
            params["min_ts"] = min_ts
        if max_ts:
            params["max_ts"] = max_ts

        data = self._request("GET", f"/markets/{ticker}/history", params=params)
        history = []

        for h in data.get("history", []):
            ts = h.get("ts")
            if isinstance(ts, (int, float)):
                timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)
            else:
                timestamp = datetime.fromisoformat(str(ts))

            history.append(
                MarketHistory(
                    ticker=ticker,
                    timestamp=timestamp,
                    yes_price=h.get("yes_price", h.get("price", 0)),
                    volume=h.get("volume", 0),
                    open_interest=h.get("open_interest", 0),
                )
            )

        return history

    def get_trades(
        self, ticker: str, limit: int = 100, cursor: Optional[str] = None
    ) -> tuple[list[Trade], Optional[str]]:
        """Fetch recent trades for a market."""
        params = {"ticker": ticker, "limit": limit}
        if cursor:
            params["cursor"] = cursor

        data = self._request("GET", "/markets/trades", params=params)
        trades = []

        for t in data.get("trades", []):
            ts = t.get("created_time", t.get("ts"))
            if isinstance(ts, (int, float)):
                timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)
            else:
                timestamp = datetime.fromisoformat(str(ts))

            trades.append(
                Trade(
                    ticker=t.get("ticker", ticker),
                    trade_id=str(t.get("trade_id", "")),
                    timestamp=timestamp,
                    side=Side.YES if t.get("taker_side") == "yes" else Side.NO,
                    action=OrderAction.BUY,
                    price=t.get("yes_price", t.get("price", 0)),
                    count=t.get("count", 1),
                )
            )

        next_cursor = data.get("cursor")
        return trades, next_cursor

    def get_events(
        self, limit: int = 50, status: Optional[str] = None
    ) -> list[Event]:
        """Fetch events."""
        params = {"limit": limit}
        if status:
            params["status"] = status

        data = self._request("GET", "/events", params=params)
        events = []

        for e in data.get("events", []):
            events.append(
                Event(
                    event_ticker=e.get("event_ticker", ""),
                    title=e.get("title", ""),
                    category=e.get("category", ""),
                )
            )

        return events

    # --- Trading ---

    def get_balance(self) -> float:
        """Get account balance in dollars."""
        data = self._request("GET", "/portfolio/balance")
        return data.get("balance", 0) / 100.0  # cents to dollars

    def place_order(
        self,
        ticker: str,
        side: Side,
        action: OrderAction,
        count: int,
        price: int,  # in cents (1-99)
    ) -> dict:
        """Place a limit order on Kalshi.

        Args:
            ticker: Market ticker
            side: "yes" or "no"
            action: "buy" or "sell"
            count: Number of contracts
            price: Limit price in cents (1-99)
        """
        body = {
            "ticker": ticker,
            "action": action.value,
            "side": side.value,
            "count": count,
            "type": "limit",
            "yes_price": price if side == Side.YES else None,
            "no_price": price if side == Side.NO else None,
        }
        # Remove None values
        body = {k: v for k, v in body.items() if v is not None}

        return self._request("POST", "/portfolio/orders", json_body=body)

    def get_positions(self) -> list[dict]:
        """Get current open positions."""
        data = self._request("GET", "/portfolio/positions")
        return data.get("market_positions", [])

    def get_orders(self, ticker: Optional[str] = None) -> list[dict]:
        """Get open orders."""
        params = {}
        if ticker:
            params["ticker"] = ticker
        data = self._request("GET", "/portfolio/orders", params=params)
        return data.get("orders", [])

    # --- Data Collection for Backtesting ---

    def collect_settled_markets(
        self, limit: int = 200, category: Optional[str] = None
    ) -> list[Market]:
        """Collect settled (resolved) markets for backtesting.

        These are markets where the outcome is known, which is essential
        for measuring how well our AI predictions would have performed.
        """
        all_markets = []
        cursor = None

        while len(all_markets) < limit:
            batch_limit = min(100, limit - len(all_markets))
            markets, cursor = self.get_markets(
                limit=batch_limit, cursor=cursor, status="settled"
            )
            all_markets.extend(markets)

            if not cursor or not markets:
                break

            time.sleep(0.5)  # respect rate limits

        return all_markets

    def save_market_data(self, markets: list[Market], filepath: str):
        """Save market data to JSON for offline backtesting."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [m.model_dump(mode="json") for m in markets]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info("Saved %d markets to %s", len(markets), filepath)

    def load_market_data(self, filepath: str) -> list[Market]:
        """Load market data from JSON."""
        with open(filepath) as f:
            data = json.load(f)
        return [Market(**m) for m in data]
