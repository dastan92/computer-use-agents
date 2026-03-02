"""Data models for the Kalshi trading system."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Side(str, Enum):
    YES = "yes"
    NO = "no"


class OrderAction(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    RESTING = "resting"
    FILLED = "filled"
    CANCELED = "canceled"
    PENDING = "pending"


class MarketStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    SETTLED = "settled"


# --- Kalshi API Response Models ---


class Market(BaseModel):
    """A Kalshi prediction market."""

    ticker: str
    event_ticker: str
    title: str
    subtitle: str = ""
    status: str
    yes_bid: float = 0  # best bid for YES in cents
    yes_ask: float = 0  # best ask for YES in cents
    no_bid: float = 0
    no_ask: float = 0
    last_price: float = 0
    volume: int = 0
    volume_24h: int = 0
    open_interest: int = 0
    close_time: Optional[datetime] = None
    expiration_time: Optional[datetime] = None
    result: Optional[str] = None  # "yes", "no", or None if unsettled
    category: str = ""

    @property
    def yes_mid(self) -> float:
        if self.yes_bid and self.yes_ask:
            return (self.yes_bid + self.yes_ask) / 2
        return self.last_price

    @property
    def implied_probability(self) -> float:
        """Market-implied probability from yes mid price (cents -> probability)."""
        return self.yes_mid / 100.0

    @property
    def spread(self) -> float:
        return self.yes_ask - self.yes_bid if self.yes_ask and self.yes_bid else 0


class Event(BaseModel):
    """A Kalshi event containing multiple markets."""

    event_ticker: str
    title: str
    category: str = ""
    markets: list[Market] = Field(default_factory=list)


class MarketHistory(BaseModel):
    """Historical snapshot of a market at a point in time."""

    ticker: str
    timestamp: datetime
    yes_price: float  # in cents
    volume: int = 0
    open_interest: int = 0


class Trade(BaseModel):
    """A single trade record."""

    ticker: str
    trade_id: str = ""
    timestamp: datetime
    side: Side
    action: OrderAction
    price: float  # cents
    count: int  # number of contracts


# --- Analysis & Decision Models ---


class MarketAnalysis(BaseModel):
    """Claude's analysis of a market opportunity."""

    ticker: str
    market_title: str
    ai_probability: float = Field(
        ge=0, le=1, description="AI-estimated true probability"
    )
    market_probability: float = Field(
        ge=0, le=1, description="Current market-implied probability"
    )
    edge: float = Field(description="AI prob - market prob (positive = underpriced YES)")
    confidence: float = Field(
        ge=0, le=1, description="How confident the AI is in its estimate"
    )
    reasoning: str = Field(description="AI's reasoning for this assessment")
    recommended_side: Optional[Side] = None
    recommended_size: float = 0  # fraction of bankroll
    key_factors: list[str] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list)

    @property
    def abs_edge(self) -> float:
        return abs(self.edge)


class TradeSignal(BaseModel):
    """A concrete trade signal from the strategy."""

    ticker: str
    side: Side
    action: OrderAction
    target_price: float  # cents
    size: int  # number of contracts
    edge: float
    confidence: float
    reasoning: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# --- Backtest Models ---


class BacktestTrade(BaseModel):
    """A trade executed during backtesting."""

    ticker: str
    entry_time: datetime
    exit_time: Optional[datetime] = None
    side: Side
    entry_price: float  # cents
    exit_price: Optional[float] = None  # cents
    size: int  # contracts
    pnl: float = 0  # dollars
    settled_result: Optional[str] = None  # "yes" or "no"


class BacktestResult(BaseModel):
    """Summary of a backtest run."""

    start_date: datetime
    end_date: datetime
    initial_balance: float
    final_balance: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: Optional[float] = None
    win_rate: float = 0
    avg_edge: float = 0
    trades: list[BacktestTrade] = Field(default_factory=list)

    @property
    def return_pct(self) -> float:
        return (self.final_balance - self.initial_balance) / self.initial_balance * 100
