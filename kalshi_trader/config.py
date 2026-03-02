"""Configuration management for the Kalshi trading system."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class KalshiConfig:
    """Kalshi API configuration."""

    api_base_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    demo_base_url: str = "https://demo-api.kalshi.co/trade-api/v2"
    api_key_id: str = field(default_factory=lambda: os.getenv("KALSHI_API_KEY_ID", ""))
    private_key_path: str = field(
        default_factory=lambda: os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
    )
    use_demo: bool = field(
        default_factory=lambda: os.getenv("KALSHI_USE_DEMO", "true").lower() == "true"
    )

    @property
    def base_url(self) -> str:
        return self.demo_base_url if self.use_demo else self.api_base_url


@dataclass
class ClaudeConfig:
    """Claude API configuration."""

    api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    model: str = "claude-sonnet-4-6-20250514"
    max_tokens: int = 4096
    temperature: float = 0.3


@dataclass
class TradingConfig:
    """Trading parameters."""

    max_position_pct: float = 0.05  # max 5% of balance per position
    max_concurrent_positions: int = 10
    min_edge_threshold: float = 0.05  # 5% minimum edge to trade
    kelly_fraction: float = 0.25  # quarter-Kelly for conservative sizing
    stop_loss_pct: float = 0.15  # 15% stop loss
    take_profit_pct: float = 0.20  # 20% take profit
    min_volume: int = 100  # minimum contract volume
    max_days_to_expiry: int = 30


@dataclass
class BacktestConfig:
    """Backtesting parameters."""

    initial_balance: float = 10_000.0
    commission_per_contract: float = 0.01  # Kalshi fees
    slippage_cents: int = 1  # 1 cent slippage assumption
    data_dir: str = "backtest_data"


@dataclass
class AppConfig:
    """Root configuration."""

    kalshi: KalshiConfig = field(default_factory=KalshiConfig)
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
