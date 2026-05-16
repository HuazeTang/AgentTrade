"""Global configuration for the backtesting system."""

from __future__ import annotations

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "cache"
FACTOR_DIR = PROJECT_ROOT / "data" / "factors"
MODEL_DIR = PROJECT_ROOT / "data" / "models"
RESULT_DIR = PROJECT_ROOT / "data" / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FACTOR_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ── A-share constants ────────────────────────────────────────────────────────

LOT_SIZE = 100                  # minimum shares per order (multiples of 100)
MIN_LISTING_DAYS = 60           # exclude stocks listed < 60 trading days

# Price limit fractions per board
PRICE_LIMITS = {
    "main_board": 0.10,
    "star_market": 0.20,
    "chinext": 0.20,
    "beijing": 0.30,
    "st": 0.05,
}

# Transaction costs
COMMISSION_RATE = 0.00025       # 0.025% per side
MIN_COMMISSION = 5.0            # minimum 5 CNY per trade
STAMP_TAX_RATE = 0.001          # 0.1% sell only
TRANSFER_FEE_RATE = 0.00001     # 0.001% per side

# Default slippage in basis points
DEFAULT_SLIPPAGE_BPS = 5.0

# Default benchmark index
DEFAULT_BENCHMARK = "000300.SH"  # CSI 300

# Max position as fraction of portfolio
DEFAULT_MAX_POSITION_PCT = 0.10

# Trading days per year (approximate)
TRADING_DAYS_PER_YEAR = 252

# Column names used across the system
COL_TRADE_DATE = "trade_date"
COL_SYMBOL = "symbol"
COL_OPEN = "open"
COL_HIGH = "high"
COL_LOW = "low"
COL_CLOSE = "close"
COL_PRE_CLOSE = "pre_close"
COL_VOLUME = "volume"
COL_AMOUNT = "amount"
COL_ADJ_FACTOR = "adj_factor"
COL_TURNOVER = "turnover"
COL_MARKET_CAP = "market_cap"
COL_TRADABLE_SHARES = "tradable_shares"
COL_IS_SUSPENDED = "is_suspended"
COL_IS_ST = "is_st"
COL_PRICE_LIMIT = "price_limit_frac"
COL_BOARD = "board"
COL_INDUSTRY = "industry"
