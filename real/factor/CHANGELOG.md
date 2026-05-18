# Factor Changelog

## Versioning Convention

- **MAJOR** (X.0.0): Formula/algorithm change — results are incompatible
- **MINOR** (0.X.0): Parameter change (lookback window, threshold)
- **PATCH** (0.0.X): Bug fix, edge case, data handling improvement

---

## 1.0.0 (Initial)

All built-in factors at baseline version `1.0.0`:

| Category | Factor | Ver | Lookback | Description |
|----------|--------|-----|----------|-------------|
| momentum | `momentum_1m` | 1.0.0 | 21 | 21-day price momentum |
| momentum | `momentum_3m` | 1.0.0 | 63 | 63-day price momentum |
| momentum | `momentum_6m` | 1.0.0 | 126 | 126-day price momentum |
| momentum | `momentum_12m1m` | 1.0.0 | 252 | 12m-1m momentum (skip recent month) |
| momentum | `reversal_5d` | 1.0.0 | 5 | 5-day short-term reversal |
| momentum | `reversal_10d` | 1.0.0 | 10 | 10-day short-term reversal |
| value | `ep` | 1.0.0 | 0 | Earnings yield (1/PE) |
| value | `bp` | 1.0.0 | 0 | Book-to-price (1/PB) |
| value | `sp` | 1.0.0 | 0 | Sales-to-price (1/PS) |
| volatility | `volatility_20d` | 1.0.0 | 20 | 20-day historical volatility |
| volatility | `volatility_60d` | 1.0.0 | 60 | 60-day historical volatility |
| volatility | `beta_60d` | 1.0.0 | 60 | 60-day rolling market beta |
| size | `ln_market_cap` | 1.0.0 | 0 | Natural log of market cap |
| liquidity | `turnover_20d` | 1.0.0 | 20 | 20-day average turnover |
| liquidity | `amihud_20d` | 1.0.0 | 20 | Amihud illiquidity (20-day) |
| growth | `revenue_growth_yoy` | 1.0.0 | 0 | Revenue growth YoY |
| growth | `earnings_growth_yoy` | 1.0.0 | 0 | Earnings growth YoY |
| trend | `trend_efficiency_20d` | 1.0.0 | 20 | Kaufman efficiency ratio |
| trend | `ma_trend_5_20` | 1.0.0 | 20 | (close - MA20) / MA20 |
| trend | `donchian_pct_20d` | 1.0.0 | 20 | Donchian channel position |
| trend | `up_days_ratio_20d` | 1.0.0 | 20 | Ratio of up-close days |
| trend | `ma_cross_5_20` | 1.0.0 | 20 | MA5 / MA20 - 1 |
| leader | `limit_up_freq_20d` | 1.0.0 | 20 | Limit-up frequency (20 days) |
| leader | `relative_strength_10d` | 1.0.0 | 10 | Excess return vs sector avg |
| leader | `volume_surge_5d` | 1.0.0 | 20 | Volume surge ratio (5d/20d) |
| leader | `close_position_5d` | 1.0.0 | 5 | Intraday close position (5d avg) |
| volume_price | `vol_weighted_mom_5d` | 1.0.0 | 20 | Volume-weighted 5-day momentum |
| volume_price | `money_flow_ratio_20d` | 1.0.0 | 20 | Positive money flow ratio |
| volume_price | `vwap_delta_5d` | 1.0.0 | 20 | (close - 5d VWAP) / 5d VWAP |
| volume_price | `vol_price_div_5d` | 1.0.0 | 20 | Price-volume divergence (5d) |
| risk | `downside_vol_20d` | 1.0.0 | 20 | Downside-only semi-deviation |
| risk | `max_dd_20d` | 1.0.0 | 20 | Drawdown from 20-day peak |
| risk | `risk_adj_mom_20d` | 1.0.0 | 20 | Risk-adjusted momentum (Sharpe-like) |
| risk | `dd_recovery_5d` | 1.0.0 | 20 | Bounce from 20-day low |
| risk | `market_dd_beta_20d` | 1.0.0 | 60 | Market drawdown * stock beta |

---

## 2026-05-17 — v1.0.0 New Factors Added

### momentum_accel_20d (momentum) — v1.0.0
- **What**: Price acceleration — 2nd derivative of price (mom_20d - mom_20d.shift(20))
- **Why**: Captures trend inflection points before they show in price level
- **IC weight**: 0.013

### vol_ratio_20_60 (volatility) — v1.0.0
- **What**: 20-day vol / 60-day vol — volatility regime change detector
- **Why**: A-shares often experience explosive vol expansion at trend starts; >1 signals breakout regime
- **IC weight**: 0.044

### daily_amplitude_20d (volatility) — v1.0.0
- **What**: 20-day average (high-low)/close — intraday range intensity
- **Why**: A-shares have higher intraday volatility than US; wide-range stocks attract speculative capital
- **IC weight**: 0.059 (6th of 28 active)

### overnight_gap_5d (overnight) — v1.0.0
- **What**: 5-day average overnight gap return (open vs prev_close)
- **Why**: A-share 9:15-9:25 call auction aggregates overnight information; persistent gap direction is a powerful signal
- **IC weight**: 0.010

### gap_strength_5d (overnight) — v1.0.0
- **What**: Ratio of positive-gap days to total days over 5 days
- **Why**: Consistency of upward openings matters more than magnitude
- **IC weight**: 0.007

### Active Baseline (used in backtest)

26 of 35 factors active, 9 excluded:
- **Value (3)**: `ep`, `bp`, `sp` — need fundamental data not in daily cache
- **Size (1)**: `ln_market_cap` — needs market cap data
- **Liquidity (1)**: `amihud_20d` — highly correlated with turnover_20d
- **Growth (2)**: `revenue_growth_yoy`, `earnings_growth_yoy` — need financial data
- **Risk (4)**: `downside_vol_20d`, `max_dd_20d`, `risk_adj_mom_20d`, `dd_recovery_5d` — disabled for performance reasons

---

## Template for future entries

```
## [YYYY-MM-DD] — Version X.Y.Z

### factor_name (category) — vX.Y.Z
- **What changed**: ...
- **Why**: ...
- **Impact**: IC_mean: A → B, IC_IR: X → Y
```
