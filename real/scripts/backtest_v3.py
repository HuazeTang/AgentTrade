"""V3 Backtest: daily top-3, trailing stop + hard stop + model override, 2025 full year.
Includes: limit-down unsellable handling, limit-up unbuyable handling.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from dl import DualTowerModel
from dl_v3.derived_features import ALL_V3_COLUMNS, build_v3_feature_cache
from data.cache import read_daily

# ── Config ──
SEQUENCE_LENGTH = 20
TOP_K = 3
MAX_HOLD_DAYS = 15
TRAILING_STOP = 0.12
MA_PERIOD = 20  # trend line period for exit signal
HARD_STOP = -0.08
MAX_LIMIT_DOWN_DAYS = 3  # force sell after 3 consecutive limit-downs
CHECKPOINT = "data/models/yaogu_v3_20260530_0532_best.pt"

BACKTEST_START = date(2021, 1, 1)
BACKTEST_END = date(2026, 5, 29)

# ── Load data ──
load_start = date(2019, 10, 1)
load_end = date(2026, 5, 29)

logging.info("Loading data: %s ~ %s", load_start, load_end)
daily = read_daily(load_start, load_end, prefix="daily_badj")

stock_list = pd.read_parquet("data/cache/stock_list.parquet")
st_symbols = set(stock_list[stock_list["name"].str.contains(r"\*?ST", na=False)]["symbol"])
valid_symbols = set(stock_list["symbol"]) - st_symbols

logging.info("Building V3 feature cache...")
cache = build_v3_feature_cache(daily)
feature_cols = [c for c in ALL_V3_COLUMNS if c in cache.columns]

dates_all = sorted(cache.index.get_level_values("trade_date").unique())
symbols_all = sorted(cache.index.get_level_values("symbol").unique())

feat_mats = []
for col in feature_cols:
    mat = cache[col].unstack()
    mat = mat.reindex(index=dates_all, columns=symbols_all)
    feat_mats.append(mat.values)
feat_tensor = np.stack(feat_mats, axis=-1).astype(np.float32)
feat_tensor = np.nan_to_num(feat_tensor, nan=0.0, posinf=0.0, neginf=0.0)

close = daily["close"].unstack()
close_mat = close.reindex(index=dates_all, columns=symbols_all).values.astype(np.float32)
open_mat = daily["open"].unstack().reindex(index=dates_all, columns=symbols_all).values.astype(np.float32)
high_mat = daily["high"].unstack().reindex(index=dates_all, columns=symbols_all).values.astype(np.float32)
volume = daily["volume"].unstack()
vol_mat = volume.reindex(index=dates_all, columns=symbols_all).values.astype(np.float32)

# MA10 of close (for trend exit) and MA20 of volume (for volume expansion check)
def rolling_mean(mat: np.ndarray, window: int) -> np.ndarray:
    """NaN-safe rolling mean along axis=0."""
    result = np.full_like(mat, np.nan, dtype=np.float32)
    for i in range(window - 1, mat.shape[0]):
        sl = mat[i - window + 1:i + 1, :]
        result[i, :] = np.nanmean(sl, axis=0)
    return result

ma20_mat = rolling_mean(close_mat, MA_PERIOD)
ma20_vol_mat = rolling_mean(vol_mat, 20)

# Pre-close: day t's pre_close is day t-1's close (for limit detection)
pre_close_mat = np.roll(close_mat, 1, axis=0)
pre_close_mat[0, :] = np.nan

symbol_to_idx = {s: i for i, s in enumerate(symbols_all)}
non_st_indices = np.array([symbol_to_idx[s] for s in symbols_all if s in valid_symbols])
non_st_symbols = [s for s in symbols_all if s in valid_symbols]

stock_info = stock_list.set_index("symbol")[["name"]]

# ── Load model ──
device = "mps" if torch.backends.mps.is_available() else "cpu"
ck = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model = DualTowerModel(in_features=ck["model_kwargs"]["in_features"])
model.load_state_dict(ck["model_state_dict"])
model.to(device)
model.eval()

# ── Daily inference ──
logging.info("Running daily inference...")
start_test = max(SEQUENCE_LENGTH, next(i for i, d in enumerate(dates_all)
                if pd.Timestamp(d).date() >= BACKTEST_START))
end_test = len(dates_all) - 1

daily_topk = {}
for idx in range(start_test, end_test + 1):
    td = dates_all[idx]
    if td.date() < BACKTEST_START or td.date() > BACKTEST_END:
        continue
    seq_slice = feat_tensor[idx - SEQUENCE_LENGTH:idx, :, :]
    seq_slice = seq_slice[:, non_st_indices, :]
    batch = np.transpose(seq_slice, (1, 0, 2))
    X_tensor = torch.from_numpy(batch).to(device)
    with torch.no_grad():
        scores = model.predict_proba(X_tensor).cpu().numpy().flatten()
    top_k_idx = np.argsort(scores)[-TOP_K:][::-1]
    daily_topk[idx] = [(non_st_indices[ti], float(scores[ti])) for ti in top_k_idx]

logging.info("Inference done: %d trading days", len(daily_topk))

# ── Helpers ──
def is_limit_down(idx: int, sym_idx: int) -> bool:
    """Stock closed at/near limit-down → can't sell at close."""
    pc = pre_close_mat[idx, sym_idx]
    px = close_mat[idx, sym_idx]
    if np.isnan(pc) or pc <= 0 or np.isnan(px) or px <= 0:
        return False
    return (px - pc) / pc <= -0.095

def is_limit_up(idx: int, sym_idx: int) -> bool:
    """Stock opened at/near limit-up → can't buy."""
    pc = pre_close_mat[idx, sym_idx]
    px = open_mat[idx, sym_idx]
    if np.isnan(pc) or pc <= 0 or np.isnan(px) or px <= 0:
        return False
    return (px - pc) / pc >= 0.095

# ── Backtest ──
@dataclass
class Position:
    sym_idx: int
    buy_date_idx: int
    buy_date_str: str
    cost: float
    shares: float
    capital: float
    peak_high: float
    pending_sell: bool = False
    pending_reason: str = ""
    pending_since: str = ""
    pending_ld_count: int = 0

@dataclass
class Trade:
    sym: str
    name: str
    buy_date: str
    sell_date: str
    buy_price: float
    sell_price: float
    ret: float
    hold_days: int
    exit_reason: str
    limit_down_delay: int = 0  # extra days delayed by limit-down

cash = 1_000_000.0
initial_cash = cash
positions: dict[int, Position] = {}
trade_log: list[Trade] = []
daily_equity: list[dict] = []

test_dates = sorted(daily_topk.keys())

def pos_value(idx: int, pos: Position) -> float:
    px = close_mat[idx, pos.sym_idx]
    if np.isnan(px) or px <= 0:
        px = pos.cost
    return pos.shares * px

def total_eq(idx: int) -> float:
    return cash + sum(pos_value(idx, p) for p in positions.values())

for i, idx in enumerate(test_dates):
    td = dates_all[idx]
    td_str = str(td)[:10]
    next_idx = idx + 1 if idx + 1 < len(dates_all) else idx
    topk_syms_today = set(sym_idx for sym_idx, _ in daily_topk[idx])

    # ── Step 1: Execute pending sells (from prior limit-down delays) at open ──
    done_pending = []
    for sym_idx, pos in list(positions.items()):
        if not pos.pending_sell:
            continue
        sell_price = open_mat[idx, sym_idx]
        if np.isnan(sell_price) or sell_price <= 0:
            sell_price = close_mat[idx, sym_idx] if not np.isnan(close_mat[idx, sym_idx]) else pos.cost

        # Check if STILL limit-down at open
        if is_limit_down(idx, sym_idx) and pos.pending_ld_count < MAX_LIMIT_DOWN_DAYS:
            pos.pending_ld_count += 1
            continue  # still stuck

        # Execute pending sell
        proceeds = pos.shares * sell_price
        cash += proceeds
        ret = (sell_price - pos.cost) / pos.cost
        trade_log.append(Trade(
            sym=symbols_all[sym_idx],
            name=stock_info.loc[symbols_all[sym_idx], "name"] if symbols_all[sym_idx] in stock_info.index else "?",
            buy_date=pos.buy_date_str,
            sell_date=td_str,
            buy_price=pos.cost,
            sell_price=sell_price,
            ret=ret,
            hold_days=idx - pos.buy_date_idx,
            exit_reason=pos.pending_reason,
            limit_down_delay=pos.pending_ld_count,
        ))
        done_pending.append(sym_idx)

    for sym_idx in done_pending:
        del positions[sym_idx]

    # ── Step 2: Gap-down stop at next open ──
    gap_stops = []
    for sym_idx, pos in list(positions.items()):
        if pos.pending_sell:
            continue
        if next_idx >= len(dates_all):
            continue
        open_px = open_mat[next_idx, sym_idx]
        if np.isnan(open_px) or open_px <= 0:
            continue
        if (open_px - pos.cost) / pos.cost <= HARD_STOP:
            gap_stops.append((sym_idx, open_px))

    for sym_idx, sell_price in gap_stops:
        if sym_idx not in topk_syms_today and sym_idx in positions:
            pos = positions.pop(sym_idx)
            proceeds = pos.shares * sell_price
            cash += proceeds
            ret = (sell_price - pos.cost) / pos.cost
            trade_log.append(Trade(
                sym=symbols_all[sym_idx],
                name=stock_info.loc[symbols_all[sym_idx], "name"] if symbols_all[sym_idx] in stock_info.index else "?",
                buy_date=pos.buy_date_str,
                sell_date=str(dates_all[next_idx])[:10],
                buy_price=pos.cost,
                sell_price=sell_price,
                ret=ret,
                hold_days=next_idx - pos.buy_date_idx,
                exit_reason="跳空止损",
            ))

    # ── Step 3: Close-based exit checks ──
    exit_candidates = []
    for sym_idx, pos in list(positions.items()):
        if pos.pending_sell:
            continue
        h = high_mat[idx, sym_idx]
        if not np.isnan(h) and h > 0:
            pos.peak_high = max(pos.peak_high, h)

        hold_days = idx - pos.buy_date_idx
        px = close_mat[idx, sym_idx]
        if np.isnan(px) or px <= 0:
            px = pos.cost

        # MA20 break (with volume confirmation)
        ma_break = False
        ma20 = ma20_mat[idx, sym_idx]
        vol = vol_mat[idx, sym_idx]
        avg_vol = ma20_vol_mat[idx, sym_idx]
        if not np.isnan(ma20) and ma20 > 0 and not np.isnan(vol) and vol > 0 and not np.isnan(avg_vol) and avg_vol > 0:
            if px < ma20 and vol > 1.5 * avg_vol:
                ma_break = True

        reason = None
        if (px - pos.cost) / pos.cost <= HARD_STOP:
            reason = "硬止损"
        elif ma_break:
            reason = "放量破MA20"
        elif pos.peak_high > 0 and (px - pos.peak_high) / pos.peak_high <= -TRAILING_STOP:
            reason = "移动止盈"
        elif hold_days >= MAX_HOLD_DAYS:
            reason = "到期"

        if reason and sym_idx not in topk_syms_today:
            exit_candidates.append((sym_idx, reason, px))

    # ── Step 4: Execute exits (check limit-down) ──
    for sym_idx, reason, sell_price in exit_candidates:
        if sym_idx not in positions:
            continue
        pos = positions[sym_idx]

        if is_limit_down(idx, sym_idx) and reason != "到期":
            # Can't sell at limit-down → defer
            pos.pending_sell = True
            pos.pending_reason = reason
            pos.pending_since = td_str
            pos.pending_ld_count = 1
            continue

        # Normal sell (or到期 forced sell even at limit-down)
        positions.pop(sym_idx)
        proceeds = pos.shares * sell_price
        cash += proceeds
        ret = (sell_price - pos.cost) / pos.cost
        trade_log.append(Trade(
            sym=symbols_all[sym_idx],
            name=stock_info.loc[symbols_all[sym_idx], "name"] if symbols_all[sym_idx] in stock_info.index else "?",
            buy_date=pos.buy_date_str,
            sell_date=td_str,
            buy_price=pos.cost,
            sell_price=sell_price,
            ret=ret,
            hold_days=idx - pos.buy_date_idx,
            exit_reason=reason,
        ))

    # ── Step 5: Buy new positions ──
    slots = TOP_K - len(positions)  # pending fills a slot
    if slots > 0 and cash > 0 and next_idx < len(dates_all):
        for sym_idx, score in daily_topk[idx]:
            if len(positions) >= TOP_K:
                break
            if sym_idx in positions:
                continue
            if is_limit_up(next_idx, sym_idx):
                continue

            buy_price = open_mat[next_idx, sym_idx]
            if np.isnan(buy_price) or buy_price <= 0:
                continue

            remaining = TOP_K - len(positions)
            if remaining == 0:
                break
            spend = cash / remaining
            shares = spend / buy_price

            cash -= shares * buy_price
            peak = buy_price
            if not np.isnan(high_mat[next_idx, sym_idx]) and high_mat[next_idx, sym_idx] > 0:
                peak = max(buy_price, high_mat[next_idx, sym_idx])

            positions[sym_idx] = Position(
                sym_idx=sym_idx,
                buy_date_idx=next_idx,
                buy_date_str=str(dates_all[next_idx])[:10],
                cost=buy_price,
                shares=shares,
                capital=spend,
                peak_high=peak,
            )

    # ── Step 6: Record equity ──
    daily_equity.append({
        "date": td,
        "cash": cash,
        "positions": len(positions),
        "pending": sum(1 for p in positions.values() if p.pending_sell),
        "equity": total_eq(idx),
    })

# ── Final: close all remaining ──
last_idx = test_dates[-1]
for sym_idx, pos in list(positions.items()):
    sell_price = close_mat[last_idx, sym_idx]
    if np.isnan(sell_price) or sell_price <= 0:
        sell_price = pos.cost
    cash += pos.shares * sell_price
    ret = (sell_price - pos.cost) / pos.cost
    reason = pos.pending_reason if pos.pending_sell else "回测结束"
    trade_log.append(Trade(
        sym=symbols_all[sym_idx],
        name=stock_info.loc[symbols_all[sym_idx], "name"] if symbols_all[sym_idx] in stock_info.index else "?",
        buy_date=pos.buy_date_str,
        sell_date=str(dates_all[last_idx])[:10],
        buy_price=pos.cost,
        sell_price=sell_price,
        ret=ret,
        hold_days=last_idx - pos.buy_date_idx,
        exit_reason=reason,
        limit_down_delay=pos.pending_ld_count if pos.pending_sell else 0,
    ))
positions.clear()

# ── Results ──
eq_df = pd.DataFrame(daily_equity)
final_equity = eq_df["equity"].iloc[-1]
total_ret = (final_equity - initial_cash) / initial_cash

# Monthly returns
eq_df["month"] = pd.to_datetime(eq_df["date"]).dt.to_period("M")
monthly_eq = eq_df.groupby("month")["equity"].last()
monthly_ret = monthly_eq.pct_change(fill_method=None)

# Drawdown
peak = eq_df["equity"].expanding().max()
dd = (eq_df["equity"] - peak) / peak
max_dd = abs(dd.min())

# Sharpe
daily_rets = eq_df["equity"].pct_change(fill_method=None).dropna()
sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if len(daily_rets) > 0 and daily_rets.std() > 0 else 0.0

# Trades
trades_df = pd.DataFrame([t.__dict__ for t in trade_log])
win_rate = (trades_df["ret"] > 0).mean() if len(trades_df) > 0 else 0
mean_ret = trades_df["ret"].mean() if len(trades_df) > 0 else 0
mean_hold = trades_df["hold_days"].mean() if len(trades_df) > 0 else 0

print(f"\n{'='*70}")
print(f"V3 Backtest — {BACKTEST_START.year}-{BACKTEST_END.year}")
print(f"Strategy: Top-{TOP_K}, MA20+vol exit, trail {TRAILING_STOP:.0%}, hard {HARD_STOP:.0%}, model override")
print(f"Protection: gap-down open-stop, limit-down defer (max {MAX_LIMIT_DOWN_DAYS}d)")
print(f"Model: {CHECKPOINT}")
print(f"{'='*70}")
print(f"  Initial:       {initial_cash:,.0f}")
print(f"  Final:         {final_equity:,.0f}")
print(f"  Total Return:  {total_ret:+.2%}")
print(f"  Max DD:        {max_dd:.2%}")
print(f"  Sharpe:        {sharpe:.2f}")
print(f"  Trades:        {len(trades_df)}")
print(f"  Win rate:      {win_rate:.1%}")
print(f"  Mean return:   {mean_ret:+.2%}")
print(f"  Mean hold:     {mean_hold:.1f}d")
if "limit_down_delay" in trades_df.columns:
    stuck = (trades_df["limit_down_delay"] > 0).sum()
    print(f"  Limit-down delayed exits: {stuck}")

# Monthly returns
print(f"\n{'='*60}")
print(f"Monthly Returns (from equity curve)")
print(f"{'='*60}")
for m, r in monthly_ret.items():
    if pd.notna(r):
        bar = "█" * max(1, int(abs(r) / 0.02))
        sign = "+" if r > 0 else "-"
        print(f"  {m}  {sign}{abs(r):.1%}  {bar}")

# Annual returns
eq_df["year"] = pd.to_datetime(eq_df["date"]).dt.year
annual_eq = eq_df.groupby("year")["equity"].last()
annual_ret = annual_eq.pct_change(fill_method=None)
print(f"\n{'='*60}")
print(f"Annual Returns")
print(f"{'='*60}")
for y, r in annual_ret.items():
    if pd.notna(r):
        bar = "█" * max(1, int(abs(r) / 0.05))
        sign = "+" if r > 0 else "-"
        print(f"  {y}  {sign}{abs(r):.1%}  {bar}")
# Compute 2021 return from initial cash (first year has NaN in pct_change)
first_year = annual_eq.index[0]
first_ret = (annual_eq.iloc[0] - initial_cash) / initial_cash
print(f"  {first_year}  {first_ret:+.1%}  {'█' * max(1, int(abs(first_ret) / 0.05))}")

# By exit reason
print(f"\n{'='*60}")
print(f"Exit Reason Breakdown")
print(f"{'='*60}")
for reason, grp in trades_df.groupby("exit_reason"):
    wr = (grp["ret"] > 0).mean()
    stuck = (grp["limit_down_delay"] > 0).sum() if "limit_down_delay" in grp.columns else 0
    extra = f" ({stuck} stuck)" if stuck > 0 else ""
    print(f"  {reason:<12} {len(grp):>4} trades  mean: {grp['ret'].mean():+.2%}  "
          f"win: {wr:.0%}  avg hold: {grp['hold_days'].mean():.1f}d{extra}")

# Top winners / losers
top_w = trades_df.nlargest(10, "ret")
top_l = trades_df.nsmallest(10, "ret")
print(f"\n{'='*60}")
print(f"Top 10 Winners")
print(f"{'='*60}")
for _, t in top_w.iterrows():
    ld = f" [{t['limit_down_delay']}d stuck]" if t.get("limit_down_delay", 0) > 0 else ""
    print(f"  {t['buy_date']}→{t['sell_date']}  {t['sym']:<8} {str(t['name']):<10}  {t['ret']:+.1%}  {t['hold_days']:.0f}d  {t['exit_reason']}{ld}")
print(f"\n{'='*60}")
print(f"Top 10 Losers")
print(f"{'='*60}")
for _, t in top_l.iterrows():
    ld = f" [{t['limit_down_delay']}d stuck]" if t.get("limit_down_delay", 0) > 0 else ""
    print(f"  {t['buy_date']}→{t['sell_date']}  {t['sym']:<8} {str(t['name']):<10}  {t['ret']:+.1%}  {t['hold_days']:.0f}d  {t['exit_reason']}{ld}")

# Benchmark
bs = min(daily_topk.keys())
be = max(daily_topk.keys())
cs = close_mat[bs, :]
ce = close_mat[be, :]
v = (cs > 0) & (~np.isnan(cs)) & (ce > 0) & (~np.isnan(ce))
bench_ret = float(np.median((ce[v] - cs[v]) / cs[v]))
print(f"\n  Benchmark (equal-weight median): {bench_ret:+.2%}")
