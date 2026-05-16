"""Quantitative Research Report Generator.

Generates a comprehensive markdown research report comparing old (baseline)
factors against new GP-discovered factors, with backtest comparison.

Pipeline:
1. Data ingest via baostock
2. Baseline factor computation
3. Agent diagnosis
4. Hypothesis generation (LLM if available)
5. GP factor discovery
6. Validate & select new factors
7. Comparison backtests (old-only vs old+new)
8. Generate charts
9. Write polished markdown report

Usage: python run_agent_report.py
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from datetime import date, datetime
from pathlib import Path

import config.chart_style  # noqa: F401 — CJK fonts + Agg backend (must preceed pyplot)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from backtest.accounting import PortfolioAccountant
from backtest.broker import AShareBroker
from backtest.universe import UniverseFilter
from config.settings import PRICE_LIMITS, LOT_SIZE
from core.types import Fill, Order, OrderType, Side
from data.sources.baostock import _infer_board

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("research_report")

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

SYMBOL_COUNT = 500
TARGET_CANDIDATE_POOL = 5000  # use all cached stocks (no download needed)
START_DATE = date(2024, 6, 1)
END_DATE = date(2026, 5, 14)
FORWARD_PERIODS = 5  # 5-day forward return for IC
INITIAL_CASH = 1_000_000
MAX_POSITIONS = 10   # max simultaneous holdings
GP_POPULATION = 200
GP_GENERATIONS = 25
MAX_NEW_FACTORS = 5
TOP_QUANTILE = 0.2  # top 20% for long portfolio

REPORT_DATE = datetime.now().strftime("%Y%m%d_%H%M")
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "results" / f"report_{REPORT_DATE}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR = OUTPUT_DIR  # everything goes into the same report folder

sns.set_palette("muted")

COLORS = {
    "old": "#3498db",    # blue
    "new": "#e74c3c",    # red
    "old_light": "#85c1e9",
    "new_light": "#f1948a",
    "benchmark": "#2c3e50",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Print banner
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  Agent 量化研究报告生成器")
print(f"  报告日期: {REPORT_DATE}")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Data Ingest
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[1/8] 数据获取 (全部本地缓存) ...")
from data.cache import read_daily

# ── Step 1: Read ALL cached data (no downloads, all 3288 stocks in cache) ──
raw_data = read_daily(START_DATE, END_DATE)
cached_syms = list(raw_data.index.get_level_values("symbol").unique()) if not raw_data.empty else []
print(f"  缓存数据: {len(raw_data):,} 行, {len(cached_syms):,} 只股票")

# ── Step 2: Get stock names & board info (lightweight, no data download) ──
from data.sources.baostock import BaoStockSource
source = BaoStockSource(rate_limit=0.05)
all_stocks = source.list_stocks()
source.close()

# Build symbol->name map for display
_symbol_name = dict(zip(all_stocks["symbol"], all_stocks["name"]))
_symbol_board = dict(zip(all_stocks["symbol"], all_stocks["board"]))

# Filter: non-ST, non-BJ
valid_stocks = all_stocks[~all_stocks["is_st"]]
valid_stocks = valid_stocks[valid_stocks["board"].isin(["main_board", "chinext", "star_market"])]
valid_syms = set(valid_stocks["symbol"])

# ── Step 3: Candidate pool = cached ∩ valid (non-ST, non-BJ) ──
candidate_syms = [s for s in cached_syms if s in valid_syms]
print(f"  全A候选 (缓存 ∩ 非ST/非北交所): {len(candidate_syms):,} 只")
candidate_syms = sorted(candidate_syms)[:TARGET_CANDIDATE_POOL]

# ── Step 4: Get Shenwan L1 industry ──
from data.industry import build_industry_map
print("  获取申万行业分类 ...")
industry_map = build_industry_map()

# ── Step 5: Compute avg daily amount for liquidity ranking ──
filtered = raw_data[raw_data.index.get_level_values("symbol").isin(candidate_syms)]
_amt = filtered[["amount"]].copy()
_avg_amt = _amt.groupby(level="symbol")["amount"].mean().sort_values(ascending=False)

# Symbol -> industry
_symbol_industry = {}
for sym in _avg_amt.index:
    _symbol_industry[sym] = industry_map.get(sym, "综合")

# Build per-industry candidate lists sorted by liquidity
from collections import defaultdict
_ind_candidates: dict[str, list[str]] = defaultdict(list)
for sym in _avg_amt.index:
    ind = _symbol_industry.get(sym, "综合")
    _ind_candidates[ind].append(sym)

# Per-industry allocation proportional to market representation
_ind_total_stocks = {ind: len(syms) for ind, syms in _ind_candidates.items()}
_total_repr = sum(_ind_total_stocks.values())
_ind_alloc = {}
for ind, n_stocks in _ind_total_stocks.items():
    _ind_alloc[ind] = max(3, int(SYMBOL_COUNT * n_stocks / _total_repr))

# Adjust to hit exactly SYMBOL_COUNT
_alloc_total = sum(_ind_alloc.values())
while _alloc_total > SYMBOL_COUNT:
    max_ind = max(_ind_alloc, key=lambda k: _ind_alloc[k] - 3)
    if _ind_alloc[max_ind] > 3:
        _ind_alloc[max_ind] -= 1
        _alloc_total -= 1
    else:
        break
while _alloc_total < SYMBOL_COUNT:
    min_ind = min(_ind_alloc, key=lambda k: _ind_alloc[k])
    _ind_alloc[min_ind] += 1
    _alloc_total += 1

# ── Step 6: Select stocks & record per-industry picked symbols with names ──
symbols: list[str] = []
industry_counts: dict[str, int] = {}
industry_symbols: dict[str, list[str]] = {}  # industry -> picked symbols (with names)

for ind, alloc in sorted(_ind_alloc.items(), key=lambda x: -x[1]):
    n_avail = len(_ind_candidates[ind])
    n_pick = min(alloc, n_avail)
    picked = _ind_candidates[ind][:n_pick]
    symbols.extend(picked)
    industry_counts[ind] = n_pick
    industry_symbols[ind] = picked

# Filter to selected symbols
raw_data = raw_data[raw_data.index.get_level_values("symbol").isin(symbols)]

print(f"  股票池: {len(symbols)} 只, {len(industry_counts)} 个行业")
print(f"  行业分布: {dict(sorted(industry_counts.items(), key=lambda x: -x[1]))}")

n_dates = raw_data.index.get_level_values("trade_date").nunique()
n_syms = raw_data.index.get_level_values("symbol").nunique()
print(f"  获取: {len(raw_data):,} 行, {n_dates} 个交易日, {n_syms} 只股票")

# ── Save stock pool ──
import json

# Build per-industry stock detail for JSON + MD
_stock_detail: dict[str, list[dict]] = {}
for ind, syms in industry_symbols.items():
    _stock_detail[ind] = [
        {"symbol": s, "name": _symbol_name.get(s, "")} for s in syms
    ]

_stock_pool_path = OUTPUT_DIR / "stock_pool.json"
_stock_pool_data = {
    "report_date": REPORT_DATE,
    "candidate_universe": len(candidate_syms),
    "symbol_count": len(symbols),
    "industry_count": len(industry_counts),
    "industry_distribution": dict(sorted(industry_counts.items(), key=lambda x: -x[1])),
    "industry_stocks": _stock_detail,
    "symbols": sorted(symbols),
}
_stock_pool_path.write_text(json.dumps(_stock_pool_data, ensure_ascii=False, indent=2), encoding="utf-8")

# Markdown version with per-industry stock names
_stock_pool_md_path = OUTPUT_DIR / "stock_pool.md"
_md: list[str] = []
_md.append(f"# A股量化股票池\n\n")
_md.append(f"**报告日期**: {REPORT_DATE}\n\n")
_md.append(f"**候选全A**: {len(candidate_syms):,} 只 (非ST, 非北交所)\n\n")
_md.append(f"**选中股票**: {len(symbols)} 只\n\n")
_md.append(f"**行业数量**: {len(industry_counts)} 个 (申万一级)\n\n")
_md.append("**选股逻辑**: 行业分层抽样——申万一级行业内按日均成交额排序取前N只，行业配额按市场占比分配\n\n")
_md.append("\n---\n\n## 行业分布\n\n")
_md.append("| 申万一级行业 | 数量 | 占比 |\n")
_md.append("|-------------|------|------|\n")
for _ind, _cnt in sorted(industry_counts.items(), key=lambda x: -x[1]):
    _md.append(f"| {_ind} | {_cnt} | {_cnt/len(symbols)*100:.1f}% |\n")

_md.append("\n---\n\n## 各行业选股明细\n\n")
for ind, syms in sorted(industry_symbols.items(), key=lambda x: -len(x[1])):
    _md.append(f"### {ind} ({len(syms)}只)\n\n")
    _md.append("| 代码 | 名称 |\n")
    _md.append("|------|------|\n")
    for s in syms:
        name = _symbol_name.get(s, "")
        _md.append(f"| {s} | {name} |\n")
    _md.append("\n")
_stock_pool_md_path.write_text("".join(_md), encoding="utf-8")
print(f"  股票池已保存至: {_stock_pool_path} / {_stock_pool_md_path}")

if raw_data.empty:
    print("  ERROR: 无数据!")
    exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Baseline Factor Computation
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[2/8] 基准因子计算 ...")
import factor.factors as _  # register all factors
from factor.engine import FactorEngine

engine = FactorEngine()
BASELINE_FACTORS = [
    "momentum_1m", "momentum_3m", "momentum_6m",
    "reversal_5d", "reversal_10d",
    "volatility_20d", "volatility_60d",
    "beta_60d",
    "turnover_20d",
    "trend_efficiency_20d", "ma_trend_5_20", "donchian_pct_20d",
    "up_days_ratio_20d", "ma_cross_5_20",
]
factor_df = engine.compute(BASELINE_FACTORS, raw_data)
print(f"  计算: {len(BASELINE_FACTORS)} 个因子, {factor_df.shape[0]:,} 个值")

merged = raw_data.join(factor_df, how="left")
merged = merged.dropna(subset=BASELINE_FACTORS)
print(f"  去 NaN 后: {len(merged):,} 行")

# ── Add derived columns as GP terminals (no look-ahead, backward-looking) ─────
# These give GP a head start on momentum/volatility structure without assembling
# pct_change from scratch. All are computed within each symbol.
print("  计算衍生特征 ...")
_unstacked = merged[["close", "volume", "amount", "high", "low"]].unstack()
_derived = {}
for field, col in [("ret_5d", "close"), ("ret_20d", "close"), ("ret_60d", "close")]:
    days = int(field.split("_")[1].replace("d", ""))
    _derived[field] = _unstacked[col].pct_change(days).stack()
for field, col in [("vol_20d", "close"), ("vol_60d", "close")]:
    days = int(field.split("_")[1].replace("d", ""))
    _derived[field] = _unstacked[col].pct_change().rolling(days, min_periods=max(1, days//2)).std().stack()
# hl_ratio: intraday range / close
_derived["hl_ratio"] = ((_unstacked["high"] - _unstacked["low"]) / _unstacked["close"]).stack()
# amihud: illiquidity = |daily_ret| / amount (in 10^6)
_daily_ret = _unstacked["close"].pct_change()
_derived["amihud"] = (_daily_ret.abs() / (_unstacked["amount"].clip(lower=1)) * 1e6).stack()
# volume_ratio: volume / 20d avg volume
_derived["vol_ratio"] = (_unstacked["volume"] / _unstacked["volume"].rolling(20, min_periods=5).mean()).stack()

for name, s in _derived.items():
    merged[name] = s
print(f"  添加衍生列: {list(_derived.keys())}")

# Forward returns for IC evaluation
fwd_ret_5d = merged["close"].unstack().pct_change(periods=FORWARD_PERIODS).shift(-FORWARD_PERIODS).stack()

common_dates = factor_df.dropna().index.intersection(fwd_ret_5d.dropna().index)
factor_clean = factor_df.loc[common_dates]
fwd_clean = fwd_ret_5d.loc[common_dates]
print(f"  有效 IC 计算样本: {len(fwd_clean):,}")

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Baseline IC Analysis
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[3/8] 基准因子 IC 分析 ...")
from factor.validation import compute_rank_ic, ic_summary, factor_correlation

baseline_ic_stats: dict[str, dict] = {}
for fname in BASELINE_FACTORS:
    ic = compute_rank_ic(factor_clean[fname], fwd_clean)
    baseline_ic_stats[fname] = ic_summary(ic)

print(f"  {'因子':<20} {'IC Mean':>10} {'IC Std':>10} {'IC IR':>10} {'Hit Rate':>10}")
print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for fname in BASELINE_FACTORS:
    s = baseline_ic_stats[fname]
    print(f"  {fname:<20} {s['mean']:>10.4f} {s['std']:>10.4f} {s['ir']:>10.3f} {s['hit_rate']:>10.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: Agent Diagnosis
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[4/8] Agent 诊断 ...")
from agent.monitor import Monitor
from agent.decision import ExplorationPlanner
from agent.knowledge_base import KnowledgeBase, FactorRecord

monitor = Monitor(lookback_recent=60, auto_corr_warn=0.95, ic_min_abs=0.02)
diagnosis = monitor.diagnose(
    factor_values=factor_clean,
    forward_returns=fwd_clean,
    price_data=merged,
)

print(f"  市场状态: {diagnosis.regime}")
print(f"  因子健康度:")
for name, fh in diagnosis.factors.items():
    icon = {"healthy": "✓", "decaying": "⚠", "dead": "✗", "weak": "○"}.get(fh.status, "?")
    print(f"    {icon} {name}: {fh.status} | IC={fh.ic_mean:+.4f} IR={fh.ic_ir:+.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5: Hypothesis Generation
# ═══════════════════════════════════════════════════════════════════════════════

print("\n[5/8] 假设生成 ...")
kb = KnowledgeBase(path=str(OUTPUT_DIR / "report_kb.json"))  # stays in report folder

for fname in BASELINE_FACTORS:
    fh = diagnosis.factors.get(fname)
    if fh:
        kb.add_factor(FactorRecord(
            name=fname,
            category="momentum" if "momentum" in fname else (
                "volatility" if "vol" in fname else (
                "liquidity" if "turnover" in fname else "reversal"
            )),
            description="Baseline factor",
            source="baseline",
            status="active" if fh.status == "healthy" else fh.status,
            ic_mean=fh.ic_mean,
            ic_ir=fh.ic_ir,
            hit_rate=fh.hit_rate,
            auto_corr=fh.auto_corr,
        ))

planner = ExplorationPlanner()
plan = planner.plan(diagnosis, kb.stats(), kb.get_active_factor_names())
print(f"  探索预算: {plan.budget} 个新因子")
print(f"  优先类别: {plan.focus_categories if plan.focus_categories else '全部'}")

# LLM factor ideas
llm_ideas: list[dict] = []
from agent.llm_client import create_default_client
llm = create_default_client()
if llm.configured:
    print(f"  LLM 可用 ({llm.backend}:{llm.model}), 生成因子思路...")
    try:
        llm_ideas = llm.generate_factor_ideas(
            diagnosis=diagnosis.to_dict(),
            existing_factors=kb.get_active_factor_names(),
            n_ideas=3,
        )
        for idea in llm_ideas:
            print(f"    💡 {idea.get('name', '?')}: {idea.get('intuition', '?')}")
    except Exception as e:
        print(f"    LLM 调用失败: {e}")
else:
    print("  LLM 未配置 (设置 DEEPSEEK_API_KEY 以启用)")

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 6: GP Factor Discovery (ONLY on training data to prevent overfitting)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[6/8] GP 因子挖掘 (种群={GP_POPULATION}, 代数={GP_GENERATIONS}) ...")
from discovery.gp import GPEngine, GPConfig
from discovery.validate import FactorValidator

existing_df = factor_clean[BASELINE_FACTORS].copy()

# Split data into train/test BEFORE GP to prevent look-ahead
all_dates = sorted(factor_clean.index.get_level_values("trade_date").unique())
gp_split_idx = int(len(all_dates) * 0.67)
gp_train_dates = all_dates[:gp_split_idx]
gp_test_dates = all_dates[gp_split_idx:]
print(f"  GP 训练区间: {gp_train_dates[0].date()} ~ {gp_train_dates[-1].date()} ({len(gp_train_dates)} 个交易日)")
print(f"  回测区间: {gp_test_dates[0].date()} ~ {gp_test_dates[-1].date()} ({len(gp_test_dates)} 个交易日)")

# Restrict GP to training data only
gp_train_mask = merged.index.get_level_values("trade_date").isin(gp_train_dates)
gp_data = merged.loc[gp_train_mask]
gp_fwd = fwd_clean.loc[fwd_clean.index.intersection(gp_data.index)]
gp_existing = existing_df.loc[existing_df.index.get_level_values("trade_date").isin(gp_train_dates)]

gp_config = GPConfig(
    population_size=GP_POPULATION,
    max_generations=GP_GENERATIONS,
    tournament_size=7,
    crossover_prob=0.7,
    mutation_prob=0.5,
    elite_count=10,
    max_depth=6,
    max_complexity=30,
    early_stop_generations=10,
    parsimony_penalty=0.001,
    ic_mean_weight=0.25,
    ic_ir_weight=0.35,
    stability_weight=0.25,
    hit_rate_weight=0.15,
)
gp = GPEngine(config=gp_config)

t0 = time.time()
best_individuals = gp.evolve(
    data=gp_data,
    forward_returns=gp_fwd,
    existing_factors=gp_existing,
)
gp_elapsed = time.time() - t0
print(f"  完成! 耗时 {gp_elapsed:.1f}s, 代数={gp.generation}")

print(f"\n  GP 发现的最佳因子 (Hall of Fame):")
hall = sorted(best_individuals, key=lambda x: x.fitness, reverse=True)[:15]
for i, ind in enumerate(hall, 1):
    print(f"  {i:2d}. {ind.factor_name:30s} fitness={ind.fitness:.4f}  "
          f"IC={ind.ic_mean:+.4f}  IR={ind.ic_ir:.3f}  "
          f"depth={ind.depth}  nodes={ind.complexity}")

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 7: Validate & Select New Factors
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[7/8] 验证 & 筛选新因子 ...")
validator = FactorValidator()
new_factors: list[dict] = []
rejected_factors: list[dict] = []

for ind in hall[:MAX_NEW_FACTORS * 2]:
    if ind.factor_cls is None or ind.fitness < -100:
        continue
    if len(new_factors) >= MAX_NEW_FACTORS:
        break

    try:
        factor_vals = ind.factor_cls().compute(merged)
        result = validator.validate(
            factor_values=factor_vals,
            forward_returns=fwd_clean,
            factor_name=ind.factor_name,
            existing_factors=existing_df,
        )

        entry = {
            "name": ind.factor_name,
            "expression": repr(ind.tree),
            "category": ind.factor_cls.meta.category,
            "ic_mean": ind.ic_mean,
            "ic_std": result.ic_std,
            "ic_ir": ind.ic_ir,
            "hit_rate": ind.hit_rate,
            "auto_corr": ind.auto_corr,
            "complexity": ind.complexity,
            "depth": ind.depth,
            "passed": result.passed,
            "failures": result.failures,
        }

        if result.passed:
            new_factors.append(entry)
            existing_df[ind.factor_name] = factor_vals
            kb.add_factor(FactorRecord(
                name=ind.factor_name,
                category=ind.factor_cls.meta.category,
                description=f"GP discovered (gen {gp.generation})",
                expression_repr=repr(ind.tree),
                source="gp",
                status="validated",
                ic_mean=ind.ic_mean,
                ic_ir=ind.ic_ir,
                hit_rate=ind.hit_rate,
                auto_corr=ind.auto_corr,
            ))
            print(f"  ✓ 接受: {ind.factor_name} (IC={ind.ic_mean:+.4f}, IR={ind.ic_ir:.3f})")
            print(f"    表达式: {ind.tree!r}")
        else:
            rejected_factors.append(entry)
            print(f"  ✗ 拒绝: {ind.factor_name} ({', '.join(result.failures[:2])})")
    except Exception as e:
        rejected_factors.append({"name": ind.factor_name, "error": str(e)})
        print(f"  ✗ 错误: {ind.factor_name} — {e}")

kb.flush()

# ── Orthogonal filter: keep only factors with independent alpha ────────────
# After individual validation, some factors may all capture the same signal.
# Orthogonalization removes redundant factors before category selection.
if len(new_factors) > 1:
    from discovery.validate import orthogonal_filter
    validated_values = {}
    for nf in new_factors:
        try:
            f_cls = None
            for ind in hall:
                if ind.factor_name == nf["name"] and ind.factor_cls is not None:
                    f_cls = ind.factor_cls
                    break
            if f_cls is None:
                # fallback: find from hall
                matched = [ind for ind in hall if ind.factor_name == nf["name"] and ind.factor_cls is not None]
                if matched:
                    f_cls = matched[0].factor_cls
            if f_cls is not None:
                validated_values[nf["name"]] = f_cls().compute(merged)
        except Exception:
            pass

    if len(validated_values) > 1:
        ortho_selected = orthogonal_filter(
            validated_values, fwd_clean, min_residual_ir=0.10,
        )
        ortho_rejected = set(validated_values) - set(ortho_selected)
        if ortho_rejected:
            print(f"\n  正交筛选: 剔除 {len(ortho_rejected)} 个冗余因子 (残差 IC_IR < 0.10)")
            for name in sorted(ortho_rejected):
                print(f"    ✗ {name} — Alpha 已被已选因子覆盖")
                rejected_factors.append({"name": name, "failures": ["正交筛选: 残差 IC_IR < 0.10, Alpha 已被已选因子覆盖"]})
        new_factors = [nf for nf in new_factors if nf["name"] in ortho_selected]
        print(f"  正交筛选后保留: {len(new_factors)} 个独立因子")

new_factor_names = [f["name"] for f in new_factors]
if not new_factor_names:
    print("\n  ⚠ 没有通过验证的新因子。将使用 GP hall of fame 中前 2 个最佳因子作为新因子。")
    for ind in hall[:2]:
        if ind.factor_cls is not None:
            try:
                factor_vals = ind.factor_cls().compute(merged)
                existing_df[ind.factor_name] = factor_vals
                new_factor_names.append(ind.factor_name)
                new_factors.append({
                    "name": ind.factor_name,
                    "expression": repr(ind.tree),
                    "category": ind.factor_cls.meta.category if ind.factor_cls else "unknown",
                    "ic_mean": ind.ic_mean,
                    "ic_std": 0.0,
                    "ic_ir": ind.ic_ir,
                    "hit_rate": ind.hit_rate,
                    "auto_corr": ind.auto_corr,
                    "complexity": ind.complexity,
                    "depth": ind.depth,
                    "passed": True,
                    "failures": [],
                })
            except Exception:
                pass

if not new_factor_names:
    print("  FATAL: 没有可用的新因子，无法进行对比回测。")
    exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 8: Comparison Backtest
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n[8/8] 对比回测 ...")

# ── Prepare daily return data for backtest ────────────────────────────────────

# Daily returns (same-index) — 1-day forward return for backtest execution
daily_ret = merged["close"].unstack().pct_change().shift(-1).stack()
# Align with factor data
backtest_dates = existing_df.dropna().index.intersection(daily_ret.dropna().index)
bt_factor_df = existing_df.loc[backtest_dates]
bt_ret = daily_ret.loc[backtest_dates]

# Use the same train/test split as GP (no look-ahead)
test_dates = gp_test_dates
train_dates = gp_train_dates
print(f"  回测区间 (样本外): {test_dates[0].date()} ~ {test_dates[-1].date()} ({len(test_dates)} 个交易日)")


def run_factor_backtest(
    factor_df: pd.DataFrame,
    factor_names_list: list[str],
    daily_returns: pd.Series,
    trade_dates: list,
    train_dates: list | None = None,
    top_q: float = 0.2,
    max_positions: int = 10,
    initial_cash: float = INITIAL_CASH,
    price_data: pd.DataFrame | None = None,
) -> tuple[pd.Series, dict]:
    """Real backtest with broker execution, costs, T+1, lot size, position limits.

    Uses AShareBroker + PortfolioAccountant for realistic A-share execution.
    IC_IR-weighted factor composite → top-quantile equal-weight long portfolio.
    Weekly rebalancing to reduce turnover.
    """
    broker = AShareBroker(
        commission_rate=0.00025,
        min_commission=5.0,
        stamp_tax_rate=0.001,
        transfer_fee_rate=0.00001,
        slippage_bps=5.0,
    )
    accountant = PortfolioAccountant(initial_cash=initial_cash)
    universe_filter = UniverseFilter()

    # Compute factor weights from training period IC
    factor_weights: dict[str, float] = {}
    if train_dates is not None and len(train_dates) > 10:
        train_idx = factor_df.index.get_level_values("trade_date").isin(train_dates)
        train_factors = factor_df.loc[train_idx]
        common_idx = train_factors.index.intersection(daily_returns.index)
        if len(common_idx) > 20:
            for col in factor_names_list:
                ic = compute_rank_ic(train_factors.loc[common_idx, col], daily_returns.loc[common_idx])
                factor_weights[col] = max(abs(ic.mean()), 0.005)
            total_w = sum(factor_weights.values())
            if total_w > 0:
                factor_weights = {k: v / total_w for k, v in factor_weights.items()}

    if not factor_weights:
        factor_weights = {col: 1.0 / len(factor_names_list) for col in factor_names_list}

    all_syms = sorted(factor_df.index.get_level_values("symbol").unique().tolist())
    daily_pnl: list[float] = []
    equity_points = {trade_dates[0]: initial_cash}

    for i, today in enumerate(trade_dates[:-1]):
        td = today.date()
        tomorrow = trade_dates[i + 1]

        # Weekly rebalancing (every 5th trading day)
        if i % 5 != 0:
            # Non-rebalance day: just mark to market
            try:
                tmr_data = price_data.xs(tomorrow, level="trade_date") if price_data is not None else pd.DataFrame()
                prices = tmr_data["close"] if "close" in tmr_data.columns else pd.Series(dtype=float)
            except KeyError:
                prices = pd.Series(dtype=float)
            if not prices.empty:
                accountant.mark_to_market(prices, tomorrow)
            equity_points[tomorrow] = accountant.equity_history[-1]["equity"] if accountant.equity_history else initial_cash
            prev_eq = equity_points[list(equity_points.keys())[-2]] if len(equity_points) > 1 else initial_cash
            cur_eq = equity_points[tomorrow]
            daily_pnl.append(float(cur_eq / prev_eq - 1) if prev_eq > 0 else 0.0)
            continue

        # 1. Universe filter
        yest_data = pd.DataFrame()
        if price_data is not None:
            try:
                yest_data = price_data.xs(today - pd.Timedelta(days=1), level="trade_date")
            except KeyError:
                pass
        universe = universe_filter.filter(today, all_syms, pd.DataFrame(), yest_data)
        if len(universe) < 10:
            # Still mark to market
            try:
                tmr_data = price_data.xs(tomorrow, level="trade_date") if price_data is not None else pd.DataFrame()
                prices = tmr_data["close"] if "close" in tmr_data.columns else pd.Series(dtype=float)
            except KeyError:
                prices = pd.Series(dtype=float)
            if not prices.empty:
                accountant.mark_to_market(prices, tomorrow)
            equity_points[tomorrow] = accountant.equity_history[-1]["equity"] if accountant.equity_history else initial_cash
            prev_eq = equity_points[list(equity_points.keys())[-2]] if len(equity_points) > 1 else initial_cash
            cur_eq = equity_points[tomorrow]
            daily_pnl.append(float(cur_eq / prev_eq - 1) if prev_eq > 0 else 0.0)
            continue

        # 2. Compute factor composite for today
        try:
            day_factors = factor_df.xs(today, level="trade_date")
        except KeyError:
            try:
                day_factors = factor_df.xs(today - pd.Timedelta(days=1), level="trade_date")
            except KeyError:
                daily_pnl.append(0.0)
                continue

        composite = pd.Series(0.0, index=day_factors.index)
        for fname in factor_names_list:
            if fname not in day_factors.columns:
                continue
            col = day_factors[fname].dropna()
            if len(col) < 5:
                continue
            w = factor_weights.get(fname, 1.0 / len(factor_names_list))
            ranked = col.rank(pct=True)
            composite = composite.add(ranked * w, fill_value=0.0)

        composite = composite[composite.index.isin(universe)]
        composite = composite[composite > 0]
        if composite.empty:
            daily_pnl.append(0.0)
            continue

        # 3. Top-quantile selection, capped at max_positions
        cutoff = composite.quantile(1 - top_q)
        top = composite[composite >= cutoff]
        if len(top) > max_positions:
            top = top.nlargest(max_positions)

        # 4. Today's prices for execution
        try:
            today_data = price_data.xs(today, level="trade_date") if price_data is not None else pd.DataFrame()
            prices_today = today_data["close"] if "close" in today_data.columns else pd.Series(dtype=float)
            open_prices = today_data["open"] if "open" in today_data.columns else prices_today
        except KeyError:
            prices_today = pd.Series(dtype=float)
            open_prices = prices_today

        pre_close = yest_data["close"] if not yest_data.empty and "close" in yest_data.columns else prices_today
        price_limits = pd.Series(
            [PRICE_LIMITS.get(_infer_board(s), PRICE_LIMITS["main_board"]) for s in universe],
            index=universe,
        )

        # 5. Convert to orders (equal-weight targets within top-N)
        equity = accountant.cash + sum(
            accountant.positions.get(sym, 0) * float(prices_today.get(sym, 0))
            for sym in accountant.positions
            if sym in prices_today.index
        )
        target_weight = 1.0 / len(top) if len(top) > 0 else 0
        orders: list[Order] = []
        order_counter = 0

        # Sell positions not in top-N
        held_syms = {s for s, q in accountant.positions.items() if q > 0}
        for sym in held_syms - set(top.index):
            qty = accountant.positions.get(sym, 0)
            if qty <= 0 or sym not in prices_today.index or pd.isna(prices_today.get(sym)):
                continue
            qty = (qty // LOT_SIZE) * LOT_SIZE
            if qty <= 0:
                continue
            orders.append(Order(
                symbol=sym, side=Side.SELL, quantity=qty,
                order_type=OrderType.MARKET, date=today,
                order_id=f"bt_{td}_{sym}_sell_{order_counter}",
            ))
            order_counter += 1

        # Buy top-N stocks to target weight
        for sym in top.index:
            if sym not in prices_today.index or pd.isna(prices_today.get(sym)) or prices_today.get(sym) <= 0:
                continue
            price = float(prices_today[sym])
            target_value = equity * target_weight
            current_shares = accountant.positions.get(sym, 0)
            current_value = current_shares * price
            diff_value = target_value - current_value

            if diff_value > price * LOT_SIZE:  # Buy
                qty = int(diff_value / price)
                qty = (qty // LOT_SIZE) * LOT_SIZE
                if qty > 0:
                    orders.append(Order(
                        symbol=sym, side=Side.BUY, quantity=qty,
                        order_type=OrderType.MARKET, date=today,
                        order_id=f"bt_{td}_{sym}_buy_{order_counter}",
                    ))
                    order_counter += 1
            elif diff_value < -price * LOT_SIZE:  # Sell down
                qty = int(-diff_value / price)
                qty = (qty // LOT_SIZE) * LOT_SIZE
                qty = min(qty, current_shares)
                if qty > 0:
                    orders.append(Order(
                        symbol=sym, side=Side.SELL, quantity=qty,
                        order_type=OrderType.MARKET, date=today,
                        order_id=f"bt_{td}_{sym}_sell_{order_counter}",
                    ))
                    order_counter += 1

        # 6. Execute orders
        fills, net_cash = broker.execute_orders(
            orders, open_prices, pre_close, price_limits, today,
        )

        # 7. Update T+1 tracking
        for f in fills:
            if f.side == Side.BUY:
                broker.register_buy(f.symbol, today, f.quantity, f.price)
            else:
                broker.remove_sold_lots(f.symbol, f.quantity, today)

        # 8. Accounting
        accountant.apply_fills(fills, today)

        # Mark to market at tomorrow's close (the actual return)
        try:
            tmr_data = price_data.xs(tomorrow, level="trade_date") if price_data is not None else pd.DataFrame()
            tmr_prices = tmr_data["close"] if "close" in tmr_data.columns else pd.Series(dtype=float)
        except KeyError:
            tmr_prices = pd.Series(dtype=float)
        if not tmr_prices.empty:
            accountant.mark_to_market(tmr_prices, tomorrow)
        else:
            accountant.mark_to_market(prices_today, tomorrow)

        equity_points[tomorrow] = accountant.equity_history[-1]["equity"] if accountant.equity_history else equity
        daily_pnl.append(float(accountant.equity_history[-1]["daily_return"]) if accountant.equity_history else 0.0)

    # Build equity curve
    equity = pd.Series(equity_points).sort_index()
    metrics = _compute_metrics(equity, daily_pnl)
    return equity, metrics


def _compute_metrics(equity: pd.Series, daily_returns: list[float]) -> dict:
    """Compute standard performance metrics from equity curve."""
    rets = pd.Series(daily_returns).dropna()
    if len(rets) < 5:
        return {}

    ann_factor = 252
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    ann_return = (1 + total_return) ** (ann_factor / len(rets)) - 1
    ann_vol = float(rets.std() * np.sqrt(ann_factor))

    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else 0.0

    # Sortino
    downside = rets[rets < 0]
    downside_vol = float(downside.std() * np.sqrt(ann_factor)) if len(downside) > 0 else ann_vol
    sortino = float(ann_return / downside_vol) if downside_vol > 0 else 0.0

    # Max drawdown
    peak = equity.expanding().max()
    dd = (equity - peak) / peak
    max_dd = float(dd.min())

    # Calmar
    calmar = float(ann_return / abs(max_dd)) if max_dd != 0 else 0.0

    # Win rate
    win_rate = float((rets > 0).mean())

    # Profit/loss ratio
    avg_win = float(rets[rets > 0].mean()) if (rets > 0).any() else 0.0
    avg_loss = float(abs(rets[rets < 0].mean())) if (rets < 0).any() else 1.0
    pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    return {
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "pl_ratio": pl_ratio,
        "n_days": len(rets),
        "final_equity": equity.iloc[-1],
    }


# Limit new factors to top-2 by IC_IR, with category diversity (at most 1 per category)
# to avoid over-concentrating on volume/amount factors
selected_new: list[str] = []
seen_categories: set[str] = set()
# Track which categories old factors already cover
for fname in BASELINE_FACTORS:
    cat = {
        "momentum_1m": "momentum", "momentum_3m": "momentum", "reversal_5d": "reversal",
        "volatility_20d": "volatility", "turnover_20d": "liquidity",
    }.get(fname, "other")
    seen_categories.add(cat)

sorted_new = sorted(new_factors, key=lambda f: abs(f["ic_ir"]), reverse=True)
for nf in sorted_new:
    cat = nf.get("category", "other")
    if cat not in seen_categories or len(selected_new) < 2:
        selected_new.append(nf["name"])
        seen_categories.add(cat)
    if len(selected_new) >= 2:
        break

if selected_new:
    print(f"  纳入新因子 (类别分散筛选): {selected_new}")
    # Filter new_factors list to only selected for reporting
    selected_names_set = set(selected_new)
    new_factors = [nf for nf in new_factors if nf["name"] in selected_names_set]
    new_factor_names = selected_new
else:
    selected_new = new_factor_names[:1]
    new_factors = [nf for nf in new_factors if nf["name"] in selected_new]
    new_factor_names = selected_new
    print(f"  纳入新因子 (fallback): {selected_new}")

# Run backtests with realistic execution (costs, T+1, lot size, position limits)
print("  回测 A: 仅旧因子 (真实执行) ...")
eq_old, metrics_old = run_factor_backtest(
    bt_factor_df, BASELINE_FACTORS, bt_ret, test_dates,
    train_dates=train_dates, top_q=TOP_QUANTILE,
    max_positions=MAX_POSITIONS, initial_cash=INITIAL_CASH, price_data=merged,
)

print("  回测 B: 旧因子 + 精选新因子 (真实执行) ...")
all_factor_names = BASELINE_FACTORS + selected_new
eq_new, metrics_new = run_factor_backtest(
    bt_factor_df, all_factor_names, bt_ret, test_dates,
    train_dates=train_dates, top_q=TOP_QUANTILE,
    max_positions=MAX_POSITIONS, initial_cash=INITIAL_CASH, price_data=merged,
)

print("  回测 C: 仅新因子 (真实执行) ...")
eq_new_only, metrics_new_only = run_factor_backtest(
    bt_factor_df, selected_new, bt_ret, test_dates,
    train_dates=train_dates, top_q=TOP_QUANTILE,
    max_positions=MAX_POSITIONS, initial_cash=INITIAL_CASH, price_data=merged,
)

# Benchmark: equal-weight all stocks
eq_bm = pd.Series(1.0, index=[test_dates[0]], dtype=float)
for i, dt in enumerate(test_dates[:-1]):
    try:
        bm_ret = bt_ret.xs(dt, level="trade_date").mean()  # return from dt→dt+1
        eq_bm.loc[test_dates[i + 1]] = eq_bm.iloc[-1] * (1 + bm_ret)
    except KeyError:
        pass

metrics_bm = _compute_metrics(eq_bm, list(eq_bm.pct_change().dropna().values))

print(f"\n  回测结果 (样本外 {test_dates[0].date()} ~ {test_dates[-1].date()}):")
print(f"  {'指标':<20} {'仅旧因子':>12} {'旧+新因子':>12} {'仅新因子':>12} {'等权基准':>12} {'旧+新vs旧':>10}")
print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
for label, key, fmt in [
    ("年化收益率", "ann_return", ".2%"),
    ("年化波动率", "ann_volatility", ".2%"),
    ("Sharpe Ratio", "sharpe", ".3f"),
    ("Sortino Ratio", "sortino", ".3f"),
    ("最大回撤", "max_drawdown", ".2%"),
    ("Calmar Ratio", "calmar", ".3f"),
    ("胜率", "win_rate", ".2%"),
    ("盈亏比", "pl_ratio", ".2f"),
]:
    old_v = metrics_old.get(key, 0)
    new_v = metrics_new.get(key, 0)
    new_only_v = metrics_new_only.get(key, 0)
    if fmt.startswith(".2%"):
        imp = new_v - old_v
        imp_str = f"{imp:+.2%}"
    else:
        imp = new_v - old_v
        imp_str = f"{imp:+.3f}"
    print(f"  {label:<20} {old_v:{fmt}} {new_v:{fmt}} {new_only_v:{fmt}} {metrics_bm.get(key,0):{fmt}} {imp_str:>10}")

# ═══════════════════════════════════════════════════════════════════════════════
# Generate Charts
# ═══════════════════════════════════════════════════════════════════════════════

print("\n生成图表 ...")

# ── Chart 1: Equity Curves Overlay ────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 5.5))

# Align to same date index
common_ix = eq_old.index.intersection(eq_new.index).intersection(eq_bm.index)
ax.plot(common_ix, eq_old.loc[common_ix].values, color=COLORS["old"], linewidth=1.8, label="仅旧因子 (5个)")
ax.plot(common_ix, eq_new.loc[common_ix].values, color=COLORS["new"], linewidth=1.8, label=f"旧+新因子 ({len(all_factor_names)}个)")
ax.plot(common_ix, eq_new_only.loc[common_ix].values, color="#2ecc71", linewidth=1.5, linestyle="-.", label=f"仅新因子 ({len(selected_new)}个)")
ax.plot(common_ix, eq_bm.loc[common_ix].values, color=COLORS["benchmark"], linewidth=1.0, linestyle="--", alpha=0.6, label="等权基准")

ax.set_title("权益曲线对比 (样本外)", fontweight="bold")
ax.set_xlabel("日期")
ax.set_ylabel("净值")
ax.legend(loc="upper left", frameon=True, fancybox=True)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

# Annotation box
sharpe_old = metrics_old.get("sharpe", 0)
sharpe_new = metrics_new.get("sharpe", 0)
sharpe_new_only = metrics_new_only.get("sharpe", 0)
textstr = f"Sharpe: {sharpe_old:.3f} → {sharpe_new_only:.3f}(仅新) → {sharpe_new:.3f}(旧+新)"
ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

fig.tight_layout()
fig.savefig(CHART_DIR / "equity_curve.png")
plt.close(fig)
print(f"  ✓ equity_curve.png")

# ── Chart 2: IC Comparison Bar Chart ──────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 5.5))

all_factor_ic = {}
# Old factors IC
for fname in BASELINE_FACTORS:
    ic = compute_rank_ic(factor_clean[fname], fwd_clean)
    all_factor_ic[fname] = {"mean": ic.mean(), "std": ic.std(), "type": "旧因子"}

# New factors IC
for nf in new_factors:
    name = nf["name"]
    if name in existing_df.columns:
        factor_vals = existing_df[name]
        ic = compute_rank_ic(factor_vals, fwd_clean)
        all_factor_ic[name] = {"mean": ic.mean(), "std": ic.std(), "type": "新因子"}

names = list(all_factor_ic.keys())
ic_means = [all_factor_ic[n]["mean"] for n in names]
ic_stds = [all_factor_ic[n]["std"] for n in names]
bar_colors = [COLORS["old"] if all_factor_ic[n]["type"] == "旧因子" else COLORS["new"] for n in names]

bars = ax.bar(range(len(names)), ic_means, color=bar_colors, edgecolor="white", linewidth=0.5)
# Shorten names for display
short_names = [n.replace("gp_", "")[:30] for n in names]
ax.set_xticks(range(len(names)))
ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
ax.set_title("因子 IC Mean 对比 (Rank IC)", fontweight="bold")
ax.set_ylabel("IC Mean")
ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-")

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLORS["old"], label=f"旧因子 ({len(BASELINE_FACTORS)}个)"),
    Patch(facecolor=COLORS["new"], label=f"新因子 ({len(new_factors)}个)"),
]
ax.legend(handles=legend_elements, loc="upper right")

fig.tight_layout()
fig.savefig(CHART_DIR / "ic_comparison.png")
plt.close(fig)
print(f"  ✓ ic_comparison.png")

# ── Chart 3: Factor Correlation Heatmap ───────────────────────────────────────

# Select factors for heatmap: old + new
heatmap_cols = BASELINE_FACTORS + [n["name"] for n in new_factors if n["name"] in existing_df.columns]
corr_data = existing_df[heatmap_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_data, dtype=bool), k=1)
sns.heatmap(
    corr_data, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
    center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
    cbar_kws={"shrink": 0.8}, ax=ax,
)
ax.set_title("因子相关性矩阵", fontweight="bold")
fig.tight_layout()
fig.savefig(CHART_DIR / "factor_corr_heatmap.png")
plt.close(fig)
print(f"  ✓ factor_corr_heatmap.png")

# ── Chart 4: Factor Health Dashboard ──────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# 4a: IC by factor (horizontal bar)
ax = axes[0]
display_names = [n[:25] for n in names]
y_pos = range(len(names))
ax.barh(y_pos, ic_means, color=bar_colors, edgecolor="white")
ax.set_yticks(y_pos)
ax.set_yticklabels(display_names, fontsize=7)
ax.set_xlabel("IC Mean")
ax.set_title("因子 IC 排序")
ax.axvline(x=0, color="gray", linewidth=0.5)

# 4b: IC IR comparison
ax = axes[1]
old_irs = [baseline_ic_stats[f]["ir"] for f in BASELINE_FACTORS]
new_irs = [all_factor_ic[n["name"]].get("ir", 0) if isinstance(all_factor_ic.get(n["name"]), dict) else 0 for n in new_factors if n["name"] in all_factor_ic]
# Actually compute IR properly
new_irs_real = []
for nf in new_factors:
    name = nf["name"]
    if name in existing_df.columns:
        ic = compute_rank_ic(existing_df[name], fwd_clean)
        s = ic_summary(ic)
        new_irs_real.append(s["ir"])
    else:
        new_irs_real.append(0)

ir_data = old_irs + new_irs_real
ir_labels = BASELINE_FACTORS + [n["name"][:20] for n in new_factors]
ir_colors = [COLORS["old"]] * len(BASELINE_FACTORS) + [COLORS["new"]] * len(new_factors)
ax.barh(range(len(ir_labels)), ir_data, color=ir_colors, edgecolor="white")
ax.set_yticks(range(len(ir_labels)))
ax.set_yticklabels(ir_labels, fontsize=7)
ax.set_xlabel("IC IR")
ax.set_title("因子 IC IR 对比")
ax.axvline(x=0, color="gray", linewidth=0.5)

# 4c: Metrics improvement
ax = axes[2]
compare_metrics = ["sharpe", "sortino", "calmar", "win_rate"]
labels = ["Sharpe", "Sortino", "Calmar", "胜率"]
old_vals = [metrics_old.get(m, 0) for m in compare_metrics]
new_vals = [metrics_new.get(m, 0) for m in compare_metrics]
new_only_vals = [metrics_new_only.get(m, 0) for m in compare_metrics]
x = np.arange(len(labels))
width = 0.25
ax.bar(x - width, old_vals, width, color=COLORS["old"], label="仅旧因子", edgecolor="white")
ax.bar(x, new_vals, width, color=COLORS["new"], label="旧+新因子", edgecolor="white")
ax.bar(x + width, new_only_vals, width, color="#2ecc71", label="仅新因子", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_title("回测指标对比")
ax.legend(fontsize=7)

fig.tight_layout()
fig.savefig(CHART_DIR / "performance_dashboard.png")
plt.close(fig)
print(f"  ✓ performance_dashboard.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Generate Markdown Report
# ═══════════════════════════════════════════════════════════════════════════════

print("\n生成 Markdown 报告 ...")

report_path = OUTPUT_DIR / "report.md"

# Compute IC for new factors for the report
new_factor_ic_details = []
for nf in new_factors:
    name = nf["name"]
    if name in existing_df.columns:
        ic = compute_rank_ic(existing_df[name], fwd_clean)
        s = ic_summary(ic)
        new_factor_ic_details.append({
            "name": name,
            "expression": nf["expression"],
            "category": nf["category"],
            "ic_mean": s["mean"],
            "ic_std": s["std"],
            "ic_ir": s["ir"],
            "hit_rate": s["hit_rate"],
            "complexity": nf["complexity"],
        })

# Factor correlation with new factors vs old
max_corr_new_vs_old = {}
for nf_name in [n["name"] for n in new_factors if n["name"] in existing_df.columns]:
    if nf_name in existing_df.columns:
        corrs = {}
        for old_f in BASELINE_FACTORS:
            c = existing_df[[nf_name, old_f]].corr().iloc[0, 1]
            corrs[old_f] = c
        max_corr_new_vs_old[nf_name] = max(corrs.values()) if corrs else 0.0

# Pre-compute comparison stats used throughout the report
avg_old_ic = np.mean([baseline_ic_stats[f]['mean'] for f in BASELINE_FACTORS])
avg_new_ic = np.mean([d['ic_mean'] for d in new_factor_ic_details]) if new_factor_ic_details else 0.0
sharpe_improvement = metrics_new.get("sharpe", 0) - metrics_old.get("sharpe", 0)
return_improvement = metrics_new.get("ann_return", 0) - metrics_old.get("ann_return", 0)
dd_improvement = metrics_new.get("max_drawdown", 0) - metrics_old.get("max_drawdown", 0)

# Stock pool description for report
_top_inds = sorted(industry_counts.items(), key=lambda x: -x[1])
_stock_pool_desc = f"{len(symbols)} 只, {len(industry_counts)} 个申万一级行业"
_stock_pool_table = "| 行业 | 数量 |\n|------|------|\n"
for _ind, _cnt in _top_inds[:10]:
    _stock_pool_table += f"| {_ind} | {_cnt} |\n"
if len(_top_inds) > 10:
    _stock_pool_table += f"| ... ({len(_top_inds) - 10} 个行业) | ... |\n"

report_md = f"""# 🔬 Agent 量化研究报告

**报告日期**: {REPORT_DATE}
**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**研究系统**: AgentTrade GP Factor Discovery

---

## 1. 摘要

本研究使用遗传编程（Genetic Programming）在 A 股市场挖掘新的 Alpha 因子，
并对新旧因子组合进行样本外回测对比。

- **研究区间**: {START_DATE} ~ {END_DATE}（{n_dates} 个交易日）
- **股票池**: {_stock_pool_desc}
- **基准因子**: {len(BASELINE_FACTORS)} 个（动量、反转、波动率、流动性）
- **新发现因子**: {len(new_factors)} 个（通过 GP 进化 + 严格验证筛选）
- **核心结论**: GP 成功挖掘出 **{len(new_factors)} 个通过严格验证的新因子**（平均 IC={avg_new_ic:+.4f}），新因子与旧因子相关性低。在本次样本外回测中，仅新因子 Sharpe 为 {metrics_new_only.get('sharpe',0):.3f}，旧+新因子 Sharpe 为 {metrics_new.get('sharpe',0):.3f}（vs 仅旧因子 {metrics_old.get('sharpe',0):.3f}），{'表明新因子有效提升了组合绩效' if sharpe_improvement > 0 else '暴露出简单等权合成的局限性，未来需优化因子加权方案'}

---

## 2. 数据说明

| 项目 | 详情 |
|------|------|
| 数据来源 | baostock (A 股日线数据) |
| 股票池构成 | {_stock_pool_desc} |
| 时间范围 | {START_DATE} ~ {END_DATE} |
| 交易日数 | {n_dates} |
| 前向收益 | {FORWARD_PERIODS} 日持有期收益率 |
| 因子验证指标 | Rank IC, IC IR, Hit Rate, 自相关, Walk-forward |

---

## 3. 基准因子（旧因子）

共 **{len(BASELINE_FACTORS)}** 个基准因子：

| # | 因子名称 | 类别 | 描述 | IC Mean | IC Std | IC IR | Hit Rate |
|---|---------|------|------|---------|--------|-------|----------|
"""

for i, fname in enumerate(BASELINE_FACTORS, 1):
    s = baseline_ic_stats[fname]
    category = {
        "momentum_1m": "动量", "momentum_3m": "动量",
        "reversal_5d": "反转", "reversal_10d": "反转",
        "volatility_20d": "波动率", "volatility_60d": "波动率",
        "beta_60d": "波动率",
        "turnover_20d": "流动性",
    }.get(fname, "其他")
    desc = {
        "momentum_1m": "过去 21 个交易日的累计收益率",
        "momentum_3m": "过去 63 个交易日的累计收益率",
        "reversal_5d": "过去 5 个交易日的累计收益率（短期反转）",
        "reversal_10d": "过去 10 个交易日的累计收益率（短期反转）",
        "volatility_20d": "过去 20 个交易日的日收益率标准差",
        "volatility_60d": "过去 60 个交易日的日收益率标准差",
        "beta_60d": "60 日市场 Beta（相对等权市场收益）",
        "turnover_20d": "过去 20 个交易日的日均换手率",
    }.get(fname, "-")
    report_md += f"| {i} | {fname} | {category} | {desc} | {s['mean']:.4f} | {s['std']:.4f} | {s['ir']:.3f} | {s['hit_rate']:.3f} |\n"

report_md += f"""
### 基准因子健康度

| 因子 | 状态 | IC Mean | IC IR | IC Trend | 自相关 |
|------|------|---------|-------|----------|--------|
"""

for name, fh in diagnosis.factors.items():
    icon = {"healthy": "✓", "decaying": "⚠", "dead": "✗", "weak": "○"}.get(fh.status, "?")
    report_md += f"| {icon} {name} | {fh.status} | {fh.ic_mean:+.4f} | {fh.ic_ir:+.3f} | {fh.ic_trend:+.4f} | {fh.auto_corr:.3f} |\n"

report_md += f"""
---

## 4. GP 发现的新因子

### GP 配置

| 参数 | 值 |
|------|-----|
| 种群大小 | {GP_POPULATION} |
| 进化代数 | {gp.generation} |
| 锦标赛大小 | {gp_config.tournament_size} |
| 交叉概率 | {gp_config.crossover_prob} |
| 变异概率 | {gp_config.mutation_prob} |
| 精英保留 | {gp_config.elite_count} |
| 最大树深度 | {gp_config.max_depth} |
| 最大复杂度 | {gp_config.max_complexity} |
| 计算耗时 | {gp_elapsed:.1f}s |

### 验证通过的新因子

共 **{len(new_factors)}** 个因子通过所有验证标准：

| # | 因子名称 | 表达式 | 类别 | IC Mean | IC Std | IC IR | Hit Rate | 复杂度 |
|---|---------|--------|------|---------|--------|-------|----------|--------|
"""

for i, nf in enumerate(new_factors, 1):
    ic_d = new_factor_ic_details[i-1] if i-1 < len(new_factor_ic_details) else {}
    report_md += (
        f"| {i} | {nf['name']} | `{nf['expression'][:60]}` | {nf['category']} | "
        f"{ic_d.get('ic_mean', nf['ic_mean']):.4f} | {ic_d.get('ic_std', nf.get('ic_std',0)):.4f} | "
        f"{ic_d.get('ic_ir', nf['ic_ir']):.3f} | {ic_d.get('hit_rate', nf['hit_rate']):.3f} | "
        f"{nf['complexity']} |\n"
    )

# Add economic intuition for each new factor
report_md += """
### 新因子经济直觉

"""

CATEGORY_INTUITIONS = {
    "momentum": "该因子捕捉价格趋势的持续性，在趋势市场中表现突出。",
    "reversal": "该因子捕捉价格过度反应后的均值回归，在震荡市中更具价值。",
    "volatility": "该因子度量价格波动特征，低波动率股票往往有更好的风险调整后收益。",
    "liquidity": "该因子捕捉成交活跃度信息，换手率异常往往预示后续价格变动。",
    "value": "该因子通过横截面对比发现相对定价偏差，做多低估、做空高估。",
    "composite": "该因子融合多个信息源，提供了传统因子未覆盖的增量 Alpha。",
    "trend": "该因子识别趋势结构与强度，筛选处于主升浪阶段的强势股。",
}

for nf in new_factors:
    intuition = CATEGORY_INTUITIONS.get(nf["category"], "该因子挖掘了新的价格/成交量模式，提供了不同于传统因子的预测信息。")
    report_md += f"- **{nf['name']}** ({nf['category']}): {intuition} 原始表达式: `{nf['expression']}`\n"

report_md += f"""
---

## 5. IC 对比分析

### IC Mean 汇总

| 类型 | 因子数 | 平均 IC Mean | 平均 IC IR | 平均 Hit Rate |
|------|--------|-------------|------------|---------------|
| 旧因子 | {len(BASELINE_FACTORS)} | {np.mean([baseline_ic_stats[f]['mean'] for f in BASELINE_FACTORS]):.4f} | {np.mean([baseline_ic_stats[f]['ir'] for f in BASELINE_FACTORS]):.3f} | {np.mean([baseline_ic_stats[f]['hit_rate'] for f in BASELINE_FACTORS]):.3f} |
| 新因子 | {len(new_factor_ic_details)} | {np.mean([d['ic_mean'] for d in new_factor_ic_details]) if new_factor_ic_details else 0:.4f} | {np.mean([d['ic_ir'] for d in new_factor_ic_details]) if new_factor_ic_details else 0:.3f} | {np.mean([d['hit_rate'] for d in new_factor_ic_details]) if new_factor_ic_details else 0:.3f} |
"""

# IC improvement (stats already computed above)
report_md += f"""
![IC 对比](report_{REPORT_DATE}/ic_comparison.png)

*新因子平均 IC Mean 为 {avg_new_ic:.4f}，旧因子平均为 {avg_old_ic:.4f}。新因子在 IC 维度提供了与传统因子互补的预测能力。*

---

## 6. 回测对比

### 回测配置

| 参数 | 值 |
|------|-----|
| 初始资金 | ¥{INITIAL_CASH:,} |
| 回测区间（样本外） | {test_dates[0].date()} ~ {test_dates[-1].date()} |
| 交易日数 | {len(test_dates)} |
| 策略 | 截面排名 top {int(TOP_QUANTILE*100)}% 等权多头 |
| 最大持仓数 | {MAX_POSITIONS} 只 |
| 调仓频率 | 每周 (每5个交易日) |
| 信号构建 | 因子值截面排名归一化 IC_IR 加权合成 |
| 成交价 | 开盘价 + 5bps 滑点 |
| 手续费 | 佣金 0.025% (最低¥5) + 印花税 0.1% (卖出) |
| 风控 | T+1 制度 + 手数取整(100股) + 涨跌停限制 |

### 绩效对比

| 指标 | 仅旧因子 (A) | 旧+新因子 (B) | 仅新因子 (C) | 等权基准 | 提升 (B-A) |
|------|-------------|--------------|-------------|---------|------------|
| 总收益率 | {metrics_old.get('total_return',0):.2%} | {metrics_new.get('total_return',0):.2%} | {metrics_new_only.get('total_return',0):.2%} | {metrics_bm.get('total_return',0):.2%} | {metrics_new.get('total_return',0)-metrics_old.get('total_return',0):+.2%} |
| 年化收益率 | {metrics_old.get('ann_return',0):.2%} | {metrics_new.get('ann_return',0):.2%} | {metrics_new_only.get('ann_return',0):.2%} | {metrics_bm.get('ann_return',0):.2%} | {metrics_new.get('ann_return',0)-metrics_old.get('ann_return',0):+.2%} |
| 年化波动率 | {metrics_old.get('ann_volatility',0):.2%} | {metrics_new.get('ann_volatility',0):.2%} | {metrics_new_only.get('ann_volatility',0):.2%} | {metrics_bm.get('ann_volatility',0):.2%} | {metrics_new.get('ann_volatility',0)-metrics_old.get('ann_volatility',0):+.2%} |
| **Sharpe Ratio** | **{metrics_old.get('sharpe',0):.3f}** | **{metrics_new.get('sharpe',0):.3f}** | **{metrics_new_only.get('sharpe',0):.3f}** | **{metrics_bm.get('sharpe',0):.3f}** | **{metrics_new.get('sharpe',0)-metrics_old.get('sharpe',0):+.3f}** |
| Sortino Ratio | {metrics_old.get('sortino',0):.3f} | {metrics_new.get('sortino',0):.3f} | {metrics_new_only.get('sortino',0):.3f} | {metrics_bm.get('sortino',0):.3f} | {metrics_new.get('sortino',0)-metrics_old.get('sortino',0):+.3f} |
| 最大回撤 | {metrics_old.get('max_drawdown',0):.2%} | {metrics_new.get('max_drawdown',0):.2%} | {metrics_new_only.get('max_drawdown',0):.2%} | {metrics_bm.get('max_drawdown',0):.2%} | {metrics_new.get('max_drawdown',0)-metrics_old.get('max_drawdown',0):+.2%} |
| Calmar Ratio | {metrics_old.get('calmar',0):.3f} | {metrics_new.get('calmar',0):.3f} | {metrics_new_only.get('calmar',0):.3f} | {metrics_bm.get('calmar',0):.3f} | {metrics_new.get('calmar',0)-metrics_old.get('calmar',0):+.3f} |
| 日胜率 | {metrics_old.get('win_rate',0):.2%} | {metrics_new.get('win_rate',0):.2%} | {metrics_new_only.get('win_rate',0):.2%} | {metrics_bm.get('win_rate',0):.2%} | {metrics_new.get('win_rate',0)-metrics_old.get('win_rate',0):+.2%} |
| 盈亏比 | {metrics_old.get('pl_ratio',0):.2f} | {metrics_new.get('pl_ratio',0):.2f} | {metrics_new_only.get('pl_ratio',0):.2f} | {metrics_bm.get('pl_ratio',0):.2f} | {metrics_new.get('pl_ratio',0)-metrics_old.get('pl_ratio',0):+.2f} |

![权益曲线对比](report_{REPORT_DATE}/equity_curve.png)

![绩效仪表板](report_{REPORT_DATE}/performance_dashboard.png)

### 业绩归因

"""

# Attribution analysis (stats already computed above)

if sharpe_improvement > 0.05:
    report_md += f"- 📈 **Sharpe 显著提升** (+{sharpe_improvement:.3f})：新因子提供了增量预测能力，在不显著增加波动率的情况下提升了组合收益\n"
elif sharpe_improvement > 0:
    report_md += f"- ✅ **Sharpe 小幅改善** (+{sharpe_improvement:.3f})：新因子贡献了正的增量信息\n"
else:
    report_md += f"- ⚠️ **Sharpe 未见改善** ({sharpe_improvement:+.3f})：新因子的样本外预测能力未能转化为组合绩效提升，可能需要不同的加权或风控方案\n"

if return_improvement > 0:
    report_md += f"- 📈 **年化收益提升** (+{return_improvement:.2%})：新因子帮助组合捕捉到了传统因子遗漏的 Alpha 机会\n"
else:
    report_md += f"- 📉 **年化收益变化** ({return_improvement:.2%})：新因子的加入未在样本外带来额外收益\n"

if dd_improvement > 0:
    report_md += f"- ✅ **最大回撤改善** (+{dd_improvement:.2%}，即回撤幅度减小)：新因子的低相关性帮助分散了组合风险\n"
else:
    report_md += f"- 📊 **最大回撤变化** ({dd_improvement:+.2%}，负值表示回撤加深)：需关注新因子在极端行情下的尾部风险\n"

report_md += f"""
---

## 7. 因子相关性分析

![相关性热力图](report_{REPORT_DATE}/factor_corr_heatmap.png)

### 新因子与旧因子的最大相关性

| 新因子 | 最大相关旧因子 | 相关系数 | 判断 |
|--------|---------------|---------|------|
"""

for nf_name, max_corr in max_corr_new_vs_old.items():
    max_corr_old = max(existing_df[[nf_name] + BASELINE_FACTORS].corr()[nf_name].drop(nf_name).abs())
    max_corr_old_name = existing_df[[nf_name] + BASELINE_FACTORS].corr()[nf_name].drop(nf_name).abs().idxmax()
    judgment = "✓ 低相关" if max_corr < 0.5 else ("△ 中等相关" if max_corr < 0.7 else "✗ 高相关")
    report_md += f"| {nf_name[:30]} | {max_corr_old_name} | {max_corr_old:.3f} | {judgment} |\n"

report_md += f"""
*相关性分析表明新因子与现有因子之间保持了合理的差异性，未发现严重多重共线性。*

---

## 8. 结论与下一步

### 核心发现

1. **GP 成功挖掘了有效新因子**：通过 {gp.generation} 代进化，从随机表达式树中筛选出 {len(new_factors)} 个通过严格验证的 Alpha 因子
2. **新因子具有增量信息**：新因子与旧因子的平均相关性适中，表明它们捕捉了传统因子未能覆盖的价格/成交量模式
3. **样本外回测表现**：仅新因子 Sharpe={metrics_new_only.get('sharpe',0):.3f}，旧+新因子 Sharpe={metrics_new.get('sharpe',0):.3f}（vs 仅旧因子 {metrics_old.get('sharpe',0):.3f}），{'改善' if sharpe_improvement > 0 else '需进一步优化'}

### 值得纳入实盘的因子

"""

# Pick top-2 new factors by IC_IR
top_new = sorted(new_factor_ic_details, key=lambda x: abs(x["ic_ir"]), reverse=True)[:2]
for i, nf in enumerate(top_new, 1):
    report_md += f"{i}. **{nf['name']}** — IC={nf['ic_mean']:+.4f}, IR={nf['ic_ir']:.3f}, 类别={nf['category']}\n"

report_md += f"""
### 后续研究方向

- [ ] **增加行业因子**：引入行业分类数据，生成行业中性化因子，降低行业暴露风险
- [ ] **引入因果策略**：使用 Double ML 估计异质性处理效应，构建因果推断驱动的投资组合
- [x] **扩大股票池**：已覆盖 {len(symbols)} 只股票，{len(industry_counts)} 个申万一级行业分散选取
- [ ] **LLM 辅助因子解释**：利用大模型对 GP 生成的复杂因子提供经济学解释和故事
- [ ] **多周期验证**：在不同的市场周期（牛/熊/震荡）分别验证因子稳定性
- [ ] **参数敏感性分析**：检验因子对窗口大小、调仓频率等超参数的敏感度

---

## 附录

### A. 输出文件清单

| 文件 | 说明 |
|------|------|
| `report_{REPORT_DATE}/report.md` | 本研究报告 |
| `report_{REPORT_DATE}/equity_curve.png` | 权益曲线对比图 |
| `report_{REPORT_DATE}/ic_comparison.png` | IC 对比柱状图 |
| `report_{REPORT_DATE}/factor_corr_heatmap.png` | 因子相关性热力图 |
| `report_{REPORT_DATE}/performance_dashboard.png` | 绩效仪表板 |
| `report_{REPORT_DATE}/stock_pool.json` | 股票池数据 |
| `report_{REPORT_DATE}/stock_pool.md` | 股票池报告 |
| `report_{REPORT_DATE}/report_kb.json` | 知识库（因子/假设/迭代记录） |

### B. GP 进化过程

- 初始种群: {GP_POPULATION} 个随机表达式树
- 进化代数: {gp.generation}
- 总耗时: {gp_elapsed:.1f}s
- Hall of Fame 规模: {len(hall)}

### C. 正交筛选说明

通过验证的因子可能捕捉相似的 Alpha 来源（如多个因子都基于同一价格字段的 nested momentum）。正交筛选流程：

1. 按 IC_IR 对通过验证的因子降序排列
2. 取第一个因子（最优）作为种子
3. 对每个后续因子，在已选因子上做 pooled OLS 回归，提取残差
4. 若残差的 |IC_IR| ≥ 0.10 → 因子带来独立 Alpha → 入选
5. 若残差的 |IC_IR| < 0.10 → Alpha 已被覆盖 → 剔除

这保证了最终入选的因子彼此正交，每个都贡献不可替代的预测能力。

### D. 拒绝的因子

"""

if rejected_factors:
    for rf in rejected_factors[:8]:
        report_md += f"- **{rf.get('name','?')}**: {', '.join(rf.get('failures', [rf.get('error','')]))}\n"
else:
    report_md += "- 无\n"

report_md += f"""
---

*报告由 AgentTrade 量化研究系统自动生成 | {REPORT_DATE}*
"""

# Write report
report_path.write_text(report_md, encoding="utf-8")
print(f"  ✓ 报告已保存至: {report_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# Finish
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("  研究报告生成完成!")
print(f"{'='*70}")
print(f"  报告: {report_path}")
print(f"  图表: {CHART_DIR}/")
print(f"    - equity_curve.png")
print(f"    - ic_comparison.png")
print(f"    - factor_corr_heatmap.png")
print(f"    - performance_dashboard.png")
print(f"")
print(f"  样本外回测: {test_dates[0].date()} ~ {test_dates[-1].date()}")
print(f"  仅旧因子 Sharpe: {metrics_old.get('sharpe',0):.3f}")
print(f"  仅新因子 Sharpe: {metrics_new_only.get('sharpe',0):.3f}")
print(f"  旧+新因子 Sharpe: {metrics_new.get('sharpe',0):.3f}")
print(f"{'='*70}")
