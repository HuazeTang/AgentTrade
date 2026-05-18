"""End-to-end agent system demo.

Pipeline:
1. Data ingest via baostock
2. Baseline factor computation
3. Agent diagnosis (Monitor)
4. Hypothesis generation (ExplorationPlanner + LLM if available)
5. GP evolution (factor discovery)
6. Validate and report

Usage: python run_agent_demo.py
"""

from __future__ import annotations

import logging
import warnings
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agent_demo")

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOL_COUNT = 30
START_DATE = date(2024, 6, 1)
END_DATE = date(2026, 5, 14)
INITIAL_CASH = 1_000_000
GP_POPULATION = 30
GP_GENERATIONS = 8
MAX_NEW_FACTORS = 5

OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("Agent 量化研究系统 — 端到端演示")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Data Ingest
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1/7] 数据获取 (baostock) ...")
from data.sources.baostock import BaoStockSource
from data.pipeline import ingest_daily

source = BaoStockSource(rate_limit=0.05)
all_stocks = source.list_stocks()

candidates = all_stocks[~all_stocks["is_st"]].copy()
main = candidates[candidates["board"] == "main_board"]["symbol"].head(20)
chinext = candidates[candidates["board"] == "chinext"]["symbol"].head(8)
star = candidates[candidates["board"] == "star_market"]["symbol"].head(2)
symbols = pd.concat([main, chinext, star]).tolist()

print(f"  股票池: {len(symbols)} 只 ({len(main)} 主板 + {len(chinext)} 创业板 + {len(star)} 科创板)")
print(f"  时间: {START_DATE} ~ {END_DATE}")

raw_data = ingest_daily(symbols, START_DATE, END_DATE, source=source)
source.close()

n_rows = len(raw_data)
date_range = raw_data.index.get_level_values("trade_date")
print(f"  获取: {n_rows:,} 行, {raw_data.index.get_level_values('symbol').nunique()} 只股票")
print(f"  日期范围: {date_range.min().date()} ~ {date_range.max().date()}")

if raw_data.empty:
    print("  ERROR: 无数据!")
    exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Baseline Factor Computation
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/7] 基准因子计算 ...")
import factor.factors as _  # register all factors
from factor.engine import FactorEngine

engine = FactorEngine()
factor_names = ["momentum_1m", "momentum_3m", "reversal_5d", "volatility_20d", "turnover_20d"]
factor_df = engine.compute(factor_names, raw_data)
print(f"  计算: {len(factor_names)} 个因子, {factor_df.shape[0]:,} 个值")

# Merge and clean
merged = raw_data.join(factor_df, how="left")
merged = merged.dropna(subset=factor_names)
print(f"  去 NaN 后: {len(merged):,} 行")

# Compute forward returns (5-day) for IC evaluation
fwd_ret_5d = merged["close"].unstack().pct_change(periods=5).shift(-5).stack()

# Only use dates where we have both factors and forward returns
common_dates = factor_df.dropna().index.intersection(fwd_ret_5d.dropna().index)
factor_clean = factor_df.loc[common_dates]
fwd_clean = fwd_ret_5d.loc[common_dates]
print(f"  有效 IC 计算样本: {len(fwd_clean):,}")

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Baseline IC Analysis
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/7] 基准因子 IC 分析 ...")
from factor.validation import compute_rank_ic, ic_summary

print(f"  {'因子':<20} {'IC Mean':>10} {'IC Std':>10} {'IC IR':>10} {'Hit Rate':>10}")
print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for fname in factor_names:
    ic = compute_rank_ic(factor_clean[fname], fwd_clean)
    s = ic_summary(ic)
    print(f"  {fname:<20} {s['mean']:>10.4f} {s['std']:>10.4f} {s['ir']:>10.3f} {s['hit_rate']:>10.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: Agent Diagnosis
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/7] Agent 诊断 ...")
from agent.monitor import Monitor
from agent.decision import ExplorationPlanner
from agent.knowledge_base import KnowledgeBase, FactorRecord, HypothesisRecord, FailureRecord

monitor = Monitor(lookback_recent=60, auto_corr_warn=0.95, ic_min_abs=0.02)
diagnosis = monitor.diagnose(
    factor_values=factor_clean,
    forward_returns=fwd_clean,
    price_data=merged,
)

print(f"  市场状态: {diagnosis.regime} (置信度={diagnosis.regime_confidence:.2f})")
print(f"  因子健康度:")
for name, fh in diagnosis.factors.items():
    icon = {"healthy": "✓", "decaying": "⚠", "dead": "✗", "weak": "○"}.get(fh.status, "?")
    print(f"    {icon} {name}: {fh.status} | IC={fh.ic_mean:+.4f} IR={fh.ic_ir:+.3f} "
          f"trend={fh.ic_trend:+.4f} auto_corr={fh.auto_corr:.3f}")
    if fh.warnings:
        for w in fh.warnings:
            print(f"      ⚠ {w}")

if diagnosis.anomalies:
    print(f"  异常: {diagnosis.anomalies}")

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5: Hypothesis Generation
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5/7] 假设生成 ...")
kb = KnowledgeBase(path=str(OUTPUT_DIR / "demo_kb.json"))

# Register baseline factors in knowledge base
for fname in factor_names:
    fh = diagnosis.factors.get(fname)
    if fh:
        kb.add_factor(FactorRecord(
            name=fname,
            category="momentum" if "momentum" in fname else (
                "volatility" if "vol" in fname else (
                "liquidity" if "turnover" in fname else "reversal"
            )),
            description=f"Baseline factor",
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
print(f"  优先探索类别: {plan.focus_categories if plan.focus_categories else '全部'}")
print(f"  终端集: {plan.focus_terminals}")
print(f"  目标 (按紧迫度排序):")
for i, t in enumerate(plan.targets, 1):
    print(f"    {i}. [{t.urgency:.2f}] {t.target_type}: {t.description}")
    for action in t.suggested_actions[:2]:
        print(f"       → {action}")

# Try LLM if available
from agent.llm_client import create_default_client
llm = create_default_client()
if llm.configured:
    print(f"\n  LLM 可用 ({llm.backend}:{llm.model}), 生成因子思路...")
    try:
        ideas = llm.generate_factor_ideas(
            diagnosis=diagnosis.to_dict(),
            existing_factors=kb.get_active_factor_names(),
            n_ideas=3,
        )
        if ideas:
            for idea in ideas:
                name = idea.get("name", "?")
                intuition = idea.get("intuition", "?")
                cat = idea.get("category", "?")
                print(f"    💡 {name} [{cat}]: {intuition}")
    except Exception as e:
        print(f"    LLM 调用失败: {e}")
else:
    print(f"\n  LLM 未配置 (设置 DEEPSEEK_API_KEY 以启用)")

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 6: GP Evolution
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n[6/7] GP 因子挖掘 (种群={GP_POPULATION}, 代数={GP_GENERATIONS}) ...")
from discovery.gp import GPEngine, GPConfig
from discovery.validate import FactorValidator

# Build existing factors DataFrame for novelty check
existing_df = factor_clean[factor_names].copy()

# Build extended terminal set: raw fields + factor names
from discovery.expr import TERMINAL_FIELDS
_factor_terminals = [f for f in factor_names if f not in set(TERMINAL_FIELDS)]
_extended_terminals = list(TERMINAL_FIELDS) + _factor_terminals

gp_config = GPConfig(
    population_size=GP_POPULATION,
    max_generations=GP_GENERATIONS,
    tournament_size=5,
    crossover_prob=0.7,
    mutation_prob=0.4,
    elite_count=3,
    max_depth=5,
    max_complexity=20,
    early_stop_generations=4,
    terminals=_extended_terminals,
)
gp = GPEngine(config=gp_config)

# Run evolution (this is the heavy part)
import time
t0 = time.time()
best_individuals = gp.evolve(
    data=merged,
    forward_returns=fwd_clean,
    existing_factors=existing_df,
)
elapsed = time.time() - t0
print(f"  完成! 耗时 {elapsed:.1f}s, 代数={gp.generation}")

# Show results
print(f"\n  GP 发现的最佳因子 (Hall of Fame):")
hall = sorted(best_individuals, key=lambda x: x.fitness, reverse=True)[:10]
for i, ind in enumerate(hall, 1):
    print(f"  {i:2d}. {ind.factor_name:30s} fitness={ind.fitness:.4f}  "
          f"IC={ind.ic_mean:+.4f}  IR={ind.ic_ir:.3f}  "
          f"depth={ind.depth}  nodes={ind.complexity}  "
          f"expr={ind.tree!r}")

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 7: Validate & Learn
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n[7/7] 验证 & 学习 ...")
validator = FactorValidator()
accepted = []
rejected = []

for ind in hall[:MAX_NEW_FACTORS]:
    if ind.factor_cls is None:
        rejected.append(ind.factor_name)
        continue

    try:
        factor_vals = ind.factor_cls().compute(merged)
        result = validator.validate(
            factor_values=factor_vals,
            forward_returns=fwd_clean,
            factor_name=ind.factor_name,
            existing_factors=existing_df,
        )
        if result.passed:
            accepted.append(ind.factor_name)
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
            existing_df[ind.factor_name] = factor_vals  # add to pool
        else:
            rejected.append(ind.factor_name)
            print(f"  ✗ 拒绝: {ind.factor_name} ({', '.join(result.failures[:2])})")
    except Exception as e:
        rejected.append(ind.factor_name)
        print(f"  ✗ 错误: {ind.factor_name} — {e}")

kb.flush()

# ═══════════════════════════════════════════════════════════════════════════════
# Summary Report
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("迭代总结")
print(f"{'='*70}")
print(f"  数据: {len(symbols)} 只股票, {len(merged):,} 条记录")
print(f"  基准因子: {len(factor_names)} 个")
print(f"  GP 进化: {gp.generation} 代, {elapsed:.1f}s")
print(f"  发现因子: {len(accepted)} 个接受, {len(rejected)} 个拒绝")
print(f"  知识库: {kb.stats()['total_factors']} 个因子, {kb.stats()['total_hypotheses']} 个假设")
print(f"  状态: 诊断完成, 因子更新, 知识库已持久化")

if accepted:
    print(f"\n  采纳的新因子:")
    for name in accepted:
        rec = kb.get_factor(name)
        if rec:
            print(f"    • {name}: IC={rec.ic_mean:+.4f}, IR={rec.ic_ir:.3f}, category={rec.category}")

print(f"\n  知识库已保存至: {OUTPUT_DIR / 'demo_kb.json'}")
print(f"\n{'='*70}")
print("Agent 端到端演示完成!")
print(f"{'='*70}")
