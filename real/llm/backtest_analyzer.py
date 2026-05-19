"""LLM-powered backtest analysis: trade journal → weakness diagnosis → new factor ideas.

Usage:
    from llm.backtest_analyzer import analyze_backtest

    analysis = analyze_backtest(
        journal=journal,
        equity_series=equity,
        factor_weights=weights,
        start=start_date,
        end=end_date,
        max_positions=1,
        sell_rank_limit=5,
        take_profit_pct=0.25,
    )
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from agent.llm_client import LLMClient, create_default_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是一个A股量化策略研究员，擅长分析回测交易日志，"
    "发现策略行为弱点并提出新因子改进方案。"
)


def _build_trade_summary(journal: list[dict]) -> tuple[list[str], list[str]]:
    """Extract trade log lines and per-symbol P&L from journal."""
    trade_lines: list[str] = []
    symbol_trades: dict[str, list[dict]] = {}

    for entry in journal:
        for f_rec in entry.get("fills", []):
            sym = f_rec["symbol"]
            trade_lines.append(
                f"  {entry['date']} {f_rec['side'].upper():4s} {sym} "
                f"{f_rec['shares']}股 @¥{f_rec['price']:.2f}"
            )
            symbol_trades.setdefault(sym, []).append({
                "date": entry["date"],
                "side": f_rec["side"],
                "shares": f_rec["shares"],
                "price": f_rec["price"],
            })

    pnl_lines: list[str] = []
    for sym, trades in symbol_trades.items():
        buys = [t for t in trades if t["side"] == "buy"]
        sells = [t for t in trades if t["side"] == "sell"]
        total_bought = sum(b["shares"] * b["price"] for b in buys)
        total_sold = sum(s["shares"] * s["price"] for s in sells)
        if total_bought > 0 and total_sold > 0:
            pnl = total_sold - total_bought
            pnl_lines.append(
                f"  {sym}: 买入{len(buys)}笔 ¥{total_bought:,.0f}, "
                f"卖出{len(sells)}笔 ¥{total_sold:,.0f}, "
                f"净盈亏 {pnl/total_bought*100:+.1f}% (¥{pnl:+,.0f})"
            )

    return trade_lines, pnl_lines


def _build_equity_story(
    equity_series: pd.Series,
) -> tuple[list[str], list[str]]:
    """Extract best/worst days and significant drawdown periods."""
    ret = equity_series.pct_change().dropna()
    if len(ret) < 2:
        return [], []

    move_lines = [
        f"  Best day: {ret.idxmax().strftime('%Y-%m-%d')} ({ret.max():+.2%})",
        f"  Worst day: {ret.idxmin().strftime('%Y-%m-%d')} ({ret.min():+.2%})",
    ]

    peak = equity_series.expanding().max()
    dd = (equity_series - peak) / peak

    dd_periods: list[tuple[pd.Timestamp, pd.Timestamp, float]] = []
    in_dd = False
    dd_start: pd.Timestamp | None = None
    dd_max = 0.0
    for d in dd.index:
        dd_val = float(dd[d])
        if dd_val < -0.10 and not in_dd:
            in_dd = True
            dd_start = d
            dd_max = dd_val
        elif dd_val < -0.10 and in_dd:
            dd_max = min(dd_max, dd_val)
        elif dd_val >= -0.02 and in_dd:
            dd_periods.append((dd_start, d, dd_max))  # type: ignore[arg-type]
            in_dd = False
    if in_dd and dd_start is not None:
        dd_periods.append((dd_start, dd.index[-1], dd_max))

    dd_lines = []
    for s, e, depth in dd_periods[:5]:
        dd_lines.append(
            f"  {s.strftime('%Y-%m-%d')} → {e.strftime('%Y-%m-%d')}: "
            f"max drawdown {depth:.1%}"
        )
    return move_lines, dd_lines


def analyze_backtest(
    journal: list[dict],
    equity_series: pd.Series,
    factor_weights: dict[str, float],
    start: date,
    end: date,
    max_positions: int = 1,
    sell_rank_limit: int = 5,
    take_profit_pct: float = 0.25,
    llm: LLMClient | None = None,
) -> dict | None:
    """Send backtest results to LLM; return diagnosis + factor proposals.

    Returns None if no LLM is available or there are no trades.
    """
    if llm is None or not llm.configured:
        try:
            llm = create_default_client()
        except Exception:
            pass
    if llm is None or not llm.configured:
        logger.info("LLM backtest analysis skipped (no API key)")
        return None

    trade_lines, pnl_lines = _build_trade_summary(journal)
    if not trade_lines:
        logger.info("LLM backtest analysis skipped (no trades)")
        return None

    move_lines, dd_lines = _build_equity_story(equity_series)

    # Factor weights (top 10 by absolute value)
    sorted_fw = sorted(factor_weights.items(), key=lambda x: abs(x[1]), reverse=True)
    factor_lines = [
        f"  {name}: {weight:.4f} {'[GP]' if name.startswith('gp_') else ''}"
        for name, weight in sorted_fw[:10]
    ]

    ret = equity_series.pct_change().dropna()
    strategy_desc = (
        f"策略：Top-{max_positions}买入，排名跌出Top-{sell_rank_limit}卖出，"
        f"止盈线{take_profit_pct*100:.0f}%，每周调仓"
    )

    prompt = f"""## 回测绩效
- 期间: {start} → {end}
- 累计收益: {equity_series.iloc[-1]/equity_series.iloc[0]-1:.2%}
- Sharpe: {ret.mean()*252/(ret.std()*252**0.5):.3f} (假设0利率)
- 最大回撤: {((equity_series/equity_series.expanding().max()-1).min()):.1%}
- 胜率: {(ret>0).mean():.1%}
- 总交易: {len(trade_lines)} 笔

## 策略规则
{strategy_desc}

## 因子权重 (Top-10)
{chr(10).join(factor_lines)}

## 极端波动
{chr(10).join(move_lines) if move_lines else '(无)'}

## 严重回撤区间
{chr(10).join(dd_lines) if dd_lines else '(无显著回撤区)'}

## 个股盈亏
{chr(10).join(pnl_lines) if pnl_lines else '(无完整交易)'}

## 完整交易日志
{chr(10).join(trade_lines)}

## 指令
你是量化研究员。请分析以上回测结果，从交易行为中诊断策略弱点（例如：是否卖飞牛股？是否止损过快？是否在回撤期买入？是否过度集中在某些行业？），然后提出 2-4 个可能改进的新因子想法。

输出 JSON：
{{
  "diagnosis": "弱点诊断（中文，2-3句话，指最严重的问题）",
  "proposed_factors": [
    {{
      "name": "因子名（如 trend_strength_hold）",
      "intuition": "这个因子要捕捉什么（中文，一句话）",
      "expression_hint": "表达式提示（如 ts_mean(close, 60) / close）"
    }}
  ]
}}"""

    logger.info("Sending backtest analysis to LLM...")
    try:
        result = llm.chat_json(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            expected_keys=["diagnosis", "proposed_factors"],
        )
        if "error" in result:
            logger.warning("LLM analysis failed: %s", result.get("error"))
            return None
        logger.info("LLM backtest analysis complete: %d proposed factors",
                   len(result.get("proposed_factors", [])))
        return result
    except Exception as e:
        logger.warning("LLM backtest analysis error: %s", e)
        return None
