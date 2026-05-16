"""
简易 LLM 延迟压测脚本：并发调用 Qwen（DashScope）并输出延迟分布/QPS，自动记录易读的缩进 JSON 日志（若指定 log-file）。
用法示例：
    python test_api_speed.py --total 50 --concurrency 10 --llm-model qwen-plus
    python test_api_speed.py --prompt-preset trade --concurrency 5
    python test_api_speed.py --prompt-file prompt.txt --log-file logs/llm.json
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Callable, Dict, List, Optional, Tuple

import requests

RequestFn = Callable[[requests.Session], Tuple[str, str]]

LLM_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    pct = max(0.0, min(100.0, pct))
    idx = int(round((len(values) - 1) * pct / 100.0))
    return sorted(values)[idx]


def summarize(latencies: List[float], total_sent: int, errors: int, wall_time: float) -> Dict[str, float]:
    if not latencies:
        return {"sent": total_sent, "ok": 0, "errors": errors, "duration": wall_time}
    total = len(latencies)
    avg = sum(latencies) / total
    return {
        "sent": total_sent,
        "ok": total,
        "errors": errors,
        "duration": wall_time,
        "qps": total / wall_time if wall_time > 0 else 0.0,
        "avg_ms": avg * 1000,
        "p50_ms": percentile(latencies, 50) * 1000,
        "p95_ms": percentile(latencies, 95) * 1000,
        "p99_ms": percentile(latencies, 99) * 1000,
        "max_ms": max(latencies) * 1000,
    }


def run_benchmark(
    name: str,
    total_requests: int,
    concurrency: int,
    fn: RequestFn,
    log_file: str = "",
    log_max_len: int = 2000,
) -> Dict[str, float]:
    """多线程循环调用 fn，收集每次请求耗时并可选落盘日志。"""
    tasks: queue.Queue[int] = queue.Queue()
    for i in range(total_requests):
        tasks.put(i)

    latencies: List[float] = []
    errors = 0
    err_lock = Lock()
    logs: List[dict] = []
    log_lock = Lock()

    def worker() -> None:
        nonlocal errors
        session = requests.Session()
        while True:
            try:
                tasks.get_nowait()
            except queue.Empty:
                return
            
            # 提前获取 prompt，确保在异常捕获中可用
            try:
                # 这里 fn 是 build_llm_call 返回的 _call，它内部会调 prompt_supplier
                # 但为了在 fn 失败时也能记录 prompt，我们需要稍作调整
                # 理想方案是让 fn 只负责网络请求，prompt 在外层获取
                # 不过为了改动最小，我们先通过 locals() 机制适配，
                # 但 fn 必须能成功返回 prompt。如果 fn 内部 prompt_supplier() 就挂了，
                # 则 used_prompt 确实拿不到。
                start = time.perf_counter()
                resp_text, used_prompt = fn(session)
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)
                if log_file:
                    entry = {
                        "ts": time.time(),
                        "status": "ok",
                        "latency_ms": elapsed * 1000,
                        "prompt": used_prompt,
                        "response": resp_text[:log_max_len],
                    }
                    with log_lock:
                        logs.append(entry)
            except Exception as exc:
                with err_lock:
                    errors += 1
                if log_file:
                    # 尝试从 locals 提取，或者如果 fn 还没返回就报错了，则记录 None
                    p = locals().get("used_prompt")
                    entry = {
                        "ts": time.time(),
                        "status": "error",
                        "error": str(exc),
                        "prompt": p,
                    }
                    with log_lock:
                        logs.append(entry)
                print(f"[{name}] error: {exc}")
            finally:
                tasks.task_done()

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker) for _ in range(concurrency)]
        for f in futures:
            f.result()
    wall = time.perf_counter() - t0
    stats = summarize(latencies, total_requests, errors, wall)
    print(
        f"[{name}] sent={stats.get('sent')} ok={stats.get('ok', 0)} "
        f"errors={stats.get('errors')} duration={stats.get('duration'):.3f}s "
        f"qps={stats.get('qps', 0):.1f} "
        f"avg={stats.get('avg_ms', 0):.2f}ms "
        f"p95={stats.get('p95_ms', 0):.2f}ms max={stats.get('max_ms', 0):.2f}ms"
    )
    if log_file and logs:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        print(f"[{name}] logs written to {path} ({len(logs)} entries, pretty JSON, truncated response to {log_max_len} chars)")
    return stats


def build_llm_call(model: str, prompt_supplier: Callable[[], str], api_key: str, timeout: float = 20.0) -> RequestFn:
    if not api_key:
        raise RuntimeError("LLM api_key missing; set --llm-key 或环境变量 QWEN_API_KEY")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    def _call(session: requests.Session) -> Tuple[str, str]:
        prompt = prompt_supplier()
        payload = {
            "model": model,
            "input": {"prompt": prompt},
            # result_format 使用 text 以减少解析时间；无需 JSON 结构化。
            "parameters": {"result_format": "text"},
        }
        resp = session.post(LLM_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.text, prompt

    return _call


@dataclass
class BenchPlan:
    name: str
    builder: Callable[[], RequestFn]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LLM latency/throughput (DashScope/Qwen).")
    parser.add_argument("--total", type=int, default=50, help="请求数量")
    parser.add_argument("--concurrency", type=int, default=10, help="并发工作线程数")
    parser.add_argument("--llm-model", type=str, default="qwen-plus", help="LLM 模型名（例如 qwen-plus）")
    parser.add_argument("--llm-prompt", type=str, default="ping", help="LLM 压测提示词（若提供 prompt-file/preset 则被覆盖）")
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="",
        help="从文件读取压测提示词（utf-8）；优先级最高",
    )
    parser.add_argument(
        "--prompt-preset",
        type=str,
        default="",
        choices=["", "trade", "trade-log"],
        help="内置提示词模板；当 prompt-file 为空且指定该值时生效",
    )
    parser.add_argument("--log-dir", type=str, default="sim_logs", help="读取仿真日志目录（trade-log 模式需要）")
    parser.add_argument("--sample-agent", type=str, default="agent-001", help="从日志中抽取的账户 ID（trade-log 模式）")
    parser.add_argument("--recent-n", type=int, default=20, help="trade-log 模式下，抽取最近 N 条成交价格作为 recent_prices")
    parser.add_argument("--risk-limit", type=float, default=5000.0, help="trade-log 模式下的持仓上限")
    parser.add_argument("--llm-timeout", type=float, default=20.0, help="LLM 请求超时秒数")
    parser.add_argument("--log-file", type=str, default="", help="将每次请求的 prompt/响应/耗时写入缩进 JSON 文件")
    parser.add_argument("--log-max-len", type=int, default=2000, help="单条响应保留的最大字符数，避免日志过大")
    parser.add_argument(
        "--llm-key",
        type=str,
        default="",
        help="LLM API Key，留空则读取环境变量 QWEN_API_KEY 或 API_KEY",
    )
    args = parser.parse_args()

    llm_key = args.llm_key or os.getenv("QWEN_API_KEY") or os.getenv("API_KEY") or ""
    if not llm_key:
        print("缺少 LLM key，请使用 --llm-key 或设置环境变量 QWEN_API_KEY/API_KEY")
        return

    def load_json(path: Path) -> Optional[dict]:
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    def _nearest_by_ts(items: List[dict], ts_key: str, target_ts: float) -> dict:
        if not items:
            return {}
        return min(items, key=lambda x: abs(float(x.get(ts_key, 0) or 0) - target_ts))

    def load_trade_log_prompt(log_dir: str, agent: str, recent_n: int, risk_limit: float) -> Optional[str]:
        """从 sim_logs 中选取一笔成交(tx)，并配套最近的盘口与账户快照生成提示词。"""
        base = Path(log_dir)
        trades_p = base / "trades.json"
        book_p = base / "book_series.json"
        agents_p = base / "agents.json"
        if not trades_p.exists() or not book_p.exists() or not agents_p.exists():
            missing = [p.name for p in [trades_p, book_p, agents_p] if not p.exists()]
            print(f"Missing files in {log_dir}: {missing}")
            return None

        trades_obj = load_json(trades_p) or {}
        trades = trades_obj.get("trades") or []
        target_trade = {}
        if not trades:
            print(f"Warning: {trades_p} contains no trades or failed to load.")
            recent_prices = []
            current_price = 100.0
        else:
            target_trade = trades[-1]  # 选取最新一笔成交作为决策场景
            idx = len(trades) - 1
            start = max(0, idx - recent_n + 1)
            window = trades[start : idx + 1]
            recent_prices = [t.get("price") for t in window if "price" in t]
            current_price = target_trade.get("price", recent_prices[-1] if recent_prices else 100.0)

        book_obj = load_json(book_p) or {}
        book_series = book_obj.get("book_series") or []
        if trades and book_series:
            book_pick = _nearest_by_ts(book_series, "ts", float(target_trade.get("ts", 0.0) or 0.0))
        else:
            if not book_series:
                print(f"Warning: {book_p} contains no book_series or failed to load.")
            book_pick = book_series[-1] if book_series else {}
        bids = book_pick.get("bids") or []
        asks = book_pick.get("asks") or []

        agents_obj = load_json(agents_p) or {}
        agents_list = agents_obj.get("agents_series") or []
        if trades and agents_list:
            pick_agents = _nearest_by_ts(agents_list, "ts", float(target_trade.get("ts", 0.0) or 0.0))
        else:
            if not agents_list:
                print(f"Warning: {agents_p} contains no agents_series or failed to load.")
            pick_agents = agents_list[-1] if agents_list else {}
        
        # 选决策主体：若 sample_agent 在成交双方中，则用它，否则默认使用买方作为决策主体
        if trades:
            buy_agent = target_trade.get("buy_agent") or ""
            sell_agent = target_trade.get("sell_agent") or ""
            decision_agent = agent if agent in (buy_agent, sell_agent) else buy_agent or agent
        else:
            decision_agent = agent
            buy_agent = ""
            sell_agent = ""

        def find_agent_state(aid: str) -> dict:
            for a in pick_agents.get("agents", []):
                if a.get("agent") == aid:
                    return {"cash": a.get("cash"), "position": a.get("position")}
            return {}

        agent_state = find_agent_state(decision_agent) or {"cash": 100000.0, "position": 0.0}
        buy_state = find_agent_state(buy_agent) or {}
        sell_state = find_agent_state(sell_agent) or {}

        trade_preset_log = f"""
你是一个交易 Agent，需输出交易决策 JSON，不要多余解释。
输入字段（已填充真实仿真数据）：
- recent_prices: {recent_prices}
- recent_prices 顺序：从旧到新（recent_prices[0] 最早，末尾为最新）
- current_price: {current_price}
- order_book: bids={bids[:5]}, asks={asks[:5]}
- my_account: {agent_state}  # 作为决策主体的账户（若 sample_agent 在成交中，则使用它，否则使用买方）
- risk_limit: {risk_limit}
- last_trade: {{"tx": "{target_trade.get("tx_hash") if trades else ""}", "buy_agent": "{buy_agent}", "sell_agent": "{sell_agent}", "price": {current_price}, "quantity": {target_trade.get("quantity") if trades else 0}, "ts": {target_trade.get("ts") if trades else 0}}}
- buy_agent_state: {buy_state}
- sell_agent_state: {sell_state}
要求：
- 根据趋势与盘口做出买入或卖出决策
- JSON 字段: side (buy/sell/hold), price (float), quantity (int>=1), reason (string，简述当前交易/不交易的理由)
- 不要超出 risk_limit 和可用现金/持仓
- 若不交易，返回 {{"side": "hold", "price": 0, "quantity": 0, "reason": "no-op"}}
示例输入已在上方提供，请仅输出 JSON。
"""
        return trade_preset_log.strip()

    trade_preset = """
你是一个交易 Agent，需输出交易决策 JSON，不要多余解释。
输入字段：
- recent_prices: 过去 N 个时间点的价格序列，例如 [100.1, 100.3, 100.0, 99.8, 100.2]
- recent_prices 顺序：从旧到新（recent_prices[0] 最早，末尾为最新）
- current_price: 当前最新成交价
- order_book: 盘口，包含 bids/asks，每个元素 {price, quantity}
- my_account: 我的账户 {cash, position}
- risk_limit: 我的持仓上限，例如 5000
要求：
- 根据趋势与盘口做出买入或卖出决策
- JSON 字段: side (buy/sell/hold), price (float), quantity (int>=1), reason (string，简述当前交易/不交易的理由)
- 不要超出 risk_limit 和可用现金/持仓
- 若不交易，返回 {"side": "hold", "price": 0, "quantity": 0, "reason": "no-op"}
示例输入：
recent_prices=[100.1,100.3,100.0,99.8,100.2]
current_price=100.05
order_book.bids=[[99.9,120],[99.8,200]], asks=[[100.1,150],[100.2,180]]
my_account={cash: 100000, position: 1200}, risk_limit=5000
请仅输出 JSON。
"""

    # 选择提示词供应器优先级：prompt_file > prompt_preset(trade-log/trade) > llm_prompt
    if args.prompt_file:
        static_prompt = Path(args.prompt_file).read_text(encoding="utf-8")

        def prompt_supplier() -> str:
            return static_prompt

    elif args.prompt_preset == "trade-log":
        # 优化：在循环外加载一次，避免压测时重复读盘解析 JSON 导致延迟测量不准
        static_prompt = load_trade_log_prompt(args.log_dir, args.sample_agent, args.recent_n, args.risk_limit)
        if not static_prompt:
            print(f"[trade-log] 无法读取日志，请检查 {args.log_dir} 下是否有 trades.json/book_series.json/agents.json")
            return

        def prompt_supplier() -> str:
            return static_prompt

    elif args.prompt_preset == "trade":
        static_prompt = trade_preset.strip()

        def prompt_supplier() -> str:
            return static_prompt

    else:

        def prompt_supplier() -> str:
            return args.llm_prompt

    plan = BenchPlan(
        "llm",
        lambda: build_llm_call(args.llm_model, prompt_supplier, llm_key, timeout=args.llm_timeout),
    )
    run_benchmark(
        plan.name,
        args.total,
        args.concurrency,
        plan.builder(),
        log_file=args.log_file,
        log_max_len=args.log_max_len,
    )


if __name__ == "__main__":
    main()

