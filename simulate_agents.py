"""
Simple simulation driver:
- Registers N agents with equal initial cash/position
- Lets them trade around a target mid price (default 100)
- Keeps local logs of all submitted orders and server trade history

Prereq: start the API locally:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
Then run:
    python simulate_agents.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import requests

API = "http://127.0.0.1:8000"
LLM_API_KEY = os.environ["API_KEY"]  # Placeholder for Tongyi/Qwen; left empty for fallback heuristic


@dataclass
class OrderDecision:
    side: Literal["buy", "sell"]
    price: float
    quantity: int


def llm_or_heuristic_decision(agent: str, mid: float = 100.0) -> OrderDecision:
    """
    Placeholder for LLM-driven strategy. If LLM_API_KEY is set, replace this body
    with a call to Tongyi/Qwen to decide side/price/qty. For now, use a simple
    stochastic strategy around the target mid price.
    """
    # Heuristic: small random skew around mid, alternate sides
    side = random.choice(["buy", "sell"])
    price = round(random.normalvariate(mid, 1.5), 2)
    price = max(0.1, price)
    qty = random.randint(1, 10)
    return OrderDecision(side=side, price=price, quantity=qty)


def llm_decision_qwen(
    agent: str,
    book: dict,
    trades: dict,
    me: dict,
    model: str = "qwen-plus",
    fallback_mid: float = 100.0,
) -> Optional[OrderDecision]:
    """
    Call Qwen for a structured trading decision.
    Falls back to heuristic if API key is missing or call fails.
    """
    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        return None

    buy1 = book.get("bids", [{}])[0] if book.get("bids") else {}
    sell1 = book.get("asks", [{}])[0] if book.get("asks") else {}
    recent = trades.get("trades", [])[-5:]
    prompt = f"""
你是一个简化的限价交易 Agent，需要输出一个 JSON 决策，不要多余文字。
字段: side(buy/sell), price(数字), quantity(整数>=1)。
约束:
- 买单价格不高于卖1价过多（可小幅跨1-2跳），卖单价格不低于买1价过多。
- 数量必须在持仓/余额允许范围内，整数手。
当前盘口:
买1={buy1.get('price')} 数量={buy1.get('quantity')}；卖1={sell1.get('price')} 数量={sell1.get('quantity')}
最近成交(价,量): {[ (t.get('price'), t.get('quantity')) for t in recent ]}
我的余额={me.get('cash')}, 持仓={me.get('position')}
仅输出 JSON，例如: {{"side":"buy","price":100.12,"quantity":3}}
"""
    try:
        resp = requests.post(
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": model,
                "input": {"prompt": prompt},
                "parameters": {"result_format": "json"},
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        # Depending on dashscope response shape
        text = (
            data.get("output", {}).get("text")
            or data.get("choices", [{}])[0].get("message", {}).get("content")
        )
        if not text:
            return None
        import json as pyjson

        parsed = pyjson.loads(text)
        side = parsed.get("side")
        price = float(parsed.get("price"))
        qty = int(parsed.get("quantity"))
        if side not in ("buy", "sell") or qty < 1 or price <= 0:
            return None
        return OrderDecision(side=side, price=price, quantity=qty)
    except Exception:
        return None


def register_agent(agent: str, cash: float, position: float) -> None:
    resp = requests.post(
        f"{API}/agents",
        json={"agent": agent, "initial_cash": cash, "initial_position": position},
        timeout=10,
    )
    if resp.status_code not in (200, 400):
        raise RuntimeError(f"register {agent} failed: {resp.text}")
    if resp.status_code == 400 and "already exists" not in resp.text:
        raise RuntimeError(f"register {agent} failed: {resp.text}")


def place_order(agent: str, decision: OrderDecision) -> dict:
    payload = {
        "agent": agent,
        "side": decision.side,
        "price": decision.price,
        "quantity": decision.quantity,
    }
    resp = requests.post(f"{API}/orders", json=payload, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"order failed for {agent}: {resp.text}")
    return resp.json()


def fetch_book(depth: int = 8) -> dict:
    return requests.get(f"{API}/book", params={"depth": depth}, timeout=10).json()


def fetch_trades(limit: int = 100000) -> dict:
    return requests.get(f"{API}/trades", params={"limit": limit}, timeout=10).json()


def fetch_agent(agent: str) -> dict:
    return requests.get(f"{API}/agents/{agent}", timeout=10).json()


def fetch_agents_all() -> dict:
    return requests.get(f"{API}/agents", timeout=10).json()


def run_simulation(
    agent_count: int = 100,
    initial_cash: float = 100_000.0,
    initial_position: float = 1_000.0,
    duration_sec: float = 60.0,
    target_mid: float = 100.0,
    orders_per_sec: float = 20.0,
    log_dir: str = "sim_logs",
    sample_agent: str = "agent-001",
    snapshot_interval: float = 0.0,  # 0 表示每次下单后都抓快照
    use_llm: bool = False,
):
    print(f"Registering {agent_count} agents...")
    for i in range(agent_count):
        aid = f"agent-{i+1:03d}"
        register_agent(aid, initial_cash, initial_position)
    print("Registration done.")

    if use_llm:
        print("Using LLM models")

    orders_log: List[dict] = []
    gap = 1.0 / orders_per_sec if orders_per_sec > 0 else 0.0
    book_series: List[dict] = []
    agents_series: List[dict] = []
    last_snapshot = 0.0
    last_trade_ts = None

    print(f"Running for {duration_sec}s at ~{orders_per_sec} orders/s around mid={target_mid} ...")
    start = time.time()
    n = 0
    while time.time() - start < duration_sec:
        aid = f"agent-{(n % agent_count) + 1:03d}"
        if use_llm:
            # Fetch context for LLM decision
            book_ctx = fetch_book(depth=5)
            trades_ctx = fetch_trades(limit=50)
            me_ctx = fetch_agent(aid)
            decision = llm_decision_qwen(aid, book_ctx, trades_ctx, me_ctx, fallback_mid=target_mid)
            if decision is None:
                decision = llm_or_heuristic_decision(aid, mid=target_mid)
        else:
            decision = llm_or_heuristic_decision(aid, mid=target_mid)
        trade_ts = None
        try:
            res = place_order(aid, decision)
            # capture server trade timestamp if present (better alignment than client time)
            if res.get("trades"):
                trade_ts = max(t.get("ts", 0) for t in res["trades"])
            orders_log.append(
                {
                    "agent": aid,
                    "decision": decision.__dict__,
                    "response": res,
                    "ts": time.time(),
                }
            )
        except Exception as exc:  # keep going even if some orders reject (e.g., balance)
            orders_log.append(
                {
                    "agent": aid,
                    "decision": decision.__dict__,
                    "error": str(exc),
                    "ts": time.time(),
                }
            )
        n += 1

        now = time.time()
        need_snap = snapshot_interval == 0 or now - last_snapshot >= snapshot_interval
        if need_snap:
            # capture book and agents with timestamp; if本次有成交，优先用本次成交 ts，否则用当前时间
            snap_ts = trade_ts or now
            try:
                book_snap = fetch_book(depth=10)
                book_series.append({"ts": snap_ts, "bids": book_snap.get("bids", []), "asks": book_snap.get("asks", [])})
            except Exception as exc:
                orders_log.append({"agent": aid, "decision": decision.__dict__, "error": f"book snapshot failed: {exc}", "ts": now})
            try:
                agents_snap = fetch_agents_all()
                agents_series.append({"ts": snap_ts, "agents": agents_snap.get("agents", [])})
            except Exception as exc:
                orders_log.append({"agent": aid, "decision": decision.__dict__, "error": f"agents snapshot failed: {exc}", "ts": now})
            last_snapshot = now

        if gap:
            time.sleep(gap)

    print("Simulation window completed. Fetching final state...")
    trades = fetch_trades(limit=50000)

    sample_state = fetch_agent(sample_agent)

    print("\n=== Summary ===")
    print(f"Total orders sent: {len(orders_log)}")
    print(f"Total trades on server: {len(trades.get('trades', []))}")
    print(f"Sample agent {sample_agent} state: {sample_state}")
    if trades.get("trades"):
        print("Last 3 trades:")
        for t in trades["trades"][-3:]:
            print(t)

    out_dir = Path(log_dir)
    out_dir.mkdir(exist_ok=True)
    Path(out_dir / "orders.json").write_text(json.dumps(orders_log, indent=2))
    Path(out_dir / "trades.json").write_text(json.dumps(trades, indent=2))
    Path(out_dir / "book_series.json").write_text(json.dumps({"book_series": book_series}, indent=2))
    Path(out_dir / "agents.json").write_text(json.dumps({"agents_series": agents_series}, indent=2))
    print(f"\nLogs written to {out_dir}/orders.json, trades.json, book_series.json, agents.json")


def parse_args():
    p = argparse.ArgumentParser(description="Continuous agent trading simulation.")
    p.add_argument("--agents", type=int, default=100, help="Number of agents to register")
    p.add_argument("--initial-cash", type=float, default=100_000.0, help="Initial cash per agent")
    p.add_argument("--initial-pos", type=float, default=1_000.0, help="Initial position per agent")
    p.add_argument("--duration", type=float, default=60.0, help="Duration in seconds to run")
    p.add_argument("--ops", type=float, default=20.0, help="Orders per second target")
    p.add_argument("--mid", type=float, default=100.0, help="Target mid price to trade around")
    p.add_argument("--log-dir", type=str, default="sim_logs", help="Directory to store logs")
    p.add_argument("--sample-agent", type=str, default="agent-001", help="Agent id to sample state")
    p.add_argument("--use-llm", action="store_true", help="Use Qwen decision; fallback to heuristic if unavailable")
    return p.parse_args()


def main():
    args = parse_args()
    run_simulation(
        agent_count=args.agents,
        initial_cash=args.initial_cash,
        initial_position=args.initial_pos,
        duration_sec=args.duration,
        target_mid=args.mid,
        orders_per_sec=args.ops,
        log_dir=args.log_dir,
        sample_agent=args.sample_agent,
        use_llm=args.use_llm,
    )


if __name__ == "__main__":
    main()

