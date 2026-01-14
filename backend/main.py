import heapq
import threading
import time
from dataclasses import dataclass, field
from typing import List, Literal, Tuple

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


class OrderIn(BaseModel):
    agent: str = Field(..., description="Agent identifier")
    side: Literal["buy", "sell"]
    price: float = Field(..., gt=0)
    quantity: int = Field(..., ge=1, description="Quantity must be integer >= 1")


class AgentCreate(BaseModel):
    agent: str
    initial_cash: float = Field(0, ge=0)
    initial_position: float = Field(0, ge=0)


class Deposit(BaseModel):
    amount: float = Field(..., gt=0)


class BookLevel(BaseModel):
    price: float
    quantity: float


class Trade(BaseModel):
    buy_agent: str
    sell_agent: str
    price: float
    quantity: float
    ts: float
    tx_hash: str


@dataclass(order=True)
class _BookEntry:
    sort_price: float
    ts: float
    order: OrderIn = field(compare=False)


class OrderBook:
    """Minimal in-memory order book with price-time priority."""

    def __init__(self) -> None:
        self._buys: List[_BookEntry] = []
        self._sells: List[_BookEntry] = []
        self._ledger = {}  # agent -> {"cash": float, "position": float}
        self._trades: List[Trade] = []
        self._tx_counter = 1
        self._lock = threading.Lock()

    def _next_tx(self) -> str:
        tx = f"tx{self._tx_counter:08d}"
        self._tx_counter += 1
        return tx

    def _touch_agent(self, agent: str) -> None:
        if agent not in self._ledger:
            raise HTTPException(status_code=400, detail=f"Agent '{agent}' not registered")

    def _settle(self, buy: OrderIn, sell: OrderIn, price: float, qty: float) -> Trade:
        self._touch_agent(buy.agent)
        self._touch_agent(sell.agent)
        # Cash decreases for buyer, increases for seller; position opposite.
        self._ledger[buy.agent]["cash"] -= price * qty
        self._ledger[buy.agent]["position"] += qty
        self._ledger[sell.agent]["cash"] += price * qty
        self._ledger[sell.agent]["position"] -= qty
        trade = Trade(
            buy_agent=buy.agent,
            sell_agent=sell.agent,
            price=price,
            quantity=qty,
            ts=time.time(),
            tx_hash=self._next_tx(),
        )
        self._trades.append(trade)
        return trade

    def _push_buy(self, order: OrderIn) -> None:
        heapq.heappush(self._buys, _BookEntry(-order.price, time.time(), order))

    def _push_sell(self, order: OrderIn) -> None:
        heapq.heappush(self._sells, _BookEntry(order.price, time.time(), order))

    def _match(self, incoming: OrderIn) -> List[Trade]:
        trades: List[Trade] = []
        if incoming.side == "buy":
            while (
                self._sells
                and incoming.quantity > 0
                and incoming.price >= self._sells[0].sort_price
            ):
                best = heapq.heappop(self._sells)
                match_qty = min(incoming.quantity, best.order.quantity)
                trade_price = (incoming.price + best.order.price) / 2
                trades.append(self._settle(incoming, best.order, trade_price, match_qty))
                incoming.quantity -= match_qty
                best.order.quantity -= match_qty
                if best.order.quantity > 0:
                    self._push_sell(best.order)
            if incoming.quantity > 0:
                self._push_buy(incoming)
        else:
            while (
                self._buys
                and incoming.quantity > 0
                and incoming.price <= -self._buys[0].sort_price
            ):
                best = heapq.heappop(self._buys)
                match_qty = min(incoming.quantity, best.order.quantity)
                trade_price = (incoming.price + best.order.price) / 2
                trades.append(self._settle(best.order, incoming, trade_price, match_qty))
                incoming.quantity -= match_qty
                best.order.quantity -= match_qty
                if best.order.quantity > 0:
                    self._push_buy(best.order)
            if incoming.quantity > 0:
                self._push_sell(incoming)
        return trades

    def submit(self, order: OrderIn) -> List[Trade]:
        with self._lock:
            if order.agent not in self._ledger:
                raise HTTPException(status_code=400, detail="Agent not registered")
            # Require sufficient balance/position
            if order.side == "buy":
                required = order.price * order.quantity
                if self._ledger[order.agent]["cash"] + 1e-9 < required:
                    raise HTTPException(status_code=400, detail="Insufficient cash balance")
            else:
                if self._ledger[order.agent]["position"] + 1e-9 < order.quantity:
                    raise HTTPException(status_code=400, detail="Insufficient position to sell")
            # Copy to avoid mutating caller's object
            incoming = OrderIn(**order.model_dump())
            return self._match(incoming)

    def best_levels(self, depth: int = 5) -> Tuple[List[BookLevel], List[BookLevel]]:
        with self._lock:
            buys = sorted((-e.sort_price, e.order.quantity) for e in self._buys)[-depth:]
            sells = sorted((e.sort_price, e.order.quantity) for e in self._sells)[:depth]
            buy_lvls = [BookLevel(price=p, quantity=q) for p, q in reversed(buys)]
            sell_lvls = [BookLevel(price=p, quantity=q) for p, q in sells]
            return buy_lvls, sell_lvls

    def last_trades(self, limit: int = 20) -> List[Trade]:
        with self._lock:
            return self._trades[-limit:]

    def agent_state(self, agent: str) -> dict:
        with self._lock:
            self._touch_agent(agent)
            return self._ledger[agent]

    def all_agents(self) -> list[dict]:
        with self._lock:
            return [
                {"agent": aid, "cash": v["cash"], "position": v["position"]}
                for aid, v in self._ledger.items()
            ]

    def register_agent(self, agent: str, initial_cash: float = 0.0, initial_position: float = 0.0) -> dict:
        with self._lock:
            if agent in self._ledger:
                raise HTTPException(status_code=400, detail="Agent already exists")
            self._ledger[agent] = {"cash": float(initial_cash), "position": float(initial_position)}
            return self._ledger[agent]

    def deposit(self, agent: str, amount: float) -> dict:
        with self._lock:
            self._touch_agent(agent)
            self._ledger[agent]["cash"] += float(amount)
            return self._ledger[agent]


book = OrderBook()
app = FastAPI(title="AgentTrade Sim")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/orders")
def place_order(order: OrderIn):
    trades = book.submit(order)
    return {"trades": trades, "message": "order accepted"}


@app.get("/book")
def get_book(depth: int = 5):
    bids, asks = book.best_levels(depth=depth)
    return {"bids": bids, "asks": asks}


@app.get("/trades")
def get_trades(limit: int = 20):
    return {"trades": book.last_trades(limit=limit)}


@app.get("/agents/{agent_id}")
def get_agent(agent_id: str):
    return book.agent_state(agent_id)


@app.get("/agents")
def list_agents():
    return {"agents": book.all_agents()}


@app.post("/agents")
def register_agent(body: AgentCreate):
    state = book.register_agent(body.agent, body.initial_cash, body.initial_position)
    return {"agent": body.agent, "state": state, "message": "agent registered"}


@app.post("/agents/{agent_id}/deposit")
def deposit(agent_id: str, body: Deposit):
    state = book.deposit(agent_id, body.amount)
    return {"agent": agent_id, "state": state, "message": "deposit success"}


@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})


# Serve the static frontend; resolve absolute path to avoid cwd issues.
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

