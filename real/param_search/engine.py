"""Search engine, worker, strategies, and result persistence."""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing
import os
import sys
import tempfile
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from param_search.config import ParameterSpec, SearchConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Worker (runs in spawned subprocess)
# ═══════════════════════════════════════════════════════════════════════════════

def _suppress_output() -> None:
    """Silence all output in worker subprocess.

    Redirects stdout → devnull, stderr → devnull, suppresses Python warnings,
    and installs NullHandler on root logger BEFORE importing simulation module.
    """
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    import warnings
    warnings.filterwarnings("ignore")

    root = logging.getLogger()
    root.handlers = []
    root.addHandler(logging.NullHandler())
    root.setLevel(logging.WARNING)


def run_one_simulation(
    params: dict[str, Any],
    start: date | None = None,
    end: date | None = None,
    initial_cash: float | None = None,
) -> dict[str, Any]:
    """Run one simulation with given parameter overrides.

    Returns dict with 'params' and raw float metrics.
    Designed to run in a spawned subprocess.
    """
    _suppress_output()

    import run_agent_simulation as sim_mod

    # Apply parameter overrides
    for name, value in params.items():
        setattr(sim_mod, name, value)

    sim = sim_mod.AgentSimulation(
        start=start or sim_mod.TRADING_PERIOD[0],
        end=end or sim_mod.TRADING_PERIOD[1],
        initial_cash=initial_cash or sim_mod.INITIAL_CASH,
        mode="factor",
        fast_mode=True,
        output_dir=sim_mod.JOURNAL_DIR,
    )

    try:
        metrics = sim.run()
    except Exception as exc:
        return {"params": params, "error": str(exc), "metrics": {}}

    metrics["params"] = params
    return metrics


def _pool_worker(args: dict[str, Any]) -> dict[str, Any]:
    """Thin wrapper for multiprocessing.Pool.map — must be module-level."""
    return run_one_simulation(args)


# ═══════════════════════════════════════════════════════════════════════════════
# Result Persistence
# ═══════════════════════════════════════════════════════════════════════════════

def _make_hash(params: dict[str, Any]) -> str:
    """Deterministic hash of a parameter combination."""
    raw = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class ResultStore:
    """Atomic save/load with resume support."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._entries: dict[str, dict] = {}  # keyed by params_hash

    # ── I/O ──────────────────────────────────────────────────────────────

    def load(self) -> int:
        """Load existing results from JSON. Returns count loaded."""
        if not self.path.exists():
            return 0
        try:
            with open(self.path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return 0
        self._entries.clear()
        for item in data:
            h = item.get("params_hash") or _make_hash(item.get("params", {}))
            self._entries[h] = item
        return len(self._entries)

    def save(self) -> None:
        """Atomic write: write to temp file then rename."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        items = list(self._entries.values())
        # Sort by Sharpe descending for readability
        items.sort(key=lambda r: r.get("metrics", {}).get("sharpe_ratio", -999),
                   reverse=True)
        fd, tmp = tempfile.mkstemp(
            suffix=".json", dir=self.path.parent, prefix=".param_search_tmp_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=2, default=str, ensure_ascii=False)
            os.replace(tmp, self.path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ── CRUD ─────────────────────────────────────────────────────────────

    def has(self, params: dict) -> bool:
        return _make_hash(params) in self._entries

    def add(self, result: dict) -> None:
        params = result.get("params", {})
        h = _make_hash(params)
        entry = {
            "params_hash": h,
            "params": params,
            "metrics": {
                k: v for k, v in result.items()
                if k not in ("params", "error")
            },
            "timestamp": datetime.now().isoformat(),
        }
        if result.get("error"):
            entry["error"] = result["error"]
        self._entries[h] = entry

    def pending(self, all_combos: list[dict]) -> list[dict]:
        """Return combos not yet evaluated."""
        return [c for c in all_combos if not self.has(c)]

    def top(self, metric: str = "sharpe_ratio", n: int = 10) -> list[dict]:
        items = list(self._entries.values())
        items.sort(
            key=lambda r: r.get("metrics", {}).get(metric, -999),
            reverse=True,
        )
        return items[:n] if n else items

    def count(self) -> int:
        return len(self._entries)


# ═══════════════════════════════════════════════════════════════════════════════
# Search Strategies
# ═══════════════════════════════════════════════════════════════════════════════

class RandomSearch:
    """Random search: sample N combos from each parameter's distribution."""

    def __init__(self, config: SearchConfig):
        self.config = config
        self._total = config.n_iterations

    @property
    def total(self) -> int:
        return self._total

    def generate(self) -> Iterator[dict[str, Any]]:
        rng = np.random.default_rng(self.config.random_seed)
        for _ in range(self.config.n_iterations):
            combo = {}
            for p in self.config.parameters:
                combo[p.name] = p.sample(rng)
            yield combo


class GridSearch:
    """Full grid over discretized parameter values."""

    def __init__(self, config: SearchConfig):
        self.config = config
        self._total = 1
        for p in config.parameters:
            self._total *= len(p.grid_values())

    @property
    def total(self) -> int:
        return self._total

    def generate(self) -> Iterator[dict[str, Any]]:
        import itertools
        grids = [(p.name, p.grid_values()) for p in self.config.parameters]
        names, value_lists = zip(*grids)
        for values in itertools.product(*value_lists):
            yield dict(zip(names, values))


class SequentialSearch:
    """Tune one param at a time in priority order, keep best value."""

    def __init__(self, config: SearchConfig):
        self.config = config
        self._total = sum(len(p.grid_values()) for p in config.parameters)

    @property
    def total(self) -> int:
        return self._total

    def generate(self) -> Iterator[dict[str, Any]]:
        # NOT used with the generate-then-execute pattern; handled specially.
        # This class generates lazily with state.
        raise NotImplementedError("SequentialSearch uses run_sequential directly")


# ═══════════════════════════════════════════════════════════════════════════════
# Search Engine
# ═══════════════════════════════════════════════════════════════════════════════

def _build_strategy(config: SearchConfig):
    if config.strategy == "random":
        return RandomSearch(config)
    if config.strategy == "grid":
        return GridSearch(config)
    if config.strategy == "sequential":
        return SequentialSearch(config)
    raise ValueError(f"Unknown strategy: {config.strategy}")


def _run_sequential(config: SearchConfig, store: ResultStore) -> None:
    """Tune one parameter at a time, keeping best value found."""
    best = {p.name: p.default for p in config.parameters}

    ctx = multiprocessing.get_context("spawn")

    for param_spec in config.parameters:
        # Build candidate configs: vary this param, fix others at best
        configs = []
        for value in param_spec.grid_values():
            cfg = dict(best)
            cfg[param_spec.name] = value
            configs.append(cfg)

        pending = store.pending(configs)
        if not pending:
            continue

        print(f"\n  Tuning {param_spec.name}: {len(pending)} candidates")
        with ctx.Pool(min(config.n_workers, len(pending))) as pool:
            results = pool.map(_pool_worker, pending)

        for r in results:
            store.add(r)
        store.save()

        # Find best
        metric = config.metric
        best_cfg = max(results, key=lambda r: r.get(metric, -999))
        best[param_spec.name] = best_cfg["params"][param_spec.name]
        score = best_cfg.get(metric, "?")
        print(f"  → {param_spec.name} = {best[param_spec.name]} "
              f"({metric}={score})")


class SearchEngine:
    """Orchestrates parameter search: strategy → workers → persistence."""

    def __init__(self, config: SearchConfig):
        self.config = config
        self.store = ResultStore(config.output_path)

    def run(self) -> list[dict]:
        """Execute the search, returning top results."""
        # Load prior results for resume
        if self.config.resume:
            n_loaded = self.store.load()
            if n_loaded:
                print(f"Loaded {n_loaded} existing results from {self.config.output_path}")

        strategy = _build_strategy(self.config)

        # Sequential search has its own orchestration
        if isinstance(strategy, SequentialSearch):
            _run_sequential(self.config, self.store)
            return self.store.top(n=None)

        # Generate all combos and filter to pending
        all_combos = list(strategy.generate())
        pending = self.store.pending(all_combos)

        print(f"Strategy:    {self.config.strategy}")
        print(f"Parameters:  {len(self.config.parameters)}")
        print(f"Total:       {len(all_combos)}")
        print(f"Done:        {len(all_combos) - len(pending)}")
        print(f"Pending:     {len(pending)}")
        print(f"Workers:     {self.config.n_workers}")
        print(f"Metric:      {self.config.metric}")
        print()

        if not pending:
            print("All combinations already evaluated.")
            return self.store.top(n=None)

        # Process in batches for crash resilience
        batch_size = self.config.n_workers * 4
        ctx = multiprocessing.get_context("spawn")
        t0 = time.time()

        with ctx.Pool(self.config.n_workers) as pool:
            for i in range(0, len(pending), batch_size):
                batch = pending[i:i + batch_size]
                results = pool.map(_pool_worker, batch)
                for r in results:
                    self.store.add(r)
                self.store.save()

                done = min(i + batch_size, len(pending))
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(pending) - done) / rate if rate > 0 else 0
                print(f"  [{done}/{len(pending)}] "
                      f"{rate:.2f} runs/s, ETA {eta:.0f}s")

        elapsed = time.time() - t0
        print(f"\nDone. {len(pending)} runs in {elapsed:.0f}s "
              f"({len(pending) / elapsed:.2f} runs/s)")

        return self.store.top(n=None)
