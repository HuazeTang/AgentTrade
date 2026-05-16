"""Event-study based causal strategy.

Uses market model event studies to estimate CAR around discrete events
(e.g., index inclusion, policy changes, regulatory events).

Ranks stocks by their expected CAR and constructs long-short portfolios.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from causal.event_study import EventStudy, EventStudyResult
from strategy.base import Strategy

logger = logging.getLogger(__name__)


class EventStudyStrategy(Strategy):
    """Strategy that trades around discrete events based on expected CAR.

    The strategy:
    1. Identifies event dates from an event column (e.g., "index_added")
    2. Runs event study on historical events to estimate typical CAR
    3. At decision time, loads stocks with upcoming/recent events
    4. Goes long on events with positive expected CAR, short on negative
    """

    name = "event_study"

    def __init__(
        self,
        event_col: str,
        price_col: str = "close",
        est_window: tuple[int, int] = (-120, -10),
        car_window: tuple[int, int] = (0, 5),
        top_n: int = 20,
        long_only: bool = False,
        refit_freq: str = "monthly",
        min_events: int = 10,
    ):
        self.event_col = event_col
        self.price_col = price_col
        self.est_window = est_window
        self.car_window = car_window
        self.top_n = top_n
        self.long_only = long_only
        self.refit_freq = refit_freq
        self.min_events = min_events

        # State
        self._event_study = EventStudy(
            est_window=est_window,
            car_window=car_window,
        )
        self._last_result: EventStudyResult | None = None
        self._expected_car: dict[str, float] = {}  # symbol → expected CAR
        self._last_refit_date: pd.Timestamp | None = None
        self._history_returns: list[pd.DataFrame] = []
        self._history_events: list[pd.Series] = []

    @property
    def allow_short(self) -> bool:
        return not self.long_only

    def generate_weights(
        self,
        date: pd.Timestamp,
        universe: list[str],
        data: pd.DataFrame,
        prices: pd.Series,
        current_positions: dict[str, float],
        cash: float,
    ) -> pd.Series:
        # Collect history (for periodic refit)
        self._collect_history(date, data)

        # Refit event study if needed
        if self._should_refit(date):
            self._refit()

        # Look for active events among universe
        if self.event_col not in data.columns:
            return pd.Series(dtype=float)

        events = data[self.event_col].fillna(0)
        active_events = events[events > 0]
        active_events = active_events[active_events.index.isin(universe)]

        if active_events.empty:
            return pd.Series(dtype=float)

        # Rank by expected CAR
        car_values = pd.Series(self._expected_car).reindex(active_events.index).dropna()
        car_values = car_values.sort_values(ascending=False)

        long_n = min(self.top_n, len(car_values))
        long_syms = car_values.head(long_n).index.tolist()
        short_syms = [] if self.long_only else car_values.tail(long_n).index.tolist()

        weights = pd.Series(0.0, index=data.index)
        if long_syms:
            weights.loc[long_syms] = 1.0 / len(long_syms)
        if short_syms:
            weights.loc[short_syms] = -1.0 / len(short_syms)

        total = weights.abs().sum()
        if total > 0:
            weights = weights / total
        return weights[weights != 0]

    def _collect_history(self, date: pd.Timestamp, data: pd.DataFrame) -> None:
        """Store data for periodic refit of the event study model."""
        if self.event_col not in data.columns or self.price_col not in data.columns:
            return

        # Compute returns from price
        returns = data[self.price_col].pct_change()

        self._history_returns.append(returns.to_frame("ret"))
        self._history_events.append(data[self.event_col].copy())

        max_hist = 500
        if len(self._history_returns) > max_hist:
            self._history_returns = self._history_returns[-max_hist:]
            self._history_events = self._history_events[-max_hist:]

    def _should_refit(self, date: pd.Timestamp) -> bool:
        if self._last_refit_date is None:
            return len(self._history_events) >= 20
        if self.refit_freq == "daily":
            return True
        if self.refit_freq == "W-MON" and date.dayofweek == 0:
            return True
        if self.refit_freq == "monthly" and date.day <= 7 and date.dayofweek == 0:
            return True
        return False

    def _refit(self) -> None:
        """Refit event study on accumulated history."""
        if len(self._history_returns) < 20:
            return

        try:
            returns = pd.concat(self._history_returns)["ret"]
            events = pd.concat(self._history_events)

            # Create a simple market return proxy (equal-weighted)
            market_ret = returns.groupby("trade_date").mean() if "trade_date" in returns.index.names else returns.mean()

            result = self._event_study.run(
                returns=returns,
                market_returns=market_ret,
                events=events,
            )

            self._last_result = result

            # Store expected CAR per event type
            if result.n_events >= self.min_events:
                self._expected_car = {
                    self.event_col: result.mean_car,
                }
                logger.info(
                    "Event study refit: CAR(%d,%d)=%.6f (p=%.4f), n=%d",
                    self.car_window[0], self.car_window[1],
                    result.mean_car, result.car_p_value, result.n_events,
                )
            else:
                logger.info(
                    "Event study: insufficient events (%d < %d), skipping",
                    result.n_events, self.min_events,
                )

            self._last_refit_date = pd.Timestamp.now()

        except Exception as e:
            logger.warning("Event study refit failed: %s", e)

    @property
    def last_result(self) -> EventStudyResult | None:
        return self._last_result
