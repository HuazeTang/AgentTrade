"""Event study for causal effect estimation around discrete events.

Estimates abnormal returns (AR) and cumulative abnormal returns (CAR)
using a market model benchmark. Supports both single-stock and portfolio
event studies.

Common use cases: index inclusion/exclusion, policy announcements,
earnings surprises, regulatory changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EventStudyResult:
    """Result of an event study.

    Attributes:
        mean_car: Mean cumulative abnormal return across all events.
        car_se: Standard error of mean CAR (cross-sectional).
        car_t_stat: t-statistic for CAR significance.
        car_p_value: p-value.
        n_events: Number of events analyzed.
        car_series: Per-event CAR values.
        ar_daily: Average abnormal return per event day (relative to event).
        ar_daily_se: Standard error of AR per event day.
        car_window: (start, end) relative to event date.
    """
    mean_car: float
    car_se: float
    car_t_stat: float
    car_p_value: float
    n_events: int = 0
    car_series: pd.Series | None = None  # per-event CAR
    ar_daily: pd.Series | None = None  # avg AR by event day
    ar_daily_se: pd.Series | None = None
    car_window: tuple[int, int] = (0, 5)

    @property
    def significant(self, alpha: float = 0.05) -> bool:
        return self.car_p_value < alpha

    def summary(self) -> str:
        w = self.car_window
        lines = [
            f"CAR({w[0]},{w[1]}): {self.mean_car:.6f} (SE={self.car_se:.6f})",
            f"t-stat: {self.car_t_stat:.3f}, p-value: {self.car_p_value:.4f}",
            f"Significant at 5%: {self.significant}",
            f"N events: {self.n_events}",
        ]
        if self.ar_daily is not None:
            lines.append("Daily AR:")
            for d, ar in self.ar_daily.items():
                se = self.ar_daily_se.get(d, np.nan)
                lines.append(f"  Day {d:>+3d}: {ar:.6f} (±{se:.6f})")
        return "\n".join(lines)


class EventStudy:
    """Market-model event study.

    Estimation window: [est_start, event - gap) used to fit market beta.
    Event window: [event + car_start, event + car_end] for AR/CAR computation.
    A gap between estimation and event window avoids contamination.

    Usage:
        es = EventStudy(est_window=(-120, -10), car_window=(0, 5))
        result = es.run(
            returns=price_returns,
            market_returns=market_returns,
            events=event_series,  # 1 where event occurs
        )
    """

    def __init__(
        self,
        est_window: tuple[int, int] = (-120, -10),
        car_window: tuple[int, int] = (0, 5),
        min_est_days: int = 30,
    ):
        self.est_window = est_window
        self.car_window = car_window
        self.min_est_days = min_est_days

    def run(
        self,
        returns: pd.Series,
        market_returns: pd.Series,
        events: pd.Series,
    ) -> EventStudyResult:
        """Run event study on panel data.

        Args:
            returns: Multi-indexed (trade_date, symbol) daily returns.
            market_returns: Series indexed by trade_date with market returns.
            events: Multi-indexed (trade_date, symbol) Series, 1 = event day.

        Returns:
            EventStudyResult with CAR statistics.
        """
        if events.sum() == 0:
            return EventStudyResult(
                mean_car=0.0, car_se=0.0, car_t_stat=0.0, car_p_value=1.0,
                car_window=self.car_window,
            )

        # Find event dates per symbol
        event_dates = events[events > 0]
        event_list = list(event_dates.index)

        # Unstack returns for time-series operations
        ret_wide = returns.unstack()  # date x symbol
        mkt = market_returns.reindex(ret_wide.index)

        all_cars = []
        ar_by_day: dict[int, list[float]] = {}

        for date, symbol in event_list:
            if symbol not in ret_wide.columns:
                continue

            sym_ret = ret_wide[symbol]

            # Find position of event date
            try:
                event_pos = list(sym_ret.index).index(date)
            except (ValueError, IndexError):
                continue

            # Estimation window
            est_start = max(0, event_pos + self.est_window[0])
            est_end = max(0, event_pos + self.est_window[1])

            if est_end - est_start < self.min_est_days:
                continue

            est_ret = sym_ret.iloc[est_start:est_end]
            est_mkt = mkt.iloc[est_start:est_end]

            # Market model regression: R_i = α + β * R_m + ε
            mask = est_ret.notna() & est_mkt.notna()
            if mask.sum() < self.min_est_days:
                continue

            beta, alpha = _ols_beta(est_mkt[mask].values, est_ret[mask].values)

            # Event window: compute AR and CAR
            ev_start = event_pos + self.car_window[0]
            ev_end = event_pos + self.car_window[1] + 1
            if ev_start < 0 or ev_end > len(sym_ret):
                continue

            ev_ret = sym_ret.iloc[ev_start:ev_end]
            ev_mkt = mkt.iloc[ev_start:ev_end]

            # Abnormal returns
            expected = alpha + beta * ev_mkt.values
            ar = ev_ret.values - expected
            car = np.nansum(ar)

            all_cars.append(car)

            # Collect AR by event day
            for j, ar_val in enumerate(ar):
                day_offset = self.car_window[0] + j
                ar_by_day.setdefault(day_offset, []).append(ar_val)

        n_events = len(all_cars)
        if n_events < 3:
            return EventStudyResult(
                mean_car=np.mean(all_cars) if all_cars else 0.0,
                car_se=np.inf,
                car_t_stat=0.0,
                car_p_value=1.0,
                n_events=n_events,
                car_window=self.car_window,
            )

        cars = np.array(all_cars)
        mean_car = float(np.mean(cars))
        car_se = float(np.std(cars, ddof=1) / np.sqrt(n_events))
        t_stat = mean_car / car_se if car_se > 0 else 0.0
        p_value = 2 * (1 - _normal_cdf(abs(t_stat)))

        # Daily AR averages
        ar_daily_dict = {}
        ar_se_dict = {}
        for d, vals in sorted(ar_by_day.items()):
            ar_daily_dict[d] = float(np.mean(vals))
            ar_se_dict[d] = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))

        return EventStudyResult(
            mean_car=mean_car,
            car_se=car_se,
            car_t_stat=t_stat,
            car_p_value=p_value,
            n_events=n_events,
            car_series=pd.Series(all_cars, name="CAR"),
            ar_daily=pd.Series(ar_daily_dict, name="AR"),
            ar_daily_se=pd.Series(ar_se_dict, name="AR_SE"),
            car_window=self.car_window,
        )

    def run_single(
        self,
        symbol_returns: pd.Series,
        market_returns: pd.Series,
        event_date: pd.Timestamp,
    ) -> dict | None:
        """Run event study for a single stock. Returns dict with CAR and ARs."""
        try:
            event_pos = list(symbol_returns.index).index(event_date)
        except (ValueError, IndexError):
            return None

        est_start = max(0, event_pos + self.est_window[0])
        est_end = max(0, event_pos + self.est_window[1])

        if est_end - est_start < self.min_est_days:
            return None

        est_ret = symbol_returns.iloc[est_start:est_end]
        est_mkt = market_returns.iloc[est_start:est_end]

        mask = est_ret.notna() & est_mkt.notna()
        if mask.sum() < self.min_est_days:
            return None

        beta, alpha = _ols_beta(est_mkt[mask].values, est_ret[mask].values)

        ev_start = event_pos + self.car_window[0]
        ev_end = event_pos + self.car_window[1] + 1

        if ev_start < 0 or ev_end > len(symbol_returns):
            return None

        ev_ret = symbol_returns.iloc[ev_start:ev_end]
        ev_mkt = market_returns.iloc[ev_start:ev_end]

        expected = alpha + beta * ev_mkt.values
        ar = (ev_ret.values - expected).tolist()
        car = float(np.nansum(ar))

        return {
            "alpha": float(alpha),
            "beta": float(beta),
            "ar": ar,
            "car": car,
            "event_date": event_date,
        }


def _ols_beta(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Simple OLS: y = α + β*x. Returns (beta, alpha)."""
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    beta = np.sum((x - x_mean) * (y - y_mean)) / max(np.sum((x - x_mean) ** 2), 1e-10)
    alpha = y_mean - beta * x_mean
    return float(beta), float(alpha)


def _normal_cdf(x: float) -> float:
    return float(0.5 * (1 + np.math.erf(x / np.sqrt(2))))
