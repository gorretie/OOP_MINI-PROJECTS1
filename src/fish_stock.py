"""Fish-stock, price, and revenue-risk models for a Lake Victoria fishery."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def fibonacci_stock_baseline(count: int = 15) -> list[int]:
    """Return the first ``count`` Fibonacci values as a deliberately naive stock."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer.")

    values = [0, 1]
    while len(values) < count:
        values.append(values[-1] + values[-2])
    return values[:count]


@dataclass(frozen=True)
class FishStockResults:
    """Weekly stock trajectory and catches; stock has one more entry than catch."""

    stock_tonnes: FloatArray
    catch_tonnes: FloatArray

    @property
    def final_stock_tonnes(self) -> float:
        return float(self.stock_tonnes[-1])


class FishStock:
    """Discrete logistic stock model with proportional weekly harvesting."""

    def __init__(
        self,
        growth_rate: float = 0.4,
        carrying_capacity_tonnes: float = 10_000.0,
        initial_stock_tonnes: float = 4_000.0,
        weeks_per_year: int = 52,
    ) -> None:
        values = (growth_rate, carrying_capacity_tonnes, initial_stock_tonnes)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Growth, carrying capacity, and stock must be finite.")
        if growth_rate <= 0 or carrying_capacity_tonnes <= 0:
            raise ValueError("Growth rate and carrying capacity must be positive.")
        if initial_stock_tonnes < 0 or initial_stock_tonnes > carrying_capacity_tonnes:
            raise ValueError("Initial stock must be between zero and carrying capacity.")
        if (
            isinstance(weeks_per_year, bool)
            or not isinstance(weeks_per_year, int)
            or weeks_per_year < 1
        ):
            raise ValueError("weeks_per_year must be a positive integer.")

        self.growth_rate = float(growth_rate)
        self.carrying_capacity_tonnes = float(carrying_capacity_tonnes)
        self.initial_stock_tonnes = float(initial_stock_tonnes)
        self.weeks_per_year = weeks_per_year

    def simulate(
        self,
        weeks: int,
        harvest_rate: float,
        closed_season_weeks: int = 0,
    ) -> FishStockResults:
        """Simulate stock and catch, closing the first weeks of each year if set."""
        if isinstance(weeks, bool) or not isinstance(weeks, int) or weeks < 1:
            raise ValueError("weeks must be a positive integer.")
        if not math.isfinite(harvest_rate) or not 0 <= harvest_rate <= 1:
            raise ValueError("harvest_rate must be finite and between zero and one.")
        if (
            isinstance(closed_season_weeks, bool)
            or not isinstance(closed_season_weeks, int)
            or not 0 <= closed_season_weeks <= self.weeks_per_year
        ):
            raise ValueError(
                "closed_season_weeks must be between zero and weeks_per_year."
            )

        stock = np.empty(weeks + 1, dtype=float)
        catch = np.empty(weeks, dtype=float)
        stock[0] = self.initial_stock_tonnes

        for week in range(weeks):
            in_closed_season = week % self.weeks_per_year < closed_season_weeks
            effective_harvest_rate = 0.0 if in_closed_season else harvest_rate
            catch[week] = effective_harvest_rate * stock[week]
            growth = self.growth_rate * stock[week] * (
                1 - stock[week] / self.carrying_capacity_tonnes
            )
            stock[week + 1] = max(
                0.0, stock[week] + growth - catch[week]
            )

        return FishStockResults(stock, catch)


class PriceModel:
    """Seeded bounded random walk for weekly fish prices in UGX per kilogram."""

    def __init__(
        self,
        initial_price_ugx_per_kg: float = 12_000.0,
        price_bounds_ugx_per_kg: tuple[float, float] = (9_000.0, 16_000.0),
        weekly_volatility_ugx_per_kg: float = 250.0,
        seed: int = 42,
    ) -> None:
        lower, upper = price_bounds_ugx_per_kg
        if (
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or lower <= 0
            or lower >= upper
        ):
            raise ValueError("Price bounds must be finite, positive, and increasing.")
        if (
            not math.isfinite(initial_price_ugx_per_kg)
            or not lower <= initial_price_ugx_per_kg <= upper
        ):
            raise ValueError("Initial price must fall within the price bounds.")
        if (
            not math.isfinite(weekly_volatility_ugx_per_kg)
            or weekly_volatility_ugx_per_kg < 0
        ):
            raise ValueError("Weekly price volatility must be finite and non-negative.")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer.")

        self.initial_price_ugx_per_kg = float(initial_price_ugx_per_kg)
        self.price_bounds_ugx_per_kg = (float(lower), float(upper))
        self.weekly_volatility_ugx_per_kg = float(weekly_volatility_ugx_per_kg)
        self.seed = seed

    def simulate_paths(self, weeks: int, n_paths: int = 1) -> FloatArray:
        """Return a reproducible ``n_paths x weeks`` array of bounded prices."""
        if isinstance(weeks, bool) or not isinstance(weeks, int) or weeks < 1:
            raise ValueError("weeks must be a positive integer.")
        if isinstance(n_paths, bool) or not isinstance(n_paths, int) or n_paths < 1:
            raise ValueError("n_paths must be a positive integer.")

        rng = np.random.default_rng(self.seed)
        lower, upper = self.price_bounds_ugx_per_kg
        prices = np.empty((n_paths, weeks), dtype=float)
        current = np.full(n_paths, self.initial_price_ugx_per_kg, dtype=float)
        for week in range(weeks):
            current += rng.normal(
                0.0, self.weekly_volatility_ugx_per_kg, size=n_paths
            )
            current = np.clip(current, lower, upper)
            prices[:, week] = current
        return prices


@dataclass(frozen=True)
class RevenueStatistics:
    """Summary of revenue values, including population variability."""

    mean: float
    median: float
    variance: float
    standard_deviation: float
    coefficient_of_variation: float


def revenue_statistics(revenues_ugx: Sequence[float]) -> RevenueStatistics:
    """Summarize annual revenue paths using population statistics."""
    try:
        values = np.asarray(revenues_ugx, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Revenue values must be numeric.") from error
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Revenue values must be a non-empty finite sequence.")

    mean_value = statistics.mean(values.tolist())
    if mean_value <= 0:
        raise ValueError("Mean revenue must be positive to calculate the CV.")
    standard_deviation = statistics.pstdev(values.tolist())
    return RevenueStatistics(
        mean=mean_value,
        median=statistics.median(values.tolist()),
        variance=statistics.pvariance(values.tolist()),
        standard_deviation=standard_deviation,
        coefficient_of_variation=standard_deviation / mean_value,
    )


class RiskAssessor:
    """Classify revenue variability using explicit, illustrative CV bands."""

    def __init__(
        self, moderate_cv_threshold: float = 0.10, high_cv_threshold: float = 0.20
    ) -> None:
        if (
            not math.isfinite(moderate_cv_threshold)
            or not math.isfinite(high_cv_threshold)
            or moderate_cv_threshold < 0
            or moderate_cv_threshold >= high_cv_threshold
        ):
            raise ValueError("CV thresholds must be finite, non-negative, and ordered.")
        self.moderate_cv_threshold = moderate_cv_threshold
        self.high_cv_threshold = high_cv_threshold

    def classify(self, coefficient_of_variation: float) -> str:
        """Return Low, Moderate, or High according to the configured CV cutoffs."""
        if not math.isfinite(coefficient_of_variation) or coefficient_of_variation < 0:
            raise ValueError("Coefficient of variation must be finite and non-negative.")
        if coefficient_of_variation < self.moderate_cv_threshold:
            return "Low"
        if coefficient_of_variation < self.high_cv_threshold:
            return "Moderate"
        return "High"


def annual_revenue_paths_ugx(
    catch_tonnes: Sequence[float], price_paths_ugx_per_kg: Sequence[Sequence[float]]
) -> FloatArray:
    """Convert weekly catches and prices to annual revenue per price path."""
    catch = np.asarray(catch_tonnes, dtype=float)
    prices = np.asarray(price_paths_ugx_per_kg, dtype=float)
    if (
        catch.ndim != 1
        or catch.size == 0
        or not np.all(np.isfinite(catch))
        or np.any(catch < 0)
    ):
        raise ValueError("Weekly catches must be a non-empty finite non-negative array.")
    if (
        prices.ndim != 2
        or prices.shape[0] == 0
        or prices.shape[1] != catch.size
        or not np.all(np.isfinite(prices))
        or np.any(prices <= 0)
    ):
        raise ValueError("Price paths must be a finite positive paths-by-weeks array.")
    return prices @ (catch * 1_000.0)
