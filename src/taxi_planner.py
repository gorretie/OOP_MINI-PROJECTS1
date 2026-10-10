"""Revenue analysis, demand forecasting, and fleet planning for taxi routes."""

from __future__ import annotations

import math
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve

FloatArray = NDArray[np.float64]
DEFAULT_ALPHA_GRID = tuple(float(alpha) for alpha in np.linspace(0.05, 1.0, 20))


def _validated_series(values: Sequence[float], name: str = "values") -> FloatArray:
    try:
        series = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric values.") from error
    if (
        series.ndim != 1
        or series.size == 0
        or not np.all(np.isfinite(series))
        or np.any(series < 0)
    ):
        raise ValueError(f"{name} must be a non-empty finite non-negative sequence.")
    return series


@dataclass(frozen=True, init=False)
class Route:
    """Daily passenger observations and a one-way fare for a taxi route."""

    name: str
    passenger_counts: FloatArray
    fare_ugx: float

    def __init__(
        self, name: str, passenger_counts: Sequence[float], fare_ugx: float
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Route name cannot be empty.")
        if isinstance(fare_ugx, bool) or not math.isfinite(fare_ugx) or fare_ugx < 0:
            raise ValueError("fare_ugx must be finite and non-negative.")
        passengers = _validated_series(passenger_counts, "passenger_counts").copy()
        passengers.setflags(write=False)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "passenger_counts", passengers)
        object.__setattr__(self, "fare_ugx", float(fare_ugx))

    def daily_revenue(self) -> FloatArray:
        """Return observed daily revenue in UGX."""
        return self.passenger_counts * self.fare_ugx

    def total_revenue(self) -> float:
        """Return total observed revenue in UGX."""
        return float(np.sum(self.daily_revenue()))

    def descriptive_statistics(self) -> dict[str, float]:
        """Return the arithmetic mean, sample variance, and sample deviation."""
        if self.passenger_counts.size < 2:
            raise ValueError("At least two days are required for sample statistics.")
        values = self.passenger_counts.tolist()
        return {
            "mean": statistics.mean(values),
            "variance": statistics.variance(values),
            "standard_deviation": statistics.stdev(values),
        }


class Forecaster(ABC):
    """Base interface for one- or multi-day passenger-count forecasts."""

    name: str

    @abstractmethod
    def forecast(self, history: Sequence[float], horizon: int = 1) -> FloatArray:
        """Forecast ``horizon`` future values from a non-empty history."""

    @staticmethod
    def _validate_history(
        history: Sequence[float], horizon: int
    ) -> tuple[FloatArray, int]:
        values = _validated_series(history, "history")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer.")
        return values, horizon


class MovingAverageForecaster(Forecaster):
    """Recursive moving-average forecast using the latest ``window`` values."""

    name = "3-day moving average"

    def __init__(self, window: int = 3) -> None:
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError("window must be a positive integer.")
        self.window = window

    def forecast(self, history: Sequence[float], horizon: int = 1) -> FloatArray:
        values, horizon = self._validate_history(history, horizon)
        extended = values.tolist()
        predictions = np.empty(horizon, dtype=float)
        for index in range(horizon):
            prediction = float(np.mean(extended[-self.window :]))
            predictions[index] = prediction
            extended.append(prediction)
        return predictions


class ExponentialSmoothingForecaster(Forecaster):
    """Simple exponential smoothing with a fixed smoothing parameter alpha."""

    name = "simple exponential smoothing"

    def __init__(self, alpha: float) -> None:
        if isinstance(alpha, bool) or not math.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("alpha must be finite and in the interval (0, 1].")
        self.alpha = float(alpha)

    def forecast(self, history: Sequence[float], horizon: int = 1) -> FloatArray:
        values, horizon = self._validate_history(history, horizon)
        level = float(values[0])
        for observation in values[1:]:
            level = self.alpha * float(observation) + (1 - self.alpha) * level
        return np.full(horizon, level, dtype=float)


class LinearTrendForecaster(Forecaster):
    """Least-squares linear trend fitted to all observed days."""

    name = "linear trend"

    def forecast(self, history: Sequence[float], horizon: int = 1) -> FloatArray:
        values, horizon = self._validate_history(history, horizon)
        if values.size < 2:
            raise ValueError("Linear trend forecasting requires at least two days.")
        days = np.arange(1, values.size + 1, dtype=float)
        slope, intercept = np.polyfit(days, values, deg=1)
        future_days = np.arange(values.size + 1, values.size + horizon + 1)
        return np.maximum(0.0, intercept + slope * future_days).astype(float)


class SeasonalNaiveForecaster(Forecaster):
    """Repeat observations from the corresponding day in the previous season."""

    name = "seasonal naive"

    def __init__(self, season_length: int = 7) -> None:
        if (
            isinstance(season_length, bool)
            or not isinstance(season_length, int)
            or season_length < 1
        ):
            raise ValueError("season_length must be a positive integer.")
        self.season_length = season_length

    def forecast(self, history: Sequence[float], horizon: int = 1) -> FloatArray:
        values, horizon = self._validate_history(history, horizon)
        if values.size < self.season_length:
            raise ValueError("History must contain at least one complete season.")
        return np.asarray(
            [
                values[-self.season_length + (index % self.season_length)]
                for index in range(horizon)
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class RouteForecastResult:
    """Rolling-origin scores and day-11 demand and revenue forecasts."""

    route_name: str
    mae_by_model: Mapping[str, float]
    alpha_scores: Mapping[float, float]
    best_alpha: float
    best_model: str
    backtest_predictions: Mapping[str, FloatArray]
    forecast_day11_passengers: float
    forecast_day11_revenue_ugx: float


def _mae(actual: Sequence[float], forecast: Sequence[float]) -> float:
    return float(np.mean(np.abs(np.asarray(actual, dtype=float) - forecast)))


def rolling_origin_backtest(
    route: Route,
    start_day: int = 4,
    alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
) -> RouteForecastResult:
    """Tune smoothing alpha and compare models using expanding rolling origins.

    A ``start_day`` of four evaluates forecasts for observed days 4 through the
    final day, using only earlier observations at each origin.
    """
    values = route.passenger_counts
    if (
        isinstance(start_day, bool)
        or not isinstance(start_day, int)
        or start_day < 4
        or start_day >= values.size
    ):
        raise ValueError("start_day must be at least 4 and before the final day.")
    try:
        alphas = tuple(float(alpha) for alpha in alpha_grid)
    except (TypeError, ValueError) as error:
        raise ValueError("alpha_grid must contain numeric values.") from error
    if not alphas or any(
        not math.isfinite(alpha) or not 0 < alpha <= 1 for alpha in alphas
    ):
        raise ValueError("alpha_grid values must be finite and in (0, 1].")

    origins = range(start_day - 1, values.size)
    actual = values[start_day - 1 :]
    moving = MovingAverageForecaster()
    trend = LinearTrendForecaster()
    moving_predictions = np.asarray(
        [moving.forecast(values[:origin])[0] for origin in origins], dtype=float
    )
    trend_predictions = np.asarray(
        [trend.forecast(values[:origin])[0] for origin in origins], dtype=float
    )
    alpha_predictions = {
        alpha: np.asarray(
            [
                ExponentialSmoothingForecaster(alpha).forecast(values[:origin])[0]
                for origin in origins
            ],
            dtype=float,
        )
        for alpha in alphas
    }
    alpha_scores = {
        alpha: _mae(actual, predictions)
        for alpha, predictions in alpha_predictions.items()
    }
    best_alpha = min(alphas, key=lambda alpha: (alpha_scores[alpha], alpha))
    best_ses_predictions = alpha_predictions[best_alpha]
    scores = {
        moving.name: _mae(actual, moving_predictions),
        ExponentialSmoothingForecaster.name: _mae(actual, best_ses_predictions),
        trend.name: _mae(actual, trend_predictions),
    }
    predictions_by_model = {
        moving.name: moving_predictions,
        ExponentialSmoothingForecaster.name: best_ses_predictions,
        trend.name: trend_predictions,
    }
    best_model = min(scores, key=scores.get)
    model_by_name: dict[str, Forecaster] = {
        moving.name: moving,
        ExponentialSmoothingForecaster.name: ExponentialSmoothingForecaster(
            best_alpha
        ),
        trend.name: trend,
    }
    forecast_day11 = float(model_by_name[best_model].forecast(values)[0])
    return RouteForecastResult(
        route_name=route.name,
        mae_by_model=scores,
        alpha_scores=alpha_scores,
        best_alpha=best_alpha,
        best_model=best_model,
        backtest_predictions=predictions_by_model,
        forecast_day11_passengers=forecast_day11,
        forecast_day11_revenue_ugx=forecast_day11 * route.fare_ugx,
    )


def solve_ntinda_equilibrium(
    demand_intercept: float = 120.0,
    demand_slope: float = 0.02,
    supply_intercept: float = 10.0,
    supply_slope: float = 0.03,
) -> tuple[float, float]:
    """Solve linear demand/supply curves with scipy.linalg.solve."""
    coefficients = (demand_intercept, demand_slope, supply_intercept, supply_slope)
    if not all(math.isfinite(value) for value in coefficients):
        raise ValueError("Supply and demand parameters must be finite.")
    if demand_slope <= 0 or supply_slope <= 0:
        raise ValueError("Demand and supply slopes must be positive.")
    matrix = np.asarray([[1.0, demand_slope], [1.0, -supply_slope]])
    right_hand_side = np.asarray([demand_intercept, supply_intercept])
    equilibrium_quantity, equilibrium_fare = solve(
        matrix, right_hand_side, assume_a="gen"
    )
    return float(equilibrium_fare), float(equilibrium_quantity)


def vehicles_required(
    forecast_passengers: float,
    seats_per_vehicle: int = 14,
    trips_per_vehicle: int = 8,
    buffer_fraction: float = 0.15,
) -> int:
    """Round up daily demand with a service-capacity buffer."""
    if (
        not math.isfinite(forecast_passengers)
        or forecast_passengers < 0
        or isinstance(seats_per_vehicle, bool)
        or not isinstance(seats_per_vehicle, int)
        or seats_per_vehicle < 1
        or isinstance(trips_per_vehicle, bool)
        or not isinstance(trips_per_vehicle, int)
        or trips_per_vehicle < 1
        or not math.isfinite(buffer_fraction)
        or buffer_fraction < 0
    ):
        raise ValueError("Fleet demand, capacity, trips, and buffer must be valid.")
    daily_vehicle_capacity = seats_per_vehicle * trips_per_vehicle
    buffered_demand = forecast_passengers * (1 + buffer_fraction)
    return int(math.ceil(buffered_demand / daily_vehicle_capacity))


def simulate_weekly_demand(
    days: int = 60, seed: int = 42
) -> FloatArray:
    """Generate seeded daily demand with a Friday peak and Sunday lull."""
    if isinstance(days, bool) or not isinstance(days, int) or days < 14:
        raise ValueError("days must be an integer of at least 14.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    weekday_effects = np.asarray([0, 0, 0, 0, 28, 4, -24], dtype=float)
    rng = np.random.default_rng(seed)
    day_index = np.arange(days)
    noise = rng.normal(0, 3.0, size=days)
    return np.maximum(
        1.0, 60 + weekday_effects[day_index % 7] + 0.08 * day_index + noise
    )


def seasonal_backtest_comparison(
    demand: Sequence[float], evaluation_days: int = 21
) -> dict[str, float]:
    """Compare weekly seasonal-naive and 3-day MA over rolling origins."""
    values = _validated_series(demand, "demand")
    if (
        isinstance(evaluation_days, bool)
        or not isinstance(evaluation_days, int)
        or evaluation_days < 1
        or evaluation_days > values.size - 7
    ):
        raise ValueError("evaluation_days must fit after at least 7 training days.")
    first_target = values.size - evaluation_days
    moving = MovingAverageForecaster(window=3)
    seasonal = SeasonalNaiveForecaster(season_length=7)
    actual = values[first_target:]
    moving_predictions = np.asarray(
        [moving.forecast(values[:index])[0] for index in range(first_target, values.size)]
    )
    seasonal_predictions = np.asarray(
        [
            seasonal.forecast(values[:index])[0]
            for index in range(first_target, values.size)
        ]
    )
    return {
        moving.name: _mae(actual, moving_predictions),
        seasonal.name: _mae(actual, seasonal_predictions),
    }


def plot_route_revenue_forecasts(
    routes: Sequence[Route],
    results: Mapping[str, RouteForecastResult],
) -> tuple[plt.Figure, NDArray[np.object_]]:
    """Plot ten observed daily revenues and each route's day-11 forecast."""
    if not routes:
        raise ValueError("At least one route is required.")
    if any(route.name not in results for route in routes):
        raise ValueError("Every route must have a matching forecast result.")
    figure, axes = plt.subplots(
        len(routes),
        1,
        figsize=(10, 3.2 * len(routes)),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, route in zip(axes[:, 0], routes):
        observed_days = np.arange(1, route.passenger_counts.size + 1)
        revenue = route.daily_revenue()
        forecast = results[route.name].forecast_day11_revenue_ugx
        axis.plot(observed_days, revenue, marker="o", label="Actual revenue")
        axis.scatter(
            [route.passenger_counts.size + 1],
            [forecast],
            marker="X",
            s=90,
            label=f"Day 11 forecast ({results[route.name].best_model})",
        )
        axis.set(
            title=route.name,
            xlabel="Day",
            ylabel="Revenue (UGX)",
            xticks=np.arange(1, route.passenger_counts.size + 2),
        )
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.suptitle("Actual and forecast daily route revenue")
    return figure, axes
