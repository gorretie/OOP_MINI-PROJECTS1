"""OOP classes for district population data and forecasting."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
PopulationArray: TypeAlias = np.ndarray
YearInput: TypeAlias = int | Sequence[int] | np.ndarray


class DistrictPopulation:
    """Validated annual population observations for one district."""

    def __init__(
        self,
        years: Sequence[int] | np.ndarray,
        population: Sequence[float] | np.ndarray,
        district_name: str = "District",
    ) -> None:
        year_values = np.asarray(years)
        population_values = np.asarray(population, dtype=float)

        if year_values.ndim != 1 or population_values.ndim != 1:
            raise ValueError("Years and populations must be one-dimensional.")
        if len(year_values) != len(population_values):
            raise ValueError("Years and populations must have equal lengths.")
        if len(year_values) == 0:
            raise ValueError("At least one year and population are required.")
        if not np.issubdtype(year_values.dtype, np.number):
            raise ValueError("Years must be numeric whole numbers.")
        if not np.all(np.isfinite(year_values)) or not np.all(
            year_values == np.floor(year_values)
        ):
            raise ValueError("Years must be finite whole numbers.")
        if not np.all(np.isfinite(population_values)):
            raise ValueError("Populations must be finite numbers.")
        if np.any(population_values < 0):
            raise ValueError("Populations cannot be negative.")
        if np.any(np.diff(year_values) <= 0):
            raise ValueError("Years must be unique and in increasing order.")
        if not district_name.strip():
            raise ValueError("District name cannot be empty.")

        self.years = year_values.astype(int)
        self.population = population_values
        self.district_name = district_name.strip()

    def get_district_name(self) -> str:
        return self.district_name

    def get_population(self) -> PopulationArray:
        return self.population

    def __len__(self) -> int:
        return len(self.years)

    def __repr__(self) -> str:
        return (
            f"DistrictPopulation(name={self.district_name!r}, "
            f"years={self.years[0]}-{self.years[-1]}, observations={len(self)})"
        )


class Forecaster(ABC):
    """Abstract forecasting model fitted to annual population observations."""

    def __init__(self) -> None:
        self.years: PopulationArray | None = None
        self.population: PopulationArray | None = None

    @abstractmethod
    def fit(
        self, years: Sequence[int] | np.ndarray, population: Sequence[float] | np.ndarray
    ) -> Forecaster:
        """Fit model parameters and return this forecaster."""

    @abstractmethod
    def predict(self, horizon: YearInput) -> PopulationArray:
        """Predict the next ``horizon`` years or the explicitly given years."""

    def _validate_training_data(
        self,
        years: Sequence[int] | np.ndarray,
        population: Sequence[float] | np.ndarray,
    ) -> tuple[PopulationArray, PopulationArray]:
        year_values = np.asarray(years, dtype=float)
        population_values = np.asarray(population, dtype=float)
        if year_values.ndim != 1 or population_values.ndim != 1:
            raise ValueError("Training years and populations must be one-dimensional.")
        if len(year_values) != len(population_values) or len(year_values) < 2:
            raise ValueError("Training data must contain at least two aligned points.")
        if not np.all(np.isfinite(year_values)) or not np.all(
            np.isfinite(population_values)
        ):
            raise ValueError("Training data must contain only finite values.")
        if np.any(np.diff(year_values) <= 0):
            raise ValueError("Training years must be unique and increasing.")
        if np.any(population_values < 0):
            raise ValueError("Training populations cannot be negative.")
        return year_values, population_values

    def _years_to_predict(self, horizon: YearInput) -> PopulationArray:
        if self.years is None:
            raise RuntimeError("Call fit() before predict().")
        if isinstance(horizon, (int, np.integer)):
            if horizon < 1:
                raise ValueError("Forecast horizon must be at least one year.")
            return self.years[-1] + np.arange(1, int(horizon) + 1, dtype=float)

        requested_years = np.asarray(horizon, dtype=float)
        if requested_years.ndim != 1 or requested_years.size == 0:
            raise ValueError("Prediction years must be a non-empty one-dimensional list.")
        if not np.all(np.isfinite(requested_years)) or not np.all(
            requested_years == np.floor(requested_years)
        ):
            raise ValueError("Prediction years must be finite whole numbers.")
        return requested_years


class LinearTrendForecaster(Forecaster):
    """Least-squares linear trend fitted with ``numpy.polyfit``."""

    def fit(
        self, years: Sequence[int] | np.ndarray, population: Sequence[float] | np.ndarray
    ) -> LinearTrendForecaster:
        self.years, self.population = self._validate_training_data(years, population)
        self._year_center = float(np.mean(self.years))
        self._slope, self._intercept = np.polyfit(
            self.years - self._year_center, self.population, 1
        )
        return self

    def predict(self, horizon: YearInput) -> PopulationArray:
        requested_years = self._years_to_predict(horizon)
        predictions = self._slope * (requested_years - self._year_center) + self._intercept
        return np.maximum(predictions, 0.0)


class CAGRGrowthForecaster(Forecaster):
    """Exponential model whose annual rate is the training-period CAGR."""

    def fit(
        self, years: Sequence[int] | np.ndarray, population: Sequence[float] | np.ndarray
    ) -> CAGRGrowthForecaster:
        self.years, self.population = self._validate_training_data(years, population)
        if self.population[0] <= 0 or self.population[-1] <= 0:
            raise ValueError("CAGR requires positive first and last populations.")
        elapsed_years = self.years[-1] - self.years[0]
        self.annual_rate = (self.population[-1] / self.population[0]) ** (
            1.0 / elapsed_years
        ) - 1.0
        return self

    def predict(self, horizon: YearInput) -> PopulationArray:
        requested_years = self._years_to_predict(horizon)
        elapsed_years = requested_years - self.years[-1]
        return self.population[-1] * (1.0 + self.annual_rate) ** elapsed_years


class FibonacciRatioForecaster(Forecaster):
    """Scale the latest value by successive ``F(n+1) / F(n)`` ratios.

    The first ratio uses ``n = number of training observations``; subsequent
    years use the next ratio. Historical fitted values are back-cast from the
    final training observation using the corresponding earlier ratios.
    """

    @staticmethod
    def _fibonacci(number: int) -> int:
        previous, current = 0, 1
        for _ in range(number):
            previous, current = current, previous + current
        return previous

    def fit(
        self, years: Sequence[int] | np.ndarray, population: Sequence[float] | np.ndarray
    ) -> FibonacciRatioForecaster:
        self.years, self.population = self._validate_training_data(years, population)
        self._origin_year = self.years[-1]
        self._origin_population = self.population[-1]
        self._first_ratio_index = max(2, len(self.years))
        return self

    def _ratio(self, index: int) -> float:
        return self._fibonacci(index + 1) / self._fibonacci(index)

    def predict(self, horizon: YearInput) -> PopulationArray:
        requested_years = self._years_to_predict(horizon)
        predictions = np.empty(requested_years.shape, dtype=float)
        for position, year in enumerate(requested_years):
            year_offset = int(year - self._origin_year)
            if year_offset >= 0:
                scale = math.prod(
                    self._ratio(index)
                    for index in range(
                        self._first_ratio_index,
                        self._first_ratio_index + year_offset,
                    )
                )
            else:
                scale = 1.0 / math.prod(
                    self._ratio(index)
                    for index in range(
                        self._first_ratio_index + year_offset,
                        self._first_ratio_index,
                    )
                )
            predictions[position] = self._origin_population * scale
        return predictions

# Preserve the capitalization used by the original starter file.
Districtpopulation = DistrictPopulation