"""Dispatch planning for the Kasese health-centre micro-grid."""

from __future__ import annotations

import csv
import timeit
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve
from scipy.optimize import nnls

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DispatchResults:
    """Daily dispatch quantities and cost after non-negative correction."""

    solar: FloatArray
    battery: FloatArray
    infeasible: NDArray[np.bool_]
    daily_cost: FloatArray

    @property
    def monthly_cost(self) -> float:
        """Total cost for the days represented by this result."""
        return float(np.sum(self.daily_cost))


class MicroGrid:
    """Solve the daily solar/battery load equations and report dispatch."""

    SOLAR_PRICE = 150.0
    BATTERY_PRICE = 450.0

    def __init__(
        self, coefficient_matrix: Sequence[Sequence[float]] | None = None
    ) -> None:
        matrix = (
            [[3.0, 2.0], [4.0, 1.0]]
            if coefficient_matrix is None
            else coefficient_matrix
        )
        self.coefficient_matrix = np.asarray(matrix, dtype=float)
        if (
            self.coefficient_matrix.shape != (2, 2)
            or not np.all(np.isfinite(self.coefficient_matrix))
        ):
            raise ValueError("The micro-grid coefficient matrix must be finite and 2x2.")
        self.determinant = float(np.linalg.det(self.coefficient_matrix))
        self.condition_number = float(np.linalg.cond(self.coefficient_matrix))
        self.matrix_rank = int(np.linalg.matrix_rank(self.coefficient_matrix))

    @property
    def is_well_posed(self) -> bool:
        """Whether the coefficient matrix is invertible and has finite conditioning."""
        return (
            self.matrix_rank == self.coefficient_matrix.shape[0]
            and np.isfinite(self.condition_number)
        )

    def _ensure_well_posed(self) -> None:
        if not self.is_well_posed:
            raise np.linalg.LinAlgError(
                "The demand system is singular or not numerically well-posed."
            )

    @staticmethod
    def _validate_demands(demands: Sequence[float]) -> FloatArray:
        try:
            values = np.asarray(demands, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("Demand values must be numeric.") from error
        if values.ndim != 1 or values.size == 0:
            raise ValueError("Demand values must be a non-empty one-dimensional array.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Demand values must be finite.")
        if np.any(values < 0):
            raise ValueError("Demand values cannot be negative.")
        return values

    def solve_day(self, d1: float, d2: float) -> FloatArray:
        """Return the raw [solar, battery] solution for one day's demands."""
        demands = self._validate_demands([d1, d2])
        if demands.size != 2:
            raise ValueError("A micro-grid day requires d1 and d2.")
        self._ensure_well_posed()
        return np.asarray(
            solve(self.coefficient_matrix, demands, assume_a="gen"), dtype=float
        )

    def _validate_demand_table(self, demands: Sequence[Sequence[float]]) -> FloatArray:
        try:
            values = np.asarray(demands, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("Demand rows must contain numeric values.") from error
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != 2:
            raise ValueError("Demands must be a non-empty table with two columns.")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Demands must be finite and non-negative.")
        return values

    def solve_days_loop(self, demands: Sequence[Sequence[float]]) -> FloatArray:
        """Solve each day separately, returning source quantities as 2xN."""
        table = self._validate_demand_table(demands)
        self._ensure_well_posed()
        return np.column_stack(
            [solve(self.coefficient_matrix, row, assume_a="gen") for row in table]
        )

    def solve_days_vectorized(self, demands: Sequence[Sequence[float]]) -> FloatArray:
        """Solve all days in one call using a 2xN right-hand side."""
        table = self._validate_demand_table(demands)
        self._ensure_well_posed()
        return np.asarray(
            solve(self.coefficient_matrix, table.T, assume_a="gen"), dtype=float
        )

    def dispatch_days(
        self, demands: Sequence[Sequence[float]], method: str = "vectorized"
    ) -> DispatchResults:
        """Solve a demand table, correcting infeasible days with NNLS.

        The infeasibility flag records days whose exact linear solution requests
        a negative source. Those days are re-dispatched to the closest
        non-negative solution in the least-squares sense.
        """
        table = self._validate_demand_table(demands)
        if method == "loop":
            raw = self.solve_days_loop(table)
        elif method == "vectorized":
            raw = self.solve_days_vectorized(table)
        else:
            raise ValueError("method must be 'loop' or 'vectorized'.")

        infeasible = np.any(raw < 0, axis=0)
        corrected = raw.copy()
        for day_index in np.flatnonzero(infeasible):
            corrected[:, day_index] = nnls(
                self.coefficient_matrix, table[day_index]
            )[0]

        solar = corrected[0]
        battery = corrected[1]
        daily_cost = self.SOLAR_PRICE * solar + self.BATTERY_PRICE * battery
        return DispatchResults(solar, battery, infeasible, daily_cost)

    def benchmark(
        self, demands: Sequence[Sequence[float]], number: int = 100
    ) -> dict[str, float]:
        """Time loop and vectorized solving with ``timeit`` (seconds per run)."""
        table = self._validate_demand_table(demands)
        if isinstance(number, bool) or not isinstance(number, int) or number < 1:
            raise ValueError("number must be a positive integer.")
        loop_seconds = timeit.timeit(
            lambda: self.solve_days_loop(table), number=number
        ) / number
        vectorized_seconds = timeit.timeit(
            lambda: self.solve_days_vectorized(table), number=number
        ) / number
        return {
            "loop_seconds": loop_seconds,
            "vectorized_seconds": vectorized_seconds,
        }

    def statistics(self, results: DispatchResults) -> dict[str, dict[str, float]]:
        """Return population mean, variance, and standard deviation by source."""
        return {
            name: {
                "mean": float(np.mean(values)),
                "variance": float(np.var(values)),
                "standard_deviation": float(np.std(values)),
            }
            for name, values in (
                ("solar", results.solar),
                ("battery", results.battery),
            )
        }

    @staticmethod
    def more_volatile_source(
        statistics: dict[str, dict[str, float]]
    ) -> str:
        """Identify the source with the larger standard deviation."""
        return max(
            ("solar", "battery"),
            key=lambda source: statistics[source]["standard_deviation"],
        )

    def sensitivity_analysis(
        self,
        d1: float,
        d2: float,
        draws: int = 1000,
        relative_change: float = 0.05,
        seed: int = 42,
    ) -> dict[str, dict[str, float]]:
        """Measure source changes under independent relative demand perturbations."""
        baseline_demands = self._validate_demands([d1, d2])
        if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
            raise ValueError("draws must be a positive integer.")
        if (
            not np.isfinite(relative_change)
            or relative_change < 0
            or relative_change >= 1
        ):
            raise ValueError("relative_change must be in the range [0, 1).")

        self._ensure_well_posed()
        baseline = self.solve_day(*baseline_demands)
        rng = np.random.default_rng(seed)
        perturbations = rng.uniform(
            -relative_change, relative_change, size=(draws, 2)
        )
        perturbed_demands = baseline_demands * (1.0 + perturbations)
        perturbed_solutions = solve(
            self.coefficient_matrix, perturbed_demands.T, assume_a="gen"
        )
        absolute_change = np.abs(perturbed_solutions - baseline[:, None])
        result: dict[str, dict[str, float]] = {}
        for index, source in enumerate(("solar", "battery")):
            baseline_value = baseline[index]
            mean_change = float(np.mean(absolute_change[index]))
            result[source] = {
                "mean_absolute_change": mean_change,
                "maximum_absolute_change": float(np.max(absolute_change[index])),
                "mean_absolute_change_percent": (
                    100.0 * mean_change / abs(baseline_value)
                    if baseline_value != 0
                    else float("nan")
                ),
            }
        return result

    @staticmethod
    def plot_dispatch(
        results: DispatchResults, output_path: str | Path | None = None
    ) -> None:
        """Plot stacked daily source use with cost on a second y-axis."""
        import matplotlib.pyplot as plt

        days = np.arange(1, len(results.solar) + 1)
        figure, usage_axis = plt.subplots(figsize=(12, 6))
        usage_axis.bar(days, results.solar, label="Solar (kWh)", color="gold")
        usage_axis.bar(
            days,
            results.battery,
            bottom=results.solar,
            label="Battery (kWh)",
            color="steelblue",
        )
        usage_axis.set_xlabel("Day")
        usage_axis.set_ylabel("Energy dispatched (kWh)")
        usage_axis.set_title("Kasese Health Centre: Daily Micro-grid Dispatch")
        usage_axis.set_xticks(days)

        cost_axis = usage_axis.twinx()
        cost_axis.plot(
            days,
            results.daily_cost,
            color="darkgreen",
            marker="o",
            label="Daily cost (UGX)",
        )
        cost_axis.set_ylabel("Daily energy cost (UGX)")
        handles, labels = usage_axis.get_legend_handles_labels()
        cost_handles, cost_labels = cost_axis.get_legend_handles_labels()
        usage_axis.legend(handles + cost_handles, labels + cost_labels, loc="upper left")
        figure.tight_layout()
        if output_path is not None:
            figure.savefig(output_path, dpi=150)
        plt.show()


class HybridMicroGrid(MicroGrid):
    """Three-source extension with a user-selected total-energy equation.

    The third equation is x + y + z = d3, the total generated energy.
    """

    DIESEL_PRICE = 900.0

    def __init__(
        self,
        coefficient_matrix: Sequence[Sequence[float]] | None = None,
    ) -> None:
        matrix = (
            [[3.0, 2.0, 1.0], [4.0, 1.0, 2.0], [1.0, 1.0, 1.0]]
            if coefficient_matrix is None
            else coefficient_matrix
        )
        self.coefficient_matrix = np.asarray(matrix, dtype=float)
        if (
            self.coefficient_matrix.shape != (3, 3)
            or not np.all(np.isfinite(self.coefficient_matrix))
        ):
            raise ValueError("The hybrid coefficient matrix must be finite and 3x3.")
        self.determinant = float(np.linalg.det(self.coefficient_matrix))
        self.condition_number = float(np.linalg.cond(self.coefficient_matrix))
        self.matrix_rank = int(np.linalg.matrix_rank(self.coefficient_matrix))

    def solve_day(self, d1: float, d2: float, d3: float) -> FloatArray:
        """Return raw [solar, battery, diesel] quantities for one day."""
        demands = self._validate_demands([d1, d2, d3])
        self._ensure_well_posed()
        return np.asarray(
            solve(self.coefficient_matrix, demands, assume_a="gen"), dtype=float
        )


def read_interactive_demands(
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], object] = print,
) -> tuple[float, float]:
    """Prompt repeatedly until both daily demands are finite, numeric, and >= 0."""
    reader = input if input_fn is None else input_fn
    values: list[float] = []
    for label in ("D1 daytime load (kWh)", "D2 critical-equipment load (kWh)"):
        while True:
            raw = reader(f"Enter {label}: ")
            try:
                value = float(raw)
            except (TypeError, ValueError):
                output_fn("Please enter a numeric value.")
                continue
            if not np.isfinite(value):
                output_fn("Please enter a finite value.")
                continue
            if value < 0:
                output_fn("Demand cannot be negative.")
                continue
            values.append(value)
            break
    return values[0], values[1]


def load_demand_csv(path: str | Path) -> FloatArray:
    """Load and validate ``day,D1_kWh,D2_kWh`` rows from a CSV file."""
    with Path(path).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_fields = {"day", "D1_kWh", "D2_kWh"}
        if reader.fieldnames is None or not required_fields.issubset(reader.fieldnames):
            raise ValueError("CSV must contain day, D1_kWh, and D2_kWh columns.")
        rows: list[list[float]] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                rows.append([float(row["D1_kWh"]), float(row["D2_kWh"])])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"CSV row {row_number} contains a non-numeric demand."
                ) from error
    if not rows:
        raise ValueError("CSV must contain at least one demand row.")
    return MicroGrid()._validate_demand_table(rows)


def generate_demo_csv(
    path: str | Path = "data/microgrid_demand.csv",
    days: int = 30,
    seed: int = 42,
) -> Path:
    """Write seeded synthetic demands with weekly load patterns and noise."""
    if isinstance(days, bool) or not isinstance(days, int) or days < 1:
        raise ValueError("days must be a positive integer.")
    rng = np.random.default_rng(seed)
    weekday = np.arange(days) % 7
    solar_pattern = np.array([0.0, 2.0, 4.0, 6.0, 7.0, 3.0, 1.0])
    battery_pattern = np.array([2.0, 1.0, 0.0, -1.0, -2.0, 1.0, 2.0])
    solar = 26.0 + solar_pattern[weekday] + rng.normal(0.0, 1.5, days)
    battery = 11.0 + battery_pattern[weekday] + rng.normal(0.0, 0.8, days)
    solar = np.maximum(solar, 1.0)
    battery = np.maximum(battery, 1.0)
    d1 = np.round(3.0 * solar + 2.0 * battery, 3)
    d2 = np.round(4.0 * solar + battery, 3)

    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["day", "D1_kWh", "D2_kWh"])
        writer.writerows(
            (day, demand_1, demand_2)
            for day, (demand_1, demand_2) in enumerate(zip(d1, d2), start=1)
        )
    return csv_path
