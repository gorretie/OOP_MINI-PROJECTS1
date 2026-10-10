"""Rainfall analysis and crop-suitability screening for Uganda."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from numpy.typing import NDArray
from scipy.signal import find_peaks

FloatArray = NDArray[np.float64]
MONTHS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


@dataclass(frozen=True, init=False)
class Region:
    """A region's twelve-month rainfall climatology in millimetres."""

    name: str
    monthly_rainfall_mm: FloatArray

    def __init__(self, name: str, monthly_rainfall_mm: Sequence[float]) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Region name cannot be empty.")
        try:
            rainfall = np.asarray(monthly_rainfall_mm, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("Rainfall values must be numeric.") from error
        if rainfall.shape != (12,) or not np.all(np.isfinite(rainfall)):
            raise ValueError("A region requires twelve finite monthly rainfall values.")
        if np.any(rainfall < 0):
            raise ValueError("Rainfall values cannot be negative.")
        rainfall = rainfall.copy()
        rainfall.setflags(write=False)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "monthly_rainfall_mm", rainfall)

    def annual_total(self) -> float:
        """Return the total of the twelve climatological monthly values."""
        return float(np.sum(self.monthly_rainfall_mm))

    def mean(self) -> float:
        """Return mean monthly rainfall."""
        return float(np.mean(self.monthly_rainfall_mm))

    def wettest_month(self) -> str:
        """Return the name of the wettest month."""
        return MONTHS[int(np.argmax(self.monthly_rainfall_mm))]

    def driest_month(self) -> str:
        """Return the name of the driest month."""
        return MONTHS[int(np.argmin(self.monthly_rainfall_mm))]

    def coefficient_of_variation(self) -> float:
        """Return population standard deviation divided by the monthly mean."""
        average = self.mean()
        if average == 0:
            raise ValueError("Coefficient of variation is undefined for zero rainfall.")
        return float(np.std(self.monthly_rainfall_mm) / average)


@dataclass(frozen=True)
class CropRule:
    """A transparent monthly rainfall band used for suitability screening."""

    crop: str
    minimum_mm: float
    maximum_mm: float

    def __post_init__(self) -> None:
        if not self.crop.strip():
            raise ValueError("Crop name cannot be empty.")
        if (
            not math.isfinite(self.minimum_mm)
            or not math.isfinite(self.maximum_mm)
            or self.minimum_mm < 0
            or self.minimum_mm >= self.maximum_mm
        ):
            raise ValueError("Rainfall thresholds must be finite and increasing.")

    def classify(self, rainfall_mm: float) -> str:
        """Classify rainfall for one month as good, drought-risk, or excess."""
        if not math.isfinite(rainfall_mm) or rainfall_mm < 0:
            raise ValueError("Monthly rainfall must be finite and non-negative.")
        if rainfall_mm < self.minimum_mm:
            return "Drought risk"
        if rainfall_mm > self.maximum_mm:
            return "Waterlogging risk"
        return f"Good for {self.crop}"


DEFAULT_CROP_RULES = (
    CropRule("maize", 80.0, 200.0),
    CropRule("beans", 50.0, 150.0),
    CropRule("coffee", 60.0, 180.0),
)


@dataclass(frozen=True)
class SimilarityMatrices:
    """Pairwise region similarity and distance matrices in matching name order."""

    region_names: tuple[str, ...]
    cosine: FloatArray
    pearson: FloatArray
    euclidean: FloatArray


@dataclass(frozen=True)
class RainySeasonSummary:
    """Detected prominent rainfall peaks and broad season-count category."""

    peak_months: tuple[str, ...]
    classification: str


def cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float:
    """Return vector cosine similarity (dot product divided by both norms)."""
    left, right = _vector_pair(first, second)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")
    return float(np.dot(left, right) / denominator)


def _vector_pair(
    first: Sequence[float], second: Sequence[float]
) -> tuple[FloatArray, FloatArray]:
    try:
        left = np.asarray(first, dtype=float)
        right = np.asarray(second, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Vectors must contain numeric values.") from error
    if (
        left.ndim != 1
        or right.ndim != 1
        or left.size == 0
        or left.shape != right.shape
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
    ):
        raise ValueError("Vectors must be non-empty, finite, and have matching shapes.")
    return left, right


def similarity_matrices(regions: Sequence[Region]) -> SimilarityMatrices:
    """Calculate cosine, Pearson, and Euclidean matrices for monthly patterns."""
    if not regions:
        raise ValueError("At least one region is required.")
    names = tuple(region.name for region in regions)
    if len(set(names)) != len(names):
        raise ValueError("Region names must be unique.")

    count = len(regions)
    cosine = np.empty((count, count), dtype=float)
    pearson = np.empty((count, count), dtype=float)
    euclidean = np.empty((count, count), dtype=float)
    for row, first in enumerate(regions):
        for column, second in enumerate(regions):
            left = first.monthly_rainfall_mm
            right = second.monthly_rainfall_mm
            cosine[row, column] = cosine_similarity(left, right)
            if np.std(left) == 0 or np.std(right) == 0:
                pearson[row, column] = np.nan
            else:
                pearson[row, column] = float(np.corrcoef(left, right)[0, 1])
            euclidean[row, column] = float(np.linalg.norm(left - right))
    return SimilarityMatrices(names, cosine, pearson, euclidean)


def load_power_rainfall_csv(
    path: str | Path, minimum_years: int = 10
) -> dict[str, dict[int, FloatArray]]:
    """Load validated region/year rainfall arrays from the NASA POWER CSV."""
    if (
        isinstance(minimum_years, bool)
        or not isinstance(minimum_years, int)
        or minimum_years < 1
    ):
        raise ValueError("minimum_years must be a positive integer.")

    records: dict[str, dict[int, dict[int, float]]] = {}
    required_columns = {"region", "year", "month", "rainfall_mm"}
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None or not required_columns.issubset(
                reader.fieldnames
            ):
                raise ValueError(
                    f"Rainfall CSV must contain columns: {', '.join(sorted(required_columns))}."
                )
            for row_number, row in enumerate(reader, start=2):
                try:
                    name = row["region"].strip()
                    year = int(row["year"])
                    month = int(row["month"])
                    rainfall = float(row["rainfall_mm"])
                except (AttributeError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid rainfall record on CSV row {row_number}."
                    ) from error
                if not name or not 1 <= month <= 12:
                    raise ValueError(f"Invalid region or month on CSV row {row_number}.")
                if not math.isfinite(rainfall) or rainfall < 0:
                    raise ValueError(
                        f"Rainfall must be finite and non-negative on CSV row {row_number}."
                    )
                year_data = records.setdefault(name, {}).setdefault(year, {})
                if month in year_data:
                    raise ValueError(
                        f"Duplicate rainfall record for {name}, {year}-{month:02d}."
                    )
                year_data[month] = rainfall
    except OSError as error:
        raise OSError(f"Could not read rainfall CSV at {path}.") from error

    if not records:
        raise ValueError("Rainfall CSV contains no observations.")
    result: dict[str, dict[int, FloatArray]] = {}
    common_years: set[int] | None = None
    for name, years in records.items():
        if len(years) < minimum_years:
            raise ValueError(
                f"{name} has {len(years)} years; at least {minimum_years} are required."
            )
        region_years = set(years)
        if common_years is None:
            common_years = region_years
        elif region_years != common_years:
            raise ValueError("All regions must cover the same years.")
        result[name] = {}
        for year, months in years.items():
            if set(months) != set(range(1, 13)):
                raise ValueError(f"{name} {year} must contain exactly months 1 to 12.")
            result[name][year] = np.asarray(
                [months[month] for month in range(1, 13)], dtype=float
            )
    return result


def regions_from_yearly_rainfall(
    observations: Mapping[str, Mapping[int, Sequence[float]]]
) -> list[Region]:
    """Build climatological regions by averaging monthly totals across years."""
    if not observations:
        raise ValueError("At least one region's rainfall observations are required.")
    regions: list[Region] = []
    for name, yearly_values in observations.items():
        if not yearly_values:
            raise ValueError(f"{name} has no annual rainfall observations.")
        arrays: list[FloatArray] = []
        for year, values in yearly_values.items():
            try:
                annual = np.asarray(values, dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} {year} rainfall must be numeric.") from error
            if annual.shape != (12,) or not np.all(np.isfinite(annual)):
                raise ValueError(f"{name} {year} must have twelve finite monthly totals.")
            if np.any(annual < 0):
                raise ValueError(f"{name} {year} rainfall cannot be negative.")
            arrays.append(annual)
        regions.append(Region(name, np.mean(np.stack(arrays), axis=0)))
    return regions


def classify_crop_suitability(
    regions: Sequence[Region], crop_rules: Sequence[CropRule]
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Classify each region's twelve months against every crop rule."""
    if not regions or not crop_rules:
        raise ValueError("At least one region and one crop rule are required.")
    if len({rule.crop for rule in crop_rules}) != len(crop_rules):
        raise ValueError("Crop names must be unique.")
    return {
        region.name: {
            rule.crop: tuple(
                rule.classify(float(amount)) for amount in region.monthly_rainfall_mm
            )
            for rule in crop_rules
        }
        for region in regions
    }


def detect_rainy_seasons(
    region: Region,
    prominence_mm: float = 25.0,
    dry_month_threshold_mm: float = 80.0,
) -> RainySeasonSummary:
    """Find prominent circular peaks and count peaks separated by a dry month.

    Peaks within one extended wet season are not counted as separate seasons
    unless the intervening monthly climatology drops below the dry-month
    threshold. The thresholds are explicit heuristics, not official zone limits.
    """
    if not math.isfinite(prominence_mm) or prominence_mm < 0:
        raise ValueError("prominence_mm must be finite and non-negative.")
    if not math.isfinite(dry_month_threshold_mm) or dry_month_threshold_mm < 0:
        raise ValueError("dry_month_threshold_mm must be finite and non-negative.")

    rainfall = np.asarray(region.monthly_rainfall_mm)
    tiled = np.tile(rainfall, 3)
    peak_indices, properties = find_peaks(
        tiled, prominence=prominence_mm, distance=2
    )
    middle_peaks = peak_indices[(peak_indices >= 12) & (peak_indices < 24)]
    if middle_peaks.size == 0:
        return RainySeasonSummary((), "unimodal")

    months = sorted(int(index % 12) for index in middle_peaks)
    if len(months) > 1:
        dry_gaps: list[bool] = []
        for position, month_index in enumerate(months):
            next_index = months[(position + 1) % len(months)]
            if position == len(months) - 1:
                next_index += 12
            intervening = [
                rainfall[index % 12]
                for index in range(month_index + 1, next_index)
            ]
            dry_gaps.append(
                bool(intervening and min(intervening) < dry_month_threshold_mm)
            )
        middle_prominences = properties["prominences"][
            (peak_indices >= 12) & (peak_indices < 24)
        ]
        prominence_by_month = dict(zip(months, middle_prominences))
        if not all(dry_gaps):
            if any(dry_gaps):
                groups: list[list[int]] = [[months[0]]]
                for position in range(len(months) - 1):
                    if dry_gaps[position]:
                        groups.append([])
                    groups[-1].append(months[position + 1])
                if not dry_gaps[-1]:
                    groups[0] = groups[-1] + groups[0]
                    groups.pop()
                months = [
                    max(group, key=lambda month: prominence_by_month[month])
                    for group in groups
                ]
            else:
                months = [max(months, key=lambda month: prominence_by_month[month])]
        months.sort()

    peak_months = tuple(MONTHS[index] for index in months)
    category = "bimodal" if len(peak_months) >= 2 else "unimodal"
    return RainySeasonSummary(peak_months, category)


def farmer_advisory(
    region: Region, crop_rules: Sequence[CropRule], crop: str | None = None
) -> str:
    """Return a concise crop and planting-window advisory for one region."""
    suitability = classify_crop_suitability([region], crop_rules)[region.name]
    if crop is None:
        best_rule = max(
            crop_rules,
            key=lambda rule: suitability[rule.crop].count(f"Good for {rule.crop}"),
        )
    else:
        best_rule = next(
            (rule for rule in crop_rules if rule.crop.casefold() == crop.casefold()),
            None,
        )
        if best_rule is None:
            raise ValueError(f"No crop rule is defined for {crop!r}.")
    good_months = [
        index
        for index, label in enumerate(suitability[best_rule.crop])
        if label == f"Good for {best_rule.crop}"
    ]
    if not good_months:
        return (
            f"{region.name}: none of the screened crops has a month within its "
            "rainfall band. Use local extension advice, soil-moisture checks, "
            "and seasonal forecasts before choosing a planting date."
        )
    starts = [
        index
        for index in good_months
        if region.monthly_rainfall_mm[(index - 1) % 12] < best_rule.minimum_mm
    ]
    planting_months = starts or [good_months[0]]
    plant_window = ", ".join(MONTHS[index] for index in planting_months)
    good_window = ", ".join(MONTHS[index] for index in good_months)
    note = (
        f"{region.name}: crop screening suggests {best_rule.crop}. Consider planting "
        f"at the start of a suitable spell (around {plant_window}); "
        f"the climatological months "
        f"within the rainfall band are {good_window}. These are 2015-2024 "
        "NASA POWER reanalysis estimates, not a field forecast. Confirm onset "
        "with local rain, soil moisture, seed variety, drainage, and extension "
        "advice; avoid planting on one shower alone."
    )
    if len(note.split()) > 200:
        raise RuntimeError("Generated advisory exceeds the 200-word limit.")
    return note


def plot_rainfall_lines(regions: Sequence[Region]) -> tuple[plt.Figure, plt.Axes]:
    """Plot climatological rainfall for all regions on one line chart."""
    if not regions:
        raise ValueError("At least one region is required.")
    figure, axis = plt.subplots(figsize=(10, 5))
    for region in regions:
        axis.plot(MONTHS, region.monthly_rainfall_mm, marker="o", label=region.name)
    axis.set(title="Monthly rainfall climatology", ylabel="Rainfall (mm)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure, axis


def plot_crop_suitability(
    regions: Sequence[Region], crop_rules: Sequence[CropRule]
) -> tuple[plt.Figure, NDArray[np.object_]]:
    """Plot one month-by-region suitability heatmap for each crop."""
    suitability = classify_crop_suitability(regions, crop_rules)
    figure, axes = plt.subplots(
        len(crop_rules),
        1,
        figsize=(12, max(3.2, 2.8 * len(crop_rules))),
        squeeze=False,
        constrained_layout=True,
    )
    colors = ListedColormap(["#d95f5f", "#68a96b", "#5279ad"])
    normalization = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], colors.N)
    labels = ("Drought risk", "Good", "Waterlogging risk")
    encodings = {
        "Drought risk": -1,
        "Waterlogging risk": 1,
        **{f"Good for {rule.crop}": 0 for rule in crop_rules},
    }
    for axis, rule in zip(axes[:, 0], crop_rules):
        matrix = np.asarray(
            [
                [encodings[label] for label in suitability[region.name][rule.crop]]
                for region in regions
            ],
            dtype=float,
        )
        image = axis.imshow(matrix, cmap=colors, norm=normalization, aspect="auto")
        axis.set(
            title=rule.crop.title(),
            xticks=np.arange(12),
            xticklabels=MONTHS,
            yticks=np.arange(len(regions)),
            yticklabels=[region.name for region in regions],
        )
        axis.set_xticks(np.arange(-0.5, 12, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(regions), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.5)
        axis.tick_params(which="minor", bottom=False, left=False)
    colorbar = figure.colorbar(
        image, ax=axes[:, 0].tolist(), ticks=[-1, 0, 1], fraction=0.025, pad=0.02
    )
    colorbar.ax.set_yticklabels(labels)
    figure.suptitle("Monthly crop rainfall suitability screen")
    return figure, axes


def plot_monthly_variability(
    observations: Mapping[str, Mapping[int, Sequence[float]]]
) -> tuple[plt.Figure, NDArray[np.object_]]:
    """Show year-to-year rainfall variability with month-wise box plots."""
    regions = list(observations)
    if not regions:
        raise ValueError("At least one region's rainfall observations are required.")
    figure, axes = plt.subplots(
        len(regions), 1, figsize=(12, max(4, 3.5 * len(regions))), squeeze=False
    )
    for axis, name in zip(axes[:, 0], regions):
        years = observations[name]
        if not years:
            raise ValueError(f"{name} has no annual observations.")
        values = [np.asarray(months, dtype=float) for months in years.values()]
        if any(
            month_values.shape != (12,)
            or not np.all(np.isfinite(month_values))
            or np.any(month_values < 0)
            for month_values in values
        ):
            raise ValueError(f"{name} years must contain twelve finite non-negative values.")
        monthly_samples = np.stack(values, axis=0)
        axis.boxplot(
            [monthly_samples[:, month] for month in range(12)],
            tick_labels=MONTHS,
            showfliers=False,
        )
        axis.set(title=name, ylabel="Rainfall (mm)")
        axis.grid(axis="y", alpha=0.3)
    figure.suptitle("Year-to-year monthly rainfall variability (2015-2024)")
    figure.tight_layout()
    return figure, axes
