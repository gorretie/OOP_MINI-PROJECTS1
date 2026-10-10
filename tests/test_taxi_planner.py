import statistics

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.taxi_planner import (
    DEFAULT_ALPHA_GRID,
    ExponentialSmoothingForecaster,
    LinearTrendForecaster,
    MovingAverageForecaster,
    Route,
    SeasonalNaiveForecaster,
    rolling_origin_backtest,
    seasonal_backtest_comparison,
    simulate_weekly_demand,
    solve_ntinda_equilibrium,
    vehicles_required,
    plot_route_revenue_forecasts,
)


@pytest.fixture
def sample_routes():
    return [
        Route(
            "Kampala-Ntinda",
            [35, 40, 42, 50, 55, 60, 48, 52, 47, 45],
            2_000,
        ),
        Route(
            "Kampala-Entebbe",
            [60, 58, 65, 70, 72, 80, 75, 68, 66, 64],
            5_000,
        ),
        Route(
            "Kampala-Mukono",
            [45, 47, 50, 49, 55, 62, 58, 53, 51, 50],
            3_000,
        ),
    ]


def test_route_revenue_and_descriptive_statistics():
    route = Route("Test", [2, 4, 6], 1_500)

    np.testing.assert_array_equal(route.daily_revenue(), [3_000, 6_000, 9_000])
    assert route.total_revenue() == 18_000
    assert route.descriptive_statistics() == {
        "mean": statistics.mean([2, 4, 6]),
        "variance": statistics.variance([2, 4, 6]),
        "standard_deviation": statistics.stdev([2, 4, 6]),
    }
    assert not route.passenger_counts.flags.writeable


@pytest.mark.parametrize(
    ("name", "counts", "fare"),
    [
        ("", [1, 2], 100),
        ("Route", [], 100),
        ("Route", [1, np.inf], 100),
        ("Route", [-1, 2], 100),
        ("Route", [1, 2], -1),
    ],
)
def test_route_rejects_invalid_parameters(name, counts, fare):
    with pytest.raises(ValueError):
        Route(name, counts, fare)


def test_route_requires_two_observations_for_sample_statistics():
    with pytest.raises(ValueError, match="At least two days"):
        Route("One day", [1], 1_000).descriptive_statistics()


def test_ntinda_equilibrium_uses_linear_system_and_interprets_fare():
    equilibrium_fare, equilibrium_quantity = solve_ntinda_equilibrium()

    assert equilibrium_fare == pytest.approx(2_200)
    assert equilibrium_quantity == pytest.approx(76)
    assert 2_000 < equilibrium_fare


@pytest.mark.parametrize(
    ("forecaster", "expected"),
    [
        (MovingAverageForecaster(), [11 / 3, 38 / 9]),
        (ExponentialSmoothingForecaster(alpha=1), [6.0, 6.0]),
        (LinearTrendForecaster(), [7.0, 8.6]),
        (SeasonalNaiveForecaster(3), [2.0, 3.0, 6.0, 2.0]),
    ],
)
def test_forecasters_predict_expected_horizon(forecaster, expected):
    history = [1, 2, 3, 6]
    np.testing.assert_allclose(forecaster.forecast(history, len(expected)), expected)


def test_forecasters_validate_parameters_and_insufficient_history():
    with pytest.raises(ValueError, match="alpha"):
        ExponentialSmoothingForecaster(0)
    with pytest.raises(ValueError, match="window"):
        MovingAverageForecaster(0)
    with pytest.raises(ValueError, match="at least two"):
        LinearTrendForecaster().forecast([4])
    with pytest.raises(ValueError, match="complete season"):
        SeasonalNaiveForecaster(7).forecast([1, 2, 3])
    with pytest.raises(ValueError, match="horizon"):
        MovingAverageForecaster().forecast([1, 2, 3], horizon=0)


def test_rolling_origin_backtest_evaluates_days_four_to_ten_and_tunes_alpha(
    sample_routes,
):
    route = sample_routes[0]
    result = rolling_origin_backtest(route)

    assert set(result.mae_by_model) == {
        "3-day moving average",
        "simple exponential smoothing",
        "linear trend",
    }
    assert all(np.isfinite(score) and score >= 0 for score in result.mae_by_model.values())
    assert len(result.backtest_predictions["3-day moving average"]) == 7
    assert len(result.alpha_scores) == len(DEFAULT_ALPHA_GRID)
    assert result.best_alpha in DEFAULT_ALPHA_GRID
    assert result.best_model in result.mae_by_model
    assert result.forecast_day11_passengers >= 0
    assert result.forecast_day11_revenue_ugx == pytest.approx(
        result.forecast_day11_passengers * route.fare_ugx
    )


def test_backtesting_rejects_invalid_origins_and_alpha_grids(sample_routes):
    with pytest.raises(ValueError, match="start_day"):
        rolling_origin_backtest(sample_routes[0], start_day=3)
    with pytest.raises(ValueError, match="alpha_grid"):
        rolling_origin_backtest(sample_routes[0], alpha_grid=[])
    with pytest.raises(ValueError, match="alpha_grid"):
        rolling_origin_backtest(sample_routes[0], alpha_grid=[0, 1])


@pytest.mark.parametrize(
    ("passengers", "expected"),
    [(112, 2), (113, 2), (200, 3), (0, 0)],
)
def test_fleet_capacity_adds_buffer_and_rounds_up(passengers, expected):
    assert vehicles_required(passengers) == expected


def test_fleet_planning_validates_inputs():
    with pytest.raises(ValueError):
        vehicles_required(-1)
    with pytest.raises(ValueError):
        vehicles_required(10, seats_per_vehicle=0)


def test_weekly_simulation_is_reproducible_and_seasonal_naive_wins():
    first = simulate_weekly_demand(seed=8)
    second = simulate_weekly_demand(seed=8)
    np.testing.assert_array_equal(first, second)
    scores = seasonal_backtest_comparison(first, evaluation_days=21)

    assert scores["seasonal naive"] < scores["3-day moving average"]


def test_revenue_forecast_plot_includes_actual_and_day11(sample_routes):
    results = {route.name: rolling_origin_backtest(route) for route in sample_routes}

    figure, axes = plot_route_revenue_forecasts(sample_routes, results)

    assert axes.shape == (3, 1)
    assert len(axes[0, 0].lines) == 1
    assert len(axes[0, 0].collections) == 1
    plt.close(figure)
