import numpy as np
import pytest

from src.fish_stock import (
    FishStock,
    PriceModel,
    RiskAssessor,
    annual_revenue_paths_ugx,
    fibonacci_stock_baseline,
    revenue_statistics,
)


def test_fibonacci_baseline_returns_15_values_and_validates_count():
    assert fibonacci_stock_baseline() == [
        0,
        1,
        1,
        2,
        3,
        5,
        8,
        13,
        21,
        34,
        55,
        89,
        144,
        233,
        377,
    ]
    with pytest.raises(ValueError):
        fibonacci_stock_baseline(0)


def test_logistic_stock_simulation_applies_growth_and_harvest_for_52_weeks():
    results = FishStock().simulate(52, harvest_rate=0.10)

    expected_first_step = 4_000 + 0.4 * 4_000 * (1 - 4_000 / 10_000) - 0.10 * 4_000
    assert results.stock_tonnes.shape == (53,)
    assert results.catch_tonnes.shape == (52,)
    assert results.stock_tonnes[1] == pytest.approx(expected_first_step)
    assert results.final_stock_tonnes > 4_000
    assert np.all(results.stock_tonnes >= 0)


def test_stock_simulation_validates_rates_and_supports_eight_week_closed_season():
    fishery = FishStock()
    results = fishery.simulate(104, harvest_rate=0.1, closed_season_weeks=8)

    np.testing.assert_array_equal(results.catch_tonnes[:8], np.zeros(8))
    np.testing.assert_array_equal(results.catch_tonnes[52:60], np.zeros(8))
    assert np.all(results.catch_tonnes[8:52] > 0)
    with pytest.raises(ValueError, match="harvest_rate"):
        fishery.simulate(52, harvest_rate=1.1)
    with pytest.raises(ValueError, match="closed_season_weeks"):
        fishery.simulate(52, harvest_rate=0.1, closed_season_weeks=53)


def test_price_paths_are_seeded_and_stay_within_bounds():
    model = PriceModel(seed=19)
    prices = model.simulate_paths(52, n_paths=1_000)

    assert prices.shape == (1_000, 52)
    np.testing.assert_array_equal(prices, model.simulate_paths(52, n_paths=1_000))
    assert np.all(prices >= 9_000)
    assert np.all(prices <= 16_000)


def test_revenue_statistics_use_population_variance_and_calculate_cv():
    summary = revenue_statistics([1, 2, 3])

    assert summary.mean == pytest.approx(2)
    assert summary.median == pytest.approx(2)
    assert summary.variance == pytest.approx(2 / 3)
    assert summary.standard_deviation == pytest.approx(np.sqrt(2 / 3))
    assert summary.coefficient_of_variation == pytest.approx(np.sqrt(2 / 3) / 2)
    with pytest.raises(ValueError):
        revenue_statistics([])


def test_risk_assessor_uses_ordered_cv_bands():
    assessor = RiskAssessor()

    assert assessor.classify(0.099) == "Low"
    assert assessor.classify(0.10) == "Moderate"
    assert assessor.classify(0.20) == "High"
    with pytest.raises(ValueError):
        assessor.classify(-0.1)


def test_annual_revenue_converts_tonnes_to_kilograms():
    revenue = annual_revenue_paths_ugx([1, 2], [[10_000, 12_000], [9_000, 10_000]])

    np.testing.assert_array_equal(revenue, [34_000_000, 29_000_000])
    with pytest.raises(ValueError, match="paths-by-weeks"):
        annual_revenue_paths_ugx([1, 2], [[10_000]])


def test_invalid_fish_stock_and_price_model_parameters_are_rejected():
    with pytest.raises(ValueError, match="carrying capacity"):
        FishStock(initial_stock_tonnes=10_001)
    with pytest.raises(ValueError, match="Price bounds"):
        PriceModel(price_bounds_ugx_per_kg=(16_000, 9_000))
    with pytest.raises(ValueError, match="n_paths"):
        PriceModel().simulate_paths(52, n_paths=0)
