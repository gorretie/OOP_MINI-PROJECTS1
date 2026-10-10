import numpy as np
import pytest

from src.ubos import (
    CAGRGrowthForecaster,
    DistrictPopulation,
    Districtpopulation,
    FibonacciRatioForecaster,
    Forecaster,
    LinearTrendForecaster,
)


@pytest.fixture
def sample_years():
    return np.array([2015, 2016, 2017])


@pytest.fixture
def sample_population():
    return np.array([100.0, 110.0, 121.0])


def test_district_population_stores_numpy_arrays_and_exposes_properties():
    district = DistrictPopulation([2015, 2016], [100, 110], "Kampala")

    assert isinstance(district.years, np.ndarray)
    assert isinstance(district.population, np.ndarray)
    np.testing.assert_array_equal(district.years, [2015, 2016])
    np.testing.assert_array_equal(district.population, [100.0, 110.0])
    assert district.get_district_name() == "Kampala"
    np.testing.assert_array_equal(district.get_population(), [100.0, 110.0])
    assert len(district) == 2
    assert repr(district) == (
        "DistrictPopulation(name='Kampala', years=2015-2016, observations=2)"
    )


def test_district_population_default_name_and_legacy_alias():
    district = Districtpopulation([2015], [100])

    assert isinstance(district, DistrictPopulation)
    assert district.get_district_name() == "District"


@pytest.mark.parametrize(
    ("years", "population"),
    [
        ([2015, 2016], [100]),
        ([], []),
        ([2015, 2016], [100, -1]),
        ([[2015, 2016]], [100, 110]),
        ([2015, 2016], [[100, 110]]),
        ([2015, 2016], [100, np.nan]),
        ([2015, 2016], [100, np.inf]),
        ([2015, 2015], [100, 110]),
        ([2016, 2015], [100, 110]),
        ([2015, 2016.5], [100, 110]),
    ],
)
def test_district_population_rejects_invalid_data(years, population):
    with pytest.raises(ValueError):
        DistrictPopulation(years, population)


def test_district_population_rejects_empty_name():
    with pytest.raises(ValueError, match="District name"):
        DistrictPopulation([2015], [100], "  ")


def test_forecaster_is_abstract():
    with pytest.raises(TypeError):
        Forecaster()


@pytest.mark.parametrize(
    "model_type",
    [
        LinearTrendForecaster,
        CAGRGrowthForecaster,
        FibonacciRatioForecaster,
    ],
)
def test_forecaster_requires_fit_before_predict(model_type):
    with pytest.raises(RuntimeError, match="Call fit"):
        model_type().predict(1)


@pytest.mark.parametrize(
    "model_type",
    [
        LinearTrendForecaster,
        CAGRGrowthForecaster,
        FibonacciRatioForecaster,
    ],
)
def test_fit_returns_model_and_predict_accepts_horizon_or_years(
    model_type, sample_years, sample_population
):
    model = model_type()

    assert model.fit(sample_years, sample_population) is model
    by_horizon = model.predict(2)
    by_years = model.predict([2018, 2019])

    assert by_horizon.shape == (2,)
    np.testing.assert_allclose(by_horizon, by_years)
    assert np.all(np.isfinite(by_horizon))


@pytest.mark.parametrize(
    "model_type",
    [
        LinearTrendForecaster,
        CAGRGrowthForecaster,
        FibonacciRatioForecaster,
    ],
)
@pytest.mark.parametrize(
    "bad_years,bad_population",
    [
        ([2015, 2016], [100]),
        ([2015], [100]),
        ([2015, 2015], [100, 110]),
        ([2016, 2015], [100, 110]),
        ([2015, 2016], [100, -1]),
        ([2015, 2016], [100, np.nan]),
        ([[2015, 2016]], [100, 110]),
    ],
)
def test_forecaster_fit_rejects_invalid_training_data(
    model_type, bad_years, bad_population
):
    with pytest.raises(ValueError):
        model_type().fit(bad_years, bad_population)


@pytest.mark.parametrize(
    "bad_horizon",
    [
        0,
        -1,
        [],
        [[2018]],
        [2018.5],
        [np.nan],
    ],
)
@pytest.mark.parametrize(
    "model_type",
    [
        LinearTrendForecaster,
        CAGRGrowthForecaster,
        FibonacciRatioForecaster,
    ],
)
def test_forecaster_rejects_invalid_horizons(
    model_type, bad_horizon, sample_years, sample_population
):
    model = model_type().fit(sample_years, sample_population)

    with pytest.raises(ValueError):
        model.predict(bad_horizon)


def test_linear_trend_forecaster_predicts_linear_series(sample_years):
    model = LinearTrendForecaster().fit(sample_years, [100, 110, 120])

    np.testing.assert_allclose(model.predict([2018, 2019]), [130, 140])


def test_linear_trend_forecaster_does_not_predict_negative_population():
    model = LinearTrendForecaster().fit([2015, 2016, 2017], [10, 5, 0])

    np.testing.assert_array_equal(model.predict([2018, 2019]), [0, 0])


def test_cagr_forecaster_predicts_constant_percentage_growth(sample_years):
    model = CAGRGrowthForecaster().fit(sample_years, [100, 110, 121])

    assert model.annual_rate == pytest.approx(0.1)
    np.testing.assert_allclose(model.predict([2018, 2019]), [133.1, 146.41])


@pytest.mark.parametrize(
    "population",
    [
        [0, 10],
        [10, 0],
    ],
)
def test_cagr_forecaster_requires_positive_endpoint_populations(population):
    with pytest.raises(ValueError, match="positive"):
        CAGRGrowthForecaster().fit([2015, 2016], population)


def test_fibonacci_forecaster_uses_successive_fibonacci_ratios(sample_years):
    model = FibonacciRatioForecaster().fit(sample_years, [100, 110, 121])

    np.testing.assert_allclose(model.predict([2018, 2019]), [181.5, 302.5])

