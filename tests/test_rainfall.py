from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.spatial.distance import cosine as scipy_cosine_distance

from src.rainfall import (
    DEFAULT_CROP_RULES,
    CropRule,
    Region,
    classify_crop_suitability,
    cosine_similarity,
    detect_rainy_seasons,
    farmer_advisory,
    load_power_rainfall_csv,
    plot_crop_suitability,
    plot_monthly_variability,
    plot_rainfall_lines,
    regions_from_yearly_rainfall,
    similarity_matrices,
)


ILLUSTRATIVE_REGIONS = [
    Region("Kampala", [120, 140, 180, 200, 220, 180, 90, 70, 60, 100, 110, 130]),
    Region("Gulu", [8, 25, 75, 160, 190, 145, 170, 215, 175, 150, 60, 15]),
    Region("Mbarara", [70, 85, 120, 140, 90, 25, 20, 55, 100, 125, 120, 90]),
]


def test_region_statistics_and_extreme_months():
    kampala = ILLUSTRATIVE_REGIONS[0]

    assert isinstance(kampala.monthly_rainfall_mm, np.ndarray)
    assert not kampala.monthly_rainfall_mm.flags.writeable
    assert kampala.annual_total() == pytest.approx(1_600)
    assert kampala.mean() == pytest.approx(1_600 / 12)
    assert kampala.wettest_month() == "May"
    assert kampala.driest_month() == "Sep"
    assert kampala.coefficient_of_variation() == pytest.approx(
        np.std(kampala.monthly_rainfall_mm) / np.mean(kampala.monthly_rainfall_mm)
    )


@pytest.mark.parametrize(
    "rainfall",
    [
        [1] * 11,
        [1] * 13,
        [1] * 11 + [np.nan],
        [1] * 11 + [-1],
    ],
)
def test_region_rejects_invalid_monthly_rainfall(rainfall):
    with pytest.raises(ValueError):
        Region("Invalid", rainfall)


def test_region_coefficient_of_variation_rejects_zero_mean():
    with pytest.raises(ValueError, match="zero rainfall"):
        Region("Dry", [0] * 12).coefficient_of_variation()


def test_crop_rule_classifies_boundaries_and_all_months_for_every_region():
    maize = CropRule("maize", 80, 200)

    assert maize.classify(79.99) == "Drought risk"
    assert maize.classify(80) == "Good for maize"
    assert maize.classify(200) == "Good for maize"
    assert maize.classify(200.01) == "Waterlogging risk"
    results = classify_crop_suitability(ILLUSTRATIVE_REGIONS, DEFAULT_CROP_RULES)
    assert set(results) == {"Kampala", "Gulu", "Mbarara"}
    assert all(
        len(months) == 12
        for region_results in results.values()
        for months in region_results.values()
    )


def test_cosine_similarity_uses_vector_dot_product_and_matches_scipy():
    first = ILLUSTRATIVE_REGIONS[0].monthly_rainfall_mm
    second = ILLUSTRATIVE_REGIONS[1].monthly_rainfall_mm

    assert cosine_similarity(first, second) == pytest.approx(
        1 - scipy_cosine_distance(first, second)
    )
    assert cosine_similarity(first, 2 * first) == pytest.approx(1)
    with pytest.raises(ValueError, match="zero vector"):
        cosine_similarity(first, np.zeros(12))


def test_similarity_matrices_have_correct_diagonal_symmetry_and_scale_behavior():
    pattern = np.arange(1, 13, dtype=float)
    regions = [Region("Original", pattern), Region("Wetter", 3 * pattern)]

    matrices = similarity_matrices(regions)
    np.testing.assert_allclose(np.diag(matrices.cosine), 1)
    np.testing.assert_allclose(matrices.cosine, matrices.cosine.T)
    np.testing.assert_allclose(matrices.pearson, 1)
    np.testing.assert_allclose(matrices.euclidean, matrices.euclidean.T)
    assert matrices.cosine[0, 1] == pytest.approx(1)
    assert matrices.euclidean[0, 1] > 0


def test_yearly_loader_builds_ten_year_monthly_climatologies():
    path = Path(__file__).parents[1] / "data" / "rainfall_power_2015_2024.csv"

    observations = load_power_rainfall_csv(path)
    regions = regions_from_yearly_rainfall(observations)

    assert set(observations) == {"Kampala", "Gulu", "Mbarara"}
    assert all(len(years) == 10 for years in observations.values())
    assert all(
        values.shape == (12,)
        for yearly_values in observations.values()
        for values in yearly_values.values()
    )
    assert [region.name for region in regions] == ["Kampala", "Gulu", "Mbarara"]
    assert all(np.all(region.monthly_rainfall_mm > 0) for region in regions)


def test_yearly_loader_rejects_short_or_malformed_data(tmp_path):
    short_data = tmp_path / "short.csv"
    short_data.write_text(
        "region,year,month,rainfall_mm\n"
        + "".join(f"Test,{year},{month},10\n" for year in (2020, 2021) for month in range(1, 13)),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="at least 10"):
        load_power_rainfall_csv(short_data)

    duplicate_data = tmp_path / "duplicate.csv"
    duplicate_data.write_text(
        "region,year,month,rainfall_mm\nTest,2020,1,10\nTest,2020,1,11\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_power_rainfall_csv(duplicate_data, minimum_years=1)


def test_rainy_season_detection_finds_two_seasons_but_merges_one_long_wet_spell():
    bimodal = Region("South", [30, 30, 120, 200, 120, 30, 20, 120, 190, 120, 30, 30])
    unimodal = Region("North", [10, 10, 30, 100, 180, 140, 130, 150, 200, 160, 70, 20])

    assert detect_rainy_seasons(bimodal).classification == "bimodal"
    assert len(detect_rainy_seasons(bimodal).peak_months) == 2
    assert detect_rainy_seasons(unimodal).classification == "unimodal"
    assert len(detect_rainy_seasons(unimodal).peak_months) == 1


def test_farmer_advisory_supports_a_selected_crop_and_enforces_word_limit():
    observations = load_power_rainfall_csv(
        Path(__file__).parents[1] / "data" / "rainfall_power_2015_2024.csv"
    )
    gulu = regions_from_yearly_rainfall(observations)[1]

    note = farmer_advisory(gulu, DEFAULT_CROP_RULES, crop="maize")

    assert "maize" in note
    assert "Mar" in note
    assert "Nov" not in note.split(";")[0]
    assert len(note.split()) <= 200
    with pytest.raises(ValueError, match="No crop rule"):
        farmer_advisory(gulu, DEFAULT_CROP_RULES, crop="rice")


def test_visualization_functions_return_expected_axes():
    observations = {
        region.name: {
            year: region.monthly_rainfall_mm
            for year in range(2015, 2025)
        }
        for region in ILLUSTRATIVE_REGIONS
    }

    line_figure, _ = plot_rainfall_lines(ILLUSTRATIVE_REGIONS)
    heatmap_figure, heatmap_axes = plot_crop_suitability(
        ILLUSTRATIVE_REGIONS, DEFAULT_CROP_RULES
    )
    boxplot_figure, boxplot_axes = plot_monthly_variability(observations)

    assert len(line_figure.axes) == 1
    assert heatmap_axes.shape == (3, 1)
    assert boxplot_axes.shape == (3, 1)
    plt.close(line_figure)
    plt.close(heatmap_figure)
    plt.close(boxplot_figure)
