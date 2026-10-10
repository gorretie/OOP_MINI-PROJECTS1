import numpy as np
import pytest

from src.microgrid import (
    HybridMicroGrid,
    MicroGrid,
    generate_demo_csv,
    load_demand_csv,
    read_interactive_demands,
)


def test_microgrid_matrix_is_invertible_and_solves_one_day():
    grid = MicroGrid()

    assert grid.determinant == pytest.approx(-5.0)
    assert grid.condition_number == pytest.approx(np.linalg.cond([[3, 2], [4, 1]]))
    assert grid.is_well_posed
    np.testing.assert_allclose(grid.solve_day(100, 90), [16.0, 26.0])


@pytest.mark.parametrize("demands", [(-1, 4), (np.nan, 4), (np.inf, 4), ("", 4)])
def test_solve_day_rejects_invalid_demands(demands):
    with pytest.raises(ValueError):
        MicroGrid().solve_day(*demands)


def test_solve_day_rejects_singular_system():
    grid = MicroGrid([[1, 2], [2, 4]])

    assert not grid.is_well_posed
    with pytest.raises(np.linalg.LinAlgError, match="singular"):
        grid.solve_day(10, 20)


def test_interactive_input_retries_empty_non_numeric_and_negative_values():
    responses = iter(["", "abc", "-4", "100", "90"])
    messages = []

    demands = read_interactive_demands(
        input_fn=lambda _prompt: next(responses), output_fn=messages.append
    )

    assert demands == (100.0, 90.0)
    assert len(messages) == 3


def test_seeded_csv_has_30_reproducible_demands(tmp_path):
    first = generate_demo_csv(tmp_path / "first.csv", seed=17)
    second = generate_demo_csv(tmp_path / "second.csv", seed=17)

    first_demands = load_demand_csv(first)
    second_demands = load_demand_csv(second)
    assert first_demands.shape == (30, 2)
    np.testing.assert_array_equal(first_demands, second_demands)
    assert np.all(first_demands >= 0)


def test_csv_loader_rejects_missing_columns_and_invalid_demand(tmp_path):
    missing_column = tmp_path / "missing.csv"
    missing_column.write_text("day,D1_kWh\n1,10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns"):
        load_demand_csv(missing_column)

    invalid_demand = tmp_path / "invalid.csv"
    invalid_demand.write_text(
        "day,D1_kWh,D2_kWh\n1,10,not-a-number\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="non-numeric"):
        load_demand_csv(invalid_demand)


def test_loop_vectorized_and_nnls_dispatch_agree():
    grid = MicroGrid()
    demands = [[100, 90], [10, 50], [120, 80]]

    loop_solutions = grid.solve_days_loop(demands)
    vectorized_solutions = grid.solve_days_vectorized(demands)
    np.testing.assert_allclose(loop_solutions, vectorized_solutions)

    results = grid.dispatch_days(demands)
    np.testing.assert_array_equal(results.infeasible, [False, True, False])
    assert np.all(results.solar >= 0)
    assert np.all(results.battery >= 0)
    assert results.daily_cost.shape == (3,)
    assert results.monthly_cost == pytest.approx(np.sum(results.daily_cost))
    np.testing.assert_allclose(
        [results.solar[0], results.battery[0]], loop_solutions[:, 0]
    )


def test_statistics_and_sensitivity_are_reproducible():
    grid = MicroGrid()
    results = grid.dispatch_days([[100, 90], [110, 100], [120, 105]])
    stats = grid.statistics(results)

    assert stats["solar"]["mean"] == pytest.approx(np.mean(results.solar))
    assert stats["battery"]["variance"] == pytest.approx(np.var(results.battery))
    assert grid.more_volatile_source(stats) in {"solar", "battery"}
    first = grid.sensitivity_analysis(100, 90, seed=9)
    second = grid.sensitivity_analysis(100, 90, seed=9)
    assert first == second
    assert first["solar"]["mean_absolute_change"] > 0


def test_hybrid_microgrid_solves_three_source_system_and_rejects_dependent_equation():
    hybrid = HybridMicroGrid()
    solution = hybrid.solve_day(100, 90, 40)

    assert solution.shape == (3,)
    np.testing.assert_allclose(hybrid.coefficient_matrix @ solution, [100, 90, 40])

    dependent = HybridMicroGrid(
        [[3, 2, 1], [4, 1, 2], [7, 3, 3]]
    )
    assert not dependent.is_well_posed
    with pytest.raises(np.linalg.LinAlgError, match="singular"):
        dependent.solve_day(100, 90, 40)
