# OOP Mini-Projects

## Mini-Project 1: UBOS District Population Forecast

See [ubos_analysis.ipynb](./ubos_analysis.ipynb) for the population forecasting analysis.

## Mini-Project 2: Solar Micro-Grid Dispatch Planner

The Kasese health-centre planner is implemented in [src/microgrid.py](./src/microgrid.py), with a runnable analysis in [solar_microgrid.ipynb](./solar_microgrid.ipynb). It solves the two daily load equations, accepts validated interactive or CSV input, benchmarks loop and vectorized solving, flags negative-source solutions and re-dispatches them with NNLS, reports usage statistics and UGX costs, plots daily dispatch, and includes Monte Carlo sensitivity and a three-source hybrid extension.

The seeded synthetic 30-day demand dataset is at [data/microgrid_demand.csv](./data/microgrid_demand.csv). To regenerate it reproducibly, run:

```python
from src.microgrid import generate_demo_csv

generate_demo_csv("data/microgrid_demand.csv", days=30, seed=42)
```

Install the project requirements and run the tests from the repository root:

```text
python -m pip install -r requirements.txt
python -m pytest -q
```

## Mini-Project 3: Lake Victoria Fish Stock & Export Risk Model

The Jinja fishery model uses a weekly discrete logistic stock equation with
proportional harvesting, a seeded bounded price random walk, revenue statistics,
CV-based risk classes, and Monte Carlo downside VaR. It also compares harvest
rates with the weekly MSY and models an eight-week closed season over five years.
The implementation is in [src/fish_stock.py](./src/fish_stock.py), tests are in
[tests/test_fish_stock.py](./tests/test_fish_stock.py), and the complete analysis
and charts are in [lake_victoria_fishery.ipynb](./lake_victoria_fishery.ipynb).

The values are illustrative rather than Lake Victoria field data. In particular,
the CV thresholds are example cooperative risk-policy bands, and the price-walk
volatility is an explicit modeling assumption.
