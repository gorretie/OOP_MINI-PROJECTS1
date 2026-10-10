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
