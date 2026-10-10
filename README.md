# OOP Mini-Projects

A collection of small, self-contained Python projects demonstrating object-oriented
design, numerical analysis, forecasting, testing, and data visualization. Most
analyses use illustrative or short-sample data; results should be interpreted
with the project-specific assumptions and limitations below.

## Getting started

Use Python 3.12 or newer. From the repository root, create and activate a virtual
environment, then install the dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS/Linux, activate the environment with `source .venv/bin/activate`.

Run all automated tests from the repository root:

```text
python -m pytest -q
```

The analyses are Jupyter notebooks. Open the project notebook in VS Code with
the Jupyter extension, select the virtual environment above as the kernel, and
run all cells from top to bottom. Alternatively, install Jupyter with
`python -m pip install jupyter` and start it with `python -m jupyter lab`.
Notebooks use relative paths and are intended to run from the repository root.

## Projects

### 1. UBOS District Population Forecast

- Analysis: [ubos_analysis.ipynb](./ubos_analysis.ipynb)
- Overview: population-data analysis and forecasting using the data and
  assumptions documented in the notebook.
- Findings and limitations: consult the notebook outputs and accompanying
  discussion. Forecasts depend on the input data, selected model, and
  assumptions; they are not official UBOS projections.

### 2. Solar Micro-Grid Dispatch Planner

- Model: [src/microgrid.py](./src/microgrid.py)
- Analysis: [solar_microgrid.ipynb](./solar_microgrid.ipynb)
- Demo data: [data/microgrid_demand.csv](./data/microgrid_demand.csv)
- Tests: [tests/test_microgrid.py](./tests/test_microgrid.py)

Plans solar/battery dispatch for the Kasese health-centre example. It solves
daily load equations, accepts validated input, compares loop and vectorized
solvers, flags infeasible negative-source solutions and applies non-negative
least squares, reports usage and UGX cost, and includes sensitivity analysis
and a three-source extension.

**Limitations:** the demand data are seeded synthetic examples, not measured
facility demand. Prices, equations, and dispatch assumptions need local
validation before operational use.

### 3. Lake Victoria Fish Stock & Export Risk Model

- Model: [src/fish_stock.py](./src/fish_stock.py)
- Analysis: [lake_victoria_fishery.ipynb](./lake_victoria_fishery.ipynb)
- Tests: [tests/test_fish_stock.py](./tests/test_fish_stock.py)

Simulates a weekly logistic fish-stock model with proportional harvesting,
seeded bounded price paths, revenue statistics, coefficient-of-variation risk
bands, Monte Carlo downside VaR, MSY comparisons, and an eight-week closed
season scenario.

**Limitations:** model parameters, prices, and stock values are illustrative,
not validated Lake Victoria observations. Risk bands and price volatility are
example assumptions, not official cooperative policy or a market forecast.

### 4. Rainfall Pattern & Crop Suitability Analyser

- Model: [src/rainfall.py](./src/rainfall.py)
- Analysis: [rainfall_analysis.ipynb](./rainfall_analysis.ipynb)
- Ten-year dataset: [data/rainfall_power_2015_2024.csv](./data/rainfall_power_2015_2024.csv)
- Tests: [tests/test_rainfall.py](./tests/test_rainfall.py)

Compares rainfall statistics and pairwise cosine, Pearson, and Euclidean
measures; screens monthly crop suitability; detects rainy-season peaks; and
plots rainfall, suitability, and interannual variability. The notebook
includes both the assignment's illustrative values and ten years of NASA
POWER monthly precipitation estimates for Kampala, Gulu, and Mbarara.

**Findings:** the 2015–2024 POWER climatologies are classified as bimodal for
Kampala and Mbarara, and unimodal for Gulu, using the notebook's stated
heuristic.

**Limitations:** POWER `PRECTOTCORR` values are MERRA-2 reanalysis estimates,
not rain-gauge measurements. Monthly crop screening bands (maize 80–200 mm,
beans 50–150 mm, coffee 60–180 mm) are analyst-selected demonstration
thresholds, not validated agronomic limits. The peak/dry-month season
classification is also a heuristic. Crop and planting decisions need local
forecasts, soils, varieties, and extension advice. See the
[NASA POWER Monthly API](https://power.larc.nasa.gov/docs/services/api/temporal/monthly/)
and [FAO Irrigation and Drainage Paper 56](https://www.fao.org/4/X0490E/X0490E00.htm).

### 5. Taxi Route Revenue, Pricing & Fleet Planner

- Model: [src/taxi_planner.py](./src/taxi_planner.py)
- Analysis: [taxi_planner_analysis.ipynb](./taxi_planner_analysis.ipynb)
- Tests: [tests/test_taxi_planner.py](./tests/test_taxi_planner.py)

Reports route revenue and passenger statistics; solves the Ntinda
supply-demand equilibrium; evaluates moving-average, tuned exponential
smoothing, and linear-trend forecasts with rolling-origin days 4–10; forecasts
day-11 revenue; and plans fleet size with a 15% capacity buffer. Its extension
simulates 60 days with weekday seasonality and compares seasonal-naïve with
moving-average forecasts.

**Findings from the supplied sample:** the illustrative supply-demand curves
intersect at UGX 2,200 and 76 passengers per trip-hour, putting the stated
UGX 2,000 Ntinda fare below equilibrium under those assumptions. Tuned
exponential smoothing gives the lowest days 4–10 MAE on all three routes:

| Route | Moving-average MAE (passengers) | Tuned smoothing MAE | Linear-trend MAE | Day-11 forecast (passengers) | Forecast revenue (UGX) | Vehicles with buffer |
|---|---:|---:|---:|---:|---:|---:|
| Kampala–Ntinda | 7.52 | 5.86 | 7.87 | 45 | 90,000 | 1 |
| Kampala–Entebbe | 7.19 | 4.43 | 7.71 | 64 | 320,000 | 1 |
| Kampala–Mukono | 5.33 | 3.71 | 6.39 | 50 | 150,000 | 1 |

The seeded weekly simulation also gives lower MAE for seasonal-naïve
(3.89 passengers/day) than the 3-day moving average (12.95).

**Limitations:** each route forecast is based on only ten days; the alpha grid
is tuned and scored on the same short rolling-origin sample. The supply and
demand curves are stipulated assumptions, not estimated market relationships.
Fleet sizing assumes eight fully usable one-way trips per vehicle per day and
14 occupied seats on every trip; it ignores scheduling, return-leg loads,
downtime, traffic, and service constraints. The extension is synthetic, not
observed passenger demand.

## Tests and continuous integration

GitHub Actions runs `pytest` on pushes, pull requests, and manual dispatch,
using Python 3.12. To run the same suite locally, use `python -m pytest -q`
from the repository root. The workflow is configured in
[.github/workflows/tests.yml](./.github/workflows/tests.yml).

## AI usage

AI assistance was used to draft and refine this README and project
documentation, help develop unit tests, and assist with the GitHub Actions
workflow configuration. AI-generated suggestions and code were reviewed and
validated by running the project test suite and executing the analysis
notebooks. The numerical outputs remain dependent on the supplied datasets and
the assumptions stated in each project; AI assistance does not independently
validate those assumptions or establish that illustrative data represent
real-world conditions.
