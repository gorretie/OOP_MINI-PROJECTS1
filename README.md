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

## Mini-Project 4: Rainfall Pattern & Crop Suitability Analyser

The implementation is in [src/rainfall.py](./src/rainfall.py), with tests in
[tests/test_rainfall.py](./tests/test_rainfall.py) and the complete analysis in
[rainfall_analysis.ipynb](./rainfall_analysis.ipynb). It includes validated
`Region` and `CropRule` classes, crop-screen labels for every month, NumPy
cosine similarity (verified against SciPy), pairwise cosine/Pearson/Euclidean
matrices, circular rainy-season peak detection, and rainfall, suitability, and
year-to-year variability plots.

The notebook retains the assignment's illustrative Kampala, Gulu, and Mbarara
series for the core comparison. The extension uses ten complete years of NASA
POWER monthly `PRECTOTCORR` estimates (2015-2024), saved in
[data/rainfall_power_2015_2024.csv](./data/rainfall_power_2015_2024.csv).
Locations use NASA POWER's point grid nearest the city coordinates. Its monthly
precipitation values are mean mm/day; the CSV converts them to monthly totals
by multiplying by calendar days. The request coordinates are Kampala
(0.3476, 32.5825), Gulu (2.7746, 32.2990), and Mbarara (-0.6072, 30.6545).
These MERRA-2 reanalysis estimates are not rain-gauge measurements. See the
[NASA POWER Monthly API documentation](https://power.larc.nasa.gov/docs/services/api/temporal/monthly/).

The default monthly screening bands are maize 80-200 mm, beans 50-150 mm, and
coffee 60-180 mm. These broad, analyst-selected demonstration bands are informed
by crop water needs and growth-stage guidance; the specific monthly cutoffs are
not published FAO thresholds. FAO Irrigation and Drainage Paper 56 describes
stage-specific crop coefficients and growth periods (including East African
maize-grain and dry-bean examples in Table 11); it also explains that water
requirements depend on reference evapotranspiration and local conditions.
Consult [FAO-56](https://www.fao.org/4/X0490E/X0490E00.htm) and local extension
services before making planting decisions. The automatic mode classification
likewise uses stated peak-prominence and dry-month heuristics.

## Mini-Project 5: Taxi Route Revenue, Pricing & Fleet Planner

The route model and forecasts are implemented in [src/taxi_planner.py](./src/taxi_planner.py), with tests in [tests/test_taxi_planner.py](./tests/test_taxi_planner.py) and the complete analysis in [taxi_planner_analysis.ipynb](./taxi_planner_analysis.ipynb). It reports revenue and sample descriptive statistics, solves the Ntinda supply-demand equations with `scipy.linalg.solve`, compares moving-average, tuned simple-exponential-smoothing, and linear-trend forecasts using rolling-origin days 4-10, forecasts day-11 revenue, and rounds fleet capacity up after a 15% buffer. Its seeded 60-day extension compares seasonal-naive and 3-day moving-average forecasts.

The equilibrium model gives a fare of UGX 2,200 and 76 passengers per trip-hour; the stated UGX 2,000 fare is below that illustrative equilibrium. Forecast demand and fleet needs rely on just ten observations, and the fleet calculation assumes 8 fully usable one-way trips of 14 occupied seats per vehicle per day. The extension is synthetic and only demonstrates a weekly seasonal pattern; it is not observed taxi demand.
