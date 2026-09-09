# Retail Demand Intelligence

An end-to-end retail demand forecasting system built using the Walmart M5 dataset. The project demonstrates a production-oriented data science workflow spanning data preparation, exploratory analysis, leakage-safe feature engineering, machine learning, model evaluation, persistence, inference, REST API development, SQL analytics, and automated testing.

## Project Overview

Retail demand forecasting is challenging because product demand is intermittent, prices change over time, products are not always available, and demand is influenced by calendar effects, events, and purchasing patterns.

This project develops a 28-day demand forecasting system for Walmart store `CA_1`.

The analytical grain is:

> **one item × one store × one day**

The modeling approach explicitly distinguishes between:

- products that are unavailable,
- available products with zero observed demand,
- available products with positive demand.

Unavailable product-days receive a deterministic forecast of zero, while available product-days are handled by machine learning models.

## Dataset

The project uses data from the **M5 Forecasting - Accuracy** competition.

Primary source files:

- `calendar.csv`
- `sales_train_evaluation.csv`
- `sell_prices.csv`

For the current implementation, the analysis focuses on store `CA_1`:

- **3,049 products**
- **1,941 days**
- **5,918,109 item-day observations**

Raw and processed datasets are intentionally excluded from Git because of their size.

## System Architecture

```text
M5 Raw Data
     |
     v
Data Preparation
     |
     v
Exploratory Data Analysis
     |
     v
Leakage-Safe Feature Engineering
     |
     v
Chronological Train / Validation / Test Split
     |
     v
+---------------------------+
|   Demand Modeling         |
|                           |
|  Poisson LightGBM         |
|           +               |
|  Demand Classifier        |
+---------------------------+
     |
     v
Two-Stage Forecasting
     |
     v
Forecast Evaluation
     |
     v
Model Persistence
     |
     v
Inference Pipeline
     |
     +-----------> FastAPI
     |
     +-----------> SQL Analytics
```

## Exploratory Findings

Several demand patterns emerged during exploratory analysis.

### Weekly demand

Weekend demand was substantially higher than midweek demand. Average Sunday demand was approximately **54.5% higher than Wednesday demand**.

### Product categories

Observed unit-demand share:

| Category | Share |
|---|---:|
| FOODS | 69.86% |
| HOUSEHOLD | 18.75% |
| HOBBIES | 11.39% |

### Intermittent demand

A major modeling challenge is the large number of zero-demand observations.

Across the complete dataset:

| Demand state | Share |
|---|---:|
| Available with zero sales | 44.67% |
| Available with positive sales | 36.24% |
| Unavailable | 19.09% |

Among available product-days alone, approximately **55.21% recorded zero sales**.

This motivated the later two-stage modeling architecture.

### SNAP

Average demand was higher on California SNAP days. This is treated as a **descriptive association rather than a causal effect**.

### Price behavior

Price analysis showed that demand tended to be higher when products were priced below their product-specific historical reference price.

Again, this relationship is descriptive and should not be interpreted as causal price elasticity.

## Leakage-Safe Feature Engineering

The forecast horizon is **28 days**.

A key design requirement is that every predictor used for a forecast must be available at the forecast origin.

For this reason, short lags such as `sales_lag_1` or `sales_lag_7` are not used for the non-recursive 28-day forecasting design.

Demand-history features include:

- `sales_lag_28`
- `sales_lag_35`
- `sales_lag_42`
- `sales_lag_56`
- 7-day rolling mean shifted by 28 days
- 28-day rolling mean shifted by 28 days
- 7-day rolling standard deviation shifted by 28 days
- 28-day rolling standard deviation shifted by 28 days

Additional predictors include:

- calendar variables,
- weekday and seasonal information,
- event indicators,
- SNAP,
- product hierarchy,
- current selling price,
- historical price changes,
- product price ratios.

The final ML feature contract contains **30 predictors**.

## Chronological Validation

Random train/test splitting is inappropriate for this forecasting problem because it would allow future observations to influence model development.

The data is therefore split chronologically:

```text
Training
2011-01-29 to 2016-03-27

Validation
2016-03-28 to 2016-04-24
28 days

Test
2016-04-25 to 2016-05-22
28 days
```

Rows lacking sufficient historical demand information are removed from training only.

The validation and test horizons remain intact.

## Forecasting Models

Three forecasting approaches were evaluated.

### 1. Seasonal Naive Baseline

The baseline assumes:

```text
forecast(t) = sales(t - 28)
```

This provides a transparent benchmark for determining whether machine learning adds value.

### 2. LightGBM Poisson Regressor

A LightGBM model with a Poisson objective predicts demand magnitude.

This model is appropriate for non-negative count-like demand and handles nonlinear relationships between historical demand, price, calendar variables, events, and product attributes.

### 3. Two-Stage LightGBM

Retail demand is highly intermittent, so a second model explicitly estimates whether demand will be positive.

The two-stage system works as follows:

```text
LightGBM Classifier
        |
        v
P(sales > 0)
        |
        | probability >= 0.50
        v
Poisson Demand Forecast
        |
        v
Final Two-Stage Forecast
```

If the classifier probability is below the validation-selected threshold of **0.50**, the forecast is set to zero.

The threshold was selected using the validation set and frozen before test evaluation.

## Final Test Results

The untouched 28-day test horizon produced the following results:

| Model | MAE | RMSE | WMAPE | Bias |
|---|---:|---:|---:|---:|
| Seasonal Naive | 1.4102 | 2.9357 | 89.82% | -4.53% |
| LightGBM Poisson | 1.1605 | **2.2639** | 73.92% | **-2.95%** |
| Two-Stage LightGBM | **1.1025** | 2.3300 | **70.22%** | -20.16% |

Relative to the seasonal naive baseline, the two-stage model improved:

- **MAE by 21.82%**
- **WMAPE by 21.82%**
- **RMSE by 20.63%**

The standalone Poisson model achieved the best RMSE and substantially better aggregate calibration.

This illustrates an important modeling trade-off:

> The two-stage model performs better on typical item-day absolute error, while the standalone Poisson model is preferable when large misses and aggregate forecast bias are more costly.

The test set is treated as a final evaluation horizon rather than a model-tuning dataset.

## Feature Importance

The strongest LightGBM feature was the 28-day rolling demand average shifted by the forecast horizon.

Other important predictors included:

- product identity,
- recent historical demand,
- rolling demand variability,
- department,
- selling price,
- weekday,
- event information,
- historical price relationships.

Historical demand therefore remains the dominant predictive signal, while product, calendar, and price information provide additional predictive value.

## Production Pipeline

Reusable production code is organized under `src/`.

### Data preparation

`src/data/prepare_data.py`

Handles:

- raw M5 loading,
- dtype optimization,
- store filtering,
- wide-to-long reshaping,
- calendar joins,
- price joins,
- availability construction,
- data integrity validation.

### Feature engineering

`src/features/build_features.py`

Builds:

- calendar features,
- horizon-safe demand lags,
- rolling statistics,
- price-history features,
- price-change features,
- availability-aware price history.

### Training

`src/models/train.py`

Handles:

- chronological splitting,
- feature-contract validation,
- categorical alignment,
- training-row filtering,
- LightGBM fitting,
- validation forecast assembly,
- forecast evaluation,
- end-to-end training orchestration.

### Model persistence

`src/models/persistence.py`

Persists:

- Poisson model,
- demand classifier,
- selected threshold,
- ordered feature contract.

Generated model binaries are stored locally under:

```text
artifacts/models/
```

and excluded from Git.

### Inference

`src/models/predict.py`

Provides reusable production inference with:

- feature-contract validation,
- exact feature ordering,
- availability-aware scoring,
- Poisson prediction,
- positive-demand probability,
- two-stage forecasting.

Unavailable products bypass machine learning and receive deterministic zero forecasts.

## REST API

The project includes a FastAPI inference service in:

```text
api/main.py
```

Available endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Service information |
| GET | `/health` | API and model readiness |
| POST | `/forecast` | Generate demand forecasts |

Run locally with:

```bash
python -m uvicorn api.main:app --reload
```

Interactive API documentation is then available at:

```text
http://127.0.0.1:8000/docs
```

The API loads persisted models once during application startup and delegates forecasting to the reusable inference module.

## SQL Analytics

The `sql/` directory contains a tested SQLite-compatible analytical layer.

### Schema

`sql/schema.sql`

Defines:

- `daily_demand`
- `demand_forecasts`

with primary keys, data-quality constraints, and analytical indexes.

### Business queries

`sql/analysis_queries.sql`

Includes queries for:

- daily store demand,
- category demand share,
- weekday patterns,
- availability and zero-demand states,
- SNAP demand,
- category pricing,
- highest-demand products,
- forecast MAE,
- forecast WMAPE,
- forecast bias,
- category-level forecast performance.

All SQL statements are automatically executed against an in-memory SQLite database during testing.

## Automated Testing

The project currently contains **94 passing automated tests** covering:

- evaluation metrics,
- feature engineering,
- data preparation,
- forecasting utilities,
- chronological training logic,
- LightGBM fitting,
- training orchestration,
- model persistence,
- production inference,
- FastAPI endpoints,
- SQL schema and analytical queries.

Run the complete test suite with:

```bash
python -m pytest -q
```

## Project Structure

```text
retail-demand-intelligence/
|
|-- api/
|   `-- main.py
|
|-- data/
|   |-- raw/
|   |-- interim/
|   `-- processed/
|
|-- notebooks/
|   |-- 01_data_inspection.ipynb
|   |-- 02_exploratory_data_analysis.ipynb
|   |-- 03_feature_engineering.ipynb
|   `-- 04_baseline_modeling.ipynb
|
|-- sql/
|   |-- schema.sql
|   `-- analysis_queries.sql
|
|-- src/
|   |-- data/
|   |   `-- prepare_data.py
|   |
|   |-- evaluation/
|   |   `-- metrics.py
|   |
|   |-- features/
|   |   `-- build_features.py
|   |
|   `-- models/
|       |-- forecast_models.py
|       |-- persistence.py
|       |-- predict.py
|       `-- train.py
|
|-- tests/
|
|-- .gitignore
|-- LICENSE
|-- README.md
`-- requirements.txt
```

## Reproducing the Project

### 1. Clone the repository

```bash
git clone https://github.com/dhirajworks7/retail-demand-intelligence.git
cd retail-demand-intelligence
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the M5 dataset

Place the required M5 files under:

```text
data/raw/
```

Required files:

```text
calendar.csv
sales_train_evaluation.csv
sell_prices.csv
```

### 5. Run automated tests

```bash
python -m pytest -q
```

## Technology Stack

- Python
- pandas
- NumPy
- LightGBM
- scikit-learn
- PyArrow / Parquet
- FastAPI
- Uvicorn
- SQLite / SQL
- pytest
- JupyterLab
- Git / GitHub

## Key Engineering Decisions

Several design decisions were made to keep the project realistic and reproducible:

- chronological validation instead of random splitting,
- horizon-safe features to prevent forecasting leakage,
- explicit separation of product unavailability from zero demand,
- deterministic zero forecasts for unavailable products,
- validation-only selection of the two-stage threshold,
- preservation of the final test horizon for out-of-sample evaluation,
- persisted model metadata and feature contracts,
- reusable inference rather than notebook-only prediction,
- API logic separated from ML inference logic,
- automated testing of both Python and SQL components,
- large datasets and generated model artifacts excluded from Git.

## Future Improvements

Potential extensions include:

- rolling-origin backtesting,
- hierarchical forecast reconciliation,
- probabilistic prediction intervals,
- model monitoring and drift detection,
- containerization with Docker,
- cloud deployment,
- CI/CD with GitHub Actions,
- multi-store forecasting,
- richer model explainability.

## Author

**Dhiraj**

Master's graduate in Data Science from Dalarna University, Sweden.

This project demonstrates an end-to-end data science and machine learning workflow, from raw data preparation and exploratory analysis to forecasting, evaluation, production inference, API development, SQL analytics, and automated testing.