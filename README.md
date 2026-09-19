# NIBRS Crime Forecasting Dashboard

An interactive Streamlit dashboard for exploring monthly crime incidents from FBI NIBRS data and evaluating short-term SARIMA forecasts.

## Features

- Jurisdiction-level filtering across 41 eligible jurisdictions
- Historical monthly incident analysis
- Yearly and seasonal trend visualizations
- Three-month SARIMA forecasts
- 95% model-based prediction intervals
- Comparison of SARIMA against a Seasonal Naive benchmark
- Forecast performance analysis by horizon, origin, and jurisdiction
- Interactive Plotly visualizations
- Downloadable filtered historical data

## Data

The project uses a cleaned monthly NIBRS dataset covering 2015–2024, together with precomputed forecast and model-evaluation outputs.

The repository includes:

- `data/forecasting_dataset.csv`
- `data/final_forecasts_jan_mar_2025.csv`
- `data/final_sarima_selected_models.csv`
- `data/model_comparison_by_horizon.csv`
- `data/model_comparison_by_origin.csv`
- `data/model_comparison_by_state.csv`

Source: FBI National Incident-Based Reporting System (NIBRS).

> Crime counts represent recorded NIBRS incidents. Changes over time can partly reflect differences in agency participation and reporting coverage.

## Methodology

SARIMA models were selected for each eligible jurisdiction using the development data and then evaluated using leakage-controlled rolling-origin validation. A Seasonal Naive model was used as the benchmark.

The dashboard presents the precomputed forecasts and evaluation results; it does not retrain the SARIMA models when the Streamlit app is opened.

## Run Locally

```bash
git clone <YOUR_REPOSITORY_URL>
cd nibrs-crime-forecasting

python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

## Project Structure

```text
nibrs-crime-forecasting/
├── app.py
├── data/
│   ├── forecasting_dataset.csv
│   ├── final_forecasts_jan_mar_2025.csv
│   ├── final_sarima_selected_models.csv
│   ├── model_comparison_by_horizon.csv
│   ├── model_comparison_by_origin.csv
│   └── model_comparison_by_state.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## Technologies

Python · Streamlit · Pandas · NumPy · Plotly · Time-Series Forecasting · SARIMA
