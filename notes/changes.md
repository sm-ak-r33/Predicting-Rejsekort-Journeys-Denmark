# Daily validation zoom patch

This patch fixes the daily chart zoom panel so it no longer displays future forecasts.

## What changed

- `daily_forecast.py`
  - Keeps the top panel as daily history + future 30-day forecast.
  - Changes the lower-left zoom panel to validation-only: actual validation data vs validation predictions.
  - Continues copying `results/daily/daily_trends.png` to `docs/assets/daily_trends.png`.

- `selected_arima.py`
  - Saves `results/daily/sarima_validation.csv`.

- `autoarima.py`
  - Saves `results/daily/autoarima_validation.csv`.

- `prophet_model.py`
  - Saves `results/daily/prophet_validation.csv`.

- `BiLSTM.py`
  - Saves `results/daily/bilstm_validation_predictions.csv`.
  - Keeps the validation PNG and README asset copy.

## Why validation CSVs are needed

The existing daily scripts saved metrics and future forecasts, but the daily plotter had no model-level validation prediction CSVs to draw in the zoom panel. The patch adds only those validation-output files; it does not change the model formulas, split logic, ingestion, or future forecast logic.
