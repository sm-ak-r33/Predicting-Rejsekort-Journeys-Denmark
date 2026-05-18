from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from forecast_utils import apply_weekday_floor, inverse_log1p_forecast

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DAILY = ROOT / "results" / "daily"
DATA = RESULTS_DAILY / "data_cleaned.csv"
OUT = RESULTS_DAILY / "sarima_metrics.csv"
OUT_FORECAST = RESULTS_DAILY / "sarima_forecast.csv"
OUT_VALIDATION = RESULTS_DAILY / "sarima_validation.csv"
HORIZON_DAYS = 30


def eval_metrics(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "algorithm": "SARIMA",
        "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
        "mae": float(mean_absolute_error(actual, pred)),
        "mape": float(np.mean(np.abs((actual - pred) / np.maximum(actual, 1))) * 100),
        "r2": float(r2_score(actual, pred)),
    }


def load_series():
    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date")
    df = df[df["date"] >= "2021-01-01"]
    if df.empty:
        raise ValueError("results/daily/data_cleaned.csv has no usable rows from 2021 onward.")
    series = df.set_index("date")["passengers"].asfreq("D").interpolate(limit_direction="both")
    return series.clip(lower=1)


def fit_model(series_log):
    return SARIMAX(
        series_log,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)


if __name__ == "__main__":
    RESULTS_DAILY.mkdir(parents=True, exist_ok=True)
    series = load_series()
    log_series = np.log1p(series)

    split = max(int(len(series) * 0.8), 1)
    train, test = series.iloc[:split], series.iloc[split:]
    train_log = log_series.iloc[:split]

    if len(test) >= 2:
        model = fit_model(train_log)
        pred_log = model.forecast(len(test))
        pred = inverse_log1p_forecast(pred_log.values)
        pred = apply_weekday_floor(pred, test.index, train)

        pd.DataFrame([eval_metrics(test.values, pred)]).to_csv(OUT, index=False)
        pd.DataFrame({
            "date": test.index,
            "actual": test.values,
            "sarima": pred,
        }).to_csv(OUT_VALIDATION, index=False)
    else:
        pd.DataFrame([{
            "algorithm": "SARIMA",
            "rmse": np.nan,
            "mae": np.nan,
            "mape": np.nan,
            "r2": np.nan,
        }]).to_csv(OUT, index=False)
        pd.DataFrame(columns=["date", "actual", "sarima"]).to_csv(OUT_VALIDATION, index=False)

    full_model = fit_model(log_series)
    future_idx = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=HORIZON_DAYS, freq="D")
    future_log = full_model.forecast(HORIZON_DAYS)
    future = inverse_log1p_forecast(future_log.values)
    future = apply_weekday_floor(future, future_idx, series)
    pd.DataFrame({"date": future_idx, "sarima": future}).to_csv(OUT_FORECAST, index=False)
