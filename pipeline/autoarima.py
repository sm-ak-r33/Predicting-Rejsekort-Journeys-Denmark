from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from forecast_utils import apply_weekday_floor, inverse_log1p_forecast

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DAILY = ROOT / "results" / "daily"
DATA = RESULTS_DAILY / "data_cleaned.csv"
OUT = RESULTS_DAILY / "autoarima_metrics.csv"
OUT_FORECAST = RESULTS_DAILY / "autoarima_forecast.csv"
HORIZON_DAYS = 30


def eval_metrics(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "algorithm": "AutoARIMA",
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


def fit_auto_arima(values):
    from pmdarima import auto_arima

    return auto_arima(
        values,
        seasonal=True,
        m=7,
        start_p=0,
        start_q=0,
        max_p=10,
        max_q=10,
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )


if __name__ == "__main__":
    RESULTS_DAILY.mkdir(parents=True, exist_ok=True)
    series = load_series()
    log_series = np.log1p(series)
    split = max(int(len(series) * 0.8), 1)
    train, test = series.iloc[:split], series.iloc[split:]
    train_log = log_series.iloc[:split]

    model = fit_auto_arima(train_log)

    if len(test) >= 2:
        pred_log = model.predict(n_periods=len(test))
        pred = inverse_log1p_forecast(pred_log)
        pred = apply_weekday_floor(pred, test.index, train)
        pd.DataFrame([eval_metrics(test.values, pred)]).to_csv(OUT, index=False)
    else:
        pd.DataFrame([{"algorithm": "AutoARIMA", "rmse": np.nan, "mae": np.nan, "mape": np.nan, "r2": np.nan}]).to_csv(OUT, index=False)

    full_model = fit_auto_arima(log_series)
    future_idx = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=HORIZON_DAYS, freq="D")
    future_log = full_model.predict(n_periods=HORIZON_DAYS)
    future = inverse_log1p_forecast(future_log)
    future = apply_weekday_floor(future, future_idx, series)
    pd.DataFrame({"date": future_idx, "autoarima": future}).to_csv(OUT_FORECAST, index=False)
