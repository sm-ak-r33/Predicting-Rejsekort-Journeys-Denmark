from pathlib import Path
import logging
import warnings
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DAILY = ROOT / "results" / "daily"
DATA = RESULTS_DAILY / "data_cleaned.csv"
OUT = RESULTS_DAILY / "prophet_metrics.csv"
OUT_FORECAST = RESULTS_DAILY / "prophet_forecast.csv"
HORIZON_DAYS = 30


def eval_metrics(actual, pred):
    return {
        "algorithm": "Prophet",
        "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
        "mae": float(mean_absolute_error(actual, pred)),
        "mape": float(np.mean(np.abs((actual - pred) / np.maximum(actual, 1))) * 100),
        "r2": float(r2_score(actual, pred)),
    }


if __name__ == "__main__":
    RESULTS_DAILY.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date")
    df = df[df["date"] >= "2021-01-01"][["date", "passengers"]]
    if df.empty:
        raise ValueError("results/daily/data_cleaned.csv has no usable rows from 2021 onward.")

    prophet_df = df.rename(columns={"date": "ds", "passengers": "y"})
    split = max(int(len(prophet_df) * 0.8), 1)
    train, test = prophet_df.iloc[:split], prophet_df.iloc[split:]

    model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    model.fit(train)

    if len(test) >= 2:
        forecast = model.predict(pd.DataFrame({"ds": test["ds"]}))
        pd.DataFrame([eval_metrics(test["y"].values, forecast["yhat"].values)]).to_csv(OUT, index=False)
    else:
        pd.DataFrame([{"algorithm": "Prophet", "rmse": np.nan, "mae": np.nan, "mape": np.nan, "r2": np.nan}]).to_csv(OUT, index=False)

    full_model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    full_model.fit(prophet_df)
    future_idx = pd.date_range(prophet_df["ds"].max() + pd.Timedelta(days=1), periods=HORIZON_DAYS, freq="D")
    future = full_model.predict(pd.DataFrame({"ds": future_idx}))
    pd.DataFrame({"date": future_idx, "prophet": np.maximum(future["yhat"].values, 0)}).to_csv(OUT_FORECAST, index=False)
