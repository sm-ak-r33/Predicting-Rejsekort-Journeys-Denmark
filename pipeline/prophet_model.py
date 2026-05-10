from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_cleaned.csv"
OUT = ROOT / "prophet_metrics.csv"


def eval_metrics(actual, pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
        "mae": float(mean_absolute_error(actual, pred)),
        "mape": float(np.mean(np.abs((actual - pred) / np.maximum(actual, 1))) * 100),
        "r2": float(r2_score(actual, pred)),
    }


if __name__ == "__main__":
    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date")
    df = df[df["date"] >= "2021-01-01"][["date", "passengers"]]
    prophet_df = df.rename(columns={"date": "ds", "passengers": "y"})
    split = int(len(prophet_df) * 0.8)
    train, test = prophet_df.iloc[:split], prophet_df.iloc[split:]
    model = Prophet()
    model.fit(train)
    forecast = model.predict(pd.DataFrame({"ds": test["ds"]}))
    pd.DataFrame([eval_metrics(test["y"].values, forecast["yhat"].values)]).to_csv(OUT, index=False)
