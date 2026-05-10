from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_cleaned.csv"
OUT = ROOT / "sarima_metrics.csv"


def eval_metrics(actual, pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
        "mae": float(mean_absolute_error(actual, pred)),
        "mape": float(np.mean(np.abs((actual - pred) / np.maximum(actual, 1))) * 100),
        "r2": float(r2_score(actual, pred)),
    }


if __name__ == "__main__":
    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date")
    df = df[df["date"] >= "2021-01-01"]
    series = df.set_index("date")["passengers"].asfreq("D").interpolate(limit_direction="both")
    split = int(len(series) * 0.8)
    train, test = series.iloc[:split], series.iloc[split:]
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)).fit(disp=False)
    pred = model.forecast(len(test))
    pd.DataFrame([eval_metrics(test.values, pred.values)]).to_csv(OUT, index=False)
