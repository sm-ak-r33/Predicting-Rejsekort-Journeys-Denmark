from pathlib import Path
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import logging
import shutil
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DAILY = ROOT / "results" / "daily"
DOCS_ASSETS = ROOT / "docs" / "assets"
DATA = RESULTS_DAILY / "data_cleaned.csv"
OUT_PNG = RESULTS_DAILY / "bilstm_validation.png"
OUT_README_PNG = DOCS_ASSETS / "bilstm_validation.png"
OUT_METRICS = RESULTS_DAILY / "bilstm_metrics.csv"
OUT_FORECAST = RESULTS_DAILY / "bilstm_forecast.csv"
OUT_VALIDATION = RESULTS_DAILY / "bilstm_validation_predictions.csv"
HORIZON_DAYS = 30


def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)


def _format_axis(ax):
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.grid(True, alpha=0.25)


def _copy_readme_asset():
    DOCS_ASSETS.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUT_PNG, OUT_README_PNG)
    print("Saved README preview plot: %s" % OUT_README_PNG)


if __name__ == "__main__":
    RESULTS_DAILY.mkdir(parents=True, exist_ok=True)
    np.random.seed(40)

    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date")
    if df.empty:
        raise ValueError("results/daily/data_cleaned.csv has no usable rows.")

    series = df.set_index("date")["passengers"].asfreq("D").interpolate(limit_direction="both")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))

    look_back = min(30, max(7, len(series) // 10))
    X, y = create_sequences(scaled, look_back)
    if len(X) < 10:
        raise ValueError("Not enough daily rows to train BiLSTM. Need at least 10 sequences after look_back.")

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(look_back, 1)),
        Dropout(0.2),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mape")
    model.fit(
        X_train,
        y_train,
        epochs=75,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
        verbose=0,
    )

    y_pred = scaler.inverse_transform(model.predict(X_test, verbose=0)).flatten()
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    metrics = {
        "algorithm": "BiLSTM",
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100),
        "r2": float(r2_score(y_true, y_pred)),
    }
    pd.DataFrame([metrics]).to_csv(OUT_METRICS, index=False)

    dates = series.index[look_back + split:]
    validation = pd.DataFrame({"date": dates, "actual": y_true, "bilstm": y_pred}).dropna()
    validation.to_csv(OUT_VALIDATION, index=False)

    last_year_start = validation["date"].max() - pd.Timedelta(days=365)
    validation_tail = validation[validation["date"] >= last_year_start]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(validation_tail["date"], validation_tail["actual"], linewidth=1.8, label="Actual validation data")
    rolling = validation_tail.set_index("date")["actual"].rolling(7, min_periods=1).mean()
    ax.plot(rolling.index, rolling.values, linewidth=2.1, label="Actual 7-day average")
    ax.plot(validation_tail["date"], validation_tail["bilstm"], linestyle="--", linewidth=1.8, label="BiLSTM validation forecast")
    ax.set_title("Rejsekort daily BiLSTM validation")
    ax.set_xlabel("Date")
    ax.set_ylabel("Passenger journeys")
    _format_axis(ax)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)
    _copy_readme_asset()

    rolling_values = scaled.copy().tolist()
    future_scaled = []
    for _ in range(HORIZON_DAYS):
        x = np.array(rolling_values[-look_back:]).reshape(1, look_back, 1)
        nxt = float(model.predict(x, verbose=0)[0, 0])
        future_scaled.append(nxt)
        rolling_values.append([nxt])

    future_idx = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=HORIZON_DAYS, freq="D")
    future = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()
    pd.DataFrame({"date": future_idx, "bilstm": np.maximum(future, 0)}).to_csv(OUT_FORECAST, index=False)
