from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_cleaned.csv"
OUT_PNG = ROOT / "output.png"
OUT_METRICS = ROOT / "bilstm_metrics.csv"


def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)


if __name__ == "__main__":
    np.random.seed(40)
    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date")
    series = df.set_index("date")["passengers"].asfreq("D").interpolate(limit_direction="both")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))
    look_back = 30
    X, y = create_sequences(scaled, look_back)
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
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100),
        "r2": float(r2_score(y_true, y_pred)),
    }
    pd.DataFrame([metrics]).to_csv(OUT_METRICS, index=False)

    dates = series.index[look_back + split:]
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_true, label="Actual")
    plt.plot(dates, y_pred, label="BiLSTM forecast")
    plt.title("Rejsekort daily forecast")
    plt.xlabel("Date")
    plt.ylabel("Passenger journeys")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    plt.close()
