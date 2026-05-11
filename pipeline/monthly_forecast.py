from pathlib import Path
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import logging
import math
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

from forecast_utils import apply_positive_history_floor

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = Path(__file__).resolve().parent
MONTHLY_FILES = [
    PIPELINE / "rejsekort_monthly_chart_export.xlsx",
    PIPELINE / "rejsekort_hentdata.xlsx",
    PIPELINE / "rejsekort_monthly_export_extension_data.xlsx",
]
RESULTS_MONTHLY = ROOT / "results" / "monthly"
OUT_CSV = RESULTS_MONTHLY / "monthly_cleaned.csv"
OUT_FORECAST = RESULTS_MONTHLY / "monthly_forecast.csv"
OUT_METRICS = RESULTS_MONTHLY / "monthly_model_metrics.csv"
OUT_PNG = RESULTS_MONTHLY / "monthly_trends.png"

MONTHS_DA = {
    "jan": 1, "januar": 1,
    "feb": 2, "februar": 2,
    "mar": 3, "marts": 3,
    "apr": 4, "april": 4,
    "maj": 5,
    "jun": 6, "juni": 6,
    "jul": 7, "juli": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "okt": 10, "oktober": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def flatten_columns(columns):
    result = []
    for col in columns:
        if isinstance(col, tuple):
            parts = [str(x).strip() for x in col if str(x).strip() and not str(x).startswith("Unnamed")]
            result.append(" ".join(parts))
        else:
            result.append(str(col).strip())
    return result


def parse_number(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip()
    text = text.replace("\u00a0", " ")
    text = text.replace(".", "").replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else np.nan


def parse_month(value):
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, (pd.Timestamp,)):
        return pd.Timestamp(value.year, value.month, 1)
    text = str(value).strip().lower().replace("\u00a0", " ")
    dt = pd.to_datetime(text, dayfirst=True, errors="coerce")
    if pd.notna(dt):
        return pd.Timestamp(dt.year, dt.month, 1)
    year_match = re.search(r"(20\d{2}|19\d{2})", text)
    if not year_match:
        return pd.NaT
    year = int(year_match.group(1))
    for name, month in MONTHS_DA.items():
        if re.search(rf"\b{name}\b", text):
            return pd.Timestamp(year, month, 1)
    numeric_month = re.search(r"(?:^|\D)(1[0-2]|0?[1-9])(?:\D|$)", text.replace(str(year), " "))
    if numeric_month:
        return pd.Timestamp(year, int(numeric_month.group(1)), 1)
    return pd.NaT


def read_candidate_excel(path: Path):
    frames = []
    if not path.exists():
        return frames
    excel = pd.ExcelFile(path)
    for sheet in excel.sheet_names:
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        for header in range(min(12, len(raw))):
            try:
                df = pd.read_excel(path, sheet_name=sheet, header=header)
                df.columns = flatten_columns(df.columns)
                frames.append(df)
            except Exception:
                continue
    return frames


def extract_monthly_from_frame(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame(columns=["month", "passengers"])
    df = df.dropna(how="all")
    date_candidates = []
    value_candidates = []
    for col in df.columns:
        low = str(col).lower()
        parsed_dates = df[col].apply(parse_month)
        date_score = parsed_dates.notna().sum()
        nums = df[col].apply(parse_number)
        num_score = nums.notna().sum()
        if any(token in low for token in ["måned", "maaned", "month", "dato", "date", "år"]):
            date_score += 5
        if any(token in low for token in ["personrejser", "passenger", "antal", "rejser"]):
            num_score += 5
        date_candidates.append((date_score, col, parsed_dates))
        value_candidates.append((num_score, col, nums))
    date_candidates.sort(reverse=True, key=lambda x: x[0])
    value_candidates.sort(reverse=True, key=lambda x: x[0])
    for _, date_col, dates in date_candidates[:4]:
        for _, value_col, values in value_candidates[:6]:
            if date_col == value_col:
                continue
            tmp = pd.DataFrame({"month": dates, "passengers": values}).dropna()
            tmp = tmp[tmp["passengers"] > 0]
            if len(tmp) >= 6:
                return tmp.groupby("month", as_index=False)["passengers"].sum()
    return pd.DataFrame(columns=["month", "passengers"])


def load_monthly_data():
    pieces = []
    for path in MONTHLY_FILES:
        for frame in read_candidate_excel(path):
            extracted = extract_monthly_from_frame(frame)
            if not extracted.empty:
                extracted["source_file"] = path.name
                pieces.append(extracted)
    if not pieces:
        raise FileNotFoundError("No usable monthly export found. Run both monthly download scripts first.")
    df = pd.concat(pieces, ignore_index=True)
    df = df.sort_values(["month", "source_file"])
    df = df.drop_duplicates(subset=["month"], keep="last")
    df = df.groupby("month", as_index=False)["passengers"].sum()
    df = df.sort_values("month")
    return df


def metric_frame(actual, predictions):
    rows = []
    for name, pred in predictions.items():
        aligned = pd.Series(pred, index=actual.index).replace([np.inf, -np.inf], np.nan).dropna()
        common = actual.loc[aligned.index]
        rmse = math.sqrt(mean_squared_error(common, aligned)) if len(aligned) else np.nan
        mae = mean_absolute_error(common, aligned) if len(aligned) else np.nan
        rows.append({"algorithm": name, "rmse": rmse, "mae": mae})
    return pd.DataFrame(rows).sort_values("rmse")


def forecasts(df: pd.DataFrame, horizon=12):
    series = df.set_index("month")["passengers"].asfreq("MS")
    series = series.interpolate(limit_direction="both")
    future_idx = pd.date_range(series.index.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    split = max(int(len(series) * 0.8), len(series) - min(12, max(3, len(series)//4)))
    train, test = series.iloc[:split], series.iloc[split:]

    eval_predictions = {}
    future_predictions = {}

    try:
        sarima = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
        eval_predictions["SARIMA"] = sarima.forecast(len(test))
        full = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
        future_predictions["SARIMA"] = full.forecast(horizon).values
    except Exception:
        pass

    try:
        from pmdarima import auto_arima
        model = auto_arima(train, seasonal=True, m=12, trace=False, suppress_warnings=True, error_action="ignore", stepwise=True)
        eval_predictions["AutoARIMA"] = pd.Series(model.predict(n_periods=len(test)), index=test.index)
        model_full = auto_arima(series, seasonal=True, m=12, trace=False, suppress_warnings=True, error_action="ignore", stepwise=True)
        future_predictions["AutoARIMA"] = model_full.predict(n_periods=horizon)
    except Exception:
        pass

    try:
        from prophet import Prophet
        train_p = train.reset_index().rename(columns={"month": "ds", "passengers": "y"})
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(train_p)
        future_eval = pd.DataFrame({"ds": test.index})
        eval_predictions["Prophet"] = pd.Series(model.predict(future_eval)["yhat"].values, index=test.index)
        all_p = series.reset_index().rename(columns={"month": "ds", "passengers": "y"})
        model_full = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model_full.fit(all_p)
        future_predictions["Prophet"] = model_full.predict(pd.DataFrame({"ds": future_idx}))["yhat"].values
    except Exception:
        pass

    try:
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        values = series.values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(values)
        look_back = min(12, max(3, len(scaled)//5))
        X, y = [], []
        for i in range(len(scaled) - look_back):
            X.append(scaled[i:i + look_back])
            y.append(scaled[i + look_back])
        X, y = np.array(X), np.array(y)
        split_seq = max(1, split - look_back)
        X_train, y_train = X[:split_seq], y[:split_seq]
        X_test = X[split_seq:]
        if len(X_train) and len(X_test):
            model = Sequential([
                Bidirectional(LSTM(32, return_sequences=True), input_shape=(look_back, 1)),
                Dropout(0.1),
                Bidirectional(LSTM(16)),
                Dense(1),
            ])
            model.compile(optimizer="adam", loss="mse")
            model.fit(X_train, y_train, epochs=60, batch_size=8, verbose=0, callbacks=[EarlyStopping(monitor="loss", patience=8, restore_best_weights=True)])
            pred = scaler.inverse_transform(model.predict(X_test, verbose=0)).flatten()
            eval_predictions["BiLSTM"] = pd.Series(pred[-len(test):], index=test.index)
            rolling = scaled.copy().tolist()
            future_scaled = []
            for _ in range(horizon):
                x = np.array(rolling[-look_back:]).reshape(1, look_back, 1)
                nxt = float(model.predict(x, verbose=0)[0, 0])
                future_scaled.append(nxt)
                rolling.append([nxt])
            future_predictions["BiLSTM"] = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()
    except Exception:
        pass

    if not future_predictions:
        seasonal_naive = series.iloc[-12:].values if len(series) >= 12 else np.repeat(series.iloc[-1], horizon)
        future_predictions["SeasonalNaive"] = np.resize(seasonal_naive, horizon)
        eval_predictions["SeasonalNaive"] = pd.Series(np.resize(train.iloc[-12:].values, len(test)), index=test.index)

    metrics = metric_frame(test, eval_predictions) if len(test) else pd.DataFrame()
    forecast_df = pd.DataFrame({"month": future_idx})
    for name, values in future_predictions.items():
        forecast_df[name] = apply_positive_history_floor(values, series, quantile=0.05, multiplier=0.70)
    return series, forecast_df, metrics


def plot_monthly(series, forecast_df, metrics):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(series.index, series.values, linewidth=2.0, label="Actual monthly passengers")

    # Bridge each forecast from the final observed month so the forecast horizon
    # is visually connected to the historical trend.
    last_actual = pd.DataFrame({"month": [series.index.max()], "value": [series.iloc[-1]]})
    for col in forecast_df.columns:
        if col == "month":
            continue
        plotted = forecast_df[["month", col]].dropna().rename(columns={col: "value"})
        plotted = pd.concat([last_actual, plotted], ignore_index=True)
        ax.plot(plotted["month"], plotted["value"], linestyle="--", marker="o", markersize=3, label=f"{col} forecast")

    if not forecast_df.empty:
        ax.axvline(series.index.max(), linestyle=":", linewidth=1.2, label="Forecast starts")

    if not metrics.empty:
        best = metrics.iloc[0]["algorithm"]
        ax.set_title(f"Rejsekort monthly trend and 12-month forecast — best validation RMSE: {best}")
    else:
        ax.set_title("Rejsekort monthly trend and 12-month forecast")
    ax.set_xlabel("Month")
    ax.set_ylabel("Passenger journeys")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    RESULTS_MONTHLY.mkdir(parents=True, exist_ok=True)
    monthly = load_monthly_data()
    monthly.to_csv(OUT_CSV, index=False)
    series, forecast_df, metrics = forecasts(monthly)
    forecast_df.to_csv(OUT_FORECAST, index=False)
    if metrics.empty:
        metrics = pd.DataFrame(columns=["algorithm", "rmse", "mae"])
    metrics.to_csv(OUT_METRICS, index=False)
    plot_monthly(series, forecast_df, metrics)
