from pathlib import Path
import logging
import math
import os
import shutil
import warnings
from typing import Dict, Iterable, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_MONTHLY = ROOT / "results" / "monthly"
DOCS_ASSETS = ROOT / "docs" / "assets"

INPUT_CANDIDATES = [
    RESULTS_MONTHLY / "monthly_cleaned.csv",
    ROOT / "data_monthly_cleaned.csv",
]

OUT_CLEAN = RESULTS_MONTHLY / "monthly_cleaned.csv"
OUT_FORECAST = RESULTS_MONTHLY / "monthly_forecast.csv"
OUT_METRICS = RESULTS_MONTHLY / "monthly_model_metrics.csv"
OUT_PNG = RESULTS_MONTHLY / "monthly_trends.png"
OUT_README_PNG = DOCS_ASSETS / "monthly_trends.png"

MODEL_ORDER = ["SARIMA", "AutoARIMA", "Prophet", "BiLSTM", "Ensemble"]
MODEL_COLORS = {
    "SARIMA": "#1f77b4",
    "AutoARIMA": "#ff7f0e",
    "Prophet": "#2ca02c",
    "BiLSTM": "#9467bd",
    "Ensemble": "#d62728",
}


def _to_float(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip().replace("\u00a0", " ")
    text = text.replace(".", "").replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return np.nan


def load_monthly_data() -> pd.DataFrame:
    source = None
    for candidate in INPUT_CANDIDATES:
        if candidate.exists():
            source = candidate
            break

    if source is None:
        raise FileNotFoundError(
            "No clean monthly CSV found. Run: python pipeline/ingest_monthly.py"
        )

    df = pd.read_csv(source)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "date" not in df.columns:
        for candidate in ["month", "ds", "afgangsdato"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "date"})
                break

    if "journeys" not in df.columns:
        for candidate in ["passengers", "rejser", "personrejser", "antal"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "journeys"})
                break

    if "date" not in df.columns or "journeys" not in df.columns:
        raise ValueError("Monthly CSV must contain date and journeys columns.")

    df = df[["date", "journeys"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df["journeys"] = df["journeys"].apply(_to_float)
    df = df.dropna(subset=["date", "journeys"])
    df = df[df["journeys"] > 0]
    df = df.groupby("date", as_index=False)["journeys"].sum()
    df = df.sort_values("date").reset_index(drop=True)
    return df


def clean_regular_series(df: pd.DataFrame) -> pd.Series:
    series = df.set_index("date")["journeys"].sort_index().asfreq("MS")
    series = series.interpolate(limit_direction="both")
    series = series[series > 0]
    if len(series) < 24:
        raise ValueError("Need at least 24 monthly observations for a stable 12-month forecast.")
    return series


def _seasonal_anchor(series: pd.Series, future_idx: pd.DatetimeIndex) -> np.ndarray:
    last_12 = series.tail(min(12, len(series)))
    recent_level = float(last_12.median())
    long_level = float(series.median())
    month_medians = series.groupby(series.index.month).median()

    seasonal = []
    for dt in future_idx:
        same_month = float(month_medians.get(dt.month, long_level))
        seasonal.append(0.55 * recent_level + 0.45 * same_month)
    return np.array(seasonal, dtype=float)


def _stabilize_forecast(values: Iterable[float], series: pd.Series, future_idx: pd.DatetimeIndex) -> np.ndarray:
    values = np.asarray(list(values), dtype=float)
    values = pd.Series(values).replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both")
    values = values.fillna(float(series.tail(12).median())).values

    anchor = _seasonal_anchor(series, future_idx)
    lower = np.maximum(1.0, anchor * 0.45)
    upper = anchor * 1.65
    return np.clip(values, lower, upper)


def _rmse(actual: pd.Series, pred: pd.Series) -> float:
    pred = pd.Series(pred, index=actual.index).replace([np.inf, -np.inf], np.nan).dropna()
    common = actual.loc[pred.index]
    if len(pred) == 0:
        return np.nan
    return math.sqrt(mean_squared_error(common, pred))


def _mae(actual: pd.Series, pred: pd.Series) -> float:
    pred = pd.Series(pred, index=actual.index).replace([np.inf, -np.inf], np.nan).dropna()
    common = actual.loc[pred.index]
    if len(pred) == 0:
        return np.nan
    return mean_absolute_error(common, pred)


def _sarima_forecast(train: pd.Series, test: pd.Series, full: pd.Series, horizon: int) -> Tuple[pd.Series, np.ndarray]:
    model = SARIMAX(
        np.log1p(train),
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    eval_pred = np.expm1(model.forecast(len(test)))
    eval_pred = pd.Series(eval_pred.values, index=test.index)

    full_model = SARIMAX(
        np.log1p(full),
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    future = np.expm1(full_model.forecast(horizon).values)
    return eval_pred, future


def _autoarima_forecast(train: pd.Series, test: pd.Series, full: pd.Series, horizon: int) -> Tuple[pd.Series, np.ndarray]:
    from pmdarima import auto_arima

    model = auto_arima(
        np.log1p(train),
        seasonal=True,
        m=12,
        start_p=0,
        start_q=0,
        max_p=2,
        max_q=2,
        max_P=1,
        max_Q=1,
        d=None,
        D=1,
        trace=False,
        suppress_warnings=True,
        error_action="ignore",
        stepwise=True,
    )
    eval_pred = pd.Series(np.expm1(model.predict(n_periods=len(test))), index=test.index)

    full_model = auto_arima(
        np.log1p(full),
        seasonal=True,
        m=12,
        start_p=0,
        start_q=0,
        max_p=2,
        max_q=2,
        max_P=1,
        max_Q=1,
        d=None,
        D=1,
        trace=False,
        suppress_warnings=True,
        error_action="ignore",
        stepwise=True,
    )
    future = np.expm1(full_model.predict(n_periods=horizon))
    return eval_pred, future


def _prophet_forecast(train: pd.Series, test: pd.Series, full: pd.Series, horizon: int, future_idx: pd.DatetimeIndex) -> Tuple[pd.Series, np.ndarray]:
    from prophet import Prophet

    def fit_predict(input_series: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
        frame = input_series.reset_index()
        frame.columns = ["ds", "y"]
        model = Prophet(
            yearly_seasonality=6,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.08,
            seasonality_prior_scale=8.0,
            interval_width=0.80,
        )
        model.fit(frame)
        return model.predict(pd.DataFrame({"ds": dates}))["yhat"].values

    eval_pred = pd.Series(fit_predict(train, test.index), index=test.index)
    future = fit_predict(full, future_idx)
    return eval_pred, future


def _bilstm_forecast(train: pd.Series, test: pd.Series, full: pd.Series, horizon: int) -> Tuple[pd.Series, np.ndarray]:
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Bidirectional, Dense, Dropout, LSTM
    from tensorflow.keras.models import Sequential

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(full.values.reshape(-1, 1))
    look_back = 12

    X, y = [], []
    for i in range(len(scaled) - look_back):
        X.append(scaled[i:i + look_back])
        y.append(scaled[i + look_back])
    X = np.asarray(X)
    y = np.asarray(y)

    split_pos = len(train) - look_back
    if split_pos <= 0 or len(X) <= split_pos:
        raise ValueError("Not enough monthly observations for BiLSTM validation.")

    X_train = X[:split_pos]
    y_train = y[:split_pos]
    X_test = X[split_pos:split_pos + len(test)]

    model = Sequential([
        Bidirectional(LSTM(24, return_sequences=True), input_shape=(look_back, 1)),
        Dropout(0.10),
        Bidirectional(LSTM(12)),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        X_train,
        y_train,
        epochs=80,
        batch_size=8,
        verbose=0,
        callbacks=[EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)],
    )

    eval_values = scaler.inverse_transform(model.predict(X_test, verbose=0)).flatten()
    eval_pred = pd.Series(eval_values[-len(test):], index=test.index)

    rolling = scaled.copy().tolist()
    future_scaled = []
    for _ in range(horizon):
        x = np.asarray(rolling[-look_back:]).reshape(1, look_back, 1)
        nxt = float(model.predict(x, verbose=0)[0, 0])
        future_scaled.append(nxt)
        rolling.append([nxt])
    future = scaler.inverse_transform(np.asarray(future_scaled).reshape(-1, 1)).flatten()
    return eval_pred, future


def build_forecasts(series: pd.Series, horizon: int = 12) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    future_idx = pd.date_range(series.index.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")

    test_size = 12 if len(series) >= 48 else max(6, len(series) // 5)
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]

    eval_predictions: Dict[str, pd.Series] = {}
    future_predictions: Dict[str, np.ndarray] = {}

    model_functions = {
        "SARIMA": _sarima_forecast,
        "AutoARIMA": _autoarima_forecast,
        "Prophet": _prophet_forecast,
        "BiLSTM": _bilstm_forecast,
    }

    for name, fn in model_functions.items():
        try:
            if name == "Prophet":
                eval_pred, future = fn(train, test, series, horizon, future_idx)
            else:
                eval_pred, future = fn(train, test, series, horizon)
            eval_predictions[name] = eval_pred.clip(lower=1)
            future_predictions[name] = _stabilize_forecast(future, series, future_idx)
        except Exception as exc:
            print(f"Skipping {name}: {exc}")

    if not future_predictions:
        base = series.tail(12).values
        future_predictions["SeasonalNaive"] = np.resize(base, horizon)
        eval_predictions["SeasonalNaive"] = pd.Series(np.resize(train.tail(12).values, len(test)), index=test.index)

    rows = []
    for name, pred in eval_predictions.items():
        rows.append({
            "model": name,
            "rmse": _rmse(test, pred),
            "mae": _mae(test, pred),
        })
    metrics = pd.DataFrame(rows).sort_values("rmse", na_position="last").reset_index(drop=True)

    ranked = [m for m in metrics["model"].tolist() if m in future_predictions]
    ensemble_members = ranked[:2] if len(ranked) >= 2 else ranked
    if ensemble_members:
        future_predictions["Ensemble"] = np.mean([future_predictions[m] for m in ensemble_members], axis=0)
        eval_predictions["Ensemble"] = pd.concat([eval_predictions[m] for m in ensemble_members], axis=1).mean(axis=1)
        metrics = pd.concat([
            metrics,
            pd.DataFrame([{
                "model": "Ensemble",
                "rmse": _rmse(test, eval_predictions["Ensemble"]),
                "mae": _mae(test, eval_predictions["Ensemble"]),
            }]),
        ], ignore_index=True).sort_values("rmse", na_position="last").reset_index(drop=True)

    forecast = pd.DataFrame({"date": future_idx})
    for name in MODEL_ORDER:
        if name in future_predictions:
            forecast[name] = future_predictions[name]
    for name, values in future_predictions.items():
        if name not in forecast.columns:
            forecast[name] = values

    validation = pd.DataFrame({"date": test.index, "Actual": test.values})
    for name in MODEL_ORDER:
        if name in eval_predictions:
            validation[name] = eval_predictions[name].reindex(test.index).values
    for name, values in eval_predictions.items():
        if name not in validation.columns:
            validation[name] = values.reindex(test.index).values

    return forecast, metrics, validation


def _millions(values: Iterable[float]) -> np.ndarray:
    return np.asarray(values, dtype=float) / 1_000_000.0


def _format_date_axis(ax, monthly: bool = True) -> None:
    ax.grid(True, alpha=0.22)
    if monthly:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    for label in ax.get_xticklabels():
        label.set_rotation(35)
        label.set_ha("right")


def _copy_readme_asset() -> None:
    DOCS_ASSETS.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUT_PNG, OUT_README_PNG)
    print(f"Saved README preview plot: {OUT_README_PNG}")


def _plot_forecast_lines(ax, series: pd.Series, forecast: pd.DataFrame, include_legend: bool = True) -> None:
    last_date = series.index.max()
    last_value = series.iloc[-1]

    forecast_cols = [c for c in forecast.columns if c != "date"]
    model_cols = [c for c in forecast_cols if c != "Ensemble"]

    if model_cols:
        forecast_min = forecast[model_cols].min(axis=1).values
        forecast_max = forecast[model_cols].max(axis=1).values
        ax.fill_between(
            forecast["date"],
            _millions(forecast_min),
            _millions(forecast_max),
            alpha=0.12,
            color="#d62728",
            label="Model range" if include_legend else None,
        )

    for col in model_cols:
        bridge_dates = pd.concat([pd.Series([last_date]), forecast["date"]], ignore_index=True)
        bridge_values = np.concatenate([[last_value], forecast[col].values])
        ax.plot(
            bridge_dates,
            _millions(bridge_values),
            linestyle="--",
            linewidth=1.4,
            alpha=0.70,
            color=MODEL_COLORS.get(col, "#777777"),
            label=col if include_legend else None,
        )

    if "Ensemble" in forecast.columns:
        bridge_dates = pd.concat([pd.Series([last_date]), forecast["date"]], ignore_index=True)
        bridge_values = np.concatenate([[last_value], forecast["Ensemble"].values])
        ax.plot(
            bridge_dates,
            _millions(bridge_values),
            linewidth=2.8,
            marker="o",
            markersize=4,
            color=MODEL_COLORS["Ensemble"],
            label="Ensemble" if include_legend else None,
        )


def _plot_validation_lines(ax, validation: pd.DataFrame, include_legend: bool = True) -> None:
    if validation is None or validation.empty or "Actual" not in validation.columns:
        ax.text(0.5, 0.5, "No validation data available", ha="center", va="center", transform=ax.transAxes)
        return

    model_cols = [c for c in validation.columns if c not in ["date", "Actual", "Ensemble"]]
    dates = pd.to_datetime(validation["date"])

    ax.fill_between(dates, _millions(validation["Actual"].values), alpha=0.18, color="#3a3a3a")
    ax.plot(
        dates,
        _millions(validation["Actual"].values),
        color="#2f2f2f",
        linewidth=2.2,
        label="Actual validation" if include_legend else None,
    )

    if model_cols:
        pred_min = validation[model_cols].min(axis=1).values
        pred_max = validation[model_cols].max(axis=1).values
        ax.fill_between(
            dates,
            _millions(pred_min),
            _millions(pred_max),
            alpha=0.12,
            color="#d62728",
            label="Validation model range" if include_legend else None,
        )

    for col in model_cols:
        ax.plot(
            dates,
            _millions(validation[col].values),
            linestyle="--",
            linewidth=1.4,
            alpha=0.70,
            color=MODEL_COLORS.get(col, "#777777"),
            label=f"{col} validation" if include_legend else None,
        )

    if "Ensemble" in validation.columns:
        ax.plot(
            dates,
            _millions(validation["Ensemble"].values),
            linewidth=2.8,
            marker="o",
            markersize=4,
            color=MODEL_COLORS["Ensemble"],
            label="Ensemble validation" if include_legend else None,
        )


def _draw_metrics_table(ax, metrics: pd.DataFrame) -> None:
    ax.axis("off")
    if metrics.empty:
        ax.text(0.5, 0.5, "No validation metrics", ha="center", va="center")
        return

    display_metrics = metrics.copy()
    display_metrics["RMSE"] = display_metrics["rmse"].map(lambda x: "" if pd.isna(x) else f"{x / 1_000_000:.2f}M")
    display_metrics["MAE"] = display_metrics["mae"].map(lambda x: "" if pd.isna(x) else f"{x / 1_000_000:.2f}M")
    display_metrics = display_metrics[["model", "RMSE", "MAE"]].head(6)
    table = ax.table(
        cellText=display_metrics.values,
        colLabels=["Model", "RMSE", "MAE"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_facecolor("#edf1fb")
            cell.set_text_props(weight="bold")
    best = metrics.iloc[0]["model"]
    ax.set_title(f"Validation metrics\nBest: {best}", fontweight="bold")


def plot_monthly(series: pd.Series, forecast: pd.DataFrame, metrics: pd.DataFrame, validation: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    })

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[2.1, 1.15], width_ratios=[1.25, 1.25, 1.0])
    ax_main = fig.add_subplot(grid[0, :])
    ax_zoom = fig.add_subplot(grid[1, :2])
    ax_table = fig.add_subplot(grid[1, 2])

    fig.suptitle("Rejsekort — Monthly Journeys & 12-Month Forecast", fontsize=18, fontweight="bold")

    ax_main.fill_between(series.index, _millions(series.values), alpha=0.18, color="#3a3a3a")
    ax_main.plot(series.index, _millions(series.values), color="#2f2f2f", linewidth=2.0, label="Historical")
    ax_main.axvline(series.index.max(), linestyle=":", linewidth=1.4, color="#777777", label="Forecast starts")
    _plot_forecast_lines(ax_main, series, forecast, include_legend=True)
    ax_main.set_title("Full monthly history with stabilized forecast range")
    ax_main.set_ylabel("Journeys (millions)")
    ax_main.grid(True, alpha=0.22)
    ax_main.legend(ncol=4, loc="upper left", frameon=True)
    ax_main.xaxis.set_major_locator(mdates.YearLocator(1))
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    _plot_validation_lines(ax_zoom, validation, include_legend=False)
    ax_zoom.set_title("Validation close-up only: actual vs model validation predictions")
    ax_zoom.set_ylabel("Journeys (millions)")
    _format_date_axis(ax_zoom, monthly=True)

    _draw_metrics_table(ax_table, metrics)

    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    _copy_readme_asset()


def main() -> None:
    RESULTS_MONTHLY.mkdir(parents=True, exist_ok=True)

    monthly = load_monthly_data()
    series = clean_regular_series(monthly)

    clean_df = series.reset_index()
    clean_df.columns = ["date", "journeys"]
    clean_df.to_csv(OUT_CLEAN, index=False)

    forecast, metrics, validation = build_forecasts(series, horizon=12)
    forecast.to_csv(OUT_FORECAST, index=False)
    metrics.to_csv(OUT_METRICS, index=False)
    plot_monthly(series, forecast, metrics, validation)


if __name__ == "__main__":
    main()
