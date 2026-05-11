from pathlib import Path
import logging
import warnings

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd

from forecast_utils import apply_weekday_floor

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DAILY = ROOT / "results" / "daily"
DATA = RESULTS_DAILY / "data_cleaned.csv"

FORECAST_FILES = [
    (RESULTS_DAILY / "sarima_forecast.csv", "sarima"),
    (RESULTS_DAILY / "autoarima_forecast.csv", "autoarima"),
    (RESULTS_DAILY / "prophet_forecast.csv", "prophet"),
    (RESULTS_DAILY / "bilstm_forecast.csv", "bilstm"),
]

METRIC_FILES = [
    RESULTS_DAILY / "sarima_metrics.csv",
    RESULTS_DAILY / "autoarima_metrics.csv",
    RESULTS_DAILY / "prophet_metrics.csv",
    RESULTS_DAILY / "bilstm_metrics.csv",
]

OUT_FORECAST = RESULTS_DAILY / "daily_forecast.csv"
OUT_METRICS = RESULTS_DAILY / "daily_model_metrics.csv"
OUT_PNG = RESULTS_DAILY / "daily_trends.png"

MODEL_LABELS = {
    "sarima": "SARIMA",
    "autoarima": "AutoARIMA",
    "prophet": "Prophet",
    "bilstm": "BiLSTM",
    "ensemble_mean": "Ensemble mean",
}


def _to_datetime(series):
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().sum() == 0:
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return parsed


def clean_forecast_frame(path, model_name):
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "date" not in df.columns:
        possible_dates = [c for c in df.columns if c == "ds" or "date" in c]
        if possible_dates:
            df = df.rename(columns={possible_dates[0]: "date"})
        else:
            raise ValueError("No date column found in %s" % path.name)

    df["date"] = _to_datetime(df["date"])
    df = df.dropna(subset=["date"])

    if model_name not in df.columns:
        candidates = [
            c for c in df.columns
            if c != "date" and pd.to_numeric(df[c], errors="coerce").notna().sum() > 0
        ]
        if not candidates:
            raise ValueError("No numeric forecast value column found in %s" % path.name)
        df = df.rename(columns={candidates[0]: model_name})

    df[model_name] = pd.to_numeric(df[model_name], errors="coerce")
    df = df.dropna(subset=[model_name])
    return df[["date", model_name]].drop_duplicates("date").sort_values("date")


def load_forecasts(actual=None):
    combined = None

    for path, model_name in FORECAST_FILES:
        if not path.exists():
            continue
        df = clean_forecast_frame(path, model_name)
        if df.empty:
            continue
        combined = df if combined is None else combined.merge(df, on="date", how="outer")

    if combined is None or combined.empty:
        expected = ", ".join([str(p) for p, _ in FORECAST_FILES])
        raise FileNotFoundError(
            "No usable daily forecast files found. Expected at least one of: %s. "
            "Run selected_arima, autoarima, prophet, and bilstm first." % expected
        )

    combined = combined.sort_values("date")
    model_cols = [c for c in combined.columns if c != "date"]

    if actual is not None and not actual.empty and model_cols:
        history = actual.set_index("date")["passengers"].sort_index()
        for col in model_cols:
            mask = combined[col].notna()
            if mask.any():
                combined.loc[mask, col] = apply_weekday_floor(
                    combined.loc[mask, col].values,
                    combined.loc[mask, "date"],
                    history,
                )

    if model_cols:
        combined["ensemble_mean"] = combined[model_cols].mean(axis=1, skipna=True)
        if actual is not None and not actual.empty:
            history = actual.set_index("date")["passengers"].sort_index()
            mask = combined["ensemble_mean"].notna()
            combined.loc[mask, "ensemble_mean"] = apply_weekday_floor(
                combined.loc[mask, "ensemble_mean"].values,
                combined.loc[mask, "date"],
                history,
            )

    combined.to_csv(OUT_FORECAST, index=False)
    return combined


def load_metrics():
    rows = []
    for path in METRIC_FILES:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        if "algorithm" not in df.columns:
            df.insert(0, "algorithm", path.stem.replace("_metrics", ""))
        rows.append(df)

    if not rows:
        metrics = pd.DataFrame(columns=["algorithm", "rmse", "mae", "mape", "r2"])
    else:
        metrics = pd.concat(rows, ignore_index=True, sort=False)
        for col in ["rmse", "mae", "mape", "r2"]:
            if col in metrics.columns:
                metrics[col] = pd.to_numeric(metrics[col], errors="coerce")
        if "rmse" in metrics.columns:
            metrics = metrics.sort_values("rmse", na_position="last")

    metrics.to_csv(OUT_METRICS, index=False)
    return metrics


def load_actuals():
    if not DATA.exists():
        return pd.DataFrame(columns=["date", "passengers"])

    df = pd.read_csv(DATA)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "date" not in df.columns:
        raise ValueError("results/daily/data_cleaned.csv must contain a date column.")
    if "passengers" not in df.columns:
        numeric_cols = [c for c in df.columns if c != "date"]
        if not numeric_cols:
            raise ValueError("results/daily/data_cleaned.csv must contain passengers or another numeric value column.")
        df = df.rename(columns={numeric_cols[0]: "passengers"})

    df["date"] = _to_datetime(df["date"])
    df["passengers"] = pd.to_numeric(df["passengers"], errors="coerce")
    df = df.dropna(subset=["date", "passengers"])
    if df.empty:
        return pd.DataFrame(columns=["date", "passengers"])

    df = df.groupby("date", as_index=False)["passengers"].sum().sort_values("date")
    return df


def _bridge_forecast(actual_tail, forecast_df, col):
    plotted = forecast_df[["date", col]].dropna().sort_values("date").copy()
    if plotted.empty:
        return plotted
    if actual_tail is not None and not actual_tail.empty:
        last_actual = actual_tail.iloc[-1]
        bridge = pd.DataFrame({"date": [last_actual["date"]], col: [last_actual["passengers"]]})
        plotted = pd.concat([bridge, plotted], ignore_index=True)
    return plotted


def _best_model_name(metrics):
    if metrics.empty or "algorithm" not in metrics.columns or "rmse" not in metrics.columns:
        return None
    ranked = metrics.dropna(subset=["rmse"])
    if ranked.empty:
        return None
    return str(ranked.iloc[0]["algorithm"])


def _format_axis(ax):
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.grid(True, alpha=0.25)


def plot_daily(actual, forecasts, metrics):
    fig, ax = plt.subplots(figsize=(15, 8))

    actual_tail = pd.DataFrame(columns=["date", "passengers"])
    if actual is not None and not actual.empty:
        last_date = actual["date"].max()
        start_date = last_date - pd.Timedelta(days=365)
        actual_tail = actual[actual["date"] >= start_date].copy()
        ax.plot(
            actual_tail["date"],
            actual_tail["passengers"],
            linewidth=1.4,
            alpha=0.75,
            label="Actual daily passengers",
        )

        smoothed = actual_tail.set_index("date")["passengers"].rolling(7, min_periods=1).mean()
        ax.plot(smoothed.index, smoothed.values, linewidth=2.4, label="Actual 7-day average")

    model_cols = [c for c in forecasts.columns if c != "date" and c != "ensemble_mean"]
    for col in model_cols:
        plotted = _bridge_forecast(actual_tail, forecasts, col)
        if plotted.empty:
            continue
        ax.plot(
            plotted["date"],
            plotted[col],
            linestyle="--",
            linewidth=1.6,
            marker="o",
            markersize=2.8,
            label="%s forecast" % MODEL_LABELS.get(col, col),
        )

    if "ensemble_mean" in forecasts.columns:
        plotted = _bridge_forecast(actual_tail, forecasts, "ensemble_mean")
        if not plotted.empty:
            ax.plot(
                plotted["date"],
                plotted["ensemble_mean"],
                linewidth=2.6,
                marker="o",
                markersize=3.2,
                label="Ensemble mean forecast",
            )

    if actual is not None and not actual.empty:
        forecast_dates = forecasts["date"].dropna()
        if not forecast_dates.empty:
            ax.axvline(actual["date"].max(), linestyle=":", linewidth=1.3, label="Forecast starts")

    best = _best_model_name(metrics)
    if best:
        title = "Rejsekort daily trend and 30-day forecast — best validation RMSE: %s" % best
    else:
        title = "Rejsekort daily trend and 30-day forecast"
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Passenger journeys")
    _format_axis(ax)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    RESULTS_DAILY.mkdir(parents=True, exist_ok=True)
    actual = load_actuals()
    forecasts = load_forecasts(actual)
    metrics = load_metrics()
    plot_daily(actual, forecasts, metrics)
