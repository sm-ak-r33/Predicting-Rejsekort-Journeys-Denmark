from pathlib import Path
import logging
import shutil
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from forecast_utils import apply_weekday_floor

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DAILY = ROOT / "results" / "daily"
DOCS_ASSETS = ROOT / "docs" / "assets"

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
OUT_README_PNG = DOCS_ASSETS / "daily_trends.png"

MODEL_LABELS = {
    "sarima": "SARIMA",
    "autoarima": "AutoARIMA",
    "prophet": "Prophet",
    "bilstm": "BiLSTM",
    "ensemble_mean": "Ensemble mean",
}
MODEL_COLORS = {
    "sarima": "#1f77b4",
    "autoarima": "#ff7f0e",
    "prophet": "#2ca02c",
    "bilstm": "#9467bd",
    "ensemble_mean": "#d62728",
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
            "No usable daily forecast files found.\n"
            "Expected at least one of: %s.\n"
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


def _millions(values):
    return np.asarray(values, dtype=float) / 1_000_000.0


def _copy_readme_asset():
    DOCS_ASSETS.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUT_PNG, OUT_README_PNG)
    print("Saved README preview plot: %s" % OUT_README_PNG)


def _format_date_axis(ax, interval=14):
    ax.grid(True, alpha=0.22)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    for label in ax.get_xticklabels():
        label.set_rotation(35)
        label.set_ha("right")


def _plot_daily_forecast_lines(ax, actual_tail, forecasts, include_legend=True):
    forecast_cols = [c for c in forecasts.columns if c != "date"]
    model_cols = [c for c in forecast_cols if c != "ensemble_mean"]

    if model_cols:
        model_values = forecasts[model_cols]
        forecast_min = model_values.min(axis=1).values
        forecast_max = model_values.max(axis=1).values
        ax.fill_between(
            forecasts["date"],
            _millions(forecast_min),
            _millions(forecast_max),
            alpha=0.12,
            color="#d62728",
            label="Model range" if include_legend else None,
        )

    for col in model_cols:
        plotted = _bridge_forecast(actual_tail, forecasts, col)
        if plotted.empty:
            continue
        ax.plot(
            plotted["date"],
            _millions(plotted[col].values),
            linestyle="--",
            linewidth=1.4,
            alpha=0.70,
            color=MODEL_COLORS.get(col, "#777777"),
            label="%s forecast" % MODEL_LABELS.get(col, col) if include_legend else None,
        )

    if "ensemble_mean" in forecasts.columns:
        plotted = _bridge_forecast(actual_tail, forecasts, "ensemble_mean")
        if not plotted.empty:
            ax.plot(
                plotted["date"],
                _millions(plotted["ensemble_mean"].values),
                linewidth=2.8,
                marker="o",
                markersize=4,
                color=MODEL_COLORS["ensemble_mean"],
                label="Ensemble mean forecast" if include_legend else None,
            )


def _draw_metrics_table(ax, metrics):
    ax.axis("off")
    if metrics.empty:
        ax.text(0.5, 0.5, "No validation metrics", ha="center", va="center")
        return

    display_metrics = metrics.copy()
    if "algorithm" not in display_metrics.columns:
        display_metrics.insert(0, "algorithm", "")
    for col in ["rmse", "mae", "mape"]:
        if col not in display_metrics.columns:
            display_metrics[col] = np.nan

    display_metrics["Model"] = display_metrics["algorithm"].map(lambda x: MODEL_LABELS.get(str(x).lower(), str(x)))
    display_metrics["RMSE"] = display_metrics["rmse"].map(lambda x: "" if pd.isna(x) else "%.2fM" % (x / 1_000_000))
    display_metrics["MAE"] = display_metrics["mae"].map(lambda x: "" if pd.isna(x) else "%.2fM" % (x / 1_000_000))
    display_metrics = display_metrics[["Model", "RMSE", "MAE"]].head(6)

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

    best = _best_model_name(metrics)
    if best:
        best = MODEL_LABELS.get(best.lower(), best)
        ax.set_title("Validation metrics\nBest: %s" % best, fontweight="bold")
    else:
        ax.set_title("Validation metrics", fontweight="bold")


def plot_daily(actual, forecasts, metrics):
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

    best = _best_model_name(metrics)
    best_label = MODEL_LABELS.get(best.lower(), best) if best else None
    if best_label:
        fig.suptitle("Rejsekort — Daily Journeys & 30-Day Forecast | Best validation RMSE: %s" % best_label, fontsize=18, fontweight="bold")
    else:
        fig.suptitle("Rejsekort — Daily Journeys & 30-Day Forecast", fontsize=18, fontweight="bold")

    actual_tail = pd.DataFrame(columns=["date", "passengers"])
    if actual is not None and not actual.empty:
        last_date = actual["date"].max()
        start_date = last_date - pd.Timedelta(days=365)
        actual_tail = actual[actual["date"] >= start_date].copy()
        ax_main.fill_between(actual_tail["date"], _millions(actual_tail["passengers"].values), alpha=0.18, color="#3a3a3a")
        ax_main.plot(actual_tail["date"], _millions(actual_tail["passengers"].values), color="#2f2f2f", linewidth=1.4, alpha=0.72, label="Actual daily")
        smoothed = actual_tail.set_index("date")["passengers"].rolling(7, min_periods=1).mean()
        ax_main.plot(smoothed.index, _millions(smoothed.values), color="#111111", linewidth=2.2, label="Actual 7-day average")

        forecast_dates = forecasts["date"].dropna()
        if not forecast_dates.empty:
            ax_main.axvline(last_date, linestyle=":", linewidth=1.4, color="#777777", label="Forecast starts")

    _plot_daily_forecast_lines(ax_main, actual_tail, forecasts, include_legend=True)
    ax_main.set_title("Latest daily history with forecast range")
    ax_main.set_ylabel("Journeys (millions)")
    ax_main.grid(True, alpha=0.22)
    ax_main.legend(ncol=4, loc="upper left", frameon=True)
    ax_main.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    for label in ax_main.get_xticklabels():
        label.set_rotation(35)
        label.set_ha("right")

    if actual_tail is not None and not actual_tail.empty:
        zoom_start = actual_tail["date"].max() - pd.Timedelta(days=90)
        zoom_actual = actual_tail[actual_tail["date"] >= zoom_start].copy()
        ax_zoom.fill_between(zoom_actual["date"], _millions(zoom_actual["passengers"].values), alpha=0.18, color="#3a3a3a")
        ax_zoom.plot(zoom_actual["date"], _millions(zoom_actual["passengers"].values), color="#2f2f2f", linewidth=1.4, alpha=0.72, label="Actual daily")
        zoom_smoothed = zoom_actual.set_index("date")["passengers"].rolling(7, min_periods=1).mean()
        ax_zoom.plot(zoom_smoothed.index, _millions(zoom_smoothed.values), color="#111111", linewidth=2.2, label="Actual 7-day average")
        ax_zoom.axvline(actual_tail["date"].max(), linestyle=":", linewidth=1.4, color="#777777")
    else:
        zoom_actual = actual_tail

    _plot_daily_forecast_lines(ax_zoom, zoom_actual, forecasts, include_legend=False)
    ax_zoom.set_title("Recent close-up + 30-day forecast")
    ax_zoom.set_ylabel("Journeys (millions)")
    _format_date_axis(ax_zoom, interval=14)

    _draw_metrics_table(ax_table, metrics)

    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    _copy_readme_asset()


if __name__ == "__main__":
    RESULTS_DAILY.mkdir(parents=True, exist_ok=True)
    actual = load_actuals()
    forecasts = load_forecasts(actual)
    metrics = load_metrics()
    plot_daily(actual, forecasts, metrics)
