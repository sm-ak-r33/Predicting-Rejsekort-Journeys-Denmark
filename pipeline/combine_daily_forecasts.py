"""Backward-compatible wrapper for the combined daily forecast stage."""
from daily_forecast import load_actuals, load_forecasts, load_metrics, plot_daily

if __name__ == "__main__":
    actual = load_actuals()
    forecasts = load_forecasts()
    metrics = load_metrics()
    plot_daily(actual, forecasts, metrics)
