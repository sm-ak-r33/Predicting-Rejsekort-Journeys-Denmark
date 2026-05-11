from __future__ import annotations

import numpy as np
import pandas as pd


def finite_array(values):
    arr = np.asarray(values, dtype=float)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    return arr


def positive_history(series):
    hist = pd.Series(series).copy()
    hist = pd.to_numeric(hist, errors="coerce").dropna()
    hist = hist[hist > 0]
    return hist.astype(float)


def apply_positive_history_floor(values, history, quantile=0.05, multiplier=0.70, minimum=1.0):
    """Keep forecasts positive without forcing them to an unrealistic constant.

    ARIMA-family models can occasionally emit negative values. Previous versions
    clipped those negatives to zero, which made Sunday/weekend forecasts look as
    if no Rejsekort journeys would happen. This helper replaces invalid/negative
    forecasts with a conservative historical floor based on recent positive
    observations.
    """
    arr = finite_array(values)
    hist = positive_history(history)
    if hist.empty:
        floor = minimum
    else:
        floor = float(hist.quantile(quantile)) * float(multiplier)
        if not np.isfinite(floor) or floor <= 0:
            floor = float(hist.min())
        floor = max(float(minimum), floor)
    arr = np.where(np.isfinite(arr), arr, floor)
    return np.maximum(arr, floor)


def apply_weekday_floor(values, future_index, history, quantile=0.10, multiplier=0.75, minimum=1.0, lookback_days=730):
    """Apply a day-of-week-aware lower bound to daily forecasts.

    The bound uses recent same-weekday observations when available and falls
    back to a conservative global floor. This keeps ARIMA forecasts from becoming
    zero while preserving normal weekend/weekday seasonality.
    """
    arr = finite_array(values)
    dates = pd.DatetimeIndex(pd.to_datetime(future_index))
    hist = positive_history(history)

    if hist.empty:
        floors = np.full(len(arr), float(minimum), dtype=float)
    else:
        if isinstance(hist.index, pd.DatetimeIndex):
            hist = hist.sort_index()
            max_date = hist.index.max()
            recent = hist[hist.index >= max_date - pd.Timedelta(days=lookback_days)]
            if recent.empty:
                recent = hist
        else:
            recent = hist

        global_floor = float(recent.quantile(0.05)) * float(multiplier)
        if not np.isfinite(global_floor) or global_floor <= 0:
            global_floor = float(recent.min())
        global_floor = max(float(minimum), global_floor)

        floor_values = []
        for date in dates:
            floor = global_floor
            if isinstance(recent.index, pd.DatetimeIndex):
                same_weekday = recent[recent.index.dayofweek == date.dayofweek]
                if len(same_weekday) >= 4:
                    candidate = float(same_weekday.quantile(quantile)) * float(multiplier)
                    if np.isfinite(candidate) and candidate > 0:
                        floor = max(global_floor, candidate)
            floor_values.append(max(float(minimum), floor))
        floors = np.asarray(floor_values, dtype=float)

    arr = np.where(np.isfinite(arr), arr, floors)
    return np.maximum(arr, floors)


def inverse_log1p_forecast(values):
    arr = finite_array(values)
    return np.expm1(arr)
