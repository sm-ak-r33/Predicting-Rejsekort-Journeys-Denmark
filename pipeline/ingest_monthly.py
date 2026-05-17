import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = Path(__file__).resolve().parent
RESULTS_MONTHLY = ROOT / "results" / "monthly"
RAW_MONTHLY = RESULTS_MONTHLY / "raw"
DEBUG_MONTHLY = RAW_MONTHLY / "debug"

CHART_EXPORT = RAW_MONTHLY / "rejsekort_monthly_chart_export.xlsx"
EXTENSION_EXPORT = RAW_MONTHLY / "rejsekort_monthly_export_extension_data.xlsx"
HENTDATA_EXPORT = RAW_MONTHLY / "rejsekort_hentdata.xlsx"
OUT_CLEAN = RESULTS_MONTHLY / "monthly_cleaned.csv"

MONTH_MAP = {
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


def _ensure_dirs() -> None:
    RAW_MONTHLY.mkdir(parents=True, exist_ok=True)
    DEBUG_MONTHLY.mkdir(parents=True, exist_ok=True)


def _run_node(script_name: str, expected_any) -> None:
    script = PIPELINE / script_name
    result = subprocess.run(["node", str(script)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed:\n{result.stderr.strip() or result.stdout.strip()}")

    expected = [Path(p) for p in expected_any]
    deadline = time.time() + 30
    while not any(path.exists() for path in expected) and time.time() < deadline:
        time.sleep(1)

    if not any(path.exists() for path in expected):
        raise FileNotFoundError("Expected monthly export not found. Checked: " + ", ".join(str(p) for p in expected))


def _to_number(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip().replace("\u00a0", " ")
    text = text.replace(".", "").replace(",", ".")
    import re
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else np.nan


def _parse_month_label(value, current_year=None):
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        return pd.Timestamp(value.year, value.month, 1)

    import re
    text = str(value).strip().lower().replace("\u00a0", " ")
    dt = pd.to_datetime(text, dayfirst=True, errors="coerce")
    if pd.notna(dt):
        return pd.Timestamp(dt.year, dt.month, 1)

    year_match = re.search(r"(20\d{2}|19\d{2})", text)
    year = int(year_match.group(1)) if year_match else current_year
    if year is None:
        return pd.NaT

    for name, month in MONTH_MAP.items():
        if re.search(rf"\b{name}\b", text):
            return pd.Timestamp(year, month, 1)

    clean = text.replace(str(year), "")
    month_match = re.search(r"(?:^|\D)(1[0-2]|0?[1-9])(?:\D|$)", clean)
    if month_match:
        return pd.Timestamp(year, int(month_match.group(1)), 1)
    return pd.NaT


def _flatten_columns(columns) -> list:
    flattened = []
    for col in columns:
        if isinstance(col, tuple):
            parts = [str(x).strip() for x in col if str(x).strip() and not str(x).startswith("Unnamed")]
            flattened.append(" ".join(parts))
        else:
            flattened.append(str(col).strip())
    return flattened


def _specific_chart_export(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    records = []
    current_year = None

    for _, row in raw.iterrows():
        label = str(row.iloc[0]).strip()
        value = row.iloc[1] if len(row) > 1 else np.nan
        if label.isdigit() and len(label) == 4:
            current_year = int(label)
            continue
        dt = _parse_month_label(label, current_year=current_year)
        journeys = _to_number(value)
        if pd.notna(dt) and pd.notna(journeys) and journeys > 0:
            records.append({"date": dt, "journeys": journeys})

    return pd.DataFrame(records)


def _specific_extension_export(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    if raw.empty:
        return pd.DataFrame(columns=["date", "journeys"])

    year_cols = {}
    for col_idx, header in enumerate(raw.iloc[0].tolist()):
        year = _to_number(header)
        if pd.notna(year) and 2000 <= int(year) <= 2100:
            year_cols[col_idx] = int(year)

    records = []
    for row_idx in range(1, len(raw)):
        row = raw.iloc[row_idx]
        month_name = str(row.iloc[0]).strip().lower()
        if month_name not in MONTH_MAP:
            continue
        month = MONTH_MAP[month_name]
        for col_idx, year in year_cols.items():
            journeys = _to_number(row.iloc[col_idx])
            if pd.notna(journeys) and journeys > 0:
                records.append({"date": pd.Timestamp(year, month, 1), "journeys": journeys})

    return pd.DataFrame(records)


def _generic_extract(path: Path) -> pd.DataFrame:
    pieces = []
    excel = pd.ExcelFile(path)

    for sheet in excel.sheet_names:
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        for header in range(min(12, len(raw))):
            try:
                frame = pd.read_excel(path, sheet_name=sheet, header=header)
                frame.columns = _flatten_columns(frame.columns)
            except Exception:
                continue

            date_candidates = []
            value_candidates = []
            for col in frame.columns:
                low = str(col).lower()
                parsed_dates = frame[col].apply(_parse_month_label)
                nums = frame[col].apply(_to_number)

                date_score = parsed_dates.notna().sum()
                value_score = nums.notna().sum()
                if any(token in low for token in ["måned", "maaned", "month", "dato", "date", "år", "year"]):
                    date_score += 5
                if any(token in low for token in ["personrejser", "passenger", "antal", "rejser", "journeys", "total"]):
                    value_score += 5

                date_candidates.append((date_score, col, parsed_dates))
                value_candidates.append((value_score, col, nums))

            date_candidates.sort(reverse=True, key=lambda x: x[0])
            value_candidates.sort(reverse=True, key=lambda x: x[0])

            for _, date_col, dates in date_candidates[:4]:
                for _, value_col, values in value_candidates[:8]:
                    if date_col == value_col:
                        continue
                    temp = pd.DataFrame({"date": dates, "journeys": values}).dropna()
                    temp = temp[temp["journeys"] > 0]
                    if len(temp) >= 6:
                        pieces.append(temp[["date", "journeys"]])
                        break
                if pieces:
                    break

    if not pieces:
        return pd.DataFrame(columns=["date", "journeys"])
    return pd.concat(pieces, ignore_index=True)


def _read_export(path: Path, preferred: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "journeys"])

    parsers = [_specific_chart_export, _generic_extract] if preferred == "chart" else [_specific_extension_export, _generic_extract]
    for parser in parsers:
        try:
            df = parser(path)
            if not df.empty:
                df = df[["date", "journeys"]].copy()
                df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
                df["journeys"] = df["journeys"].apply(_to_number)
                df = df.dropna(subset=["date", "journeys"])
                df = df[df["journeys"] > 0]
                if not df.empty:
                    return df
        except Exception:
            continue
    return pd.DataFrame(columns=["date", "journeys"])


def _combine(chart_df: pd.DataFrame, extension_df: pd.DataFrame) -> pd.DataFrame:
    chart_df = chart_df.copy()
    extension_df = extension_df.copy()

    if not chart_df.empty and not extension_df.empty:
        old_history = chart_df[chart_df["date"].dt.year < 2022]
        newer_history = extension_df[extension_df["date"].dt.year >= 2022]
        combined = pd.concat([old_history, newer_history], ignore_index=True)
        if len(combined) >= 24:
            return _finalize(combined)

    combined = pd.concat([chart_df, extension_df], ignore_index=True)
    return _finalize(combined)


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("No usable monthly rows were parsed from the downloaded Excel files.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    df["journeys"] = df["journeys"].apply(_to_number)
    df = df.dropna(subset=["date", "journeys"])
    df = df[df["journeys"] > 0]
    df = (
        df.sort_values("date")
        .drop_duplicates("date", keep="last")
        .groupby("date", as_index=False)["journeys"]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )
    if len(df) < 24:
        raise ValueError(f"Only {len(df)} monthly observations found; need at least 24 for the monthly forecast.")
    return df


def run() -> None:
    _ensure_dirs()

    _run_node("download-monthly.js", [CHART_EXPORT])
    _run_node("download-mothly-script 2.js", [EXTENSION_EXPORT, HENTDATA_EXPORT])

    # If the UI's top-right Hent data path succeeds, keep a canonical copy for the parser/DVC output.
    if not EXTENSION_EXPORT.exists() and HENTDATA_EXPORT.exists():
        shutil.copy2(HENTDATA_EXPORT, EXTENSION_EXPORT)

    chart = _read_export(CHART_EXPORT, preferred="chart")
    extension = _read_export(EXTENSION_EXPORT, preferred="extension")
    monthly = _combine(chart, extension)

    RESULTS_MONTHLY.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUT_CLEAN, index=False)
    print(f"Saved monthly cleaned data: {OUT_CLEAN} ({len(monthly)} rows)")


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
