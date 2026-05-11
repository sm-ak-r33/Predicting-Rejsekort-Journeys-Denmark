from pathlib import Path
from datetime import date, datetime
import re
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Could not infer format.*")

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "Data.xlsx"
UPDATE = ROOT / "Data(update).xlsx"
RESULTS_DAILY = ROOT / "results" / "daily"
OUT = RESULTS_DAILY / "data_cleaned.csv"

DATE_ALIASES = [
    "Afgangsdato År - Måned - Dato",
    "Afgangsdato Aar - Maaned - Dato",
    "Afgangsdato År Måned Dato",
    "Afgangsdato",
    "Dato",
    "date",
    "Date",
]
VALUE_ALIASES = [
    "Antal Personrejser",
    "Antal personrejser",
    "Personrejser",
    "Antal rejser",
    "passengers",
    "Passengers",
]

DANISH_MONTHS = {
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

# The daily export sometimes contains hierarchy totals such as a whole year/month.
# Those totals are not real daily observations and create the huge spikes seen in plots.
MAX_REASONABLE_DAILY_PASSENGERS = 5_000_000


def _flatten_columns(columns):
    cleaned = []
    for col in columns:
        if isinstance(col, tuple):
            parts = [str(x).strip() for x in col if str(x).strip() and not str(x).startswith("Unnamed")]
            cleaned.append(" ".join(parts))
        else:
            cleaned.append(str(col).strip())
    return cleaned


def _normalize_text(value):
    return str(value).strip().replace("\u00a0", " ")


def parse_number(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = _normalize_text(value)
    text = text.replace(".", "").replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else np.nan


def _is_excel_serial(value):
    try:
        number = float(value)
    except Exception:
        return False
    # Excel date serials for 2000-01-01 to roughly 2070 are in this range.
    # Four-digit years like 2024/2025 must be rejected because they are hierarchy totals.
    return 20_000 <= number <= 65_000


def parse_daily_date(value):
    """Return a real daily date, rejecting year/month hierarchy labels.

    The TARGIT Excel export can include expandable hierarchy rows:
    2024 -> Jan -> 01-01-2024. Pandas/dateutil may convert the year row
    "2024" to 2024-01-01, which makes annual totals look like one daily value.
    This parser only accepts genuine day-level dates.
    """
    if pd.isna(value):
        return pd.NaT

    if isinstance(value, pd.Timestamp):
        return value.normalize()

    if isinstance(value, (datetime, date)):
        return pd.Timestamp(value).normalize()

    if isinstance(value, (int, float, np.integer, np.floating)):
        if not _is_excel_serial(value):
            return pd.NaT
        return pd.to_datetime(value, unit="D", origin="1899-12-30", errors="coerce").normalize()

    text = _normalize_text(value)
    if not text or text.lower() in {"nan", "nat", "none", "total", "i alt", "grand total"}:
        return pd.NaT

    low = text.lower().strip()
    low = re.sub(r"\s+", " ", low)

    # Reject pure hierarchy labels before any flexible parsing.
    if re.fullmatch(r"\d{4}(?:\.0)?", low):
        return pd.NaT
    if low in DANISH_MONTHS:
        return pd.NaT
    if re.fullmatch(r"(?:19|20)\d{2}[-/. ](?:0?[1-9]|1[0-2])", low):
        return pd.NaT
    if re.fullmatch(r"(?:0?[1-9]|1[0-2])[-/. ](?:19|20)\d{2}", low):
        return pd.NaT
    for month_name in DANISH_MONTHS:
        if re.fullmatch(rf"(?:19|20)\d{{2}}\s+{month_name}|{month_name}\s+(?:19|20)\d{{2}}", low):
            return pd.NaT

    # ISO/European numeric dates with day, month, and year.
    numeric_patterns = [
        (r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}(?:\s+.*)?$", True),
        (r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?:\s+.*)?$", False),
    ]
    for pattern, dayfirst in numeric_patterns:
        if re.match(pattern, low):
            parsed = pd.to_datetime(text, dayfirst=dayfirst, errors="coerce")
            return parsed.normalize() if pd.notna(parsed) else pd.NaT

    # Danish/English month-name dates such as "1 jan 2025" or "2025 jan 1".
    month_alt = "|".join(sorted(DANISH_MONTHS, key=len, reverse=True))
    m = re.search(rf"\b(\d{{1,2}})\s+({month_alt})\s+((?:19|20)\d{{2}})\b", low)
    if m:
        day = int(m.group(1))
        month = DANISH_MONTHS[m.group(2)]
        year = int(m.group(3))
        return pd.Timestamp(year, month, day)
    m = re.search(rf"\b((?:19|20)\d{{2}})\s+({month_alt})\s+(\d{{1,2}})\b", low)
    if m:
        year = int(m.group(1))
        month = DANISH_MONTHS[m.group(2)]
        day = int(m.group(3))
        return pd.Timestamp(year, month, day)

    # Numeric Excel serial stored as a string.
    if re.fullmatch(r"\d+(?:\.0)?", low) and _is_excel_serial(low):
        return pd.to_datetime(float(low), unit="D", origin="1899-12-30", errors="coerce").normalize()

    return pd.NaT


def _choose_column(df, aliases, prefer_date=False):
    normalized = {re.sub(r"\s+", " ", str(c)).strip().lower(): c for c in df.columns}
    for alias in aliases:
        key = alias.lower()
        if key in normalized:
            return normalized[key]
    best_col = None
    best_score = -1
    for col in df.columns:
        low = str(col).lower()
        if prefer_date:
            score = df[col].apply(parse_daily_date).notna().sum()
            if any(token in low for token in ["dato", "date", "måned", "maaned", "aar", "år"]):
                score += 5
        else:
            score = df[col].apply(parse_number).notna().sum()
            if any(token in low for token in ["personrejser", "rejser", "passenger", "antal"]):
                score += 5
        if score > best_score:
            best_score = score
            best_col = col
    return best_col if best_score > 0 else None


def _read_excel(path):
    if not path.exists():
        return pd.DataFrame(columns=["date", "passengers"])

    candidates = []
    workbook = pd.ExcelFile(path)
    for sheet in workbook.sheet_names:
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        for header_idx in range(min(18, len(raw))):
            try:
                df = pd.read_excel(path, sheet_name=sheet, header=header_idx)
                df.columns = _flatten_columns(df.columns)
                date_col = _choose_column(df, DATE_ALIASES, prefer_date=True)
                value_col = _choose_column(df, VALUE_ALIASES, prefer_date=False)
                if date_col is None or value_col is None or date_col == value_col:
                    continue
                daily_count = df[date_col].apply(parse_daily_date).notna().sum()
                value_count = df[value_col].apply(parse_number).notna().sum()
                score = daily_count * 2 + value_count
                candidates.append((score, df))
            except Exception:
                continue

    if not candidates:
        return pd.DataFrame(columns=["date", "passengers"])

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _remove_aggregate_spikes(df):
    if df.empty:
        return df
    cleaned = df.copy()
    cleaned = cleaned[cleaned["passengers"] > 0]
    cleaned = cleaned[cleaned["passengers"] <= MAX_REASONABLE_DAILY_PASSENGERS]

    if len(cleaned) >= 30:
        median = cleaned["passengers"].median()
        if pd.notna(median) and median > 0:
            cleaned = cleaned[cleaned["passengers"] <= max(MAX_REASONABLE_DAILY_PASSENGERS, median * 25)]
    return cleaned


def clean_dataframe(df):
    if df.empty:
        return pd.DataFrame(columns=["date", "passengers"])

    date_col = _choose_column(df, DATE_ALIASES, prefer_date=True)
    value_col = _choose_column(df, VALUE_ALIASES, prefer_date=False)
    if date_col is None or value_col is None or date_col == value_col:
        raise ValueError("Could not identify separate date/passenger columns. Found columns: %s" % list(df.columns))

    out = df[[date_col, value_col]].copy()
    out.columns = ["date", "passengers"]
    out["date"] = out["date"].apply(parse_daily_date)
    out["passengers"] = out["passengers"].apply(parse_number)
    out = out.dropna(subset=["date", "passengers"])
    out = _remove_aggregate_spikes(out)
    out = out.groupby("date", as_index=False)["passengers"].sum()
    out = _remove_aggregate_spikes(out)
    return out.sort_values("date")


def append_and_update(base, update):
    frames = [frame for frame in [base, update] if frame is not None and not frame.empty]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "passengers"])
    if df.empty:
        raise ValueError("No usable day-level rows found in Data.xlsx or Data(update).xlsx")
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    df = _remove_aggregate_spikes(df)
    return df.sort_values("date")


if __name__ == "__main__":
    RESULTS_DAILY.mkdir(parents=True, exist_ok=True)
    base = clean_dataframe(_read_excel(BASELINE))
    update = clean_dataframe(_read_excel(UPDATE))
    final = append_and_update(base, update)
    final.to_csv(OUT, index=False)
