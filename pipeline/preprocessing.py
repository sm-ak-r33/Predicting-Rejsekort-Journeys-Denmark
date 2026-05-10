from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "Data.xlsx"
UPDATE = ROOT / "Data(update).xlsx"
OUT = ROOT / "data_cleaned.csv"

DATE_ALIASES = [
    "Afgangsdato År - Måned - Dato",
    "Afgangsdato",
    "Dato",
    "date",
    "Date",
]
VALUE_ALIASES = [
    "Antal Personrejser",
    "Personrejser",
    "Antal rejser",
    "passengers",
    "Passengers",
]


def _flatten_columns(columns):
    cleaned = []
    for col in columns:
        if isinstance(col, tuple):
            parts = [str(x).strip() for x in col if str(x).strip() and not str(x).startswith("Unnamed")]
            cleaned.append(" ".join(parts))
        else:
            cleaned.append(str(col).strip())
    return cleaned


def _read_excel(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "passengers"])
    raw = pd.read_excel(path, header=None)
    header_idx = 0
    for idx in range(min(15, len(raw))):
        row_text = " ".join(raw.iloc[idx].astype(str).fillna(""))
        if any(alias in row_text for alias in DATE_ALIASES) or re.search(r"\d{2}-\d{2}-\d{4}", row_text):
            header_idx = idx
            break
    df = pd.read_excel(path, header=header_idx)
    df.columns = _flatten_columns(df.columns)
    return df


def _choose_column(df: pd.DataFrame, aliases, prefer_date=False):
    normalized = {re.sub(r"\s+", " ", str(c)).strip().lower(): c for c in df.columns}
    for alias in aliases:
        key = alias.lower()
        if key in normalized:
            return normalized[key]
    for col in df.columns:
        low = str(col).lower()
        if prefer_date and any(token in low for token in ["dato", "date", "måned", "aar", "år"]):
            return col
        if not prefer_date and any(token in low for token in ["personrejser", "rejser", "passenger", "antal"]):
            return col
    return None


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "passengers"])

    date_col = _choose_column(df, DATE_ALIASES, prefer_date=True)
    value_col = _choose_column(df, VALUE_ALIASES, prefer_date=False)
    if date_col is None or value_col is None:
        raise ValueError(f"Could not identify date/passenger columns. Found columns: {list(df.columns)}")

    out = df[[date_col, value_col]].copy()
    out.columns = ["date", "passengers"]
    out["date"] = pd.to_datetime(out["date"], dayfirst=True, errors="coerce")
    out["passengers"] = (
        out["passengers"].astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
    )
    out["passengers"] = pd.to_numeric(out["passengers"], errors="coerce")
    out = out.dropna(subset=["date", "passengers"])
    out = out.groupby("date", as_index=False)["passengers"].sum()
    return out


def append_and_update(base: pd.DataFrame, update: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([base, update], ignore_index=True)
    if df.empty:
        raise ValueError("No usable rows found in Data.xlsx or Data(update).xlsx")
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return df


if __name__ == "__main__":
    base = clean_dataframe(_read_excel(BASELINE))
    update = clean_dataframe(_read_excel(UPDATE))
    final = append_and_update(base, update)
    final.to_csv(OUT, index=False)
