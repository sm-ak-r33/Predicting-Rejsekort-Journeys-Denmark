from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = Path(__file__).resolve().parent

SOURCE = PIPELINE / "rejsekort_latest_year_daily_export.xlsx"
TARGET = ROOT / "Data(update).xlsx"

if not SOURCE.exists():
    raise FileNotFoundError(f"Daily export was not created: {SOURCE}")

# Reuse the robust daily parsing already used by preprocessing.py
from pipeline.preprocessing import _read_excel, clean_dataframe, append_and_update

try:
    # New latest-year export from the JS scraper
    latest = clean_dataframe(_read_excel(SOURCE))

    # Existing retained update workbook from previous years, if present
    existing = clean_dataframe(_read_excel(TARGET)) if TARGET.exists() else pd.DataFrame(
        columns=["date", "passengers"]
    )

    # Keep old years, replace overlapping dates with the newest scraped values
    retained = append_and_update(existing, latest)

    retained.to_excel(TARGET, index=False)

    print(f"Saved retained daily update workbook: {TARGET} ({len(retained)} rows)")
    print(f"Date range: {retained['date'].min().date()} to {retained['date'].max().date()}")

except Exception as exc:
    raise RuntimeError(f"Could not retain/merge daily update workbook: {TARGET}") from exc