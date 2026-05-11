from pathlib import Path
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = Path(__file__).resolve().parent
SOURCE = PIPELINE / "rejsekort_latest_year_daily_export.xlsx"
TARGET = ROOT / "Data(update).xlsx"

if not SOURCE.exists():
    raise FileNotFoundError(f"Daily export was not created: {SOURCE}")

try:
    workbook = pd.ExcelFile(SOURCE)
    if not workbook.sheet_names:
        raise ValueError("Daily export contains no worksheets")
except Exception as exc:
    raise RuntimeError(f"Daily export could not be opened as Excel: {SOURCE}") from exc

shutil.copy2(SOURCE, TARGET)
