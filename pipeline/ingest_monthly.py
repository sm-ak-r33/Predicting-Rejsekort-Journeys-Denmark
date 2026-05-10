import subprocess
import sys
import os
import time
import pandas as pd


MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "Maj": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Okt": 10, "Nov": 11, "Dec": 12,
}

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def _run_node(script_name, output_file):
    script = os.path.join(os.path.dirname(__file__), script_name)
    result = subprocess.run(["node", script], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed:\n{result.stderr.strip()}")

    deadline = time.time() + 30
    while not os.path.exists(output_file) and time.time() < deadline:
        time.sleep(1)

    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Expected output not found: {output_file}")


def _find(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find any of: {candidates}")


def parse_chart_export(path):
    """
    Parses the hierarchical year → months structure produced by download-monthly.js.

    Layout (no header row is usable as-is):
      Row 0 : column headers
      Row 1 : Grand total  → skip
      Row N : "2013"       → set current_year
      Row N+1: "Jan" 1411718
      …
    """
    df_raw = pd.read_excel(path, header=None)
    records = []
    current_year = None

    for _, row in df_raw.iterrows():
        label = str(row.iloc[0]).strip()
        value = row.iloc[1]

        if label.isdigit() and len(label) == 4:
            current_year = int(label)
        elif label in MONTH_MAP and current_year is not None:
            try:
                records.append(
                    {"year": current_year, "month": MONTH_MAP[label], "journeys": int(value)}
                )
            except (ValueError, TypeError):
                pass

    return pd.DataFrame(records)


def parse_extension_data(path):
    """
    Parses the cross-tabular (wide) format produced by download-mothly-script 2.js.

    Layout:
      Row 0 : "Afgangsdato Måned", "Total", 2022, 2023, 2024, 2025, 2026
      Row 1 : totals → skip
      Rows 2-13 : Jan … Dec with one value per year column
    """
    df_raw = pd.read_excel(path, header=None)
    header_row = df_raw.iloc[0].tolist()

    year_cols = {}
    for col_idx, h in enumerate(header_row):
        try:
            y = int(float(str(h)))
            if 2000 <= y <= 2100:
                year_cols[col_idx] = y
        except (ValueError, TypeError):
            pass

    records = []
    for row_idx in range(2, len(df_raw)):
        row = df_raw.iloc[row_idx]
        month_name = str(row.iloc[0]).strip()
        if month_name not in MONTH_MAP:
            continue
        for col_idx, year in year_cols.items():
            val = row.iloc[col_idx]
            if pd.notna(val):
                try:
                    records.append(
                        {"year": year, "month": MONTH_MAP[month_name], "journeys": int(val)}
                    )
                except (ValueError, TypeError):
                    pass

    return pd.DataFrame(records)


def run():
    chart_dest = os.path.join(ROOT, "rejsekort_monthly_chart_export.xlsx")
    ext_dest = os.path.join(ROOT, "rejsekort_monthly_export_extension_data.xlsx")

    _run_node("download-monthly.js", chart_dest)
    _run_node("download-mothly-script 2.js", ext_dest)

    chart_path = _find([
        chart_dest,
        os.path.join(ROOT, "data", "rejsekort_monthly_chart_export.xlsx"),
    ])
    ext_path = _find([
        ext_dest,
        os.path.join(ROOT, "data", "rejsekort_monthly_export_extension_data.xlsx"),
    ])

    df_chart = parse_chart_export(chart_path)
    df_ext = parse_extension_data(ext_path)

    # Strategy: download-monthly.js covers 2013-2025 (full history).
    # download-mothly-script 2.js covers 2022-2026 (adds current year).
    # Use chart for pre-2022, extension for 2022+ so that 2026 partial data is included.
    df_old = df_chart[df_chart["year"] < 2022].copy()
    df_new = df_ext[df_ext["year"] >= 2022].copy()

    df_combined = (
        pd.concat([df_old, df_new], ignore_index=True)
        .sort_values(["year", "month"])
        .reset_index(drop=True)
    )

    df_combined["date"] = pd.to_datetime(
        df_combined[["year", "month"]].assign(day=1)
    )
    df_combined = df_combined[["date", "journeys"]]
    df_combined.to_csv(os.path.join(ROOT, "data_monthly_cleaned.csv"), index=False)


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
