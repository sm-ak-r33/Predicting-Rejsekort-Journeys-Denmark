import subprocess
import sys
import os
import time


def run_daily_ingest():
    script = os.path.join(os.path.dirname(__file__), "download-daily.js")
    result = subprocess.run(["node", script], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"download-daily.js failed:\n{result.stderr.strip()}")

    output = os.path.join(os.path.dirname(__file__), "..", "Data(update).xlsx")
    output = os.path.normpath(output)

    deadline = time.time() + 30
    while not os.path.exists(output) and time.time() < deadline:
        time.sleep(1)

    if not os.path.exists(output):
        raise FileNotFoundError(f"Expected output not found: {output}")

    size = os.path.getsize(output)
    if size < 1024:
        raise ValueError(f"Output file suspiciously small ({size} bytes): {output}")


if __name__ == "__main__":
    try:
        run_daily_ingest()
    except Exception as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
