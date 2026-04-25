import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

WORKDIR = Path(__file__).resolve().parent
MODEL_SCRIPT = WORKDIR / "model_c_plus_hybrid_multi_asset_defensive_compare.py"
ALERT_SCRIPT = WORKDIR / "live_model_telegram_alert_v2.py"
LOG_FILE = WORKDIR / "run_model_and_alert_log.txt"

def write_log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

def run_script(script_path: Path, script_name: str) -> None:
    write_log(f"Starting {script_name}: {script_path}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(WORKDIR),
        text=True,
        capture_output=True,
    )

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed")

def main():
    write_log("=== AUTO MODEL + ALERT RUN START ===")

    run_script(MODEL_SCRIPT, "model")
    run_script(ALERT_SCRIPT, "alert")

    write_log("=== COMPLETE ===")

if __name__ == "__main__":
    main()
