import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime


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
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(result.stdout + "\n")

    if result.stderr:
        print(result.stderr)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(result.stderr + "\n")

    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {result.returncode}")

    write_log(f"Finished {script_name} successfully")


def main() -> None:
    write_log("=== AUTO MODEL + ALERT RUN START ===")
    write_log(f"Working directory: {WORKDIR}")
    write_log(f"Python executable: {sys.executable}")

    # Telegram is optional: the alert script should print a message if not configured.
    if os.environ.get("BOT_TOKEN") and os.environ.get("CHAT_ID"):
        write_log("Telegram environment variables found.")
    else:
        write_log("Telegram environment variables not found. Alert script may skip Telegram send.")

    run_script(MODEL_SCRIPT, "production model")
    run_script(ALERT_SCRIPT, "Telegram alert")

    write_log("=== AUTO MODEL + ALERT RUN COMPLETE ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        write_log(f"ERROR: {e}")
        raise
