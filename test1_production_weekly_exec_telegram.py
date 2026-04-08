import requests
import subprocess
import sys
import pandas as pd
import os

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    response = requests.post(
        url,
        data={"chat_id": CHAT_ID, "text": message},
        timeout=20,
    )
    print("Telegram status:", response.status_code)
    print("Telegram body:", response.text)
    response.raise_for_status()

def run_model():
    script_path = "model_c_plus_usd_hyg_short_conviction.py"

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True
    )

    print("MODEL STDOUT:\n", result.stdout)
    print("MODEL STDERR:\n", result.stderr)

    if result.returncode != 0:
        raise RuntimeError("Model failed")

def load_current():
    df = pd.read_csv(
        "model_c_plus_usd_hyg_short_conviction_latest_recommendation.csv"
    )
    row = df.iloc[-1]
    return {
        "date": row["date"],
        "top": row["top_asset"],
        "second": row["second_asset"],
        "top_score": float(row["top_score"]),
        "second_score": float(row["second_score"])
    }

def load_previous():
    df = pd.read_csv(
        "model_c_plus_usd_hyg_short_conviction_rebalance_log.csv"
    )
    row = df.iloc[-1]
    return {
        "date": row["date"],
        "top": row["top_asset"],
        "second": row["second_asset"]
    }

def main():
    print("Running model...")
    run_model()

    current = load_current()
    previous = load_previous()

    gap = current["top_score"] - current["second_score"]

    # ===== ALERT CONDITIONS =====
    strong_rotation = current["top"] != previous["top"]
    strong_gap = gap > 0.03
    conviction_jump = current["top_score"] > 0.04

    alert = False

    if strong_rotation and strong_gap:
        alert = True
    elif conviction_jump:
        alert = True

    # ===== MESSAGE =====
    message = (
        f"📊 Rotation Signal\n"
        f"Date: {current['date']}\n"
        f"\nTop: {current['top']} ({current['top_score']:.4f})\n"
        f"Second: {current['second']} ({current['second_score']:.4f})\n"
        f"Gap: {gap:.4f}\n"
        f"\nPrevious Top: {previous['top']}"
    )

    if alert:
        send_telegram(message)
        print("✅ ALERT SENT")
    else:
        print("❌ No alert")

if __name__ == "__main__":
    main()
