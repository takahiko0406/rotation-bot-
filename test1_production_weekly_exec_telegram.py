import subprocess
import pandas as pd
import requests
import os

BOT_TOKEN = os.environ["BOT_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]


# =========================
# RUN MODEL
# =========================
def run_model():
    result = subprocess.run(
        ["python", "model_c_plus_usd_hyg_short_conviction.py"],
        capture_output=True,
        text=True,
        check=True
    )

    print("MODEL STDOUT:\n", result.stdout)
    print("MODEL STDERR:\n", result.stderr)

    return result


# =========================
# LOAD CURRENT SIGNAL
# =========================
def load_current():
    df = pd.read_csv(
        "model_c_plus_usd_hyg_short_conviction_latest_recommendation.csv"
    )

    row = df.iloc[-1]

    return {
        "date": row.get("date", row.get("signal_date", "N/A")),
        "top": row["top_asset"],
        "second": row["second_asset"],
        "top_score": float(row["top_score"]),
        "second_score": float(row["second_score"])
    }


# =========================
# LOAD PREVIOUS SIGNAL
# =========================
def load_previous():
    df = pd.read_csv(
        "model_c_plus_usd_hyg_short_conviction_rebalance_log.csv"
    )

    row = df.iloc[-1]

    return {
        "date": row.get("date", row.get("signal_date", "N/A")),
        "top": row["top_asset"],
        "second": row["second_asset"]
    }


# =========================
# SHOULD ALERT?
# =========================
def should_alert(current, previous):
    score_gap = current["top_score"] - current["second_score"]

    # alert only if STRONG signal AND change
    if score_gap > 0.02 and current["top"] != previous["top"]:
        return True

    return False


# =========================
# SEND TELEGRAM
# =========================
def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": CHAT_ID,
        "text": message
    }

    requests.post(url, json=payload)


# =========================
# MAIN
# =========================
def main():
    print("Running model...")

    run_model()

    current = load_current()
    previous = load_previous()

    if should_alert(current, previous):
        msg = f"""
🚨 ROTATION ALERT 🚨

Date: {current['date']}

Top: {current['top']}
Second: {current['second']}

Score gap: {current['top_score'] - current['second_score']:.4f}

Previous Top: {previous['top']}
"""
        send_telegram(msg)
        print("ALERT SENT")
    else:
        print("No strong signal → no alert")


if __name__ == "__main__":
    main()
