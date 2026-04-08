import os
import requests
import pandas as pd
from datetime import datetime

# =========================
# 🔑 TELEGRAM SETTINGS
# =========================
BOT_TOKEN = "8676450564:AAFjONeYvYEJcpXyo0CTGm6JBGVS39ED8v8"
CHAT_ID = "8679100036"

# =========================
# 📁 FILE PATH
# =========================
SIGNAL_FILE = "latest_signal.csv"

# =========================
# 📩 TELEGRAM FUNCTION
# =========================

# =========================
# 🧠 MOCK MODEL (REPLACE WITH YOUR MODEL)
# =========================
def run_model():
    # 👉 replace this with your real model output
    # example output:
    return {
        "date": "2026-04-08",
        "top_asset": "XLE",
        "second_asset": "QQQM",
        "top_score": 0.0178,
        "second_score": 0.0054,
        "gap": 0.0124
    }

# =========================
# 📊 LOAD PREVIOUS SIGNAL
# =========================
def load_previous():
    if not os.path.exists(SIGNAL_FILE):
        return None
    return pd.read_csv(SIGNAL_FILE)

# =========================
# 💾 SAVE SIGNAL
# =========================
def save_signal(data):
    df = pd.DataFrame([data])
    df.to_csv(SIGNAL_FILE, index=False)

# =========================
# 🔍 CHECK CHANGE
# =========================
def is_changed(prev, new):
    if prev is None:
        return True
    return prev.iloc[0]["top_asset"] != new["top_asset"]

# =========================
# 🚀 MAIN
# =========================
def main():
    print("Running model...")

    result = run_model()

    top_asset = result.get("top_asset")
    second_asset = result.get("second_asset")
    top_score = result.get("top_score")
    second_score = result.get("second_score")

    # ❌ INVALID DATA CHECK
    if top_asset is None or top_score is None:
        print("Invalid signal → no alert sent")
        return

    prev = load_previous()
    changed = is_changed(prev, result)

    # 📩 SEND ONLY IF CHANGED
    if changed:
        message = f"""
📊 Rotation change detected
Time: {datetime.now()}

Signal date: {result['date']}
Top asset: {top_asset}
Second asset: {second_asset}

Top score: {top_score}
Second score: {second_score}
Score gap: {result['gap']}
"""
        send_telegram(message)
        print("Telegram sent.")

    else:
        print("No change.")

    save_signal(result)
    print("Done.")

# =========================
# ▶ RUN
# =========================
if __name__ == "__main__":
    main()