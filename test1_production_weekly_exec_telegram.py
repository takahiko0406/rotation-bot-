import os
from datetime import datetime
import requests
import pandas as pd

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
SIGNAL_FILE = "latest_signal.csv"


def send_telegram(message: str):
    if not BOT_TOKEN:
        raise ValueError("BOT_TOKEN missing")
    if not CHAT_ID:
        raise ValueError("CHAT_ID missing")

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    response = requests.post(
        url,
        data={"chat_id": CHAT_ID, "text": message},
        timeout=20,
    )
    print("Telegram response:", response.status_code, response.text)
    response.raise_for_status()


def run_model():
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "top_asset": "XLE",
        "second_asset": "QQQM",
        "top_score": 0.0178,
        "second_score": 0.0054,
        "gap": 0.0124,
    }


def load_previous():
    if not os.path.exists(SIGNAL_FILE):
        return None
    return pd.read_csv(SIGNAL_FILE)


def save_signal(data):
    pd.DataFrame([data]).to_csv(SIGNAL_FILE, index=False)


def is_changed(prev, new):
    if prev is None or prev.empty:
        return True
    return prev.iloc[0]["top_asset"] != new["top_asset"]


def main():
    print("Running model...")
    result = run_model()

    prev = load_previous()
    changed = is_changed(prev, result)

    message = f"""📊 Rotation update
Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Signal date: {result['date']}
Top asset: {result['top_asset']}
Second asset: {result['second_asset']}
Top score: {result['top_score']}
Second score: {result['second_score']}
Score gap: {result['gap']}
Changed: {'YES' if changed else 'NO'}
"""

    if changed:
        send_telegram(message)
        print("Telegram sent.")
    else:
        print("No change.")

    save_signal(result)
    print("Done.")


if __name__ == "__main__":
    main()
