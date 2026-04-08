import os
from datetime import datetime
import requests
import pandas as pd

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
SIGNAL_FILE = "latest_signal.csv"

# optimized alert thresholds
ROTATION_GAP_MIN = 0.010
CONVICTION_JUMP_MIN = 0.015
EXTREME_GAP_MIN = 0.030
DOWNTURN_TOP_SCORE = -0.003
RISK_OFF_MIN = 1.20
NOISE_GAP_MAX = 0.008


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
    print("Telegram status:", response.status_code)
    print("Telegram body:", response.text)
    response.raise_for_status()


import subprocess
import sys

def run_model():
    script_path = "model_c_plus_usd_hyg_short_conviction.py"
    csv_path = "model_c_plus_usd_hyg_short_conviction_latest_recommendation.csv"

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
        check=True,
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    df = pd.read_csv(csv_path)
    row = df.iloc[-1]

    return {
        "date": str(row["signal_date"])[:10],
        "top_asset": row["top_asset"],
        "second_asset": row["second_asset"],
        "top_score": float(row["top_score"]),
        "second_score": float(row["second_score"]),
        "gap": float(row["score_gap"]),
        "risk_off_strength": float(row["risk_off_strength"]),
    }
def load_previous():
    if not os.path.exists(SIGNAL_FILE):
        return None
    df = pd.read_csv(SIGNAL_FILE)
    if df.empty:
        return None
    return df


def save_signal(data: dict):
    pd.DataFrame([data]).to_csv(SIGNAL_FILE, index=False)


def get_prev_value(prev_df, col, default=None):
    if prev_df is None or prev_df.empty or col not in prev_df.columns:
        return default
    return prev_df.iloc[0][col]


def should_alert(prev_df, result: dict):
    prev_top = get_prev_value(prev_df, "top_asset", None)
    prev_gap = get_prev_value(prev_df, "gap", None)

    new_top = result["top_asset"]
    gap = result["gap"]
    top_score = result["top_score"]
    risk_off_strength = result["risk_off_strength"]

    strong_rotation = (prev_top != new_top) and (gap >= ROTATION_GAP_MIN)

    conviction_jump = False
    if prev_gap is not None:
        conviction_jump = abs(gap - prev_gap) >= CONVICTION_JUMP_MIN

    extreme_conviction = gap >= EXTREME_GAP_MIN
    downturn = top_score <= DOWNTURN_TOP_SCORE
    risk_off = risk_off_strength >= RISK_OFF_MIN

    alert = (
        strong_rotation
        or conviction_jump
        or extreme_conviction
        or downturn
        or risk_off
    )

    if (gap < NOISE_GAP_MAX) and (top_score > DOWNTURN_TOP_SCORE) and not risk_off:
        alert = False

    reasons = []
    if strong_rotation:
        reasons.append(f"strong rotation (gap >= {ROTATION_GAP_MIN:.3f})")
    if conviction_jump:
        reasons.append(f"conviction jump (|Δgap| >= {CONVICTION_JUMP_MIN:.3f})")
    if extreme_conviction:
        reasons.append(f"extreme conviction (gap >= {EXTREME_GAP_MIN:.3f})")
    if downturn:
        reasons.append(f"downturn (top_score <= {DOWNTURN_TOP_SCORE:.3f})")
    if risk_off:
        reasons.append(f"risk-off (risk_off_strength >= {RISK_OFF_MIN:.2f})")

    return alert, reasons, prev_top, prev_gap


def main():
    print("Running model...")
    result = run_model()

    prev_df = load_previous()
    alert, reasons, prev_top, prev_gap = should_alert(prev_df, result)

    changed = prev_top != result["top_asset"] if prev_top is not None else True
    reason_text = ", ".join(reasons) if reasons else "no alert condition"

    message = f"""📊 Rotation update
Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Signal date: {result['date']}

Previous top asset: {prev_top if prev_top is not None else 'None'}
Current top asset: {result['top_asset']}
Second asset: {result['second_asset']}

Top score: {result['top_score']:.4f}
Second score: {result['second_score']:.4f}
Gap: {result['gap']:.4f}
Previous gap: {prev_gap if prev_gap is not None else 'None'}
Risk-off strength: {result['risk_off_strength']:.3f}

Changed: {'YES' if changed else 'NO'}
Alert reason: {reason_text}
"""

    if alert:
        send_telegram(message)
        print("Telegram sent.")
    else:
        print("No alert condition met.")

    save_signal(result)
    print("Done.")


if __name__ == "__main__":
    main()
