import json
import os
from pathlib import Path

import pandas as pd
import requests

CSV = "model_c_plus_usd_hyg_short_conviction_latest_recommendation.csv"
STATE_FILE = Path("last_alert_state.json")

BIG_TURNOVER_THRESHOLD = 0.50

def send_telegram(message: str):
    bot_token = os.environ["BOT_TOKEN"]
    chat_id = os.environ["CHAT_ID"]

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    r = requests.post(
        url,
        data={"chat_id": chat_id, "text": message},
        timeout=20,
    )
    print(r.status_code, r.text)
    r.raise_for_status()

def load_prev():
    if not STATE_FILE.exists():
        return None
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return None

def calc_turnover(prev_weights, curr_weights):
    assets = set(prev_weights.keys()) | set(curr_weights.keys())
    return sum(abs(curr_weights.get(a, 0.0) - prev_weights.get(a, 0.0)) for a in assets)

def main():
    df = pd.read_csv(CSV)
    row = df.iloc[-1]

    preds = {
        "QQQM": float(row.get("adj_pred_QQQM", 0.0)),
        "XLE": float(row.get("adj_pred_XLE", 0.0)),
        "XSOE": float(row.get("adj_pred_XSOE", 0.0)),
    }

    weights = {
        "QQQM": float(row.get("QQQM", row.get("w_QQQM", 0.0))),
        "XLE": float(row.get("XLE", row.get("w_XLE", 0.0))),
        "XSOE": float(row.get("XSOE", row.get("w_XSOE", 0.0))),
        "BIL": float(row.get("BIL", row.get("w_BIL", 0.0))),
    }

    curr = {
        "signal_date": str(row["signal_date"]),
        "latest_data_date": str(row["latest_data_date"]),
        "top": str(row["top_asset"]),
        "second": str(row["second_asset"]),
        "top_score": float(row["top_score"]),
        "score_gap": float(row["score_gap"]),
        "risk_off": float(row["risk_off_strength"]),
        "growth": float(row["growth_strength"]),
        "soxx": float(row["soxx_strength"]),
        "preds": preds,
        "weights": weights,
    }

    prev = load_prev()

    all_negative = all(v <= 0 for v in preds.values())

    exit_signal = (
        curr["top_score"] <= 0
        or all_negative
        or curr["risk_off"] >= 1.0
    )

    # TQQQ danger proxy:
    # If QQQM is top, you may be tempted to use TQQQ.
    # Alert when QQQM regime becomes unsafe.
    tqqq_danger = (
        curr["top"] == "QQQM"
        and (
            curr["growth"] < 0
            or curr["soxx"] < 0
            or curr["risk_off"] > 0
        )
    )

    if prev is None:
        rotation_change = True
        turnover = 999.0
    else:
        rotation_change = prev.get("top") != curr["top"]
        turnover = calc_turnover(prev.get("weights", {}), curr["weights"])

    big_allocation_change = turnover >= BIG_TURNOVER_THRESHOLD

    send_alert = False
    reason = None

    if prev is None:
        send_alert = True
        reason = "FIRST RUN"

    elif exit_signal and not prev.get("exit_signal", False):
        send_alert = True
        reason = "EXIT / CRASH"

    elif tqqq_danger and not prev.get("tqqq_danger", False):
        send_alert = True
        reason = "TQQQ DANGER"

    elif rotation_change:
        send_alert = True
        reason = "ROTATION CHANGE"

    elif big_allocation_change:
        send_alert = True
        reason = "BIG ALLOCATION CHANGE"

    curr["exit_signal"] = exit_signal
    curr["tqqq_danger"] = tqqq_danger

    if send_alert:
        msg = f"""
🚨 {reason}

Signal date: {curr['signal_date']}
Latest data: {curr['latest_data_date']}

Top: {curr['top']}
Second: {curr['second']}

Top score: {curr['top_score']:.4f}
Gap: {curr['score_gap']:.4f}

Predictions:
QQQM: {preds['QQQM']:.4f}
XLE:  {preds['XLE']:.4f}
XSOE: {preds['XSOE']:.4f}

Weights:
QQQM: {weights['QQQM']:.0%}
XLE:  {weights['XLE']:.0%}
XSOE: {weights['XSOE']:.0%}
BIL:  {weights['BIL']:.0%}

Risk:
Risk-off: {curr['risk_off']:.3f}
Growth: {curr['growth']:.3f}
SOXX: {curr['soxx']:.3f}
Turnover: {turnover:.2f}
""".strip()

        send_telegram(msg)
    else:
        print("No alert.")

    STATE_FILE.write_text(json.dumps(curr, indent=2, default=float))

if __name__ == "__main__":
    main()
