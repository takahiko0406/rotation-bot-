"""
Live Telegram alert system for the HYBRID_MULTI_ASSET_DEFENSIVE model.

Run after your main model script creates:
    model_c_plus_hybrid_multi_asset_defensive_latest_recommendation.csv

Example run:
    C:/Users/81901/venv/Scripts/python.exe C:/Users/81901/Downloads/live_model_telegram_alert_v2.py

Telegram setup:
    Set environment variables BOT_TOKEN and CHAT_ID.

Windows PowerShell example:
    setx BOT_TOKEN "123456:ABC..."
    setx CHAT_ID "123456789"

After setx, close and reopen terminal.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

try:
    import requests
except ImportError:
    requests = None


# ============================================================
# SETTINGS
# ============================================================
WORKDIR = Path("C:/Users/81901/Downloads")
CSV = WORKDIR / "model_c_plus_hybrid_multi_asset_defensive_latest_recommendation.csv"
STATE_FILE = WORKDIR / "live_alert_state_hybrid_multi_defensive.json"
LOG_FILE = WORKDIR / "live_alert_log_hybrid_multi_defensive.csv"

# Alert thresholds
BIG_TURNOVER_THRESHOLD = 0.50       # 50% total allocation change
SCORE_GAP_WEAK_THRESHOLD = 0.003    # very weak conviction
RISK_OFF_EXIT_THRESHOLD = 1.50      # strong risk-off -> exit alert
RISK_OFF_WARN_THRESHOLD = 0.75      # warning level
TQQQ_DANGER_RISK_OFF = 0.30

# Model assets
SIGNAL_ASSETS = ["QQQM", "XLE", "XSOE", "XLI", "XLB", "BIL"]
EXEC_ASSETS = ["TQQQ", "ERX", "UXI", "QQQM", "XLE", "XSOE", "XLI", "XLB", "BIL"]
PRED_ASSETS = ["QQQM", "XLE", "XSOE", "XLI", "XLB"]
REGIME_FIELDS = [
    "growth_strength",
    "soxx_strength",
    "risk_off_strength",
    "war_strength",
    "industrial_strength",
    "materials_strength",
    "copper_strength",
    "copper_3m_strength",
    "usd_3m_strength",
    "credit_strength",
]


# ============================================================
# HELPERS
# ============================================================
def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def pct(x: float) -> str:
    return f"{x:.1%}"


def fmt_weight_dict(weights: Dict[str, float], assets) -> str:
    lines = []
    for asset in assets:
        w = weights.get(asset, 0.0)
        if abs(w) > 1e-9:
            lines.append(f"{asset}: {pct(w)}")
    if not lines:
        return "None"
    return "\n".join(lines)


def calc_turnover(prev_weights: Dict[str, float], curr_weights: Dict[str, float]) -> float:
    assets = set(prev_weights.keys()) | set(curr_weights.keys())
    return float(sum(abs(curr_weights.get(a, 0.0) - prev_weights.get(a, 0.0)) for a in assets))


def load_prev() -> Optional[Dict[str, Any]]:
    if not STATE_FILE.exists():
        return None
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_state(curr: Dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(curr, indent=2), encoding="utf-8")


def append_log(curr: Dict[str, Any], reason: str, turnover: float, send_alert: bool) -> None:
    row = {
        "run_time": pd.Timestamp.now(),
        "signal_date": curr.get("signal_date"),
        "latest_data_date": curr.get("latest_data_date"),
        "top": curr.get("top"),
        "second": curr.get("second"),
        "top_score": curr.get("top_score"),
        "second_score": curr.get("second_score"),
        "score_gap": curr.get("score_gap"),
        "reason": reason,
        "turnover_vs_prev": turnover,
        "send_alert": send_alert,
        "exit_signal": curr.get("exit_signal"),
        "risk_warning": curr.get("risk_warning"),
        "leverage_danger": curr.get("leverage_danger"),
        "tqqq_weight": curr["exec_weights"].get("TQQQ", 0.0),
        "erx_weight": curr["exec_weights"].get("ERX", 0.0),
        "uxi_weight": curr["exec_weights"].get("UXI", 0.0),
        "bil_weight": curr["exec_weights"].get("BIL", 0.0),
    }
    for k, v in curr.get("regime", {}).items():
        row[k] = v
    df = pd.DataFrame([row])
    if LOG_FILE.exists():
        df.to_csv(LOG_FILE, mode="a", index=False, header=False)
    else:
        df.to_csv(LOG_FILE, index=False)


def send_telegram(message: str) -> None:
    bot_token = os.environ.get("BOT_TOKEN")
    chat_id = os.environ.get("CHAT_ID")

    if not bot_token or not chat_id:
        print("Telegram not configured. Set BOT_TOKEN and CHAT_ID. Skipping Telegram alert.")
        return

    if requests is None:
        raise ImportError("requests is not installed. Run: pip install requests")

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    response = requests.post(
        url,
        data={"chat_id": chat_id, "text": message},
        timeout=20,
    )
    print("Telegram response:", response.status_code, response.text)
    response.raise_for_status()


# ============================================================
# MAIN LOGIC
# ============================================================
def read_latest_signal() -> Dict[str, Any]:
    if not CSV.exists():
        raise FileNotFoundError(
            f"Cannot find latest recommendation file:\n{CSV}\n"
            "Run your production model first."
        )

    df = pd.read_csv(CSV)
    if df.empty:
        raise ValueError(f"Latest recommendation CSV is empty: {CSV}")
    row = df.iloc[-1]

    preds = {a: safe_float(row.get(f"adj_pred_{a}", 0.0)) for a in PRED_ASSETS}
    signal_weights = {a: safe_float(row.get(f"signal_w_{a}", 0.0)) for a in SIGNAL_ASSETS}
    exec_weights = {a: safe_float(row.get(f"exec_w_{a}", 0.0)) for a in EXEC_ASSETS}
    regime = {k: safe_float(row.get(k, 0.0)) for k in REGIME_FIELDS}

    curr = {
        "signal_date": str(row.get("signal_date", "")),
        "latest_data_date": str(row.get("latest_data_date", "")),
        "top": str(row.get("top_asset", "")),
        "second": str(row.get("second_asset", "")),
        "top_score": safe_float(row.get("top_score", 0.0)),
        "second_score": safe_float(row.get("second_score", 0.0)),
        "score_gap": safe_float(row.get("score_gap", 0.0)),
        "overlay_fraction": safe_float(row.get("overlay_fraction", 0.0)),
        "overlay_style": str(row.get("overlay_style", "")),
        "preds": preds,
        "signal_weights": signal_weights,
        "exec_weights": exec_weights,
        "regime": regime,
    }
    return curr


def classify_alert(curr: Dict[str, Any], prev: Optional[Dict[str, Any]]):
    preds = curr["preds"]
    exec_weights = curr["exec_weights"]
    regime = curr["regime"]

    all_negative = all(v <= 0 for v in preds.values())
    top_score = curr["top_score"]
    score_gap = curr["score_gap"]
    risk_off = regime.get("risk_off_strength", 0.0)
    growth = regime.get("growth_strength", 0.0)
    soxx = regime.get("soxx_strength", 0.0)

    leveraged_weight = exec_weights.get("TQQQ", 0.0) + exec_weights.get("ERX", 0.0) + exec_weights.get("UXI", 0.0)
    tqqq_now_on = exec_weights.get("TQQQ", 0.0) > 0
    erx_now_on = exec_weights.get("ERX", 0.0) > 0
    uxi_now_on = exec_weights.get("UXI", 0.0) > 0

    # Crash / exit signal
    exit_signal = (
        top_score <= 0
        or all_negative
        or risk_off >= RISK_OFF_EXIT_THRESHOLD
        or exec_weights.get("BIL", 0.0) >= 0.99
    )

    # Warning signal, not necessarily exit
    risk_warning = risk_off >= RISK_OFF_WARN_THRESHOLD

    # Leverage danger checks
    tqqq_danger = (
        tqqq_now_on
        and (
            growth < 0
            or soxx < 0
            or risk_off > TQQQ_DANGER_RISK_OFF
        )
    )
    erx_danger = erx_now_on and risk_off > 0.75
    uxi_danger = uxi_now_on and risk_off > 0.75
    leverage_danger = tqqq_danger or erx_danger or uxi_danger

    weak_conviction = score_gap < SCORE_GAP_WEAK_THRESHOLD

    if prev is None:
        turnover = 999.0
        rotation_change = True
        allocation_change = True
        reason = "FIRST RUN"
        send_alert = True
    else:
        turnover = calc_turnover(prev.get("exec_weights", {}), exec_weights)
        rotation_change = prev.get("top") != curr.get("top")
        allocation_change = turnover >= BIG_TURNOVER_THRESHOLD
        reason = "NO ALERT"
        send_alert = False

        if exit_signal and not prev.get("exit_signal", False):
            send_alert = True
            reason = "EXIT / CASH DEFENSE"
        elif leverage_danger and not prev.get("leverage_danger", False):
            send_alert = True
            reason = "LEVERAGE DANGER"
        elif rotation_change:
            send_alert = True
            reason = "ROTATION CHANGE"
        elif allocation_change:
            send_alert = True
            reason = "BIG ALLOCATION CHANGE"
        elif risk_warning and not prev.get("risk_warning", False):
            send_alert = True
            reason = "RISK WARNING"
        elif weak_conviction and not prev.get("weak_conviction", False):
            send_alert = True
            reason = "WEAK CONVICTION"

    curr["exit_signal"] = exit_signal
    curr["risk_warning"] = risk_warning
    curr["leverage_danger"] = leverage_danger
    curr["tqqq_danger"] = tqqq_danger
    curr["erx_danger"] = erx_danger
    curr["uxi_danger"] = uxi_danger
    curr["weak_conviction"] = weak_conviction
    curr["turnover_vs_prev"] = turnover

    return send_alert, reason, turnover


def build_message(curr: Dict[str, Any], reason: str, turnover: float) -> str:
    preds = curr["preds"]
    signal_weights = curr["signal_weights"]
    exec_weights = curr["exec_weights"]
    regime = curr["regime"]

    msg = f"""
🚨 {reason}

Signal date: {curr['signal_date']}
Latest data: {curr['latest_data_date']}

Top: {curr['top']}
Second: {curr['second']}
Top score: {curr['top_score']:.4f}
Second score: {curr['second_score']:.4f}
Gap: {curr['score_gap']:.4f}
Overlay fraction: {pct(curr['overlay_fraction'])}

Adjusted predictions:
{chr(10).join([f'{a}: {preds.get(a, 0.0):.4f}' for a in PRED_ASSETS])}

Signal weights:
{fmt_weight_dict(signal_weights, SIGNAL_ASSETS)}

Executed weights:
{fmt_weight_dict(exec_weights, EXEC_ASSETS)}

Regime:
Growth: {regime.get('growth_strength', 0.0):.3f}
SOXX: {regime.get('soxx_strength', 0.0):.3f}
Risk-off: {regime.get('risk_off_strength', 0.0):.3f}
War: {regime.get('war_strength', 0.0):.3f}
Industrial: {regime.get('industrial_strength', 0.0):.3f}
Materials: {regime.get('materials_strength', 0.0):.3f}
Copper: {regime.get('copper_strength', 0.0):.3f}
USD 3M: {regime.get('usd_3m_strength', 0.0):.3f}
Credit: {regime.get('credit_strength', 0.0):.3f}

Risk flags:
Exit signal: {curr['exit_signal']}
Risk warning: {curr['risk_warning']}
Leverage danger: {curr['leverage_danger']}
TQQQ danger: {curr['tqqq_danger']}
ERX danger: {curr['erx_danger']}
UXI danger: {curr['uxi_danger']}
Weak conviction: {curr['weak_conviction']}
Turnover vs previous: {turnover:.2f}
""".strip()
    return msg


def print_signal(curr: Dict[str, Any], reason: str, turnover: float, send_alert_flag: bool) -> None:
    print("\n=== LIVE MODEL ALERT CHECK ===")
    print(f"Signal date: {curr['signal_date']}")
    print(f"Latest data date: {curr['latest_data_date']}")
    print(f"Top: {curr['top']} | Second: {curr['second']} | Gap: {curr['score_gap']:.4f}")
    print(f"Reason: {reason}")
    print(f"Send alert: {send_alert_flag}")
    print(f"Turnover vs previous: {turnover:.2f}")
    print("\nExecuted weights:")
    print(fmt_weight_dict(curr["exec_weights"], EXEC_ASSETS))
    print("\nRisk flags:")
    for k in ["exit_signal", "risk_warning", "leverage_danger", "tqqq_danger", "erx_danger", "uxi_danger", "weak_conviction"]:
        print(f"  {k}: {curr.get(k)}")


def main() -> None:
    print("Current working directory:", os.getcwd())
    print("Monitor workdir:", WORKDIR)

    curr = read_latest_signal()
    prev = load_prev()
    send_alert_flag, reason, turnover = classify_alert(curr, prev)

    print_signal(curr, reason, turnover, send_alert_flag)

    if send_alert_flag:
        msg = build_message(curr, reason, turnover)
        send_telegram(msg)
    else:
        print("No alert triggered.")

    append_log(curr, reason, turnover, send_alert_flag)
    save_state(curr)

    print("\nSaved alert log:", LOG_FILE)
    print("Saved state:", STATE_FILE)


if __name__ == "__main__":
    main()
