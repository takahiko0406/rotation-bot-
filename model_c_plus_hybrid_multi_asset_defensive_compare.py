import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

# =============================
# SETTINGS
# =============================
sector_etfs = ["QQQM", "XLE", "XSOE"]
cash_etf = "BIL"
spy_etf = "SPY"
feature_etfs = ["SOXX", "HYG"]

all_assets = sector_etfs + [cash_etf, spy_etf] + feature_etfs + ["TQQQ"]

start_date = "2018-01-01"
forward_return_days = 10
train_window = 252 * 3

rf_params = {
    "n_estimators": 120,
    "max_depth": 6,
    "min_samples_leaf": 5,
    "random_state": 42,
    "n_jobs": -1
}

# =============================
# DOWNLOAD DATA
# =============================
print("Downloading data...")
data = yf.download(all_assets, start=start_date, auto_adjust=True, progress=False)

if isinstance(data.columns, pd.MultiIndex):
    prices = data["Close"]
else:
    prices = data

prices = prices.dropna()

returns = prices.pct_change()

spy = prices[spy_etf]
spy_ret_1m = spy.pct_change(21)

soxx_rel = prices["SOXX"].pct_change(21) - spy_ret_1m
hyg_rel = prices["HYG"].pct_change(21) - spy_ret_1m

# =============================
# BUILD FEATURES
# =============================
def build_features(asset):
    px = prices[asset]

    df = pd.DataFrame({
        "ret_1m": px.pct_change(21),
        "ret_3m": px.pct_change(63),
        "ret_6m": px.pct_change(126),
        "vol_1m": px.pct_change().rolling(21).std(),
        "soxx_rel": soxx_rel,
        "hyg_rel": hyg_rel,
    })

    df["target"] = px.shift(-forward_return_days) / px - 1
    return df.dropna()

features = {a: build_features(a) for a in sector_etfs}

# =============================
# TRAIN MODEL
# =============================
latest_date = prices.index[-1]

X_list = []
y_list = []

for asset in sector_etfs:
    df = features[asset].iloc[-train_window:]
    X_list.append(df.drop(columns=["target"]))
    y_list.append(df["target"])

X_train = pd.concat(X_list)
y_train = pd.concat(y_list)

model = RandomForestRegressor(**rf_params)
model.fit(X_train, y_train)

# =============================
# PREDICTIONS
# =============================
raw_preds = {}

for asset in sector_etfs:
    row = features[asset].iloc[-1:].drop(columns=["target"])
    raw_preds[asset] = float(model.predict(row)[0])

# =============================
# SIMPLE OVERLAY
# =============================
adjusted_preds = raw_preds.copy()

# tech boost if SOXX strong
if soxx_rel.iloc[-1] > 0:
    adjusted_preds["QQQM"] += 0.002

# risk-off reduce
if hyg_rel.iloc[-1] < 0:
    adjusted_preds["QQQM"] -= 0.002
    adjusted_preds["XSOE"] -= 0.002

# =============================
# RANKING
# =============================
ranked = sorted(adjusted_preds.items(), key=lambda x: x[1], reverse=True)

top_asset = ranked[0][0]
second_asset = ranked[1][0]

top_score = ranked[0][1]
second_score = ranked[1][1]

gap = top_score - second_score

# =============================
# WEIGHTS
# =============================
if gap < 0.01:
    w_top = 0.6
else:
    w_top = 0.7

w_second = 1 - w_top

signal_weights = {a: 0.0 for a in sector_etfs + [cash_etf]}
signal_weights[top_asset] = w_top
signal_weights[second_asset] = w_second

# TQQQ overlay
exec_weights = {
    "TQQQ": 0.0,
    "QQQM": 0.0,
    "XLE": 0.0,
    "XSOE": 0.0,
    "BIL": 0.0
}

if top_asset == "QQQM" and gap > 0.01:
    exec_weights["TQQQ"] = w_top
else:
    exec_weights[top_asset] = w_top

exec_weights[second_asset] = w_second

# =============================
# PERFORMANCE (simple)
# =============================
port_ret = returns[sector_etfs].mean(axis=1)

ann_return = (1 + port_ret).prod()**(252/len(port_ret)) - 1
vol = port_ret.std() * np.sqrt(252)
sharpe = ann_return / vol

cum = (1 + port_ret).cumprod()
drawdown = cum / cum.cummax() - 1

# =============================
# PRINT OUTPUT
# =============================
print("\n=== PERFORMANCE ===")
print(f"Annual Return: {ann_return:.3f}")
print(f"Volatility:    {vol:.3f}")
print(f"Sharpe:        {sharpe:.3f}")
print(f"Max Drawdown:  {drawdown.min():.3f}")

print("\n=== LATEST SIGNAL ===")
print("Date:", latest_date.date())

print("\nRaw predictions:")
for k, v in raw_preds.items():
    print(f"{k}: {v:.4f}")

print("\nAdjusted predictions:")
for k, v in adjusted_preds.items():
    print(f"{k}: {v:.4f}")

print(f"\nTop: {top_asset}")
print(f"Second: {second_asset}")
print(f"Gap: {gap:.4f}")

print("\nSignal weights:")
print(signal_weights)

print("\nExecuted weights:")
print(exec_weights)

# =============================
# SAVE (for alert system)
# =============================
pd.DataFrame([{
    "signal_date": latest_date,
    "top_asset": top_asset,
    "second_asset": second_asset,
    "top_score": top_score,
    "score_gap": gap,
    **{f"exec_w_{k}": v for k, v in exec_weights.items()}
}]).to_csv("latest_recommendation.csv", index=False)

print("\nSaved: latest_recommendation.csv")
