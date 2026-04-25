import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import os

print("Current working directory:", os.getcwd())

# =========================================
# 1. SETTINGS
# =========================================
sector_etfs = ["QQQM", "XLE", "XSOE"]
cash_etf = "BIL"
spy_etf = "SPY"
feature_etfs = ["ITA", "SOXX", "HYG"]

# Execution asset
execution_extra = ["TQQQ"]

all_assets = sector_etfs + [cash_etf, spy_etf] + feature_etfs + execution_extra

start_date = "2010-01-01"
end_date = None

forward_return_days = 10
rebalance_step = 10
train_window = 252 * 3
transaction_cost = 0.001

risk_free_rate_annual = 0.0

rf_params = {
    "n_estimators": 300,
    "max_depth": 6,
    "min_samples_leaf": 5,
    "random_state": 42,
    "n_jobs": -1
}

overlay_scale = 0.002
risk_off_cash_threshold = 0.01
zscore_window = 252

# =========================================
# 2. TQQQ TIERED OVERLAY RULE
# =========================================
tiered_tqqq_rule = {
    "moderate": {
        "gap_min": 0.003,
        "top_score_min": 0.010,
        "growth_min": 0.30,
        "soxx_min": 0.30,
        "risk_off_max": 1.00,
        "vix_max": 25.0,
        "replace_fraction": 0.50
    },
    "strong": {
        "gap_min": 0.004,
        "top_score_min": 0.010,
        "growth_min": 0.20,
        "soxx_min": 0.20,
        "risk_off_max": 0.75,
        "vix_max": 22.0,
        "replace_fraction": 1.00
    }
}

# =========================================
# (FULL CODE CONTINUES — EXACT SAME AS YOUR ORIGINAL)
# =========================================
