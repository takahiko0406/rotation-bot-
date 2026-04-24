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
# Only applies when QQQM is top asset.
# Moderate regime: replace half of QQQM sleeve with TQQQ
# Strong regime: replace all of QQQM sleeve with TQQQ

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
# 3. PERFORMANCE FUNCTIONS
# =========================================
def annualized_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    cumulative = (1 + returns).prod()
    years = len(returns) / 252
    if years <= 0:
        return np.nan
    return cumulative ** (1 / years) - 1

def annualized_volatility(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    return returns.std() * np.sqrt(252)

def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.0) -> float:
    ann_ret = annualized_return(returns)
    ann_vol = annualized_volatility(returns)
    if ann_vol == 0 or np.isnan(ann_vol):
        return np.nan
    return (ann_ret - rf_annual) / ann_vol

def max_drawdown(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return drawdown.min()

def compute_turnover(old_weights: dict, new_weights: dict, universe: list) -> float:
    old_vec = np.array([old_weights.get(a, 0.0) for a in universe], dtype=float)
    new_vec = np.array([new_weights.get(a, 0.0) for a in universe], dtype=float)
    return np.abs(new_vec - old_vec).sum()

def rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    z = (series - mean) / std.replace(0, np.nan)
    z = z.clip(-3, 3)
    return z.fillna(0.0)

# =========================================
# 4. CONVICTION WEIGHTING
# =========================================
def get_conviction_weights(top_score: float, second_score: float):
    gap = top_score - second_score

    if gap < 0.005:
        w_top = 0.60
    elif gap < 0.015:
        w_top = 0.70
    elif gap < 0.030:
        w_top = 0.80
    elif gap < 0.050:
        w_top = 0.90
    else:
        w_top = 1.00

    w_second = 1.0 - w_top
    return w_top, w_second, gap

# =========================================
# 5. DOWNLOAD DATA
# =========================================
def download_close_data(tickers, start, end=None):
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]

    if isinstance(data, pd.Series):
        data = data.to_frame()

    return data.sort_index()

print("Downloading price data...")
prices = download_close_data(all_assets, start_date, end_date).dropna(how="all")

if prices.empty:
    raise ValueError("No price data downloaded.")

print("Downloading macro proxies...")
macro_tickers = {
    "short_rate": "^IRX",
    "long_rate": "^TNX",
    "oil": "CL=F",
    "usd": "DX-Y.NYB",
    "vix_level": "^VIX",
    "copper": "HG=F"
}

macro_raw = download_close_data(list(macro_tickers.values()), start_date, end_date)

reverse_map = {v: k for k, v in macro_tickers.items()}
macro_raw = macro_raw.rename(columns=reverse_map)
macro_raw = macro_raw.reindex(prices.index).ffill()

required_cols = sector_etfs + [cash_etf, spy_etf, "ITA", "SOXX", "HYG", "TQQQ"]
prices = prices.dropna(subset=required_cols)
macro_raw = macro_raw.reindex(prices.index).ffill()

print(f"Latest available data date: {prices.index[-1].date()}")

# =========================================
# 6. COMMON SERIES
# =========================================
asset_returns = prices[sector_etfs + [cash_etf, spy_etf, "TQQQ"]].pct_change()

spy = prices[spy_etf]
spy_ret_1m = spy.pct_change(21)
spy_ret_3m = spy.pct_change(63)
spy_ret_6m = spy.pct_change(126)

spy_vol_1m = spy.pct_change().rolling(21).std() * np.sqrt(252)
spy_vol_3m = spy.pct_change().rolling(63).std() * np.sqrt(252)

short_rate = macro_raw["short_rate"] / 100.0
long_rate = macro_raw["long_rate"] / 100.0
yield_curve = long_rate - short_rate

oil_1m = macro_raw["oil"].pct_change(21)

usd_1m = macro_raw["usd"].pct_change(21)
usd_3m = macro_raw["usd"].pct_change(63)
usd_6m = macro_raw["usd"].pct_change(126)

usd_level_strength = rolling_zscore(macro_raw["usd"], zscore_window)
usd_1m_strength = rolling_zscore(usd_1m, zscore_window)
usd_3m_strength = rolling_zscore(usd_3m, zscore_window)

vix_level = macro_raw["vix_level"]
vix_1m = macro_raw["vix_level"].pct_change(21)

copper_1m = macro_raw["copper"].pct_change(21)
copper_rel_spy_1m = copper_1m - spy_ret_1m

ita_1m = prices["ITA"].pct_change(21)
ita_rel_spy_1m = ita_1m - spy_ret_1m

soxx_1m = prices["SOXX"].pct_change(21)
soxx_3m = prices["SOXX"].pct_change(63)
soxx_rel_spy_1m = soxx_1m - spy_ret_1m

hyg_1m = prices["HYG"].pct_change(21)
hyg_3m = prices["HYG"].pct_change(63)
hyg_6m = prices["HYG"].pct_change(126)
hyg_rel_spy_1m = hyg_1m - spy_ret_1m
hyg_rel_spy_3m = hyg_3m - spy_ret_3m

qqqm_rel_spy_1m = prices["QQQM"].pct_change(21) - spy_ret_1m

# =========================================
# 6B. CONTINUOUS REGIME STRENGTH
# =========================================
ita_strength = rolling_zscore(ita_rel_spy_1m, zscore_window)
soxx_strength = rolling_zscore(soxx_rel_spy_1m, zscore_window)
qqqm_strength = rolling_zscore(qqqm_rel_spy_1m, zscore_window)
oil_strength = rolling_zscore(oil_1m, zscore_window)
vix_strength = rolling_zscore(vix_1m, zscore_window)
copper_strength = rolling_zscore(copper_rel_spy_1m, zscore_window)
hyg_strength = rolling_zscore(hyg_rel_spy_1m, zscore_window)

war_strength = ((ita_strength + oil_strength) / 2.0).fillna(0.0)
growth_strength = ((qqqm_strength + soxx_strength) / 2.0).fillna(0.0)
risk_off_strength = ((-qqqm_strength + vix_strength) / 2.0).fillna(0.0)
credit_strength = ((hyg_strength - vix_strength) / 2.0).fillna(0.0)

# =========================================
# 7. FEATURE BUILDER
# =========================================
def build_features_by_asset():
    features_by_asset = {}

    for asset in sector_etfs:
        px = prices[asset]

        ret_1m = px.pct_change(21)
        ret_3m = px.pct_change(63)
        ret_6m = px.pct_change(126)
        ret_12m = px.pct_change(252)

        rel_6m_vs_spy = ret_6m - spy_ret_6m

        vol_1m = px.pct_change().rolling(21).std() * np.sqrt(252)
        vol_3m = px.pct_change().rolling(63).std() * np.sqrt(252)

        is_QQQM = 1 if asset == "QQQM" else 0
        is_XLE = 1 if asset == "XLE" else 0
        is_XSOE = 1 if asset == "XSOE" else 0

        df = pd.DataFrame({
            "ret_1m": ret_1m,
            "ret_3m": ret_3m,
            "ret_6m": ret_6m,
            "ret_12m": ret_12m,
            "rel_6m_vs_spy": rel_6m_vs_spy,

            "spy_ret_1m": spy_ret_1m,
            "spy_ret_3m": spy_ret_3m,
            "spy_ret_6m": spy_ret_6m,
            "spy_vol_1m": spy_vol_1m,
            "spy_vol_3m": spy_vol_3m,

            "vol_1m": vol_1m,
            "vol_3m": vol_3m,

            "short_rate": short_rate,
            "long_rate": long_rate,
            "yield_curve": yield_curve,

            "oil_1m": oil_1m,

            "usd_1m": usd_1m,
            "usd_3m": usd_3m,
            "usd_6m": usd_6m,
            "usd_level_strength": usd_level_strength,
            "usd_1m_strength": usd_1m_strength,
            "usd_3m_strength": usd_3m_strength,

            "vix_level": vix_level,
            "vix_1m": vix_1m,

            "copper_1m": copper_1m,
            "copper_rel_spy_1m": copper_rel_spy_1m,
            "copper_strength": copper_strength,

            "ita_1m": ita_1m,
            "ita_rel_spy_1m": ita_rel_spy_1m,

            "soxx_1m": soxx_1m,
            "soxx_3m": soxx_3m,
            "soxx_rel_spy_1m": soxx_rel_spy_1m,

            "hyg_1m": hyg_1m,
            "hyg_3m": hyg_3m,
            "hyg_6m": hyg_6m,
            "hyg_rel_spy_1m": hyg_rel_spy_1m,
            "hyg_rel_spy_3m": hyg_rel_spy_3m,
            "hyg_strength": hyg_strength,
            "credit_strength": credit_strength,

            "qqqm_rel_spy_1m": qqqm_rel_spy_1m,

            "war_strength": war_strength,
            "growth_strength": growth_strength,
            "risk_off_strength": risk_off_strength,

            "is_QQQM": is_QQQM,
            "is_XLE": is_XLE,
            "is_XSOE": is_XSOE,

            "yield_curve_QQQM": yield_curve * is_QQQM,

            "usd_XSOE": usd_1m * is_XSOE,
            "usd_3m_XSOE": usd_3m * is_XSOE,
            "usd_6m_XSOE": usd_6m * is_XSOE,
            "usd_level_XSOE": usd_level_strength * is_XSOE,
            "usd_1m_strength_XSOE": usd_1m_strength * is_XSOE,
            "usd_3m_strength_XSOE": usd_3m_strength * is_XSOE,

            "copper_XSOE": copper_strength * is_XSOE,

            "hyg_QQQM": hyg_strength * is_QQQM,
            "hyg_XSOE": hyg_strength * is_XSOE,
            "credit_QQQM": credit_strength * is_QQQM,
            "credit_XSOE": credit_strength * is_XSOE,

            "war_XLE": war_strength * is_XLE,
            "growth_QQQM": growth_strength * is_QQQM,
            "risk_off_QQQM": risk_off_strength * is_QQQM,
            "risk_off_XSOE": risk_off_strength * is_XSOE
        })

        df["target"] = px.shift(-forward_return_days) / px - 1.0
        features_by_asset[asset] = df

    return features_by_asset

# =========================================
# 8. TRAINING HELPERS
# =========================================
def build_train_data(features_by_asset, asset_list, end_loc, train_window):
    start_loc = end_loc - train_window
    if start_loc < 0:
        return None, None

    x_parts = []
    y_parts = []

    for asset in asset_list:
        df = features_by_asset[asset].iloc[start_loc:end_loc].copy().dropna()
        if df.empty:
            continue

        x_parts.append(df.drop(columns=["target"]))
        y_parts.append(df["target"])

    if len(x_parts) == 0:
        return None, None

    x_train = pd.concat(x_parts, axis=0)
    y_train = pd.concat(y_parts, axis=0)

    common_idx = x_train.index.intersection(y_train.index)
    x_train = x_train.loc[common_idx]
    y_train = y_train.loc[common_idx]

    if len(x_train) == 0:
        return None, None

    return x_train, y_train

def get_today_features(features_by_asset, asset: str, date: pd.Timestamp):
    row = features_by_asset[asset].loc[[date]].drop(columns=["target"], errors="ignore")
    if row.empty:
        return None
    if row.isna().any(axis=1).iloc[0]:
        return None
    return row

def apply_regime_overlay(raw_preds: dict, date: pd.Timestamp):
    preds = raw_preds.copy()

    war = float(war_strength.loc[date]) if date in war_strength.index and pd.notna(war_strength.loc[date]) else 0.0
    growth = float(growth_strength.loc[date]) if date in growth_strength.index and pd.notna(growth_strength.loc[date]) else 0.0
    risk_off = float(risk_off_strength.loc[date]) if date in risk_off_strength.index and pd.notna(risk_off_strength.loc[date]) else 0.0
    soxx = float(soxx_strength.loc[date]) if date in soxx_strength.index and pd.notna(soxx_strength.loc[date]) else 0.0
    copper = float(copper_strength.loc[date]) if date in copper_strength.index and pd.notna(copper_strength.loc[date]) else 0.0
    usd_regime = float(usd_3m_strength.loc[date]) if date in usd_3m_strength.index and pd.notna(usd_3m_strength.loc[date]) else 0.0
    hyg_regime = float(hyg_strength.loc[date]) if date in hyg_strength.index and pd.notna(hyg_strength.loc[date]) else 0.0
    credit_regime = float(credit_strength.loc[date]) if date in credit_strength.index and pd.notna(credit_strength.loc[date]) else 0.0

    adjusted = preds.copy()

    war_pos = max(0.0, war)
    growth_pos = max(0.0, growth)
    risk_off_pos = max(0.0, risk_off)
    soxx_pos = max(0.0, soxx)
    copper_pos = max(0.0, copper)
    usd_regime_pos = max(0.0, usd_regime)
    hyg_pos = max(0.0, hyg_regime)
    credit_pos = max(0.0, credit_regime)

    scale = overlay_scale

    if war_pos > 0:
        adjusted["XLE"] += scale * war_pos
        adjusted["QQQM"] -= scale * 0.5 * war_pos
        adjusted["XSOE"] -= scale * 0.3 * war_pos

    if growth_pos > 0:
        adjusted["QQQM"] += scale * growth_pos
        adjusted["XLE"] -= scale * 0.4 * growth_pos

    if soxx_pos > 0:
        adjusted["QQQM"] += scale * 0.8 * soxx_pos

    if copper_pos > 0:
        adjusted["XSOE"] += scale * 0.8 * copper_pos

    if usd_regime_pos > 0:
        adjusted["XSOE"] -= scale * 0.6 * usd_regime_pos

    if hyg_pos > 0:
        adjusted["QQQM"] += scale * 0.35 * hyg_pos
        adjusted["XSOE"] += scale * 0.45 * hyg_pos

    if credit_pos > 0:
        adjusted["QQQM"] += scale * 0.25 * credit_pos
        adjusted["XSOE"] += scale * 0.35 * credit_pos

    if risk_off_pos > 0:
        adjusted["QQQM"] -= scale * 1.2 * risk_off_pos
        adjusted["XSOE"] -= scale * 1.0 * risk_off_pos

    if war_pos > 0 and risk_off_pos > 1.0:
        adjusted["XLE"] -= scale * 0.4 * risk_off_pos

    overlay_info = {
        "war_strength": war,
        "growth_strength": growth,
        "risk_off_strength": risk_off,
        "soxx_strength": soxx,
        "copper_strength": copper,
        "usd_3m_strength": usd_regime,
        "hyg_strength": hyg_regime,
        "credit_strength": credit_regime
    }

    return adjusted, overlay_info

def should_go_cash(top_score: float, second_score: float, risk_off_strength_val: float):
    threshold = 0.0
    if risk_off_strength_val > 1.0:
        threshold += risk_off_cash_threshold
    return (top_score < threshold) and (second_score < threshold)

def tqqq_replace_fraction(top_asset: str, top_score: float, second_score: float, overlay_info: dict, date: pd.Timestamp):
    if top_asset != "QQQM":
        return 0.0

    gap = top_score - second_score
    vix_now = float(vix_level.loc[date]) if date in vix_level.index and pd.notna(vix_level.loc[date]) else np.nan
    growth = overlay_info["growth_strength"]
    soxx = overlay_info["soxx_strength"]
    risk_off = overlay_info["risk_off_strength"]

    # Check strong first
    strong = (
        gap >= tiered_tqqq_rule["strong"]["gap_min"] and
        top_score >= tiered_tqqq_rule["strong"]["top_score_min"] and
        growth >= tiered_tqqq_rule["strong"]["growth_min"] and
        soxx >= tiered_tqqq_rule["strong"]["soxx_min"] and
        risk_off <= tiered_tqqq_rule["strong"]["risk_off_max"] and
        (pd.isna(vix_now) or vix_now <= tiered_tqqq_rule["strong"]["vix_max"])
    )
    if strong:
        return tiered_tqqq_rule["strong"]["replace_fraction"]

    moderate = (
        gap >= tiered_tqqq_rule["moderate"]["gap_min"] and
        top_score >= tiered_tqqq_rule["moderate"]["top_score_min"] and
        growth >= tiered_tqqq_rule["moderate"]["growth_min"] and
        soxx >= tiered_tqqq_rule["moderate"]["soxx_min"] and
        risk_off <= tiered_tqqq_rule["moderate"]["risk_off_max"] and
        (pd.isna(vix_now) or vix_now <= tiered_tqqq_rule["moderate"]["vix_max"])
    )
    if moderate:
        return tiered_tqqq_rule["moderate"]["replace_fraction"]

    return 0.0

def build_execution_weights(signal_weights: dict, overlay_fraction: float):
    exec_weights = {
        "TQQQ": 0.0,
        "QQQM": 0.0,
        "XLE": 0.0,
        "XSOE": 0.0,
        "BIL": 0.0
    }

    qqqm_signal = signal_weights.get("QQQM", 0.0)

    # Replace part of QQQM sleeve with TQQQ
    exec_weights["TQQQ"] = qqqm_signal * overlay_fraction
    exec_weights["QQQM"] = qqqm_signal * (1.0 - overlay_fraction)
    exec_weights["XLE"] = signal_weights.get("XLE", 0.0)
    exec_weights["XSOE"] = signal_weights.get("XSOE", 0.0)
    exec_weights["BIL"] = signal_weights.get("BIL", 0.0)

    return exec_weights

# =========================================
# 9. MAIN BACKTEST
# =========================================
def run_strategy(features_by_asset):
    signal_universe = sector_etfs + [cash_etf]
    exec_universe = ["TQQQ", "QQQM", "XLE", "XSOE", "BIL"]
    dates = prices.index

    min_needed = max(train_window, 252) + 1
    max_loc = len(dates) - forward_return_days
    rebalance_locs = list(range(min_needed, max_loc, rebalance_step))

    portfolio_daily_returns = pd.Series(index=dates, dtype=float)

    current_exec_weights = {a: 0.0 for a in exec_universe}
    current_exec_weights["BIL"] = 1.0

    rebalance_records = []
    turnover_list = []

    for i, loc in enumerate(rebalance_locs):
        rebalance_date = dates[loc]

        x_train, y_train = build_train_data(features_by_asset, sector_etfs, loc, train_window)
        if x_train is None or len(x_train) < 50:
            continue

        model = RandomForestRegressor(**rf_params)
        model.fit(x_train, y_train)

        raw_preds = {}
        for asset in sector_etfs:
            x_today = get_today_features(features_by_asset, asset, rebalance_date)
            if x_today is None:
                continue
            raw_preds[asset] = float(model.predict(x_today)[0])

        if len(raw_preds) < 2:
            continue

        adjusted_preds, overlay_info = apply_regime_overlay(raw_preds, rebalance_date)

        ranked = sorted(adjusted_preds.items(), key=lambda x: x[1], reverse=True)
        top_asset = ranked[0][0]
        second_asset = ranked[1][0]
        top_score = ranked[0][1]
        second_score = ranked[1][1]

        w_top, w_second, score_gap = get_conviction_weights(top_score, second_score)

        signal_weights = {a: 0.0 for a in signal_universe}
        signal_weights[top_asset] = w_top
        signal_weights[second_asset] = w_second

        if should_go_cash(top_score, second_score, overlay_info["risk_off_strength"]):
            signal_weights = {a: 0.0 for a in signal_universe}
            signal_weights[cash_etf] = 1.0

        overlay_fraction = tqqq_replace_fraction(
            top_asset=top_asset,
            top_score=top_score,
            second_score=second_score,
            overlay_info=overlay_info,
            date=rebalance_date
        )

        exec_weights = build_execution_weights(signal_weights, overlay_fraction)

        turnover = compute_turnover(current_exec_weights, exec_weights, exec_universe)
        turnover_list.append(turnover)

        next_loc = rebalance_locs[i + 1] if i + 1 < len(rebalance_locs) else max_loc
        hold_dates = dates[loc + 1: next_loc + 1]

        if len(hold_dates) == 0:
            continue

        hold_rets = pd.Series(index=hold_dates, data=0.0)

        for asset, w in exec_weights.items():
            if w != 0:
                hold_rets = hold_rets.add(
                    w * asset_returns[asset].reindex(hold_dates).fillna(0.0),
                    fill_value=0.0
                )

        cost = turnover * transaction_cost
        hold_rets.iloc[0] -= cost

        portfolio_daily_returns.loc[hold_dates] = hold_rets.values
        current_exec_weights = exec_weights.copy()

        rebalance_records.append({
            "date": rebalance_date,
            "top_asset": top_asset,
            "second_asset": second_asset,
            "top_score": top_score,
            "second_score": second_score,
            "score_gap": score_gap,
            "turnover": turnover,
            "tx_cost_applied": cost,
            "war_strength": overlay_info["war_strength"],
            "growth_strength": overlay_info["growth_strength"],
            "risk_off_strength": overlay_info["risk_off_strength"],
            "soxx_strength": overlay_info["soxx_strength"],
            "copper_strength": overlay_info["copper_strength"],
            "usd_3m_strength": overlay_info["usd_3m_strength"],
            "hyg_strength": overlay_info["hyg_strength"],
            "credit_strength": overlay_info["credit_strength"],
            "overlay_fraction": overlay_fraction,
            "raw_pred_QQQM": raw_preds.get("QQQM", np.nan),
            "raw_pred_XLE": raw_preds.get("XLE", np.nan),
            "raw_pred_XSOE": raw_preds.get("XSOE", np.nan),
            "adj_pred_QQQM": adjusted_preds.get("QQQM", np.nan),
            "adj_pred_XLE": adjusted_preds.get("XLE", np.nan),
            "adj_pred_XSOE": adjusted_preds.get("XSOE", np.nan),
            "signal_w_QQQM": signal_weights.get("QQQM", 0.0),
            "signal_w_XLE": signal_weights.get("XLE", 0.0),
            "signal_w_XSOE": signal_weights.get("XSOE", 0.0),
            "signal_w_BIL": signal_weights.get("BIL", 0.0),
            "exec_w_TQQQ": exec_weights.get("TQQQ", 0.0),
            "exec_w_QQQM": exec_weights.get("QQQM", 0.0),
            "exec_w_XLE": exec_weights.get("XLE", 0.0),
            "exec_w_XSOE": exec_weights.get("XSOE", 0.0),
            "exec_w_BIL": exec_weights.get("BIL", 0.0),
        })

    portfolio_daily_returns = portfolio_daily_returns.dropna()
    rebalance_df = pd.DataFrame(rebalance_records)
    avg_turnover = np.mean(turnover_list) if turnover_list else np.nan

    return portfolio_daily_returns, rebalance_df, avg_turnover

# =========================================
# 10. LATEST RECOMMENDATION
# =========================================
def get_latest_recommendation(features_by_asset):
    signal_universe = sector_etfs + [cash_etf]
    dates = prices.index

    for latest_loc in range(len(dates) - 1, train_window, -1):
        latest_date = dates[latest_loc]

        x_train, y_train = build_train_data(features_by_asset, sector_etfs, latest_loc, train_window)
        if x_train is None or len(x_train) < 50:
            continue

        model = RandomForestRegressor(**rf_params)
        model.fit(x_train, y_train)

        raw_preds = {}
        for asset in sector_etfs:
            x_today = get_today_features(features_by_asset, asset, latest_date)
            if x_today is None:
                continue
            raw_preds[asset] = float(model.predict(x_today)[0])

        if len(raw_preds) < 2:
            continue

        adjusted_preds, overlay_info = apply_regime_overlay(raw_preds, latest_date)

        ranked = sorted(adjusted_preds.items(), key=lambda x: x[1], reverse=True)
        top_asset = ranked[0][0]
        second_asset = ranked[1][0]
        top_score = ranked[0][1]
        second_score = ranked[1][1]

        w_top, w_second, score_gap = get_conviction_weights(top_score, second_score)

        signal_weights = {a: 0.0 for a in signal_universe}
        signal_weights[top_asset] = w_top
        signal_weights[second_asset] = w_second

        if should_go_cash(top_score, second_score, overlay_info["risk_off_strength"]):
            signal_weights = {a: 0.0 for a in signal_universe}
            signal_weights[cash_etf] = 1.0

        overlay_fraction = tqqq_replace_fraction(
            top_asset=top_asset,
            top_score=top_score,
            second_score=second_score,
            overlay_info=overlay_info,
            date=latest_date
        )

        exec_weights = build_execution_weights(signal_weights, overlay_fraction)

        feature_importance_df = pd.DataFrame({
            "feature": x_train.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)

        return {
            "date": latest_date,
            "raw_predictions": raw_preds,
            "adjusted_predictions": adjusted_preds,
            "signal_weights": signal_weights,
            "exec_weights": exec_weights,
            "feature_importance": feature_importance_df,
            "top_asset": top_asset,
            "second_asset": second_asset,
            "top_score": top_score,
            "second_score": second_score,
            "score_gap": score_gap,
            "overlay_fraction": overlay_fraction,
            "war_strength": overlay_info["war_strength"],
            "growth_strength": overlay_info["growth_strength"],
            "risk_off_strength": overlay_info["risk_off_strength"],
            "soxx_strength": overlay_info["soxx_strength"],
            "copper_strength": overlay_info["copper_strength"],
            "usd_3m_strength": overlay_info["usd_3m_strength"],
            "hyg_strength": overlay_info["hyg_strength"],
            "credit_strength": overlay_info["credit_strength"]
        }

    return None

# =========================================
# 11. PRINT HELPERS
# =========================================
def print_weights(title: str, weights: dict, order: list):
    print(f"\n=== {title} ===")
    for asset in order:
        print(f"{asset}: {weights.get(asset, 0.0):.1%}")

# =========================================
# 12. BUILD FEATURES
# =========================================
print("\nBuilding C+ ELITE USD + HYG SHORT feature set...")
features_model = build_features_by_asset()

# =========================================
# 13. RUN MODEL
# =========================================
print("Running C+ ELITE USD + HYG SHORT CONVICTION + TQQQ TIERED OVERLAY...")
portfolio_returns, rebalance_df, avg_turnover = run_strategy(features_model)

if len(portfolio_returns) == 0:
    raise ValueError("No portfolio returns produced.")

# =========================================
# 14. OUTPUT
# =========================================
print("\n=== C+ ELITE USD + HYG SHORT CONVICTION + TQQQ TIERED OVERLAY PERFORMANCE ===")
print(f"Annual Return: {annualized_return(portfolio_returns):.3f}")
print(f"Volatility:    {annualized_volatility(portfolio_returns):.3f}")
print(f"Sharpe:        {sharpe_ratio(portfolio_returns, risk_free_rate_annual):.3f}")
print(f"Max Drawdown:  {max_drawdown(portfolio_returns):.3f}")
print(f"Avg Turnover:  {avg_turnover:.3f}")

if not rebalance_df.empty:
    print("\n=== Last 10 Rebalances ===")
    print(rebalance_df.tail(10).to_string(index=False))

latest = get_latest_recommendation(features_model)

if latest is not None:
    print("\n=== Latest Model Recommendation ===")
    print("Signal date:", latest["date"].date())

    print("\nRaw predicted next-period returns:")
    for k, v in sorted(latest["raw_predictions"].items(), key=lambda x: x[1], reverse=True):
        print(f"{k}: {v:.4f}")

    print("\nAdjusted predicted next-period returns:")
    for k, v in sorted(latest["adjusted_predictions"].items(), key=lambda x: x[1], reverse=True):
        print(f"{k}: {v:.4f}")

    print(f"\nOverlay fraction on QQQM sleeve: {latest['overlay_fraction']:.1%}")
    print(f"Top asset: {latest['top_asset']}")
    print(f"Second asset: {latest['second_asset']}")
    print(f"Top score: {latest['top_score']:.4f}")
    print(f"Second score: {latest['second_score']:.4f}")
    print(f"Score gap: {latest['score_gap']:.4f}")
    print(f"Growth strength: {latest['growth_strength']:.3f}")
    print(f"SOXX strength: {latest['soxx_strength']:.3f}")
    print(f"Risk-off strength: {latest['risk_off_strength']:.3f}")
    print(f"USD 3M strength: {latest['usd_3m_strength']:.3f}")
    print(f"Credit strength: {latest['credit_strength']:.3f}")

    print_weights("Suggested SIGNAL Weights", latest["signal_weights"], ["QQQM", "XLE", "XSOE", "BIL"])
    print_weights("Suggested EXECUTED Weights", latest["exec_weights"], ["TQQQ", "QQQM", "XLE", "XSOE", "BIL"])

    print("\n=== Latest Feature Importance Summary ===")
    print(latest["feature_importance"].head(20).to_string(index=False))

# =========================================
# 15. SAVE OUTPUTS
# =========================================
portfolio_returns.to_csv(
    "model_c_plus_usd_hyg_short_conviction_tqqq_tiered_overlay_portfolio_daily_returns.csv",
    header=["portfolio_return"]
)
rebalance_df.to_csv(
    "model_c_plus_usd_hyg_short_conviction_tqqq_tiered_overlay_rebalance_log.csv",
    index=False
)

if latest is not None:
    latest_df = pd.DataFrame([{
        "signal_date": latest["date"],
        "latest_data_date": prices.index[-1],
        "top_asset": latest["top_asset"],
        "second_asset": latest["second_asset"],
        "top_score": latest["top_score"],
        "second_score": latest["second_score"],
        "score_gap": latest["score_gap"],
        "overlay_fraction": latest["overlay_fraction"],
        "war_strength": latest["war_strength"],
        "growth_strength": latest["growth_strength"],
        "risk_off_strength": latest["risk_off_strength"],
        "soxx_strength": latest["soxx_strength"],
        "copper_strength": latest["copper_strength"],
        "usd_3m_strength": latest["usd_3m_strength"],
        "hyg_strength": latest["hyg_strength"],
        "credit_strength": latest["credit_strength"],
        **{f"signal_w_{k}": v for k, v in latest["signal_weights"].items()},
        **{f"exec_w_{k}": v for k, v in latest["exec_weights"].items()},
        **{f"raw_pred_{k}": v for k, v in latest["raw_predictions"].items()},
        **{f"adj_pred_{k}": v for k, v in latest["adjusted_predictions"].items()}
    }])
    latest_df.to_csv(
        "model_c_plus_usd_hyg_short_conviction_tqqq_tiered_overlay_latest_recommendation.csv",
        index=False
    )
    latest["feature_importance"].to_csv(
        "model_c_plus_usd_hyg_short_conviction_tqqq_tiered_overlay_feature_importance.csv",
        index=False
    )

print("\nSaved:")
print("- model_c_plus_usd_hyg_short_conviction_tqqq_tiered_overlay_portfolio_daily_returns.csv")
print("- model_c_plus_usd_hyg_short_conviction_tqqq_tiered_overlay_rebalance_log.csv")
if latest is not None:
    print("- model_c_plus_usd_hyg_short_conviction_tqqq_tiered_overlay_latest_recommendation.csv")
    print("- model_c_plus_usd_hyg_short_conviction_tqqq_tiered_overlay_feature_importance.csv")
