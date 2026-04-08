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

all_assets = sector_etfs + [cash_etf, spy_etf] + feature_etfs

start_date = "2010-01-01"
end_date = None

forward_return_days = 10
train_window = 252 * 3
rebalance_step = 10
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
# 2. PERFORMANCE FUNCTIONS
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
# 2B. CONVICTION WEIGHTING
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
# 3. DOWNLOAD DATA
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

required_cols = sector_etfs + [cash_etf, spy_etf, "ITA", "SOXX", "HYG"]
prices = prices.dropna(subset=required_cols)
macro_raw = macro_raw.reindex(prices.index).ffill()

print(f"Latest available data date: {prices.index[-1].date()}")

# =========================================
# 4. COMMON SERIES
# =========================================
asset_returns = prices[sector_etfs + [cash_etf, spy_etf]].pct_change()

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
# 4B. CONTINUOUS REGIME STRENGTH
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
# 5. FEATURE BUILDER
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
# 6. TRAINING HELPERS
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

# =========================================
# 7. MAIN BACKTEST / WALK-FORWARD
# =========================================
def run_strategy(features_by_asset):
    universe = sector_etfs + [cash_etf]
    dates = prices.index

    min_needed = max(train_window, 252) + 1
    max_loc = len(dates) - forward_return_days
    rebalance_locs = list(range(min_needed, max_loc, rebalance_step))

    portfolio_daily_returns = pd.Series(index=dates, dtype=float)
    current_weights = {a: 0.0 for a in universe}
    current_weights[cash_etf] = 1.0

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

        new_weights = {a: 0.0 for a in universe}
        new_weights[top_asset] = w_top
        new_weights[second_asset] = w_second

        if should_go_cash(top_score, second_score, overlay_info["risk_off_strength"]):
            new_weights = {a: 0.0 for a in universe}
            new_weights[cash_etf] = 1.0

        turnover = compute_turnover(current_weights, new_weights, universe)
        turnover_list.append(turnover)

        next_loc = rebalance_locs[i + 1] if i + 1 < len(rebalance_locs) else max_loc
        hold_dates = dates[loc + 1: next_loc + 1]

        if len(hold_dates) == 0:
            continue

        hold_rets = pd.Series(index=hold_dates, data=0.0)

        for asset, w in new_weights.items():
            if w != 0:
                hold_rets = hold_rets.add(
                    w * asset_returns[asset].reindex(hold_dates).fillna(0.0),
                    fill_value=0.0
                )

        cost = turnover * transaction_cost
        hold_rets.iloc[0] -= cost

        portfolio_daily_returns.loc[hold_dates] = hold_rets.values
        current_weights = new_weights.copy()

        rebalance_records.append({
            "date": rebalance_date,
            "top_asset": top_asset,
            "second_asset": second_asset,
            "top_score": top_score,
            "second_score": second_score,
            "score_gap": score_gap,
            "top_weight": new_weights.get(top_asset, 0.0),
            "second_weight": new_weights.get(second_asset, 0.0),
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
            "raw_pred_QQQM": raw_preds.get("QQQM", np.nan),
            "raw_pred_XLE": raw_preds.get("XLE", np.nan),
            "raw_pred_XSOE": raw_preds.get("XSOE", np.nan),
            "adj_pred_QQQM": adjusted_preds.get("QQQM", np.nan),
            "adj_pred_XLE": adjusted_preds.get("XLE", np.nan),
            "adj_pred_XSOE": adjusted_preds.get("XSOE", np.nan),
            "w_QQQM": new_weights.get("QQQM", 0.0),
            "w_XLE": new_weights.get("XLE", 0.0),
            "w_XSOE": new_weights.get("XSOE", 0.0),
            "w_BIL": new_weights.get("BIL", 0.0)
        })

    portfolio_daily_returns = portfolio_daily_returns.dropna()
    rebalance_df = pd.DataFrame(rebalance_records)
    avg_turnover = np.mean(turnover_list) if turnover_list else np.nan

    return portfolio_daily_returns, rebalance_df, avg_turnover

# =========================================
# 8. LATEST LIVE SIGNAL
# =========================================
def get_latest_recommendation(features_by_asset):
    universe = sector_etfs + [cash_etf]
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

        weights = {a: 0.0 for a in universe}
        weights[top_asset] = w_top
        weights[second_asset] = w_second

        if should_go_cash(top_score, second_score, overlay_info["risk_off_strength"]):
            weights = {a: 0.0 for a in universe}
            weights[cash_etf] = 1.0

        feature_importance_df = pd.DataFrame({
            "feature": x_train.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)

        return {
            "date": latest_date,
            "raw_predictions": raw_preds,
            "adjusted_predictions": adjusted_preds,
            "weights": weights,
            "feature_importance": feature_importance_df,
            "top_asset": top_asset,
            "second_asset": second_asset,
            "top_score": top_score,
            "second_score": second_score,
            "score_gap": score_gap,
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
# 9. PRINT HELPERS
# =========================================
def print_weight_block(title: str, weights: dict):
    print(f"\n=== {title} ===")
    for asset in [*sector_etfs, cash_etf]:
        print(f"{asset}: {weights.get(asset, 0.0):.1%}")

def print_live_decision_summary(rebalance_df: pd.DataFrame, latest: dict):
    if rebalance_df.empty or latest is None:
        print("\nLive decision summary unavailable.")
        return

    universe = sector_etfs + [cash_etf]

    prev = rebalance_df.iloc[-1]
    prev_weights = {
        "QQQM": float(prev["w_QQQM"]),
        "XLE": float(prev["w_XLE"]),
        "XSOE": float(prev["w_XSOE"]),
        "BIL": float(prev["w_BIL"])
    }

    new_weights = latest["weights"]
    turnover_now = compute_turnover(prev_weights, new_weights, universe)
    changed = turnover_now > 1e-12

    print("\n=== Previous Rebalance ===")
    print("Date:", pd.to_datetime(prev["date"]).date())
    print(f"Top asset: {prev['top_asset']}")
    print(f"Second asset: {prev['second_asset']}")
    print(f"Top score: {prev['top_score']:.4f}")
    print(f"Second score: {prev['second_score']:.4f}")
    print(f"Score gap: {prev['score_gap']:.4f}")
    print(f"War strength: {prev['war_strength']:.3f}")
    print(f"Growth strength: {prev['growth_strength']:.3f}")
    print(f"Risk-off strength: {prev['risk_off_strength']:.3f}")
    print(f"SOXX strength: {prev['soxx_strength']:.3f}")
    print(f"Copper strength: {prev['copper_strength']:.3f}")
    print(f"USD 3M strength: {prev['usd_3m_strength']:.3f}")
    print(f"HYG strength: {prev['hyg_strength']:.3f}")
    print(f"Credit strength: {prev['credit_strength']:.3f}")

    print_weight_block("Previous Allocation", prev_weights)

    print("\n=== Current Recommended Rebalance ===")
    print("Signal date:", latest["date"].date())
    print(f"Top asset: {latest['top_asset']}")
    print(f"Second asset: {latest['second_asset']}")
    print(f"Top score: {latest['top_score']:.4f}")
    print(f"Second score: {latest['second_score']:.4f}")
    print(f"Score gap: {latest['score_gap']:.4f}")
    print(f"War strength: {latest['war_strength']:.3f}")
    print(f"Growth strength: {latest['growth_strength']:.3f}")
    print(f"Risk-off strength: {latest['risk_off_strength']:.3f}")
    print(f"SOXX strength: {latest['soxx_strength']:.3f}")
    print(f"Copper strength: {latest['copper_strength']:.3f}")
    print(f"USD 3M strength: {latest['usd_3m_strength']:.3f}")
    print(f"HYG strength: {latest['hyg_strength']:.3f}")
    print(f"Credit strength: {latest['credit_strength']:.3f}")

    print_weight_block("New Allocation", new_weights)

    print("\n=== Rotation Decision ===")
    print("Whether changed:", "YES" if changed else "NO")
    print("Turnover from previous rebalance:", f"{turnover_now:.3f}")
    print("Action:", "ROTATE this period" if changed else "NO CHANGE this period")

    print("\n=== Latest Feature Importance Summary ===")
    print(latest["feature_importance"].head(20).to_string(index=False))

# =========================================
# 10. BUILD FEATURES
# =========================================
print("\nBuilding C+ ELITE USD + HYG SHORT feature set...")
features_model_c_plus_usd_hyg_short = build_features_by_asset()

# =========================================
# 11. RUN MODEL
# =========================================
print("Running C+ ELITE USD + HYG SHORT CONVICTION...")
portfolio_returns, rebalance_df, avg_turnover = run_strategy(features_model_c_plus_usd_hyg_short)

if len(portfolio_returns) == 0:
    raise ValueError("No portfolio returns produced for C+ ELITE USD + HYG SHORT CONVICTION.")

# =========================================
# 12. PERFORMANCE OUTPUT
# =========================================
print("\n=== C+ ELITE USD + HYG SHORT CONVICTION PERFORMANCE ===")
print(f"Annual Return: {annualized_return(portfolio_returns):.3f}")
print(f"Volatility:    {annualized_volatility(portfolio_returns):.3f}")
print(f"Sharpe:        {sharpe_ratio(portfolio_returns, risk_free_rate_annual):.3f}")
print(f"Max Drawdown:  {max_drawdown(portfolio_returns):.3f}")
print(f"Avg Turnover:  {avg_turnover:.3f}")

if not rebalance_df.empty:
    print("\n=== Last 10 Rebalances ===")
    print(rebalance_df.tail(10).to_string(index=False))

latest = get_latest_recommendation(features_by_asset=features_model_c_plus_usd_hyg_short)

if latest is None:
    print("\nNo valid latest recommendation could be produced.")
else:
    print("\n=== Latest Model Recommendation ===")
    print("Signal date:", latest["date"].date())

    print("\nRaw predicted next-period returns:")
    for k, v in sorted(latest["raw_predictions"].items(), key=lambda x: x[1], reverse=True):
        print(f"{k}: {v:.4f}")

    print("\nAdjusted predicted next-period returns:")
    for k, v in sorted(latest["adjusted_predictions"].items(), key=lambda x: x[1], reverse=True):
        print(f"{k}: {v:.4f}")

    print_weight_block("Suggested Weights", latest["weights"])
    print_live_decision_summary(rebalance_df, latest)

# =========================================
# 13. SAVE OUTPUTS
# =========================================
portfolio_returns.to_csv("model_c_plus_usd_hyg_short_conviction_portfolio_daily_returns.csv", header=["portfolio_return"])
rebalance_df.to_csv("model_c_plus_usd_hyg_short_conviction_rebalance_log.csv", index=False)

if latest is not None:
    latest_weights_df = pd.DataFrame([{
        "signal_date": latest["date"],
        "latest_data_date": prices.index[-1],
        "top_asset": latest["top_asset"],
        "second_asset": latest["second_asset"],
        "top_score": latest["top_score"],
        "second_score": latest["second_score"],
        "score_gap": latest["score_gap"],
        "war_strength": latest["war_strength"],
        "growth_strength": latest["growth_strength"],
        "risk_off_strength": latest["risk_off_strength"],
        "soxx_strength": latest["soxx_strength"],
        "copper_strength": latest["copper_strength"],
        "usd_3m_strength": latest["usd_3m_strength"],
        "hyg_strength": latest["hyg_strength"],
        "credit_strength": latest["credit_strength"],
        **latest["weights"],
        **{f"raw_pred_{k}": v for k, v in latest["raw_predictions"].items()},
        **{f"adj_pred_{k}": v for k, v in latest["adjusted_predictions"].items()}
    }])
    latest_weights_df.to_csv("model_c_plus_usd_hyg_short_conviction_latest_recommendation.csv", index=False)
    latest["feature_importance"].to_csv("model_c_plus_usd_hyg_short_conviction_feature_importance.csv", index=False)

print("\nSaved:")
print("- model_c_plus_usd_hyg_short_conviction_portfolio_daily_returns.csv")
print("- model_c_plus_usd_hyg_short_conviction_rebalance_log.csv")
if latest is not None:
    print("- model_c_plus_usd_hyg_short_conviction_latest_recommendation.csv")
    print("- model_c_plus_usd_hyg_short_conviction_feature_importance.csv")