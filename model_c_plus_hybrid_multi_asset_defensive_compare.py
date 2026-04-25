import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

print("Current working directory:", os.getcwd())

# ============================================================
# MODEL C+ UPGRADE:
# - Baseline universe: QQQM, XLE, XSOE, BIL with TQQQ overlay
# - Upgraded universe: QQQM, XLE, XSOE, XLI, XLB, BIL with TQQQ overlay
# - Adds industrial/materials regime logic:
#       XLB = copper/materials/early industrial cycle
#       XLI = industrial/reshoring/capex cycle
# - Includes old-vs-new comparison in one run
# - Adds a V2 overlay test:
#       V1 = original XLI/XLB overlay
#       V2 = stricter industrial regime classifier to reduce false signals
# ============================================================

# ============================================================
# 1. SETTINGS
# ============================================================
BASELINE_SECTOR_ETFS = ["QQQM", "XLE", "XSOE"]
UPGRADED_SECTOR_ETFS = ["QQQM", "XLE", "XSOE", "XLI", "XLB"]

cash_etf = "BIL"
spy_etf = "SPY"
feature_etfs = ["ITA", "SOXX", "HYG"]
execution_extra = ["TQQQ", "ERX", "UXI"]

# Download everything needed by both models.
all_assets = sorted(set(
    BASELINE_SECTOR_ETFS
    + UPGRADED_SECTOR_ETFS
    + [cash_etf, spy_etf]
    + feature_etfs
    + execution_extra
))

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
    "n_jobs": -1,
}

overlay_scale = 0.002
risk_off_cash_threshold = 0.01
zscore_window = 252

# TQQQ overlay only applies when QQQM is the top signal asset.
tiered_tqqq_rule = {
    "moderate": {
        "gap_min": 0.003,
        "top_score_min": 0.010,
        "growth_min": 0.30,
        "soxx_min": 0.30,
        "risk_off_max": 1.00,
        "vix_max": 25.0,
        "replace_fraction": 0.50,
    },
    "strong": {
        "gap_min": 0.004,
        "top_score_min": 0.010,
        "growth_min": 0.20,
        "soxx_min": 0.20,
        "risk_off_max": 0.75,
        "vix_max": 22.0,
        "replace_fraction": 1.00,
    },
}

# ============================================================
# 2. PERFORMANCE FUNCTIONS
# ============================================================
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


def performance_summary(name: str, returns: pd.Series, avg_turnover: float) -> dict:
    return {
        "model": name,
        "annual_return": annualized_return(returns),
        "volatility": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns, risk_free_rate_annual),
        "max_drawdown": max_drawdown(returns),
        "avg_turnover": avg_turnover,
        "start": returns.dropna().index.min(),
        "end": returns.dropna().index.max(),
        "days": len(returns.dropna()),
    }

# ============================================================
# 3. DATA DOWNLOAD
# ============================================================
def download_close_data(tickers, start, end=None):
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
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
    "copper": "HG=F",
}
macro_raw = download_close_data(list(macro_tickers.values()), start_date, end_date)
macro_raw = macro_raw.rename(columns={v: k for k, v in macro_tickers.items()})
macro_raw = macro_raw.reindex(prices.index).ffill()

required_cols = all_assets
prices = prices.dropna(subset=required_cols)
macro_raw = macro_raw.reindex(prices.index).ffill()

print(f"Latest available data date: {prices.index[-1].date()}")

# ============================================================
# 4. COMMON SERIES / MACRO FEATURES
# ============================================================
asset_returns = prices[UPGRADED_SECTOR_ETFS + [cash_etf, spy_etf, "TQQQ", "ERX", "UXI"]].pct_change()

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
oil_3m = macro_raw["oil"].pct_change(63)

usd_1m = macro_raw["usd"].pct_change(21)
usd_3m = macro_raw["usd"].pct_change(63)
usd_6m = macro_raw["usd"].pct_change(126)
usd_level_strength = rolling_zscore(macro_raw["usd"], zscore_window)
usd_1m_strength = rolling_zscore(usd_1m, zscore_window)
usd_3m_strength = rolling_zscore(usd_3m, zscore_window)

vix_level = macro_raw["vix_level"]
vix_1m = macro_raw["vix_level"].pct_change(21)

copper_1m = macro_raw["copper"].pct_change(21)
copper_3m = macro_raw["copper"].pct_change(63)
copper_rel_spy_1m = copper_1m - spy_ret_1m
copper_rel_spy_3m = copper_3m - spy_ret_3m

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
xli_rel_spy_1m = prices["XLI"].pct_change(21) - spy_ret_1m
xlb_rel_spy_1m = prices["XLB"].pct_change(21) - spy_ret_1m

# Continuous regime strengths
ita_strength = rolling_zscore(ita_rel_spy_1m, zscore_window)
soxx_strength = rolling_zscore(soxx_rel_spy_1m, zscore_window)
qqqm_strength = rolling_zscore(qqqm_rel_spy_1m, zscore_window)
xli_strength = rolling_zscore(xli_rel_spy_1m, zscore_window)
xlb_strength = rolling_zscore(xlb_rel_spy_1m, zscore_window)
oil_strength = rolling_zscore(oil_1m, zscore_window)
vix_strength = rolling_zscore(vix_1m, zscore_window)
copper_strength = rolling_zscore(copper_rel_spy_1m, zscore_window)
copper_3m_strength = rolling_zscore(copper_rel_spy_3m, zscore_window)
hyg_strength = rolling_zscore(hyg_rel_spy_1m, zscore_window)

war_strength = ((ita_strength + oil_strength) / 2.0).fillna(0.0)
growth_strength = ((qqqm_strength + soxx_strength) / 2.0).fillna(0.0)
risk_off_strength = ((-qqqm_strength + vix_strength) / 2.0).fillna(0.0)
credit_strength = ((hyg_strength - vix_strength) / 2.0).fillna(0.0)

# New: industrial/materials score.
# Idea:
#   copper = raw materials demand
#   XLB relative strength = materials confirmation
#   XLI relative strength = industrial/capex confirmation
#   credit strength = risk-on confirmation
industrial_strength = (
    0.40 * copper_strength
    + 0.25 * xlb_strength
    + 0.20 * xli_strength
    + 0.15 * credit_strength
).fillna(0.0).clip(-3, 3)

materials_strength = (
    0.60 * copper_strength
    + 0.25 * copper_3m_strength
    + 0.15 * xlb_strength
).fillna(0.0).clip(-3, 3)

# ============================================================
# 5. FEATURE BUILDER
# ============================================================
def build_features_by_asset(sector_etfs: list):
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
        is_XLI = 1 if asset == "XLI" else 0
        is_XLB = 1 if asset == "XLB" else 0

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
            "oil_3m": oil_3m,

            "usd_1m": usd_1m,
            "usd_3m": usd_3m,
            "usd_6m": usd_6m,
            "usd_level_strength": usd_level_strength,
            "usd_1m_strength": usd_1m_strength,
            "usd_3m_strength": usd_3m_strength,

            "vix_level": vix_level,
            "vix_1m": vix_1m,

            "copper_1m": copper_1m,
            "copper_3m": copper_3m,
            "copper_rel_spy_1m": copper_rel_spy_1m,
            "copper_rel_spy_3m": copper_rel_spy_3m,
            "copper_strength": copper_strength,
            "copper_3m_strength": copper_3m_strength,

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
            "xli_rel_spy_1m": xli_rel_spy_1m,
            "xlb_rel_spy_1m": xlb_rel_spy_1m,

            "war_strength": war_strength,
            "growth_strength": growth_strength,
            "risk_off_strength": risk_off_strength,
            "industrial_strength": industrial_strength,
            "materials_strength": materials_strength,

            "is_QQQM": is_QQQM,
            "is_XLE": is_XLE,
            "is_XSOE": is_XSOE,
            "is_XLI": is_XLI,
            "is_XLB": is_XLB,

            # Asset-specific interactions.
            "yield_curve_QQQM": yield_curve * is_QQQM,

            "usd_XSOE": usd_1m * is_XSOE,
            "usd_3m_XSOE": usd_3m * is_XSOE,
            "usd_6m_XSOE": usd_6m * is_XSOE,
            "usd_level_XSOE": usd_level_strength * is_XSOE,
            "usd_1m_strength_XSOE": usd_1m_strength * is_XSOE,
            "usd_3m_strength_XSOE": usd_3m_strength * is_XSOE,

            "copper_XSOE": copper_strength * is_XSOE,
            "copper_XLB": copper_strength * is_XLB,
            "copper_3m_XLB": copper_3m_strength * is_XLB,
            "copper_XLI": copper_strength * is_XLI,

            "industrial_XLI": industrial_strength * is_XLI,
            "materials_XLB": materials_strength * is_XLB,
            "growth_XLI": growth_strength * is_XLI,

            "hyg_QQQM": hyg_strength * is_QQQM,
            "hyg_XSOE": hyg_strength * is_XSOE,
            "hyg_XLI": hyg_strength * is_XLI,
            "credit_QQQM": credit_strength * is_QQQM,
            "credit_XSOE": credit_strength * is_XSOE,
            "credit_XLI": credit_strength * is_XLI,

            "war_XLE": war_strength * is_XLE,
            "growth_QQQM": growth_strength * is_QQQM,
            "risk_off_QQQM": risk_off_strength * is_QQQM,
            "risk_off_XSOE": risk_off_strength * is_XSOE,
            "risk_off_XLI": risk_off_strength * is_XLI,
            "risk_off_XLB": risk_off_strength * is_XLB,
        })

        df["target"] = px.shift(-forward_return_days) / px - 1.0
        features_by_asset[asset] = df

    return features_by_asset

# ============================================================
# 6. TRAINING HELPERS
# ============================================================
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

# ============================================================
# 7. OVERLAY / ALLOCATION LOGIC
# ============================================================
def apply_regime_overlay(raw_preds: dict, date: pd.Timestamp, sector_etfs: list, overlay_style: str = "v1"):
    def val(series, default=0.0):
        if date in series.index and pd.notna(series.loc[date]):
            return float(series.loc[date])
        return default

    war = val(war_strength)
    growth = val(growth_strength)
    risk_off = val(risk_off_strength)
    soxx = val(soxx_strength)
    copper = val(copper_strength)
    copper3 = val(copper_3m_strength)
    industrial = val(industrial_strength)
    materials = val(materials_strength)
    usd_regime = val(usd_3m_strength)
    hyg_regime = val(hyg_strength)
    credit_regime = val(credit_strength)

    adjusted = raw_preds.copy()

    war_pos = max(0.0, war)
    growth_pos = max(0.0, growth)
    risk_off_pos = max(0.0, risk_off)
    soxx_pos = max(0.0, soxx)
    copper_pos = max(0.0, copper)
    copper3_pos = max(0.0, copper3)
    industrial_pos = max(0.0, industrial)
    materials_pos = max(0.0, materials)
    usd_regime_pos = max(0.0, usd_regime)
    hyg_pos = max(0.0, hyg_regime)
    credit_pos = max(0.0, credit_regime)
    scale = overlay_scale

    def add(asset, amount):
        if asset in adjusted:
            adjusted[asset] += amount

    # Existing logic preserved.
    if war_pos > 0:
        add("XLE", scale * war_pos)
        add("QQQM", -scale * 0.5 * war_pos)
        add("XSOE", -scale * 0.3 * war_pos)

    if growth_pos > 0:
        add("QQQM", scale * growth_pos)
        add("XLE", -scale * 0.4 * growth_pos)

    if soxx_pos > 0:
        add("QQQM", scale * 0.8 * soxx_pos)

    if copper_pos > 0:
        add("XSOE", scale * 0.8 * copper_pos)

    if usd_regime_pos > 0:
        add("XSOE", -scale * 0.6 * usd_regime_pos)

    if hyg_pos > 0:
        add("QQQM", scale * 0.35 * hyg_pos)
        add("XSOE", scale * 0.45 * hyg_pos)

    if credit_pos > 0:
        add("QQQM", scale * 0.25 * credit_pos)
        add("XSOE", scale * 0.35 * credit_pos)

    if risk_off_pos > 0:
        add("QQQM", -scale * 1.2 * risk_off_pos)
        add("XSOE", -scale * 1.0 * risk_off_pos)

    if war_pos > 0 and risk_off_pos > 1.0:
        add("XLE", -scale * 0.4 * risk_off_pos)

    # New XLI/XLB overlay logic.
    # V1 = original broad boost; V2 = stricter regime classifier.
    # Keep overlays modest because ML already sees the features.
    if overlay_style == "v1":
        # XLB: early industrial/materials/copper cycle.
        if "XLB" in sector_etfs:
            if copper_pos > 0:
                add("XLB", scale * 0.90 * copper_pos)
            if copper3_pos > 0:
                add("XLB", scale * 0.35 * copper3_pos)
            if materials_pos > 0:
                add("XLB", scale * 0.50 * materials_pos)
            if risk_off_pos > 0:
                add("XLB", -scale * 0.70 * risk_off_pos)

        # XLI: industrial/reshoring/capex cycle; likes industrial acceleration and credit support.
        if "XLI" in sector_etfs:
            if industrial_pos > 0:
                add("XLI", scale * 0.80 * industrial_pos)
            if growth_pos > 0:
                add("XLI", scale * 0.35 * growth_pos)
            if credit_pos > 0:
                add("XLI", scale * 0.25 * credit_pos)
            if risk_off_pos > 0:
                add("XLI", -scale * 0.80 * risk_off_pos)

    elif overlay_style in ("v2", "hybrid"):
        # V2 philosophy:
        # - XLB should be boosted only when materials/copper strength is confirmed.
        # - XLI should be boosted only when industrial strength is positive AND risk-off is not dominant.
        # - Strong USD/risk-off gets a small penalty because it often hurts global cyclicals/materials.
        industrial_regime_on = (industrial > 0.25) and (risk_off < 0.75) and (usd_regime < 1.50)
        materials_regime_on = (materials > 0.25) and (risk_off < 1.00)
        early_cycle_on = (copper3 > 0.50) and (credit_regime > -0.50) and (risk_off < 1.00)

        if "XLB" in sector_etfs:
            if materials_regime_on:
                add("XLB", scale * 0.70 * materials_pos)
            if early_cycle_on:
                add("XLB", scale * 0.30 * copper3_pos)
            if risk_off_pos > 0:
                add("XLB", -scale * 0.85 * risk_off_pos)
            if usd_regime > 1.0:
                add("XLB", -scale * 0.20 * usd_regime_pos)

        if "XLI" in sector_etfs:
            if industrial_regime_on:
                add("XLI", scale * 0.75 * industrial_pos)
                if growth_pos > 0:
                    add("XLI", scale * 0.20 * growth_pos)
                if credit_pos > 0:
                    add("XLI", scale * 0.20 * credit_pos)
            if risk_off_pos > 0:
                add("XLI", -scale * 0.90 * risk_off_pos)
            if usd_regime > 1.5:
                add("XLI", -scale * 0.15 * usd_regime_pos)

        # HYBRID extra: preserve fast-growth tech/TQQQ engine when semis + QQQM leadership are very strong.
        # This prevents XLI/XLB from diluting the original tech engine unless industrial/materials signals are truly active.
        if overlay_style == "hybrid":
            tech_regime_on = (growth > 1.00) and (soxx > 1.00) and (risk_off < 0.50)
            industrial_regime_on_h = (industrial > 0.50) and (copper3 > 0.25) and (risk_off < 0.75) and (usd_regime < 1.25)
            materials_regime_on_h = (materials > 0.50) and (copper3 > 0.50) and (risk_off < 0.90)
            if tech_regime_on:
                add("QQQM", scale * 0.45 * min(growth_pos + soxx_pos, 6.0))
                if "XLI" in sector_etfs and not industrial_regime_on_h:
                    add("XLI", -scale * 0.25 * max(0.0, -industrial))
                if "XLB" in sector_etfs and not materials_regime_on_h:
                    add("XLB", -scale * 0.25 * max(0.0, -materials))

            # NEW: Non-tech conviction boost.
            # Purpose: reduce tech bias when oil/industrial/materials regimes are truly strong.
            oil_regime = war
            if oil_regime > 0.5:
                add("XLE", scale * 1.5 * oil_regime)
            if industrial > 0.5:
                add("XLI", scale * 1.2 * industrial)
            if copper > 0.7:
                add("XLB", scale * 1.0 * copper)
            if oil_regime > 0.5 or industrial > 0.5:
                add("QQQM", -scale * 0.8 * max(oil_regime, industrial))

    else:
        raise ValueError(f"Unknown overlay_style: {overlay_style}")

    overlay_info = {
        "war_strength": war,
        "growth_strength": growth,
        "risk_off_strength": risk_off,
        "soxx_strength": soxx,
        "copper_strength": copper,
        "copper_3m_strength": copper3,
        "industrial_strength": industrial,
        "materials_strength": materials,
        "usd_3m_strength": usd_regime,
        "hyg_strength": hyg_regime,
        "credit_strength": credit_regime,
        "overlay_style": overlay_style,
    }
    return adjusted, overlay_info


def should_go_cash(top_score: float, second_score: float, risk_off_strength_val: float):
    threshold = 0.0
    if risk_off_strength_val > 1.0:
        threshold += risk_off_cash_threshold
    return (top_score < threshold) and (second_score < threshold)


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
    return w_top, 1.0 - w_top, gap


def tqqq_replace_fraction(top_asset: str, top_score: float, second_score: float, overlay_info: dict, date: pd.Timestamp):
    if top_asset != "QQQM":
        return 0.0

    gap = top_score - second_score
    vix_now = float(vix_level.loc[date]) if date in vix_level.index and pd.notna(vix_level.loc[date]) else np.nan
    growth = overlay_info["growth_strength"]
    soxx = overlay_info["soxx_strength"]
    risk_off = overlay_info["risk_off_strength"]

    strong = (
        gap >= tiered_tqqq_rule["strong"]["gap_min"]
        and top_score >= tiered_tqqq_rule["strong"]["top_score_min"]
        and growth >= tiered_tqqq_rule["strong"]["growth_min"]
        and soxx >= tiered_tqqq_rule["strong"]["soxx_min"]
        and risk_off <= tiered_tqqq_rule["strong"]["risk_off_max"]
        and (pd.isna(vix_now) or vix_now <= tiered_tqqq_rule["strong"]["vix_max"])
    )
    if strong:
        return tiered_tqqq_rule["strong"]["replace_fraction"]

    moderate = (
        gap >= tiered_tqqq_rule["moderate"]["gap_min"]
        and top_score >= tiered_tqqq_rule["moderate"]["top_score_min"]
        and growth >= tiered_tqqq_rule["moderate"]["growth_min"]
        and soxx >= tiered_tqqq_rule["moderate"]["soxx_min"]
        and risk_off <= tiered_tqqq_rule["moderate"]["risk_off_max"]
        and (pd.isna(vix_now) or vix_now <= tiered_tqqq_rule["moderate"]["vix_max"])
    )
    if moderate:
        return tiered_tqqq_rule["moderate"]["replace_fraction"]

    return 0.0



def tqqq_dynamic_replace_fraction(top_asset: str, top_score: float, second_score: float, overlay_info: dict, date: pd.Timestamp):
    """Conviction + volatility adjusted TQQQ replacement fraction."""
    if top_asset != "QQQM":
        return 0.0

    gap = top_score - second_score
    growth = float(overlay_info.get("growth_strength", 0.0))
    soxx = float(overlay_info.get("soxx_strength", 0.0))
    risk_off = float(overlay_info.get("risk_off_strength", 0.0))
    vix_now = float(vix_level.loc[date]) if date in vix_level.index and pd.notna(vix_level.loc[date]) else np.nan

    permission = (
        top_score >= 0.008
        and gap >= 0.002
        and growth >= 0.20
        and soxx >= 0.20
        and risk_off <= 1.00
        and (pd.isna(vix_now) or vix_now <= 25.0)
    )
    if not permission:
        return 0.0

    conviction = (gap - 0.002) / (0.020 - 0.002)
    conviction = float(np.clip(conviction, 0.0, 1.0))
    replace_fraction = 0.40 + 0.60 * conviction

    if pd.isna(vix_now):
        vol_adj = 1.0
    elif vix_now < 15:
        vol_adj = 1.1
    elif vix_now < 25:
        vol_adj = 1.0
    elif vix_now < 35:
        vol_adj = 0.8
    else:
        vol_adj = 0.6
    replace_fraction *= vol_adj

    if growth > 1.0 and soxx > 1.0 and risk_off < 0.50:
        replace_fraction += 0.10

    if risk_off > 0.50:
        replace_fraction *= 0.75

    return float(np.clip(replace_fraction, 0.0, 1.0))


def multi_asset_leverage_fraction(asset: str, top_asset: str, score_gap: float, overlay_info: dict, date: pd.Timestamp):
    """Safe 2x leverage gate for XLE->ERX and XLI->UXI."""
    if asset != top_asset:
        return 0.0

    vix_now = float(vix_level.loc[date]) if date is not None and date in vix_level.index and pd.notna(vix_level.loc[date]) else np.nan
    risk_off = float(overlay_info.get("risk_off_strength", 0.0))

    if score_gap > 0.015:
        frac = 0.50
    elif score_gap > 0.006:
        frac = 0.25
    else:
        frac = 0.0

    if not pd.isna(vix_now):
        if vix_now > 30:
            frac *= 0.3
        elif vix_now > 25:
            frac *= 0.6

    if risk_off > 0.5:
        frac *= 0.5

    return float(np.clip(frac, 0.0, 0.60))


def build_execution_weights(
    signal_weights: dict,
    overlay_fraction: float,
    sector_etfs: list,
    top_asset=None,
    score_gap=0.0,
    overlay_info=None,
    date=None,
):
    exec_universe = ["TQQQ", "ERX", "UXI"] + sector_etfs + [cash_etf]
    exec_weights = {a: 0.0 for a in exec_universe}
    overlay_info = overlay_info or {}

    risk_off = float(overlay_info.get("risk_off_strength", 0.0))
    vix_now = float(vix_level.loc[date]) if date is not None and date in vix_level.index and pd.notna(vix_level.loc[date]) else np.nan

    if risk_off > 1.5 or (not pd.isna(vix_now) and vix_now > 32):
        exec_weights[cash_etf] = 1.0
        return exec_weights

    if risk_off > 1.0 or (not pd.isna(vix_now) and vix_now > 28):
        defensive_cash = 0.50
    else:
        defensive_cash = 0.0

    qqqm_signal = signal_weights.get("QQQM", 0.0)
    exec_weights["TQQQ"] = qqqm_signal * overlay_fraction
    exec_weights["QQQM"] = qqqm_signal * (1.0 - overlay_fraction)

    xle_signal = signal_weights.get("XLE", 0.0)
    if xle_signal > 0:
        frac = multi_asset_leverage_fraction("XLE", top_asset, score_gap, overlay_info, date)
        exec_weights["ERX"] = xle_signal * frac
        exec_weights["XLE"] = xle_signal * (1.0 - frac)

    xli_signal = signal_weights.get("XLI", 0.0)
    if xli_signal > 0:
        frac = multi_asset_leverage_fraction("XLI", top_asset, score_gap, overlay_info, date)
        exec_weights["UXI"] = xli_signal * frac
        exec_weights["XLI"] = xli_signal * (1.0 - frac)

    for asset in sector_etfs:
        if asset not in ["QQQM", "XLE", "XLI"]:
            exec_weights[asset] = signal_weights.get(asset, 0.0)

    exec_weights[cash_etf] = signal_weights.get(cash_etf, 0.0)

    if defensive_cash > 0:
        for asset in exec_weights:
            if asset != cash_etf:
                exec_weights[asset] *= (1.0 - defensive_cash)
        exec_weights[cash_etf] = defensive_cash

    return exec_weights

# ============================================================
# 8. STRATEGY RUNNER
# ============================================================
def run_strategy(model_name: str, sector_etfs: list, features_by_asset: dict, overlay_style: str = "v1", tqqq_style: str = "tiered"):
    signal_universe = sector_etfs + [cash_etf]
    exec_universe = ["TQQQ", "ERX", "UXI"] + sector_etfs + [cash_etf]
    dates = prices.index

    min_needed = max(train_window, 252) + 1
    max_loc = len(dates) - forward_return_days
    rebalance_locs = list(range(min_needed, max_loc, rebalance_step))

    portfolio_daily_returns = pd.Series(index=dates, dtype=float)
    current_exec_weights = {a: 0.0 for a in exec_universe}
    current_exec_weights[cash_etf] = 1.0

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

        adjusted_preds, overlay_info = apply_regime_overlay(raw_preds, rebalance_date, sector_etfs, overlay_style=overlay_style)
        ranked = sorted(adjusted_preds.items(), key=lambda x: x[1], reverse=True)
        top_asset, top_score = ranked[0]
        second_asset, second_score = ranked[1]

        w_top, w_second, score_gap = get_conviction_weights(top_score, second_score)
        signal_weights = {a: 0.0 for a in signal_universe}
        signal_weights[top_asset] = w_top
        signal_weights[second_asset] = w_second

        if should_go_cash(top_score, second_score, overlay_info["risk_off_strength"]):
            signal_weights = {a: 0.0 for a in signal_universe}
            signal_weights[cash_etf] = 1.0

        if tqqq_style == "dynamic":
            overlay_fraction = tqqq_dynamic_replace_fraction(
                top_asset=top_asset,
                top_score=top_score,
                second_score=second_score,
                overlay_info=overlay_info,
                date=rebalance_date,
            )
        else:
            overlay_fraction = tqqq_replace_fraction(
                top_asset=top_asset,
                top_score=top_score,
                second_score=second_score,
                overlay_info=overlay_info,
                date=rebalance_date,
            )
        exec_weights = build_execution_weights(
            signal_weights,
            overlay_fraction,
            sector_etfs,
            top_asset=top_asset,
            score_gap=score_gap,
            overlay_info=overlay_info,
            date=rebalance_date,
        )

        turnover = compute_turnover(current_exec_weights, exec_weights, exec_universe)
        turnover_list.append(turnover)

        next_loc = rebalance_locs[i + 1] if i + 1 < len(rebalance_locs) else max_loc
        hold_dates = dates[loc + 1: next_loc + 1]
        if len(hold_dates) == 0:
            continue

        hold_rets = pd.Series(index=hold_dates, data=0.0)
        for asset, w in exec_weights.items():
            if w != 0:
                hold_rets = hold_rets.add(w * asset_returns[asset].reindex(hold_dates).fillna(0.0), fill_value=0.0)

        cost = turnover * transaction_cost
        hold_rets.iloc[0] -= cost
        portfolio_daily_returns.loc[hold_dates] = hold_rets.values
        current_exec_weights = exec_weights.copy()

        row = {
            "model": model_name,
            "date": rebalance_date,
            "top_asset": top_asset,
            "second_asset": second_asset,
            "top_score": top_score,
            "second_score": second_score,
            "score_gap": score_gap,
            "turnover": turnover,
            "tx_cost_applied": cost,
            "overlay_fraction": overlay_fraction,
            **overlay_info,
        }
        for a in sector_etfs:
            row[f"raw_pred_{a}"] = raw_preds.get(a, np.nan)
            row[f"adj_pred_{a}"] = adjusted_preds.get(a, np.nan)
            row[f"signal_w_{a}"] = signal_weights.get(a, 0.0)
            row[f"exec_w_{a}"] = exec_weights.get(a, 0.0)
        row[f"signal_w_{cash_etf}"] = signal_weights.get(cash_etf, 0.0)
        row[f"exec_w_{cash_etf}"] = exec_weights.get(cash_etf, 0.0)
        row["exec_w_TQQQ"] = exec_weights.get("TQQQ", 0.0)
        row["exec_w_ERX"] = exec_weights.get("ERX", 0.0)
        row["exec_w_UXI"] = exec_weights.get("UXI", 0.0)
        rebalance_records.append(row)

    portfolio_daily_returns = portfolio_daily_returns.dropna()
    rebalance_df = pd.DataFrame(rebalance_records)
    avg_turnover = float(np.mean(turnover_list)) if turnover_list else np.nan
    return portfolio_daily_returns, rebalance_df, avg_turnover

# ============================================================
# 9. LATEST RECOMMENDATION
# ============================================================
def get_latest_recommendation(model_name: str, sector_etfs: list, features_by_asset: dict, overlay_style: str = "v1", tqqq_style: str = "tiered"):
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

        adjusted_preds, overlay_info = apply_regime_overlay(raw_preds, latest_date, sector_etfs, overlay_style=overlay_style)
        ranked = sorted(adjusted_preds.items(), key=lambda x: x[1], reverse=True)
        top_asset, top_score = ranked[0]
        second_asset, second_score = ranked[1]
        w_top, w_second, score_gap = get_conviction_weights(top_score, second_score)

        signal_weights = {a: 0.0 for a in signal_universe}
        signal_weights[top_asset] = w_top
        signal_weights[second_asset] = w_second

        if should_go_cash(top_score, second_score, overlay_info["risk_off_strength"]):
            signal_weights = {a: 0.0 for a in signal_universe}
            signal_weights[cash_etf] = 1.0

        if tqqq_style == "dynamic":
            overlay_fraction = tqqq_dynamic_replace_fraction(top_asset, top_score, second_score, overlay_info, latest_date)
        else:
            overlay_fraction = tqqq_replace_fraction(top_asset, top_score, second_score, overlay_info, latest_date)
        exec_weights = build_execution_weights(
            signal_weights,
            overlay_fraction,
            sector_etfs,
            top_asset=top_asset,
            score_gap=score_gap,
            overlay_info=overlay_info,
            date=latest_date,
        )

        feature_importance_df = pd.DataFrame({
            "feature": x_train.columns,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        return {
            "model": model_name,
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
            **overlay_info,
        }
    return None

# ============================================================
# 10. PRINT / SAVE HELPERS
# ============================================================
def print_weights(title: str, weights: dict, order: list):
    print(f"\n=== {title} ===")
    for asset in order:
        print(f"{asset}: {weights.get(asset, 0.0):.1%}")


def print_latest(latest: dict, sector_etfs: list):
    if latest is None:
        print("No latest recommendation available.")
        return
    print(f"\n=== Latest Recommendation: {latest['model']} ===")
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
    print(f"Overlay style: {latest.get('overlay_style', 'v1')}")
    print(f"Copper strength: {latest['copper_strength']:.3f}")
    print(f"Industrial strength: {latest['industrial_strength']:.3f}")
    print(f"Materials strength: {latest['materials_strength']:.3f}")
    print(f"USD 3M strength: {latest['usd_3m_strength']:.3f}")
    print(f"Credit strength: {latest['credit_strength']:.3f}")

    print_weights("Suggested SIGNAL Weights", latest["signal_weights"], sector_etfs + [cash_etf])
    print_weights("Suggested EXECUTED Weights", latest["exec_weights"], ["TQQQ", "ERX", "UXI"] + sector_etfs + [cash_etf])

    print("\n=== Latest Feature Importance Summary ===")
    print(latest["feature_importance"].head(25).to_string(index=False))


def save_latest(prefix: str, latest: dict):
    if latest is None:
        return
    latest_df = pd.DataFrame([{
        "signal_date": latest["date"],
        "latest_data_date": prices.index[-1],
        "top_asset": latest["top_asset"],
        "second_asset": latest["second_asset"],
        "top_score": latest["top_score"],
        "second_score": latest["second_score"],
        "score_gap": latest["score_gap"],
        "overlay_fraction": latest["overlay_fraction"],
        "overlay_style": latest.get("overlay_style", "v1"),
        "war_strength": latest["war_strength"],
        "growth_strength": latest["growth_strength"],
        "risk_off_strength": latest["risk_off_strength"],
        "soxx_strength": latest["soxx_strength"],
        "copper_strength": latest["copper_strength"],
        "copper_3m_strength": latest["copper_3m_strength"],
        "industrial_strength": latest["industrial_strength"],
        "materials_strength": latest["materials_strength"],
        "usd_3m_strength": latest["usd_3m_strength"],
        "hyg_strength": latest["hyg_strength"],
        "credit_strength": latest["credit_strength"],
        **{f"signal_w_{k}": v for k, v in latest["signal_weights"].items()},
        **{f"exec_w_{k}": v for k, v in latest["exec_weights"].items()},
        **{f"raw_pred_{k}": v for k, v in latest["raw_predictions"].items()},
        **{f"adj_pred_{k}": v for k, v in latest["adjusted_predictions"].items()},
    }])
    latest_df.to_csv(f"{prefix}_latest_recommendation.csv", index=False)
    latest["feature_importance"].to_csv(f"{prefix}_feature_importance.csv", index=False)

# ============================================================
# 11. RUN BOTH MODELS
# ============================================================
print("\nBuilding baseline feature set...")
baseline_features = build_features_by_asset(BASELINE_SECTOR_ETFS)

print("Building upgraded XLI/XLB feature set...")
upgraded_features = build_features_by_asset(UPGRADED_SECTOR_ETFS)

print("\nRunning baseline model...")
baseline_returns, baseline_rebalance, baseline_turnover = run_strategy(
    "BASELINE_QQQM_XLE_XSOE", BASELINE_SECTOR_ETFS, baseline_features
)

print("Running upgraded V1 model...")
upgraded_returns, upgraded_rebalance, upgraded_turnover = run_strategy(
    "UPGRADED_V1_QQQM_XLE_XSOE_XLI_XLB", UPGRADED_SECTOR_ETFS, upgraded_features, overlay_style="v1"
)

print("Running upgraded V2 model...")
upgraded_v2_returns, upgraded_v2_rebalance, upgraded_v2_turnover = run_strategy(
    "UPGRADED_V2_INDUSTRIAL_CLASSIFIER", UPGRADED_SECTOR_ETFS, upgraded_features, overlay_style="v2"
)

print("Running HYBRID fast-growth/adaptive model...")
hybrid_returns, hybrid_rebalance, hybrid_turnover = run_strategy(
    "HYBRID_FAST_GROWTH_ADAPTIVE", UPGRADED_SECTOR_ETFS, upgraded_features, overlay_style="hybrid"
)

print("Running HYBRID + conviction/volatility + multi-asset leverage + defensive layer model...")
hybrid_dyn_returns, hybrid_dyn_rebalance, hybrid_dyn_turnover = run_strategy(
    "HYBRID_MULTI_ASSET_DEFENSIVE",
    UPGRADED_SECTOR_ETFS,
    upgraded_features,
    overlay_style="hybrid",
    tqqq_style="dynamic",
)

if len(baseline_returns) == 0 or len(upgraded_returns) == 0 or len(upgraded_v2_returns) == 0 or len(hybrid_returns) == 0 or len(hybrid_dyn_returns) == 0:
    raise ValueError("One of the models produced no returns.")

# Common sample comparison across all models.
common_idx = baseline_returns.index.intersection(upgraded_returns.index).intersection(upgraded_v2_returns.index).intersection(hybrid_returns.index).intersection(hybrid_dyn_returns.index)
baseline_common = baseline_returns.loc[common_idx]
upgraded_common = upgraded_returns.loc[common_idx]
upgraded_v2_common = upgraded_v2_returns.loc[common_idx]
hybrid_common = hybrid_returns.loc[common_idx]
hybrid_dyn_common = hybrid_dyn_returns.loc[common_idx]

summary_df = pd.DataFrame([
    performance_summary("BASELINE_FULL", baseline_returns, baseline_turnover),
    performance_summary("UPGRADED_V1_FULL", upgraded_returns, upgraded_turnover),
    performance_summary("UPGRADED_V2_FULL", upgraded_v2_returns, upgraded_v2_turnover),
    performance_summary("HYBRID_FULL", hybrid_returns, hybrid_turnover),
    performance_summary("HYBRID_DYNAMIC_FULL", hybrid_dyn_returns, hybrid_dyn_turnover),
    performance_summary("BASELINE_COMMON", baseline_common, baseline_turnover),
    performance_summary("UPGRADED_V1_COMMON", upgraded_common, upgraded_turnover),
    performance_summary("UPGRADED_V2_COMMON", upgraded_v2_common, upgraded_v2_turnover),
    performance_summary("HYBRID_COMMON", hybrid_common, hybrid_turnover),
    performance_summary("HYBRID_DYNAMIC_COMMON", hybrid_dyn_common, hybrid_dyn_turnover),
])

print("\n=== BASELINE VS V1 VS V2 VS HYBRID VS DYNAMIC PERFORMANCE SUMMARY ===")
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n=== DELTA ON COMMON SAMPLE ===")
base = summary_df[summary_df["model"] == "BASELINE_COMMON"].iloc[0]
v1 = summary_df[summary_df["model"] == "UPGRADED_V1_COMMON"].iloc[0]
v2 = summary_df[summary_df["model"] == "UPGRADED_V2_COMMON"].iloc[0]
print("V1 vs baseline:")
print(f"  Delta annual return: {v1['annual_return'] - base['annual_return']:.4f}")
print(f"  Delta volatility:     {v1['volatility'] - base['volatility']:.4f}")
print(f"  Delta Sharpe:         {v1['sharpe'] - base['sharpe']:.4f}")
print(f"  Delta max drawdown:   {v1['max_drawdown'] - base['max_drawdown']:.4f}")
print("V2 vs baseline:")
print(f"  Delta annual return: {v2['annual_return'] - base['annual_return']:.4f}")
print(f"  Delta volatility:     {v2['volatility'] - base['volatility']:.4f}")
print(f"  Delta Sharpe:         {v2['sharpe'] - base['sharpe']:.4f}")
print(f"  Delta max drawdown:   {v2['max_drawdown'] - base['max_drawdown']:.4f}")
print("V2 vs V1:")
print(f"  Delta annual return: {v2['annual_return'] - v1['annual_return']:.4f}")
print(f"  Delta volatility:     {v2['volatility'] - v1['volatility']:.4f}")
print(f"  Delta Sharpe:         {v2['sharpe'] - v1['sharpe']:.4f}")
print(f"  Delta max drawdown:   {v2['max_drawdown'] - v1['max_drawdown']:.4f}")

print("\n=== Last 10 Upgraded V1 Rebalances ===")
print(upgraded_rebalance.tail(10).to_string(index=False))
print("\n=== Last 10 Upgraded V2 Rebalances ===")
print(upgraded_v2_rebalance.tail(10).to_string(index=False))

latest_baseline = get_latest_recommendation("BASELINE_QQQM_XLE_XSOE", BASELINE_SECTOR_ETFS, baseline_features, overlay_style="v1")
latest_upgraded = get_latest_recommendation("UPGRADED_V1_QQQM_XLE_XSOE_XLI_XLB", UPGRADED_SECTOR_ETFS, upgraded_features, overlay_style="v1")
latest_upgraded_v2 = get_latest_recommendation("UPGRADED_V2_INDUSTRIAL_CLASSIFIER", UPGRADED_SECTOR_ETFS, upgraded_features, overlay_style="v2")
latest_hybrid = get_latest_recommendation("HYBRID_FAST_GROWTH_ADAPTIVE", UPGRADED_SECTOR_ETFS, upgraded_features, overlay_style="hybrid")
latest_hybrid_dyn = get_latest_recommendation("HYBRID_MULTI_ASSET_DEFENSIVE", UPGRADED_SECTOR_ETFS, upgraded_features, overlay_style="hybrid", tqqq_style="dynamic")

print_latest(latest_baseline, BASELINE_SECTOR_ETFS)
print_latest(latest_upgraded, UPGRADED_SECTOR_ETFS)
print_latest(latest_upgraded_v2, UPGRADED_SECTOR_ETFS)
print_latest(latest_hybrid, UPGRADED_SECTOR_ETFS)
print_latest(latest_hybrid_dyn, UPGRADED_SECTOR_ETFS)

# ============================================================
# 12. SAVE OUTPUTS
# ============================================================
base_prefix = "model_c_plus_baseline_tqqq_tiered_overlay"
up_prefix = "model_c_plus_xli_xlb_industrial_tqqq_tiered_overlay_v1"
up_v2_prefix = "model_c_plus_xli_xlb_industrial_classifier_v2"
hybrid_prefix = "model_c_plus_hybrid_fast_growth_adaptive"
hybrid_dyn_prefix = "model_c_plus_hybrid_multi_asset_defensive"
compare_prefix = "compare_baseline_vs_xli_xlb_v1_v2_hybrid_multi_defensive"

baseline_returns.to_csv(f"{base_prefix}_portfolio_daily_returns.csv", header=["portfolio_return"])
baseline_rebalance.to_csv(f"{base_prefix}_rebalance_log.csv", index=False)
save_latest(base_prefix, latest_baseline)

upgraded_returns.to_csv(f"{up_prefix}_portfolio_daily_returns.csv", header=["portfolio_return"])
upgraded_rebalance.to_csv(f"{up_prefix}_rebalance_log.csv", index=False)
save_latest(up_prefix, latest_upgraded)

upgraded_v2_returns.to_csv(f"{up_v2_prefix}_portfolio_daily_returns.csv", header=["portfolio_return"])
upgraded_v2_rebalance.to_csv(f"{up_v2_prefix}_rebalance_log.csv", index=False)
save_latest(up_v2_prefix, latest_upgraded_v2)

hybrid_returns.to_csv(f"{hybrid_prefix}_portfolio_daily_returns.csv", header=["portfolio_return"])
hybrid_rebalance.to_csv(f"{hybrid_prefix}_rebalance_log.csv", index=False)
save_latest(hybrid_prefix, latest_hybrid)

hybrid_dyn_returns.to_csv(f"{hybrid_dyn_prefix}_portfolio_daily_returns.csv", header=["portfolio_return"])
hybrid_dyn_rebalance.to_csv(f"{hybrid_dyn_prefix}_rebalance_log.csv", index=False)
save_latest(hybrid_dyn_prefix, latest_hybrid_dyn)

summary_df.to_csv(f"{compare_prefix}_performance_summary.csv", index=False)
pd.DataFrame({
    "baseline_return": baseline_common,
    "upgraded_v1_return": upgraded_common,
    "upgraded_v2_return": upgraded_v2_common,
    "hybrid_return": hybrid_common,
    "hybrid_dynamic_return": hybrid_dyn_common,
    "delta_v1_vs_baseline": upgraded_common - baseline_common,
    "delta_v2_vs_baseline": upgraded_v2_common - baseline_common,
    "delta_hybrid_vs_baseline": hybrid_common - baseline_common,
    "delta_hybrid_vs_v2": hybrid_common - upgraded_v2_common,
    "delta_hybrid_dynamic_vs_baseline": hybrid_dyn_common - baseline_common,
    "delta_hybrid_dynamic_vs_hybrid": hybrid_dyn_common - hybrid_common,
    "delta_hybrid_dynamic_vs_v2": hybrid_dyn_common - upgraded_v2_common,
    "delta_v2_vs_v1": upgraded_v2_common - upgraded_common,
}).to_csv(f"{compare_prefix}_daily_returns_common_sample.csv")

print("\nSaved:")
print(f"- {base_prefix}_portfolio_daily_returns.csv")
print(f"- {base_prefix}_rebalance_log.csv")
print(f"- {base_prefix}_latest_recommendation.csv")
print(f"- {base_prefix}_feature_importance.csv")
print(f"- {up_prefix}_portfolio_daily_returns.csv")
print(f"- {up_prefix}_rebalance_log.csv")
print(f"- {up_prefix}_latest_recommendation.csv")
print(f"- {up_prefix}_feature_importance.csv")
print(f"- {up_v2_prefix}_portfolio_daily_returns.csv")
print(f"- {up_v2_prefix}_rebalance_log.csv")
print(f"- {up_v2_prefix}_latest_recommendation.csv")
print(f"- {up_v2_prefix}_feature_importance.csv")
print(f"- {hybrid_dyn_prefix}_portfolio_daily_returns.csv")
print(f"- {hybrid_dyn_prefix}_rebalance_log.csv")
print(f"- {hybrid_dyn_prefix}_latest_recommendation.csv")
print(f"- {hybrid_dyn_prefix}_feature_importance.csv")
print(f"- {compare_prefix}_performance_summary.csv")
print(f"- {compare_prefix}_daily_returns_common_sample.csv")
