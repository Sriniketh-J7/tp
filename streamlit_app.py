import os
import io
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from prophet import Prophet

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, Image as RLImage, PageBreak)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Walmart Sales Forecaster",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container { padding-top: 1.2rem; }
.winner {
    background: #E8F5E9;
    border-left: 5px solid #2E7D32;
    padding: 14px 18px;
    border-radius: 6px;
    font-size: 1.05rem;
    margin-bottom: 12px;
}
h1 { color: #1565C0; }
h2 { color: #283593; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae_score(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))

def r2_s(y_true, y_pred):
    return float(r2_score(y_true, y_pred))

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    return buf.read()

def iqr_clip(s, k=3.0):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return s.clip(q1 - k * iqr, q3 + k * iqr)

# ─────────────────────────────────────────────
# DATA LOADING & MERGING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_merge(sales_bytes, feat_bytes, store_bytes):
    sales    = pd.read_csv(io.BytesIO(sales_bytes))
    features = pd.read_csv(io.BytesIO(feat_bytes))
    stores   = pd.read_csv(io.BytesIO(store_bytes))

    sales["Date"]    = pd.to_datetime(sales["Date"])
    features["Date"] = pd.to_datetime(features["Date"])
    sales["Date"]    = sales["Date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
    features["Date"] = features["Date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()

    sales = sales[sales["Weekly_Sales"] >= 0].copy()

    sales = (sales.groupby(["Store","Dept","Date"], as_index=False)
                  .agg({"Weekly_Sales":"sum","IsHoliday":"max"}))

    feat_agg = {c: ("max" if c == "IsHoliday" else "mean")
                for c in features.columns if c not in ["Store","Date"]}
    features = features.groupby(["Store","Date"], as_index=False).agg(feat_agg)

    df = sales.merge(features, on=["Store","Date"], how="left", suffixes=("","_feat"))
    df = df.merge(stores, on="Store", how="left")

    md_cols = [c for c in df.columns if c.lower().startswith("markdown")]
    for c in md_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(lower=0)

    for c in df.select_dtypes(include=np.number).columns:
        df[c] = df[c].replace([np.inf,-np.inf], np.nan).fillna(df[c].median())

    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].fillna("Unknown")

    df["Weekly_Sales"] = df.groupby(["Store","Dept"])["Weekly_Sales"].transform(iqr_clip)
    df = df.sort_values(["Store","Dept","Date"]).reset_index(drop=True)
    return df, md_cols

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def engineer_features(df, md_cols):
    df = df.copy()
    df["Year"]         = df["Date"].dt.year
    df["Month"]        = df["Date"].dt.month
    df["Quarter"]      = df["Date"].dt.quarter
    df["WeekOfYear"]   = df["Date"].dt.isocalendar().week.astype(int)
    df["IsMonthEnd"]   = df["Date"].dt.is_month_end.astype(int)
    df["IsQuarterEnd"] = df["Date"].dt.is_quarter_end.astype(int)
    df["IsHoliday"]    = pd.to_numeric(df.get("IsHoliday", 0), errors="coerce").fillna(0).astype(int)
    df["Holiday_Prev"] = df.groupby(["Store","Dept"])["IsHoliday"].shift(1).fillna(0).astype(int)
    df["Holiday_Next"] = df.groupby(["Store","Dept"])["IsHoliday"].shift(-1).fillna(0).astype(int)
    df["MarkDown_Total"] = df[md_cols].sum(axis=1) if md_cols else 0.0
    df["MarkDown_Any"]   = (df["MarkDown_Total"] > 0).astype(int)
    df["Size"] = pd.to_numeric(df.get("Size", 1), errors="coerce").fillna(1)

    for lag in [1, 2, 4, 8, 13, 26, 52]:
        df[f"Sales_Lag_{lag}"] = df.groupby(["Store","Dept"])["Weekly_Sales"].shift(lag)

    for win in [4, 8, 13, 26]:
        df[f"Sales_RollMean_{win}"] = df.groupby(["Store","Dept"])["Weekly_Sales"].transform(
            lambda s: s.shift(1).rolling(win, min_periods=1).mean())
        df[f"Sales_RollStd_{win}"] = df.groupby(["Store","Dept"])["Weekly_Sales"].transform(
            lambda s: s.shift(1).rolling(win, min_periods=2).std())

    df["Sales_EWM_4"]  = df.groupby(["Store","Dept"])["Weekly_Sales"].transform(
        lambda s: s.shift(1).ewm(span=4, min_periods=1).mean())
    df["Sales_EWM_13"] = df.groupby(["Store","Dept"])["Weekly_Sales"].transform(
        lambda s: s.shift(1).ewm(span=13, min_periods=1).mean())

    for col in ["Temperature","Fuel_Price","CPI","Unemployment"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[f"Delta_{col}"] = df.groupby("Store")[col].diff()

    seasonal = (df.groupby(["Dept","WeekOfYear"])["Weekly_Sales"]
                  .mean().reset_index()
                  .rename(columns={"Weekly_Sales":"Dept_Seasonal_Avg"}))
    df = df.merge(seasonal, on=["Dept","WeekOfYear"], how="left")

    store_tot = (df.groupby(["Store","Date"])["Weekly_Sales"]
                   .sum().reset_index()
                   .rename(columns={"Weekly_Sales":"Store_Total_Sales"}))
    df = df.merge(store_tot, on=["Store","Date"], how="left")
    df["Dept_Share"] = df["Weekly_Sales"] / (df["Store_Total_Sales"] + 1)

    if "Type" in df.columns:
        df["Type"] = df["Type"].astype(str)
        type_dummies = pd.get_dummies(df["Type"], prefix="StoreType", drop_first=False)
        df = pd.concat([df.drop(columns=["Type"]), type_dummies], axis=1)

    lag_cols = [c for c in df.columns if any(x in c for x in
        ["Lag","Roll","EWM","Delta","Dept_Seasonal","Store_Total","Dept_Share"])]
    for c in lag_cols:
        df[c] = df[c].replace([np.inf,-np.inf], np.nan).fillna(0)

    return df.sort_values(["Store","Dept","Date"]).reset_index(drop=True)

# ─────────────────────────────────────────────
# TRAIN ML MODELS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_ml_models(_df):
    df = _df
    DROP = ["Weekly_Sales","Date","Store_Total_Sales","Dept_Share"]
    feat_cols = [c for c in df.columns if c not in DROP]

    unique_dates = sorted(df["Date"].unique())
    split_idx    = max(1, min(int(len(unique_dates) * 0.8), len(unique_dates)-1))
    split_date   = unique_dates[split_idx - 1]

    train_df = df[df["Date"] <= split_date].copy()
    test_df  = df[df["Date"] >  split_date].copy()

    X_train = pd.get_dummies(train_df[feat_cols], drop_first=True)
    X_test  = pd.get_dummies(test_df[feat_cols],  drop_first=True)
    X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)
    X_train = X_train.replace([np.inf,-np.inf], np.nan).fillna(0)
    X_test  = X_test.replace([np.inf,-np.inf],  np.nan).fillna(0)

    y_train = train_df["Weekly_Sales"].astype(float).values
    y_test  = test_df["Weekly_Sales"].astype(float).values

    xgb_model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=7,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0, eval_metric="rmse"
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    pred_xgb = xgb_model.predict(X_test).clip(min=0)

    rf_model = RandomForestRegressor(
        n_estimators=200, max_depth=20, min_samples_leaf=4,
        random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    pred_rf = rf_model.predict(X_test).clip(min=0)

    metrics_dict = {
        "XGBoost": {
            "RMSE": round(rmse(y_test, pred_xgb), 2),
            "MAE":  round(mae_score(y_test, pred_xgb), 2),
            "R2":   round(r2_s(y_test, pred_xgb), 4),
        },
        "Random Forest": {
            "RMSE": round(rmse(y_test, pred_rf), 2),
            "MAE":  round(mae_score(y_test, pred_rf), 2),
            "R2":   round(r2_s(y_test, pred_rf), 4),
        }
    }

    fi_xgb = pd.Series(xgb_model.feature_importances_, index=X_train.columns)
    fi_rf  = pd.Series(rf_model.feature_importances_,  index=X_train.columns)

    return (xgb_model, rf_model, X_train.columns.tolist(), feat_cols,
            split_date, test_df, y_test, pred_xgb, pred_rf,
            metrics_dict, fi_xgb, fi_rf)

# ─────────────────────────────────────────────
# PROPHET FORECAST
# ─────────────────────────────────────────────
def prophet_forecast(df, store_id, dept_id, n_weeks=12):
    full_series = (df[(df["Store"]==store_id) & (df["Dept"]==dept_id)]
                   .sort_values("Date").copy())

    series = full_series[["Date","Weekly_Sales"]].rename(
        columns={"Date":"ds","Weekly_Sales":"y"}).dropna()

    if len(series) < 10:
        return None, None

    exog_cols = [c for c in ["Temperature","Fuel_Price","CPI","Unemployment",
                              "IsHoliday","MarkDown_Total"] if c in df.columns]

    m = Prophet(weekly_seasonality=True, yearly_seasonality=True,
                daily_seasonality=False, seasonality_mode="multiplicative",
                changepoint_prior_scale=0.05)

    for col in exog_cols:
        m.add_regressor(col)
        series = series.copy()
        series[col] = full_series[col].values[:len(series)]

    m.fit(series)

    last_date    = series["ds"].max()
    future_dates = [last_date + pd.Timedelta(weeks=i+1) for i in range(n_weeks)]
    future_df    = pd.DataFrame({"ds": future_dates})
    for col in exog_cols:
        future_df[col] = full_series[col].iloc[-1]

    forecast = m.predict(future_df)
    forecast["yhat"] = forecast["yhat"].clip(lower=0)
    return forecast[["ds","yhat","yhat_lower","yhat_upper"]], series

# ─────────────────────────────────────────────
# XGBOOST FUTURE FORECAST
# ─────────────────────────────────────────────
def xgb_future_forecast(df, xgb_model, feat_cols, store_id, dept_id, n_weeks=12):
    series = (df[(df["Store"]==store_id) & (df["Dept"]==dept_id)]
              .sort_values("Date").copy())

    if len(series) < 10:
        return None

    last_row      = series.iloc[-1].copy()
    last_date     = series["Date"].max()
    sales_history = list(series["Weekly_Sales"].values)

    DROP = ["Weekly_Sales","Date","Store_Total_Sales","Dept_Share"]
    feat_cols_use = [c for c in feat_cols if c not in DROP]
    booster_feats = xgb_model.get_booster().feature_names

    forecasts = []
    for i in range(n_weeks):
        future_date = last_date + pd.Timedelta(weeks=i+1)
        row = last_row.copy()
        row["Date"]         = future_date
        row["Year"]         = future_date.year
        row["Month"]        = future_date.month
        row["Quarter"]      = (future_date.month - 1) // 3 + 1
        row["WeekOfYear"]   = future_date.isocalendar()[1]
        row["IsMonthEnd"]   = int(future_date.is_month_end)
        row["IsQuarterEnd"] = int(future_date.month in [3,6,9,12] and future_date.is_month_end)

        h = sales_history
        row["Sales_Lag_1"]       = h[-1]  if len(h) >= 1  else 0
        row["Sales_Lag_2"]       = h[-2]  if len(h) >= 2  else 0
        row["Sales_Lag_4"]       = h[-4]  if len(h) >= 4  else 0
        row["Sales_Lag_8"]       = h[-8]  if len(h) >= 8  else 0
        row["Sales_Lag_13"]      = h[-13] if len(h) >= 13 else 0
        row["Sales_Lag_26"]      = h[-26] if len(h) >= 26 else 0
        row["Sales_Lag_52"]      = h[-52] if len(h) >= 52 else 0
        row["Sales_RollMean_4"]  = np.mean(h[-4:])
        row["Sales_RollMean_8"]  = np.mean(h[-8:])
        row["Sales_RollMean_13"] = np.mean(h[-13:])
        row["Sales_RollMean_26"] = np.mean(h[-26:])
        row["Sales_EWM_4"]       = pd.Series(h).ewm(span=4).mean().iloc[-1]
        row["Sales_EWM_13"]      = pd.Series(h).ewm(span=13).mean().iloc[-1]

        X_row = pd.DataFrame([row[feat_cols_use]])
        X_row = pd.get_dummies(X_row, drop_first=True)
        X_row = X_row.reindex(columns=booster_feats, fill_value=0)
        X_row = X_row.replace([np.inf,-np.inf], np.nan).fillna(0)

        pred = float(xgb_model.predict(X_row)[0])
        pred = max(pred, 0)
        forecasts.append({"ds": future_date, "yhat": round(pred, 2)})
        sales_history.append(pred)

    return pd.DataFrame(forecasts)

# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────
def plot_sales_trend(df):
    weekly = df.groupby("Date")["Weekly_Sales"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(weekly["Date"], weekly["Weekly_Sales"], color="#1565C0", linewidth=1.2)
    ax.set_title("Total Weekly Sales Over Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Total Weekly Sales ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(); plt.tight_layout()
    return fig

def plot_holiday_impact(df):
    hol = df.groupby("IsHoliday")["Weekly_Sales"].mean().reset_index()
    hol["Label"] = hol["IsHoliday"].map({0:"Non-Holiday",1:"Holiday"})
    fig, ax = plt.subplots(figsize=(5,4))
    bars = ax.bar(hol["Label"], hol["Weekly_Sales"], color=["#90CAF9","#EF5350"])
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+50,
                f"${b.get_height():,.0f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_title("Holiday vs Non-Holiday: Avg Sales", fontsize=12, fontweight="bold")
    ax.set_ylabel("Avg Weekly Sales ($)"); plt.tight_layout()
    return fig

def plot_top_depts(df):
    dept_tot = df.groupby("Dept")["Weekly_Sales"].sum().sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(11,4))
    ax.bar(dept_tot.index.astype(str), dept_tot.values, color="#1E88E5")
    ax.set_title("Top 15 Departments by Total Sales", fontsize=12, fontweight="bold")
    ax.set_xlabel("Department"); ax.set_ylabel("Total Sales ($)"); plt.tight_layout()
    return fig

def plot_monthly_seasonality(df):
    monthly = df.groupby("Month")["Weekly_Sales"].mean().reset_index()
    mnames  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly["MN"] = monthly["Month"].apply(lambda x: mnames[x-1])
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(range(len(monthly)), monthly["Weekly_Sales"], marker="o", color="#43A047", linewidth=2)
    ax.fill_between(range(len(monthly)), monthly["Weekly_Sales"].values, alpha=0.15, color="#43A047")
    ax.set_xticks(range(len(monthly))); ax.set_xticklabels(monthly["MN"])
    ax.set_title("Average Weekly Sales by Month", fontsize=12, fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Avg Weekly Sales ($)"); plt.tight_layout()
    return fig

def plot_store_type(df):
    st_cols = [c for c in df.columns if c.startswith("StoreType_")]
    if not st_cols:
        return None
    tmp = df[["Weekly_Sales"] + st_cols].copy()
    tmp["StoreType"] = tmp[st_cols].idxmax(axis=1).str.replace("StoreType_","",regex=False)
    by_type = tmp.groupby("StoreType")["Weekly_Sales"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(by_type.index, by_type.values, color=["#1565C0","#FF7043","#2E7D32"])
    ax.set_title("Avg Weekly Sales by Store Type", fontsize=12, fontweight="bold")
    ax.set_xlabel("Store Type"); ax.set_ylabel("Avg Weekly Sales ($)"); plt.tight_layout()
    return fig

def plot_corr_heatmap(df):
    cols = [c for c in ["Weekly_Sales","Temperature","Fuel_Price","CPI","Unemployment",
                         "MarkDown_Total","Size","IsHoliday","WeekOfYear","Month"]
            if c in df.columns]
    fig, ax = plt.subplots(figsize=(9,7))
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, square=True, linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap", fontsize=12, fontweight="bold"); plt.tight_layout()
    return fig

def plot_model_comparison(metrics_dict):
    models = list(metrics_dict.keys())
    fig, axes = plt.subplots(1, 3, figsize=(13,4))
    bar_colors = ["#1565C0","#E65100"]
    for ax, key, label in zip(axes,
                               ["RMSE","MAE","R2"],
                               ["RMSE (lower=better)","MAE (lower=better)","R2 (higher=better)"]):
        vals = [metrics_dict[m][key] for m in models]
        bars = ax.bar(models, vals, color=bar_colors)
        ax.set_title(label, fontsize=11, fontweight="bold"); ax.set_ylabel(key)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.01,
                    f"{v:.4f}" if key=="R2" else f"{v:,.0f}",
                    ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    return fig

def plot_feature_importance(fi_xgb, fi_rf):
    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    for ax, fi, title, col in zip(axes,
                                   [fi_xgb, fi_rf],
                                   ["XGBoost Feature Importance","Random Forest Feature Importance"],
                                   ["#E65100","#1565C0"]):
        top = fi.sort_values(ascending=True).tail(15)
        ax.barh(top.index, top.values, color=col)
        ax.set_title(title, fontsize=11, fontweight="bold"); ax.set_xlabel("Importance")
    plt.tight_layout()
    return fig

def plot_actual_vs_pred(test_df, y_test, pred_xgb, pred_rf):
    pair = (test_df.groupby(["Store","Dept"])["Weekly_Sales"]
                   .count().sort_values(ascending=False).index[0])
    ms, md = int(pair[0]), int(pair[1])
    sample = test_df[(test_df["Store"]==ms) & (test_df["Dept"]==md)].sort_values("Date")
    pos    = [test_df.index.get_loc(i) for i in sample.index]

    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(sample["Date"], y_test[pos], label="Actual", color="black", linewidth=1.5)
    ax.plot(sample["Date"], pred_xgb[pos], label="XGBoost",
            color="#E65100", linestyle="--", linewidth=1.3)
    ax.plot(sample["Date"], pred_rf[pos],  label="Random Forest",
            color="#1565C0", linestyle="--", linewidth=1.3)
    ax.set_title(f"Actual vs Predicted — Store {ms}, Dept {md}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Weekly Sales ($)")
    ax.legend(); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(); plt.tight_layout()
    return fig

def plot_prophet_forecast(history_series, forecast_df, store_id, dept_id):
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(history_series["ds"], history_series["y"],
            label="Historical", color="#546E7A", linewidth=1.2)
    ax.plot(forecast_df["ds"], forecast_df["yhat"],
            label="Prophet Forecast", color="#AD1457", linewidth=2, marker="o", markersize=4)
    ax.fill_between(forecast_df["ds"],
                    forecast_df["yhat_lower"], forecast_df["yhat_upper"],
                    alpha=0.2, color="#AD1457", label="Confidence Interval")
    ax.axvline(x=history_series["ds"].max(), color="red", linestyle=":", alpha=0.7,
               label="Forecast Start")
    ax.set_title(f"Prophet 12-Week Forecast — Store {store_id}, Dept {dept_id}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Weekly Sales ($)")
    ax.legend(); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(); plt.tight_layout()
    return fig

def plot_xgb_forecast(history_df, xgb_fc, store_id, dept_id):
    fig, ax = plt.subplots(figsize=(12,5))
    tail = history_df.tail(52)
    ax.plot(tail["Date"], tail["Weekly_Sales"],
            label="Historical (last 52w)", color="#546E7A", linewidth=1.2)
    ax.plot(xgb_fc["ds"], xgb_fc["yhat"],
            label="XGBoost Forecast", color="#E65100", linewidth=2, marker="o", markersize=4)
    ax.axvline(x=history_df["Date"].max(), color="red", linestyle=":", alpha=0.7,
               label="Forecast Start")
    ax.set_title(f"XGBoost 12-Week Forecast — Store {store_id}, Dept {dept_id}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Weekly Sales ($)")
    ax.legend(); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(); plt.tight_layout()
    return fig

def plot_combined_forecast(history_df, prophet_fc, xgb_fc, store_id, dept_id):
    fig, ax = plt.subplots(figsize=(13,5))
    tail = history_df.tail(52)
    ax.plot(tail["Date"], tail["Weekly_Sales"],
            label="Historical (last 52w)", color="#546E7A", linewidth=1.2)
    if prophet_fc is not None:
        ax.plot(prophet_fc["ds"], prophet_fc["yhat"], label="Prophet",
                color="#AD1457", linewidth=2, linestyle="--", marker="o", markersize=4)
    if xgb_fc is not None:
        ax.plot(xgb_fc["ds"], xgb_fc["yhat"], label="XGBoost",
                color="#E65100", linewidth=2, linestyle="--", marker="s", markersize=4)
    ax.axvline(x=history_df["Date"].max(), color="red", linestyle=":", alpha=0.7,
               label="Forecast Start")
    ax.set_title(f"Prophet vs XGBoost — Store {store_id}, Dept {dept_id}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Weekly Sales ($)")
    ax.legend(); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(); plt.tight_layout()
    return fig

# ─────────────────────────────────────────────
# PDF REPORT
# ─────────────────────────────────────────────
def generate_pdf(df, metrics_dict, forecast_prophet, forecast_xgb,
                 store_id, dept_id, figs_dict):
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=letter,
                                rightMargin=0.7*inch, leftMargin=0.7*inch,
                                topMargin=0.7*inch, bottomMargin=0.7*inch)
    styles = getSampleStyleSheet()
    story  = []

    title_s = ParagraphStyle("T", parent=styles["Title"], fontSize=20,
                              textColor=colors.HexColor("#1565C0"), spaceAfter=6)
    h1_s    = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=14,
                              textColor=colors.HexColor("#283593"), spaceBefore=14, spaceAfter=6)
    h2_s    = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11,
                              textColor=colors.HexColor("#37474F"), spaceBefore=8, spaceAfter=4)
    normal  = styles["Normal"]

    def add_fig(fig, w=6.5*inch, h=3.0*inch):
        if fig is None:
            return
        img_buf = io.BytesIO(fig_to_bytes(fig))
        story.append(RLImage(img_buf, width=w, height=h))
        story.append(Spacer(1, 6))

    def make_table(data, header_color="#1565C0", row_colors=None):
        t = Table(data)
        row_colors = row_colors or ["#EEF2FF", "#FFFFFF"]
        t.setStyle(TableStyle([
            ("BACKGROUND",     (0,0), (-1,0), colors.HexColor(header_color)),
            ("TEXTCOLOR",      (0,0), (-1,0), colors.white),
            ("FONTNAME",       (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",       (0,0), (-1,-1), 9),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor(c) for c in row_colors]),
            ("GRID",           (0,0), (-1,-1), 0.4, colors.HexColor("#CFD8DC")),
            ("PADDING",        (0,0), (-1,-1), 5),
        ]))
        return t

    # Cover
    story.append(Paragraph("Walmart Store Sales Forecasting Report", title_s))
    story.append(Paragraph(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", normal))
    story.append(Spacer(1, 10))

    # Dataset summary
    story.append(Paragraph("Dataset Summary", h1_s))
    summary = [
        ["Metric","Value"],
        ["Total Records",    f"{len(df):,}"],
        ["Stores",           str(df['Store'].nunique())],
        ["Departments",      str(df['Dept'].nunique())],
        ["Date Range",       f"{df['Date'].min().date()} to {df['Date'].max().date()}"],
        ["Total Sales",      f"${df['Weekly_Sales'].sum():,.0f}"],
        ["Avg Weekly Sales", f"${df['Weekly_Sales'].mean():,.2f}"],
    ]
    story.append(make_table(summary))
    story.append(Spacer(1, 10))

    # EDA plots
    story.append(Paragraph("Exploratory Data Analysis", h1_s))
    for key in ["trend","monthly","holiday","top_depts","store_type","corr"]:
        if figs_dict.get(key):
            add_fig(figs_dict[key])

    story.append(PageBreak())

    # Model comparison
    story.append(Paragraph("Model Performance", h1_s))
    story.append(Paragraph(
        "Trained on 80% of historical data. Evaluated on 20% held-out test set.", normal))
    story.append(Spacer(1, 6))

    comp_data = [["Model","RMSE","MAE","R2"]]
    for m, v in metrics_dict.items():
        comp_data.append([m, f"${v['RMSE']:,.2f}", f"${v['MAE']:,.2f}", f"{v['R2']:.4f}"])
    story.append(make_table(comp_data))
    story.append(Spacer(1, 8))

    for key in ["comparison","feature_imp","actual_vs_pred"]:
        if figs_dict.get(key):
            add_fig(figs_dict[key], h=3.3*inch)

    story.append(PageBreak())

    # Forecasts
    story.append(Paragraph("12-Week Sales Forecast", h1_s))
    story.append(Paragraph(f"Store {store_id} — Department {dept_id}", h2_s))

    for key in ["prophet_fc","xgb_fc","combined_fc"]:
        if figs_dict.get(key):
            add_fig(figs_dict[key], h=3.2*inch)

    if forecast_prophet is not None:
        story.append(Paragraph("Prophet Weekly Forecast", h2_s))
        fc_data = [["Week","Date","Forecast ($)","Lower ($)","Upper ($)"]]
        for i, row in forecast_prophet.iterrows():
            fc_data.append([str(i+1), row["ds"].strftime("%Y-%m-%d"),
                            f"${row['yhat']:,.2f}", f"${row['yhat_lower']:,.2f}",
                            f"${row['yhat_upper']:,.2f}"])
        story.append(make_table(fc_data, header_color="#AD1457",
                                row_colors=["#FCE4EC","white"]))
        story.append(Spacer(1, 10))

    if forecast_xgb is not None:
        story.append(Paragraph("XGBoost Weekly Forecast", h2_s))
        fc_data2 = [["Week","Date","Forecast ($)"]]
        for i, row in forecast_xgb.iterrows():
            fc_data2.append([str(i+1), row["ds"].strftime("%Y-%m-%d"),
                             f"${row['yhat']:,.2f}"])
        story.append(make_table(fc_data2, header_color="#E65100",
                                row_colors=["#FBE9E7","#FFFFFF"]))

    doc.build(story)
    buf.seek(0)
    return buf.read()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("🛒 Walmart Sales\nForecaster")
st.sidebar.markdown("---")
st.sidebar.markdown("### Upload Your Data")

sales_file = st.sidebar.file_uploader("sales_data.csv",    type="csv", key="sales")
feat_file  = st.sidebar.file_uploader("features_data.csv", type="csv", key="feat")
store_file = st.sidebar.file_uploader("stores_data.csv",   type="csv", key="store")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Expected Files:**
- `sales_data.csv` — Store, Dept, Date, Weekly_Sales, IsHoliday
- `features_data.csv` — Store, Date, Temperature, Fuel_Price, CPI, Unemployment, MarkDowns
- `stores_data.csv` — Store, Type, Size

*Use Walmart competition dataset from Kaggle*
""")

# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
st.title("🛒 Walmart Store Sales Forecasting")
st.markdown("Upload your 3 CSV files → Models train → Get 12-week forecasts + downloadable PDF report")

if not (sales_file and feat_file and store_file):
    st.info("👈 Upload all 3 CSV files from the sidebar to get started.")
    st.markdown("""
    ### What this app does:
    - Cleans and merges your 3 data files
    - Engineers 30+ features (lags, rolling means, holidays, markdowns, seasonality)
    - Trains **XGBoost** and **Random Forest** on 80% of data, evaluates on 20%
    - Forecasts **next 12 weeks** of sales using XGBoost and Prophet
    - Generates a **downloadable PDF report** with all charts and tables

    ### Get the data:
    1. [Kaggle Walmart Competition](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data)
    2. `train.csv` → rename `sales_data.csv`
    3. `features.csv` → rename `features_data.csv`
    4. `stores.csv` → rename `stores_data.csv`
    """)
    st.stop()

# Load data
with st.spinner("Loading and merging data..."):
    df_raw, md_cols = load_and_merge(
        sales_file.read(), feat_file.read(), store_file.read()
    )

with st.spinner("Engineering features..."):
    df = engineer_features(df_raw, md_cols)

st.success(f"✅ Data ready: {len(df):,} rows | {df['Store'].nunique()} stores | {df['Dept'].nunique()} departments")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🤖 Model Performance", "📈 Forecast", "📄 PDF Report"])

# ══════════════════════════
# TAB 1 — EDA
# ══════════════════════════
with tab1:
    st.header("Exploratory Data Analysis")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Stores",         df["Store"].nunique())
    c3.metric("Departments",    df["Dept"].nunique())
    c4.metric("Total Sales",    f"${df['Weekly_Sales'].sum()/1e9:.2f}B")

    st.markdown("---")
    st.subheader("Total Weekly Sales Trend")
    fig_trend = plot_sales_trend(df)
    st.pyplot(fig_trend)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Monthly Seasonality")
        fig_monthly = plot_monthly_seasonality(df)
        st.pyplot(fig_monthly)
    with col2:
        st.subheader("Holiday Impact")
        fig_holiday = plot_holiday_impact(df)
        st.pyplot(fig_holiday)

    st.subheader("Top 15 Departments by Total Sales")
    fig_depts = plot_top_depts(df)
    st.pyplot(fig_depts)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Sales by Store Type")
        fig_storetype = plot_store_type(df)
        if fig_storetype:
            st.pyplot(fig_storetype)
        else:
            fig_storetype = None
    with col4:
        st.subheader("Correlation Heatmap")
        fig_corr = plot_corr_heatmap(df)
        st.pyplot(fig_corr)

# ══════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════
with tab2:
    st.header("Model Training & Evaluation")
    st.info("XGBoost and Random Forest trained on 80% of data, tested on 20%.")

    with st.spinner("Training models... (1-2 minutes)"):
        (xgb_model, rf_model, feat_names, feat_cols,
         split_date, test_df, y_test, pred_xgb, pred_rf,
         metrics_dict, fi_xgb, fi_rf) = train_ml_models(df)

    best_name = min(metrics_dict, key=lambda m: metrics_dict[m]["RMSE"])
    best      = metrics_dict[best_name]

    st.markdown(f"""
    <div class="winner">
    🏆 Best Model: <strong>{best_name}</strong>
    &nbsp;|&nbsp; RMSE: <strong>${best['RMSE']:,.2f}</strong>
    &nbsp;|&nbsp; MAE: <strong>${best['MAE']:,.2f}</strong>
    &nbsp;|&nbsp; R²: <strong>{best['R2']:.4f}</strong>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Performance Table")
    comp_df = (pd.DataFrame(metrics_dict).T
                 .reset_index().rename(columns={"index":"Model"}))
    st.dataframe(
        comp_df.style
               .background_gradient(subset=["RMSE","MAE"], cmap="RdYlGn_r")
               .background_gradient(subset=["R2"], cmap="RdYlGn")
               .format({"RMSE":"{:,.2f}","MAE":"{:,.2f}","R2":"{:.4f}"}),
        use_container_width=True
    )

    fig_comp = plot_model_comparison(metrics_dict)
    st.pyplot(fig_comp)

    st.subheader("Feature Importances")
    fig_fi = plot_feature_importance(fi_xgb, fi_rf)
    st.pyplot(fig_fi)

    st.subheader("Actual vs Predicted (Test Set Sample)")
    try:
        fig_avp = plot_actual_vs_pred(test_df, y_test, pred_xgb, pred_rf)
        st.pyplot(fig_avp)
    except Exception as e:
        st.warning(f"Could not render plot: {e}")
        fig_avp = None

# ══════════════════════════
# TAB 3 — FORECAST
# ══════════════════════════
with tab3:
    st.header("12-Week Future Sales Forecast")

    col1, col2 = st.columns(2)
    sel_store = col1.selectbox("Store", sorted(df["Store"].unique()))
    sel_dept  = col2.selectbox("Department", sorted(df["Dept"].unique()))

    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Running Prophet forecast..."):
            prophet_fc, history_series = prophet_forecast(df, sel_store, sel_dept, n_weeks=12)

        with st.spinner("Running XGBoost forecast..."):
            xgb_fc = xgb_future_forecast(
                df, xgb_model, feat_cols, sel_store, sel_dept, n_weeks=12)

        history_df = (df[(df["Store"]==sel_store) & (df["Dept"]==sel_dept)]
                      .sort_values("Date")[["Date","Weekly_Sales"]].copy())

        if prophet_fc is not None:
            st.subheader("Prophet Forecast")
            fig_pfc = plot_prophet_forecast(history_series, prophet_fc, sel_store, sel_dept)
            st.pyplot(fig_pfc)
            st.dataframe(
                prophet_fc.rename(columns={"ds":"Date","yhat":"Forecast ($)",
                                            "yhat_lower":"Lower ($)","yhat_upper":"Upper ($)"})
                          .assign(Date=lambda d: d["Date"].dt.strftime("%Y-%m-%d"))
                          .round(2),
                use_container_width=True)
        else:
            fig_pfc = None
            st.warning("Not enough data for Prophet forecast on this Store-Dept.")

        if xgb_fc is not None:
            st.subheader("XGBoost Forecast")
            fig_xfc = plot_xgb_forecast(history_df, xgb_fc, sel_store, sel_dept)
            st.pyplot(fig_xfc)
            st.dataframe(
                xgb_fc.rename(columns={"ds":"Date","yhat":"Forecast ($)"})
                       .assign(Date=lambda d: d["Date"].dt.strftime("%Y-%m-%d"))
                       .round(2),
                use_container_width=True)
        else:
            fig_xfc = None
            st.warning("Not enough data for XGBoost forecast on this Store-Dept.")

        if prophet_fc is not None and xgb_fc is not None:
            st.subheader("Prophet vs XGBoost Comparison")
            fig_comb = plot_combined_forecast(history_df, prophet_fc, xgb_fc, sel_store, sel_dept)
            st.pyplot(fig_comb)
        else:
            fig_comb = None

        # Save to session state for PDF tab
        st.session_state.update({
            "forecast_done":  True,
            "sel_store":      sel_store,
            "sel_dept":       sel_dept,
            "prophet_fc":     prophet_fc,
            "xgb_fc":         xgb_fc,
            "history_df":     history_df,
            "history_series": history_series,
            "fig_pfc":        fig_pfc,
            "fig_xfc":        fig_xfc,
            "fig_comb":       fig_comb,
        })

        st.success("Forecast complete! Go to the **PDF Report** tab to download.")
    else:
        st.info("Select a store and department, then click **Generate Forecast**.")

# ══════════════════════════
# TAB 4 — PDF REPORT
# ══════════════════════════
with tab4:
    st.header("Download PDF Report")

    if not st.session_state.get("forecast_done"):
        st.warning("Generate a forecast first in the **Forecast** tab.")
    else:
        st.success("All data ready. Click below to build and download your report.")

        if st.button("Generate PDF Report", type="primary"):
            with st.spinner("Building PDF..."):
                figs_dict = {
                    "trend":          fig_trend,
                    "monthly":        fig_monthly,
                    "holiday":        fig_holiday,
                    "top_depts":      fig_depts,
                    "store_type":     fig_storetype,
                    "corr":           fig_corr,
                    "comparison":     fig_comp,
                    "feature_imp":    fig_fi,
                    "actual_vs_pred": fig_avp,
                    "prophet_fc":     st.session_state.get("fig_pfc"),
                    "xgb_fc":         st.session_state.get("fig_xfc"),
                    "combined_fc":    st.session_state.get("fig_comb"),
                }
                pdf_bytes = generate_pdf(
                    df=df,
                    metrics_dict=metrics_dict,
                    forecast_prophet=st.session_state.get("prophet_fc"),
                    forecast_xgb=st.session_state.get("xgb_fc"),
                    store_id=st.session_state["sel_store"],
                    dept_id=st.session_state["sel_dept"],
                    figs_dict=figs_dict,
                )

            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name="walmart_sales_forecast_report.pdf",
                mime="application/pdf",
            )
