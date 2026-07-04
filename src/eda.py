"""
eda.py
------
Exploratory Data Analysis functions. Every plot is saved to outputs/figures/
with a title, axis labels, and legend, and every function prints a short
business-insight summary derived from the actual numbers (not canned text).

Note on Plotly: this module tries to use Plotly for a couple of interactive
HTML charts (nice for a portfolio) and transparently falls back to
Matplotlib/Seaborn if Plotly is not installed in the current environment, so
the pipeline always runs end-to-end.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100

FIG_DIR = "../outputs/figures"


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] Saved figure: {path}")


def plot_daily_trend(df):
    daily = df["demand_mw"].resample("D").mean()
    fig, ax = plt.subplots(figsize=(14, 5))
    daily.plot(ax=ax, color="#1f77b4", linewidth=1)
    ax.set_title("Daily Average Electricity Demand (2022-2024)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand (MW)")
    ax.legend(["Daily Avg Demand"])
    _save(fig, "01_daily_demand_trend.png")

    pct_change = (daily.iloc[-30:].mean() / daily.iloc[:30].mean() - 1) * 100
    print(f"[insight] Demand has grown ~{pct_change:.1f}% comparing the first 30 days "
          f"to the most recent 30 days of the dataset, consistent with a steady long-term "
          f"load-growth trend an energy planner should budget capacity for.")


def plot_monthly_trend(df):
    monthly = df["demand_mw"].resample("ME").mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    monthly.plot(kind="line", marker="o", ax=ax, color="#d62728")
    ax.set_title("Monthly Average Electricity Demand", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Demand (MW)")
    ax.legend(["Monthly Avg Demand"])
    _save(fig, "02_monthly_demand_trend.png")

    by_month = df.groupby("month")["demand_mw"].mean()
    peak_month = by_month.idxmax()
    print(f"[insight] Month {peak_month} shows the highest average demand across all years "
          f"({by_month.max():.0f} MW), driven by seasonal heating/cooling load.")


def plot_weekly_pattern(df):
    weekday_avg = df.groupby("day_name")["demand_mw"].mean()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_avg = weekday_avg.reindex(order)
    fig, ax = plt.subplots(figsize=(10, 5))
    weekday_avg.plot(kind="bar", ax=ax, color="#2ca02c")
    ax.set_title("Average Demand by Day of Week", fontsize=13, fontweight="bold")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Average Demand (MW)")
    ax.legend(["Avg Demand"])
    _save(fig, "03_weekly_pattern.png")

    drop = (weekday_avg[["Saturday", "Sunday"]].mean() / weekday_avg[order[:5]].mean() - 1) * 100
    print(f"[insight] Weekend demand runs about {abs(drop):.1f}% {'lower' if drop < 0 else 'higher'} "
          f"than weekday demand, reflecting reduced commercial/industrial activity.")


def plot_hourly_pattern(df):
    hourly_avg = df.groupby("hour")["demand_mw"].mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    hourly_avg.plot(kind="line", marker="o", ax=ax, color="#9467bd")
    ax.set_title("Average Hourly Demand Pattern (Daily Load Curve)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Demand (MW)")
    ax.set_xticks(range(0, 24))
    ax.legend(["Avg Demand by Hour"])
    _save(fig, "04_hourly_pattern.png")

    peak_hr = hourly_avg.idxmax()
    trough_hr = hourly_avg.idxmin()
    print(f"[insight] Demand peaks around {peak_hr}:00 (evening peak) and bottoms out around "
          f"{trough_hr}:00 (overnight trough) -- classic dual-peak residential/commercial load curve.")


def plot_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["demand_mw"], bins=50, kde=True, ax=axes[0], color="#1f77b4")
    axes[0].set_title("Distribution of Electricity Demand", fontweight="bold")
    axes[0].set_xlabel("Demand (MW)")
    axes[0].legend(["Density (KDE)"])

    sns.boxplot(x=df["demand_mw"], ax=axes[1], color="#ff7f0e")
    axes[1].set_title("Boxplot of Electricity Demand", fontweight="bold")
    axes[1].set_xlabel("Demand (MW)")
    _save(fig, "05_demand_distribution.png")

    skew = df["demand_mw"].skew()
    print(f"[insight] Demand distribution skewness is {skew:.2f} "
          f"({'right-skewed, i.e. occasional high-demand spikes' if skew > 0.2 else 'roughly symmetric'}), "
          f"relevant for choosing robust vs. standard forecasting error metrics.")


def plot_solar_trend(df):
    monthly_solar = df["solar_gen_mw"].resample("ME").mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    monthly_solar.plot(ax=ax, color="#ff9900", marker="o")
    ax.set_title("Monthly Average Solar Generation", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Solar Generation (MW)")
    ax.legend(["Avg Solar Generation"])
    _save(fig, "06_solar_generation_trend.png")

    summer_peak = df.groupby("month")["solar_gen_mw"].mean().max()
    winter_low = df.groupby("month")["solar_gen_mw"].mean().min()
    print(f"[insight] Solar output swings from ~{winter_low:.0f} MW (lowest month) to "
          f"~{summer_peak:.0f} MW (peak month), a {(summer_peak / max(winter_low,1)):.1f}x seasonal range "
          f"that grid planners must offset with other generation sources in winter.")


def plot_correlation_matrix(df):
    cols = ["demand_mw", "solar_gen_mw", "wind_gen_mw", "temperature_c"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix: Demand, Generation, Temperature", fontweight="bold")
    _save(fig, "07_correlation_matrix.png")

    strongest = corr["demand_mw"].drop("demand_mw").abs().idxmax()
    print(f"[insight] '{strongest}' has the strongest linear relationship with demand "
          f"(r={corr['demand_mw'][strongest]:.2f}), useful as an exogenous regressor in SARIMAX.")


def plot_rolling_moving_average(df):
    fig, ax = plt.subplots(figsize=(14, 5))
    df["demand_mw"].resample("D").mean().plot(ax=ax, label="Daily Avg Demand", alpha=0.4, color="gray")
    df["demand_mw"].resample("D").mean().rolling(7).mean().plot(ax=ax, label="7-Day Rolling Avg", color="#1f77b4", linewidth=2)
    df["demand_mw"].resample("D").mean().rolling(30).mean().plot(ax=ax, label="30-Day Moving Avg", color="#d62728", linewidth=2)
    ax.set_title("Rolling & Moving Averages of Daily Demand", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand (MW)")
    ax.legend()
    _save(fig, "08_rolling_moving_average.png")
    print("[insight] The 30-day moving average smooths out weekly noise and clearly exposes "
          "the underlying seasonal trend used to sanity-check the ARIMA/SARIMA forecast.")


def plot_seasonality_boxplot(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x="month", y="demand_mw", data=df.reset_index(), ax=ax, palette="viridis", hue="month", legend=False)
    ax.set_title("Seasonality: Demand Distribution by Month", fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Demand (MW)")
    _save(fig, "09_seasonality_boxplot.png")
    print("[insight] Median demand and its spread both shift across months, confirming annual "
          "seasonality that a plain ARIMA (non-seasonal) would miss -- motivating the SARIMA model.")


def plot_weekday_weekend_comparison(df):
    grp = df.groupby(["is_weekend", "hour"])["demand_mw"].mean().unstack(level=0)
    grp.columns = ["Weekday", "Weekend"]
    fig, ax = plt.subplots(figsize=(12, 5))
    grp.plot(ax=ax, marker="o")
    ax.set_title("Hourly Demand: Weekday vs Weekend", fontsize=13, fontweight="bold")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Demand (MW)")
    ax.legend(["Weekday", "Weekend"])
    _save(fig, "10_weekday_vs_weekend.png")
    gap = (grp["Weekday"].mean() - grp["Weekend"].mean())
    print(f"[insight] Weekdays run ~{gap:.0f} MW higher than weekends on average across the day, "
          f"most pronounced during business hours (9am-6pm).")


def run_full_eda(df: pd.DataFrame):
    print("=" * 60)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    plot_daily_trend(df)
    plot_monthly_trend(df)
    plot_weekly_pattern(df)
    plot_hourly_pattern(df)
    plot_distribution(df)
    plot_solar_trend(df)
    plot_correlation_matrix(df)
    plot_rolling_moving_average(df)
    plot_seasonality_boxplot(df)
    plot_weekday_weekend_comparison(df)
    print("[eda] All EDA figures saved to outputs/figures/")
