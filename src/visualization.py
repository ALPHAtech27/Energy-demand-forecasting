"""
visualization.py
------------------
Plotting helpers for stationarity analysis, ACF/PACF, seasonal decomposition,
and model forecast/residual diagnostics.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
FIG_DIR = "../outputs/figures"


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualization] Saved figure: {path}")


def plot_before_after_differencing(original: pd.Series, differenced: pd.Series, name_suffix=""):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    original.plot(ax=axes[0], color="#1f77b4")
    axes[0].set_title("Before Differencing (Original Series)", fontweight="bold")
    axes[0].set_ylabel("Demand (MW)")
    axes[0].legend(["Original"])

    differenced.plot(ax=axes[1], color="#d62728")
    axes[1].set_title("After Differencing (d=1)", fontweight="bold")
    axes[1].set_ylabel("Differenced Demand (MW)")
    axes[1].legend(["Differenced"])
    fig.tight_layout()
    _save(fig, f"11_stationarity_before_after{name_suffix}.png")


def plot_acf_pacf(acf_vals, pacf_vals):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].stem(range(len(acf_vals)), acf_vals)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Autocorrelation Function (ACF)", fontweight="bold")
    axes[0].set_xlabel("Lag (hours)")
    axes[0].set_ylabel("Correlation")

    axes[1].stem(range(len(pacf_vals)), pacf_vals)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Partial Autocorrelation Function (PACF)", fontweight="bold")
    axes[1].set_xlabel("Lag (hours)")
    axes[1].set_ylabel("Partial Correlation")
    fig.tight_layout()
    _save(fig, "12_acf_pacf.png")


def plot_decomposition(decomposition):
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    decomposition.observed.plot(ax=axes[0], color="#1f77b4")
    axes[0].set_ylabel("Observed")
    axes[0].set_title("Seasonal Decomposition of Electricity Demand", fontweight="bold")

    decomposition.trend.plot(ax=axes[1], color="#2ca02c")
    axes[1].set_ylabel("Trend")

    decomposition.seasonal.plot(ax=axes[2], color="#9467bd")
    axes[2].set_ylabel("Seasonal")

    decomposition.resid.plot(ax=axes[3], color="#d62728", linestyle="None", marker=".", markersize=2)
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("Date")
    fig.tight_layout()
    _save(fig, "13_seasonal_decomposition.png")


def plot_actual_vs_predicted(actual, predicted, dates, model_name):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, actual, label="Actual", color="#1f77b4", linewidth=1.5)
    ax.plot(dates, predicted, label=f"{model_name} Forecast", color="#d62728", linewidth=1.5, linestyle="--")
    ax.set_title(f"Actual vs Predicted Demand -- {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand (MW)")
    ax.legend()
    fig.autofmt_xdate()
    _save(fig, f"14_actual_vs_predicted_{model_name.lower()}.png")


def plot_residuals(actual, predicted, dates, model_name):
    residuals = np.asarray(actual) - np.asarray(predicted)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(dates, residuals, color="#8c564b")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title(f"{model_name} Residuals Over Time", fontweight="bold")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Residual (MW)")

    sns.histplot(residuals, bins=30, kde=True, ax=axes[1], color="#8c564b")
    axes[1].set_title(f"{model_name} Residual Distribution", fontweight="bold")
    axes[1].set_xlabel("Residual (MW)")
    fig.tight_layout()
    _save(fig, f"15_residuals_{model_name.lower()}.png")


def plot_model_comparison(metrics_dict: dict):
    df = pd.DataFrame(metrics_dict).T
    fig, ax = plt.subplots(figsize=(10, 5))
    df[["MAE", "RMSE"]].plot(kind="bar", ax=ax, color=["#1f77b4", "#d62728"])
    ax.set_title("Model Comparison: MAE & RMSE", fontsize=13, fontweight="bold")
    ax.set_ylabel("Error (MW)")
    ax.set_xlabel("Model")
    ax.legend(["MAE", "RMSE"])
    _save(fig, "16_model_comparison.png")


def plot_forecast_future(history: pd.Series, forecast: np.ndarray, forecast_dates, model_name):
    fig, ax = plt.subplots(figsize=(14, 5))
    history.iloc[-24 * 14:].plot(ax=ax, label="Historical Demand (last 14 days)", color="#1f77b4")
    ax.plot(forecast_dates, forecast, label=f"{model_name} Forecast (future)", color="#2ca02c", linewidth=2)
    ax.set_title(f"Future Demand Forecast -- {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand (MW)")
    ax.legend()
    fig.autofmt_xdate()
    _save(fig, f"17_future_forecast_{model_name.lower()}.png")
