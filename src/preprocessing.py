"""
preprocessing.py
-----------------
Data cleaning and preprocessing pipeline for the Energy Demand Forecasting
project.

Design decisions (also explained inline at each function):
    - Timestamps are parsed explicitly and used as the DataFrame index so all
      downstream time-series operations (resampling, lag features, rolling
      windows) work correctly.
    - Duplicates are dropped on the datetime column since each hour should
      have exactly one reading; keeping the first occurrence assumes the
      first log is the original write.
    - Missing values are filled with time-based interpolation rather than
      mean/median imputation, because electricity demand and temperature are
      strongly autocorrelated in time -- interpolation preserves the local
      trend far better than a flat fill would.
    - Outliers are detected with the IQR method per numeric column and capped
      (winsorized) rather than dropped, because dropping rows would break the
      hourly continuity that the forecasting models depend on.
    - Physically impossible values (negative generation) are corrected by
      taking the absolute value, since they are logging sign-errors, not
      genuine negative generation.
"""

import numpy as np
import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw CSV and parse the datetime column."""
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate timestamp rows, keeping the first occurrence.

    Rationale: each hour should have exactly one demand reading. Duplicate
    rows are almost always a double-write from the ingestion/logging system.
    """
    before = len(df)
    df = df.drop_duplicates(subset="datetime", keep="first")
    removed = before - len(df)
    print(f"[preprocessing] Removed {removed} duplicate rows.")
    return df


def fix_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    """Correct physically impossible values (e.g. negative generation)."""
    for col in ["solar_gen_mw", "wind_gen_mw"]:
        n_neg = (df[col] < 0).sum()
        if n_neg:
            df[col] = df[col].abs()
            print(f"[preprocessing] Corrected {n_neg} negative values in {col} (sign errors).")
    return df


def sort_and_reindex(df: pd.DataFrame) -> pd.DataFrame:
    """Sort chronologically and set a complete hourly DateTimeIndex.

    Reindexing onto a full hourly range exposes any *missing* timestamps
    (gaps in the log) as NaN rows, so they get treated by the same
    interpolation step as other missing values instead of silently
    shrinking the series.
    """
    df = df.sort_values("datetime").set_index("datetime")
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_range)
    df.index.name = "datetime"
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using time-based interpolation.

    Time interpolation is chosen over mean/median fill because demand and
    temperature both have strong short-term autocorrelation -- the true
    value at a missing hour is much closer to its neighbours than to the
    column average. Any remaining edge NaNs (start/end of series) are
    handled with forward/backward fill.
    """
    n_missing_before = df.isna().sum().sum()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="time")
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    n_missing_after = df.isna().sum().sum()
    print(f"[preprocessing] Filled missing values via time interpolation "
          f"({n_missing_before} -> {n_missing_after} remaining NaNs).")
    return df


def treat_outliers(df: pd.DataFrame, cols=None, factor: float = 3.0) -> pd.DataFrame:
    """Winsorize outliers using the IQR method (cap, don't drop).

    Capping preserves the hourly time index (required for ARIMA/SARIMA)
    while neutralizing the influence of sensor spikes. A wider factor (3.0
    instead of the classic 1.5) is used deliberately because demand data is
    naturally spiky at peak hours -- 1.5x IQR would flag legitimate peak
    demand as outliers.
    """
    if cols is None:
        cols = ["demand_mw", "solar_gen_mw", "wind_gen_mw", "temperature_c"]
    for col in cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower, upper)
        if n_out:
            print(f"[preprocessing] Capped {n_out} outliers in {col} "
                  f"(bounds: {lower:.1f} to {upper:.1f}).")
    return df


def validate_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all numeric columns are proper floats."""
    numeric_cols = ["demand_mw", "solar_gen_mw", "wind_gen_mw", "temperature_c"]
    for col in numeric_cols:
        df[col] = df[col].astype(float)
    return df


def run_preprocessing_pipeline(raw_path: str, processed_path: str) -> pd.DataFrame:
    """Run the full cleaning pipeline end-to-end and save the result."""
    print("=" * 60)
    print("STEP 1: DATA CLEANING")
    print("=" * 60)
    df = load_raw_data(raw_path)
    print(f"[preprocessing] Loaded {len(df)} raw rows.")
    df = remove_duplicates(df)
    df = fix_impossible_values(df)
    df = sort_and_reindex(df)
    df = handle_missing_values(df)
    df = treat_outliers(df)
    df = validate_dtypes(df)
    df.to_csv(processed_path)
    print(f"[preprocessing] Cleaned dataset saved to {processed_path} "
          f"({len(df)} rows, {df.index.min()} to {df.index.max()}).")
    return df


if __name__ == "__main__":
    run_preprocessing_pipeline(
        raw_path="../data/raw/energy_demand_raw.csv",
        processed_path="../data/processed/cleaned_data.csv",
    )
