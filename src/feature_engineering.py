"""
feature_engineering.py
------------------------
Creates time-based and statistical features used for EDA and forecasting.

Each feature is created for a specific analytical reason (documented inline):
    - Calendar features (year/month/week/day/hour/quarter) let us group and
      compare demand across different time granularities in the EDA step.
    - The weekend indicator captures the clear behavioural difference in
      industrial/commercial consumption between weekdays and weekends.
    - Lag features give the forecasting models direct access to recent
      history, which is the single strongest predictor of near-term demand.
    - Rolling mean/std summarize local trend and volatility, smoothing out
      hour-to-hour noise so the model can see the underlying signal.
    - The differenced series is required to test/achieve stationarity before
      fitting ARIMA-family models.
"""

import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add year, month, week, day, hour, quarter, weekend indicator."""
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["week"] = df.index.isocalendar().week.astype(int)
    df["day"] = df.index.day
    df["hour"] = df.index.hour
    df["quarter"] = df.index.quarter
    df["weekday"] = df.index.dayofweek
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["day_name"] = df.index.day_name()
    return df


def add_lag_features(df: pd.DataFrame, target: str = "demand_mw",
                      lags=(1, 24, 168)) -> pd.DataFrame:
    """Add lag features.

    Lags chosen:
        1   hour  -> immediate short-term persistence
        24  hours -> same hour, previous day (captures daily cycle)
        168 hours -> same hour, previous week (captures weekly cycle)
    """
    for lag in lags:
        df[f"{target}_lag_{lag}"] = df[target].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target: str = "demand_mw",
                          windows=(24, 168)) -> pd.DataFrame:
    """Add rolling mean/std to capture local trend and volatility."""
    for w in windows:
        df[f"{target}_roll_mean_{w}"] = df[target].rolling(window=w).mean()
        df[f"{target}_roll_std_{w}"] = df[target].rolling(window=w).std()
    return df


def add_differenced_series(df: pd.DataFrame, target: str = "demand_mw",
                            periods: int = 1) -> pd.DataFrame:
    """Add a differenced series (used later for stationarity)."""
    df[f"{target}_diff_{periods}"] = df[target].diff(periods)
    return df


def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 60)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 60)
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_differenced_series(df)
    print(f"[feature_engineering] Added calendar, lag, rolling, and "
          f"differenced features. Shape: {df.shape}")
    return df
