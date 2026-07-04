"""
generate_raw_data.py
---------------------
Generates a realistic synthetic hourly electricity grid dataset for the
Energy Demand Forecasting project.

WHY SYNTHETIC DATA:
This project was built in an offline sandbox with no internet access, so a
public Kaggle/government dataset could not be downloaded directly. Instead,
this script generates a physically-plausible hourly dataset (3 full years)
with the same structure, seasonality, and real-world messiness (missing
values, duplicates, outliers) that a genuine grid-operator export would have.
When you run this project locally, simply replace this file's output with a
real dataset (e.g. Kaggle "Hourly Energy Consumption", PJM Interconnection,
or your national grid operator's open data) that has the same column names,
and the rest of the pipeline (src/, notebook, dashboard) works unchanged.

Columns produced:
    datetime            - hourly timestamp
    demand_mw           - electricity demand (MW)
    solar_gen_mw         - solar generation (MW)
    wind_gen_mw          - wind generation (MW)
    temperature_c        - ambient temperature (Celsius)
"""

import numpy as np
import pandas as pd

np.random.seed(42)

START = "2022-01-01 00:00:00"
END = "2024-12-31 23:00:00"

def generate():
    idx = pd.date_range(start=START, end=END, freq="h")
    n = len(idx)

    day_of_year = idx.dayofyear.values
    hour = idx.hour.values
    weekday = idx.dayofweek.values  # 0=Mon
    year_frac = (idx - idx[0]).days / 365.25

    # --- Base load with long-term growth trend ---
    base_load = 3200 + 45 * year_frac  # slow demand growth over 3 years

    # --- Annual seasonality (higher in summer for cooling + winter heating) ---
    annual_cycle = 550 * np.cos(2 * np.pi * (day_of_year - 200) / 365.25) * -1
    annual_cycle += 300 * np.cos(2 * np.pi * (day_of_year - 15) / 365.25)  # winter peak

    # --- Daily (hourly) demand curve: morning + evening peaks ---
    daily_cycle = (
        260 * np.exp(-((hour - 9) ** 2) / (2 * 2.5 ** 2)) +
        420 * np.exp(-((hour - 19) ** 2) / (2 * 2.8 ** 2)) -
        350 * np.exp(-((hour - 4) ** 2) / (2 * 3.0 ** 2))
    )

    # --- Weekly pattern: lower demand on weekends ---
    weekend_dip = np.where(weekday >= 5, -240, 0)

    # --- Temperature (drives cooling/heating demand) ---
    temperature = (
        18 - 10 * np.cos(2 * np.pi * (day_of_year - 15) / 365.25)
        + np.random.normal(0, 2.2, n)
        + 4 * np.sin(2 * np.pi * hour / 24 - np.pi / 2)
    )
    temp_effect = 4.5 * np.abs(temperature - 21)  # AC/heating load beyond comfort temp

    # --- Noise ---
    noise = np.random.normal(0, 90, n)

    demand = base_load + annual_cycle + daily_cycle + weekend_dip + temp_effect + noise
    demand = np.clip(demand, 1800, None)

    # --- Solar generation: daylight hours, seasonal intensity, cloud noise ---
    solar_potential = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)
    seasonal_solar = 0.55 + 0.45 * np.cos(2 * np.pi * (day_of_year - 172) / 365.25)
    cloud_factor = np.clip(np.random.normal(0.85, 0.25, n), 0.05, 1.2)
    solar_gen = 700 * solar_potential * seasonal_solar * cloud_factor
    solar_gen = np.clip(solar_gen, 0, None)

    # --- Wind generation: fairly random with mild seasonal boost in winter ---
    wind_seasonal = 1.15 - 0.3 * np.cos(2 * np.pi * (day_of_year - 172) / 365.25)
    wind_gen = np.clip(np.random.weibull(1.8, n) * 220 * wind_seasonal, 0, 900)

    df = pd.DataFrame({
        "datetime": idx,
        "demand_mw": demand.round(2),
        "solar_gen_mw": solar_gen.round(2),
        "wind_gen_mw": wind_gen.round(2),
        "temperature_c": temperature.round(2),
    })

    # --- Inject realistic messiness for the cleaning step to handle ---
    rng = np.random.default_rng(7)

    # 1. Random missing values (sensor dropouts)
    missing_idx = rng.choice(df.index, size=int(0.015 * n), replace=False)
    df.loc[missing_idx, "demand_mw"] = np.nan
    missing_idx2 = rng.choice(df.index, size=int(0.01 * n), replace=False)
    df.loc[missing_idx2, "temperature_c"] = np.nan

    # 2. Duplicate rows (logging errors)
    dup_rows = df.sample(n=40, random_state=1)
    df = pd.concat([df, dup_rows], ignore_index=True)

    # 3. Outliers / sensor spikes
    spike_idx = rng.choice(df.index, size=25, replace=False)
    df.loc[spike_idx, "demand_mw"] *= rng.uniform(2.5, 4.0, size=25)

    # 4. A few negative/impossible values (data entry errors)
    err_idx = rng.choice(df.index, size=10, replace=False)
    df.loc[err_idx, "solar_gen_mw"] = -df.loc[err_idx, "solar_gen_mw"]

    # 5. Shuffle rows out of order (as real raw exports often are)
    df = df.sample(frac=1.0, random_state=3).reset_index(drop=True)

    # 6. datetime stored as inconsistent string format (realism)
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    return df


if __name__ == "__main__":
    data = generate()
    out_path = "energy_demand_raw.csv"
    data.to_csv(out_path, index=False)
    print(f"Raw dataset generated: {out_path} ({len(data)} rows)")
