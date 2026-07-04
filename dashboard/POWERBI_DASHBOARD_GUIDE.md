# Power BI Dashboard Build Guide -- Energy Demand Forecasting

This guide walks through building `EnergyDashboard.pbix` from `dashboard/dashboard_data.csv`
(generated automatically by `main.py`). A `.pbix` binary can't be authored outside Power BI
Desktop itself, so this is a precise, field-by-field build spec you can follow in ~20-30 minutes.

## 1. Data Import

1. Power BI Desktop -> **Get Data -> Text/CSV** -> select `dashboard/dashboard_data.csv`.
2. Columns: `date_time` (datetime), `demand_mw`, `solar_gen_mw`, `wind_gen_mw`, `temperature_c`,
   `year`, `month`, `week`, `hour`, `is_weekend`, `day_name`, `record_type` (`Actual`/`Forecast`).
3. In Power Query, set `date_time` to **Date/Time**, `is_weekend` to **Whole Number**, and confirm
   the rest are typed correctly, then **Close & Apply**.
4. Also import `outputs/metrics.csv` (model, MAE, RMSE, MAPE, R2) as a second table for the
   model-comparison visual.

## 2. Model / Relationships

Create a **Date table** (Modeling -> New Table):
```
DateTable = CALENDAR(MIN(dashboard_data[date_time]), MAX(dashboard_data[date_time]))
```
Mark it as a Date Table and relate it 1:* to `dashboard_data[date_time]`. This unlocks proper
time-intelligence functions and consistent slicer behavior across visuals.

## 3. DAX Measures (KPI Cards)

```DAX
Total Demand (MWh) = SUM(dashboard_data[demand_mw])

Average Demand = AVERAGE(dashboard_data[demand_mw])

Peak Demand = CALCULATE(MAX(dashboard_data[demand_mw]), dashboard_data[record_type]="Actual")

Forecasted Demand (Next 30d) =
    CALCULATE(AVERAGE(dashboard_data[demand_mw]), dashboard_data[record_type]="Forecast")

YoY Demand Growth % =
    VAR CurrYear = CALCULATE(AVERAGE(dashboard_data[demand_mw]), dashboard_data[year]=MAX(dashboard_data[year]))
    VAR PrevYear = CALCULATE(AVERAGE(dashboard_data[demand_mw]), dashboard_data[year]=MAX(dashboard_data[year])-1)
    RETURN DIVIDE(CurrYear - PrevYear, PrevYear)
```

## 4. Page 1 -- Executive Overview

**KPI Cards** (top row, Card visual): Total Demand, Average Demand, Peak Demand, Forecasted
Demand. Use a consistent number format (`#,##0 "MW"`).

**Charts:**
- **Demand Trend** -- Line chart, X=`date_time` (Date hierarchy -> Month), Y=`demand_mw`,
  Legend=`record_type` (so Actual/Forecast render in different colors/line styles automatically).
- **Monthly Demand** -- Clustered column chart, X=`month`, Y=`Average Demand` measure.
- **Solar Generation** -- Area chart, X=`date_time`, Y=`solar_gen_mw`.
- **Rolling Average** -- Line chart using a DAX measure:
  `Rolling 7d Avg = AVERAGEX(DATESINPERIOD(DateTable[Date], MAX(DateTable[Date]), -7, DAY), [Average Demand])`

## 5. Page 2 -- Forecast vs Actual

- Line chart: X=`date_time`, Y=`demand_mw`, Legend=`record_type`, filtered to the last 90 days +
  all forecast rows, so the actual/forecast transition is visible in one view.
- Table/matrix visual pulling from `metrics.csv`: rows=`model`, columns=MAE/RMSE/MAPE/R2,
  conditional-formatted (green=better) so viewers instantly see SARIMA outperforming ARIMA.

## 6. Page 3 -- Seasonal & Heatmap Analysis

- **Heatmap**: Matrix visual, rows=`hour`, columns=`day_name`, values=`Average Demand` measure,
  with conditional formatting (color scale) -- reveals the weekday evening-peak pattern visually.
- **Box-and-whisker-style spread**: Use a Line/Column combo or the "Box and Whisker" custom
  visual (import from AppSource) with `month` on the axis and `demand_mw` as the value.

## 7. Interactive Features

- **Slicers**: `year`, `is_weekend` (rename values 0/1 -> Weekday/Weekend via a calculated
  column), and a date-range slicer bound to `DateTable[Date]`.
- **Drill-through page**: create a "Day Detail" page with a table of hourly `demand_mw`,
  `temperature_c`, `solar_gen_mw`; right-click a day on any chart -> Drill Through.
- **Tooltips**: build a small tooltip page showing demand + temperature for the hovered date;
  set it as the default tooltip on the main line chart (Format -> Tooltip -> Report page).
- **Navigation buttons**: Insert -> Buttons -> Blank, bind each to "Page Navigation" for the
  three pages above; place consistently in a top nav bar shape.

## 8. Theme / Styling

Apply a custom theme (View -> Themes -> Browse for themes) with a professional palette, e.g.:
- Primary: `#1F4E79` (deep blue -- demand)
- Accent: `#ED7D31` (orange -- solar/forecast)
- Background: `#F5F5F5`, Text: `#262626`

Use a single consistent font (Segoe UI or Calibri) across all visuals, and keep KPI cards flush
along the top with charts below, matching typical executive-dashboard layouts.

## 9. Refresh

Since `main.py` regenerates `dashboard/dashboard_data.csv` and `outputs/metrics.csv` each run,
set the Power BI file's data source to that path and use **Refresh** after each pipeline run to
pull in the latest cleaned data, forecast, and metrics.
