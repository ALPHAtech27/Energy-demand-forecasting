"""
evaluation.py
--------------
Forecast evaluation metrics: MAE, RMSE, MAPE, R^2.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_forecast(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute standard forecast error metrics."""
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / np.where(actual == 0, np.nan, actual))) * 100
    r2 = r2_score(actual, predicted)

    return {"MAE": round(mae, 3), "RMSE": round(rmse, 3),
            "MAPE": round(mape, 3), "R2": round(r2, 4)}


def compare_models(metrics_dict: dict) -> str:
    """Given {model_name: metrics_dict}, return the name of the best model
    by RMSE (primary) with MAPE as a tiebreaker, and a plain-language reason."""
    best_model = min(metrics_dict, key=lambda m: metrics_dict[m]["RMSE"])
    other_models = [m for m in metrics_dict if m != best_model]
    reason_parts = []
    for m in other_models:
        rmse_diff = metrics_dict[m]["RMSE"] - metrics_dict[best_model]["RMSE"]
        reason_parts.append(f"{rmse_diff:.1f} MW lower RMSE than {m}")
    reason = f"{best_model} was selected as the better model ({', '.join(reason_parts)})."
    return best_model, reason
