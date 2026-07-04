"""
forecasting.py
----------------
Stationarity testing, ACF/PACF analysis, and ARIMA / SARIMA model fitting
and forecasting.

IMPORTANT - environment note:
This module is written to use `statsmodels` (the industry-standard library
for ADF/KPSS tests, ACF/PACF, ARIMA and SARIMAX) whenever it is installed --
which it will be in any normal environment via requirements.txt. The sandbox
this project was authored in has no internet access to install packages, so
a from-scratch fallback implementation (`_ManualADF`, `_ManualKPSS`,
`_ManualACFPACF`, `_ManualARIMA`) is included and is used automatically if
`import statsmodels` fails. This keeps the pipeline runnable everywhere while
still being 100% statsmodels-based in the target/portfolio environment.
Nothing in main.py or the notebook needs to change either way.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

try:
    from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


# ----------------------------------------------------------------------
# STATIONARITY TESTS
# ----------------------------------------------------------------------

def adf_test(series: pd.Series) -> dict:
    """Augmented Dickey-Fuller test. H0: series has a unit root (non-stationary)."""
    series = series.dropna()
    if STATSMODELS_AVAILABLE:
        result = adfuller(series, autolag="AIC")
        return {"test_stat": result[0], "p_value": result[1],
                "n_lags": result[2], "crit_values": result[4],
                "stationary": result[1] < 0.05}
    return _manual_adf(series)


def _manual_adf(series: pd.Series, max_lag: int = 24) -> dict:
    """From-scratch ADF test via OLS regression (used only if statsmodels
    is unavailable). Regression: dy_t = a + b*t + g*y_{t-1} + sum(d_i*dy_{t-i}) + e
    Test statistic = g_hat / se(g_hat), compared to MacKinnon-approximate
    critical values for a constant+trend model."""
    y = series.values
    dy = np.diff(y)
    n = len(dy) - max_lag
    X_rows = []
    y_target = []
    for t in range(max_lag, len(dy)):
        row = [1.0, t, y[t]]  # const, trend, lagged level
        row += [dy[t - i] for i in range(1, max_lag + 1)]
        X_rows.append(row)
        y_target.append(dy[t])
    X = np.array(X_rows)
    y_t = np.array(y_target)
    beta, residuals, rank, sv = np.linalg.lstsq(X, y_t, rcond=None)
    resid = y_t - X @ beta
    dof = len(y_t) - X.shape[1]
    sigma2 = (resid @ resid) / dof
    XtX_inv = np.linalg.inv(X.T @ X)
    se_gamma = np.sqrt(sigma2 * XtX_inv[2, 2])
    t_stat = beta[2] / se_gamma
    crit = {"1%": -3.96, "5%": -3.41, "10%": -3.12}  # approx, constant+trend
    return {"test_stat": t_stat, "p_value": None, "n_lags": max_lag,
            "crit_values": crit, "stationary": t_stat < crit["5%"]}


def kpss_test(series: pd.Series) -> dict:
    """KPSS test. H0: series is (trend/level) stationary -- opposite null to ADF."""
    series = series.dropna()
    if STATSMODELS_AVAILABLE:
        stat, p_value, n_lags, crit = kpss(series, regression="c", nlags="auto")
        return {"test_stat": stat, "p_value": p_value, "n_lags": n_lags,
                "crit_values": crit, "stationary": p_value > 0.05}
    return _manual_kpss(series)


def _manual_kpss(series: pd.Series, lags: int = 24) -> dict:
    """From-scratch level-stationarity KPSS test with Newey-West long-run
    variance (used only if statsmodels is unavailable)."""
    y = series.values
    n = len(y)
    resid = y - y.mean()
    S = np.cumsum(resid)
    gamma0 = np.sum(resid ** 2) / n
    lrv = gamma0
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        gamma_l = np.sum(resid[lag:] * resid[:-lag]) / n
        lrv += 2 * w * gamma_l
    kpss_stat = np.sum(S ** 2) / (n ** 2 * lrv)
    crit = {"1%": 0.739, "5%": 0.463, "10%": 0.347}
    return {"test_stat": kpss_stat, "p_value": None, "n_lags": lags,
            "crit_values": crit, "stationary": kpss_stat < crit["5%"]}


# ----------------------------------------------------------------------
# ACF / PACF
# ----------------------------------------------------------------------

def compute_acf_pacf(series: pd.Series, nlags: int = 48):
    series = series.dropna()
    if STATSMODELS_AVAILABLE:
        acf_vals = acf(series, nlags=nlags)
        pacf_vals = pacf(series, nlags=nlags)
        return acf_vals, pacf_vals
    return _manual_acf(series, nlags), _manual_pacf(series, nlags)


def _manual_acf(series: pd.Series, nlags: int) -> np.ndarray:
    y = series.values
    y = y - y.mean()
    n = len(y)
    denom = np.sum(y ** 2)
    return np.array([np.sum(y[:n - k] * y[k:]) / denom for k in range(nlags + 1)])


def _manual_pacf(series: pd.Series, nlags: int) -> np.ndarray:
    """PACF via successive OLS regressions (regress y_t on lags 1..k, take
    coefficient on lag k)."""
    y = series.values
    n = len(y)
    pacf_vals = [1.0]
    for k in range(1, nlags + 1):
        X = np.column_stack([y[k - i - 1:n - i - 1] for i in range(k)]) if k > 0 else None
        target = y[k:]
        X = np.column_stack([np.ones(len(target))] + [y[k - i - 1:n - i - 1] for i in range(k)])
        beta, *_ = np.linalg.lstsq(X, target, rcond=None)
        pacf_vals.append(beta[-1])
    return np.array(pacf_vals)


def decompose_series(series: pd.Series, period: int = 24):
    series = series.dropna()
    if STATSMODELS_AVAILABLE:
        return seasonal_decompose(series, model="additive", period=period)
    return _manual_decompose(series, period)


class _ManualDecomposeResult:
    def __init__(self, trend, seasonal, resid, observed):
        self.trend, self.seasonal, self.resid, self.observed = trend, seasonal, resid, observed


def _manual_decompose(series: pd.Series, period: int):
    """Simple additive decomposition: trend = centered moving average,
    seasonal = average detrended value per period-position, resid = rest."""
    trend = series.rolling(window=period, center=True).mean()
    detrended = series - trend
    seasonal_avg = detrended.groupby(detrended.index.hour if period == 24 else
                                      (np.arange(len(detrended)) % period)).mean()
    if period == 24:
        seasonal = series.index.hour.map(seasonal_avg)
        seasonal = pd.Series(seasonal, index=series.index)
    else:
        idx_mod = np.arange(len(series)) % period
        seasonal = pd.Series([seasonal_avg[i] for i in idx_mod], index=series.index)
    resid = series - trend - seasonal
    return _ManualDecomposeResult(trend, seasonal, resid, series)


# ----------------------------------------------------------------------
# ARIMA / SARIMA MODELS
# ----------------------------------------------------------------------

class ManualARIMA:
    """From-scratch ARIMA(p,d,q)-style model (AR + differencing, with an
    approximate MA term via a two-stage residual-regression), used only as a
    fallback when statsmodels is not installed. Optional seasonal lag terms
    approximate SARIMA behaviour.

    This is NOT a full MLE ARIMA implementation -- it is a transparent
    linear-regression-based approximation with the same conceptual
    ingredients (AR, I, MA, seasonal lags) so the pipeline can run start to
    finish without statsmodels. Swap in statsmodels' ARIMA/SARIMAX (already
    wired up above) for production-grade parameter estimation.
    """

    def __init__(self, p=24, d=1, q=1, seasonal_lag=None):
        self.p, self.d, self.q, self.seasonal_lag = p, d, q, seasonal_lag
        self.model_ = None
        self.last_values_ = None
        self.history_ = None

    def fit(self, series: pd.Series):
        y = series.values.astype(float)
        self.history_ = y.copy()
        diff = y.copy()
        for _ in range(self.d):
            diff = np.diff(diff)
        self.diff_ = diff

        lags = list(range(1, self.p + 1))
        if self.seasonal_lag:
            lags.append(self.seasonal_lag)
        max_lag = max(lags)

        X = np.column_stack([diff[max_lag - l: len(diff) - l] for l in lags])
        target = diff[max_lag:]
        ar_model = LinearRegression().fit(X, target)
        ar_resid = target - ar_model.predict(X)

        # Approximate MA(q) stage: regress residuals on their own lagged values
        if self.q > 0 and len(ar_resid) > self.q + 5:
            Xr = np.column_stack([ar_resid[self.q - i: len(ar_resid) - i] for i in range(1, self.q + 1)])
            tr = ar_resid[self.q:]
            ma_model = LinearRegression().fit(Xr, tr)
        else:
            ma_model = None

        self.lags_ = lags
        self.max_lag_ = max_lag
        self.ar_model_ = ar_model
        self.ma_model_ = ma_model
        self.last_resid_ = ar_resid[-self.q:] if (self.q > 0 and ma_model is not None) else None
        return self

    def forecast(self, steps: int) -> np.ndarray:
        diff_hist = list(self.diff_)
        resid_hist = list(self.last_resid_) if self.last_resid_ is not None else []
        forecasts_diff = []
        for _ in range(steps):
            x = [diff_hist[-l] for l in self.lags_]
            pred = self.ar_model_.predict([x])[0]
            if self.ma_model_ is not None and len(resid_hist) >= self.q:
                xr = [resid_hist[-i] for i in range(1, self.q + 1)]
                pred += self.ma_model_.predict([xr])[0]
                resid_hist.append(0.0)  # future residual assumed 0 (best estimate)
            diff_hist.append(pred)
            forecasts_diff.append(pred)

        # Integrate back (undo differencing) starting from last actual values
        result = np.array(forecasts_diff)
        last_vals = self.history_[-self.d:] if self.d > 0 else None
        for _ in range(self.d):
            result = np.cumsum(result) + last_vals[-1]
            last_vals = last_vals[:-1]
        return result


def fit_arima(train: pd.Series, order=(24, 1, 1)):
    """Fit an ARIMA model (statsmodels if available, else ManualARIMA)."""
    if STATSMODELS_AVAILABLE:
        model = ARIMA(train, order=order)
        return model.fit()
    p, d, q = order
    return ManualARIMA(p=p, d=d, q=q).fit(train)


def fit_sarima(train: pd.Series, order=(2, 1, 1), seasonal_order=(1, 1, 1, 24)):
    """Fit a SARIMA model (statsmodels if available, else ManualARIMA with a
    seasonal lag term approximating the seasonal component)."""
    if STATSMODELS_AVAILABLE:
        model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                         enforce_stationarity=False, enforce_invertibility=False)
        return model.fit(disp=False)
    p, d, q = order
    seasonal_lag = seasonal_order[3]
    return ManualARIMA(p=p, d=d, q=q, seasonal_lag=seasonal_lag).fit(train)


def forecast_model(fitted_model, steps: int) -> np.ndarray:
    """Uniform forecasting interface regardless of backend."""
    if STATSMODELS_AVAILABLE:
        return np.asarray(fitted_model.forecast(steps=steps))
    return fitted_model.forecast(steps=steps)
