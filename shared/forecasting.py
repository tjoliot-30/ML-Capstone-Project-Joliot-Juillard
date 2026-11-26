"""Public forecasting primitives for TD1, TD2, and TDo1.

The assessed ETS/SARIMA rolling loop deliberately stays in the notebooks. This
module exposes only target-safe primitives and a bounded residual diagnostic;
it does not distribute a completed model-comparison solution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

def seasonal_naive(train: np.ndarray, horizon: int, season: int = 7) -> np.ndarray:
    train = np.asarray(train, dtype=float)
    if len(train) < season:
        raise ValueError("Training history is shorter than the seasonal period")
    return np.resize(train[-season:], horizon)


def rolling_origins(n_observations: int, *, initial: int = 700, horizon: int = 14, step: int = 28, n_origins: int = 3) -> tuple[int, ...]:
    candidates = tuple(initial + step * i for i in range(n_origins))
    if candidates[-1] + horizon > n_observations:
        raise ValueError("Not enough observations for the requested rolling origins")
    return candidates


def sarima_residual_diagnostics(series: np.ndarray, *, season: int = 7, max_lag: int = 14) -> tuple[pd.DataFrame, np.ndarray]:
    """Fit the teaching SARIMA and return Ljung-Box plus residual ACF evidence.

    The first two seasonal periods are discarded as a bounded burn-in. Four
    fitted AR/MA coefficients (p+q+P+Q) consume reference degrees of freedom;
    differencing orders are not fitted coefficients. Fit warnings remain
    visible. Failure to reject is not proof that residuals are independent.
    """

    values = np.asarray(series, dtype=float)
    fitted = SARIMAX(
        values,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, season),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=50)
    residuals = np.asarray(fitted.resid)[2 * season :]
    model_df = 4  # SARIMA(1,1,1)(1,0,1,m): p + q + P + Q.
    diagnostic = acorr_ljungbox(
        residuals, lags=[season, max_lag], model_df=model_df, return_df=True
    ).reset_index(names="lag")
    diagnostic["model_df"] = model_df
    diagnostic["test_df"] = diagnostic["lag"] - model_df
    diagnostic["fit_converged"] = bool(fitted.mle_retvals.get("converged", False))
    residual_acf = acf(residuals, nlags=max_lag, fft=True)
    return diagnostic, residual_acf
