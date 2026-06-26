import numpy as np
import pandas as pd

from .base import CovarianceEstimator


class NeweyWestEstimator(CovarianceEstimator):
    """HAC (Newey-West) covariance with a Bartlett kernel.

    Sample covariance plus Bartlett-weighted lagged autocovariances, so serial
    correlation in daily changes is carried into the matrix. The Bartlett kernel
    keeps the result positive semi-definite; max_lags=0 recovers the sample covariance.
    """

    def __init__(self, max_lags: int | None = None):
        self.max_lags = max_lags

    def fit(self, series: pd.DataFrame) -> np.ndarray:
        X = series.values
        T = len(X)
        # Newey & West (1994) rule-of-thumb bandwidth; 2/9 is their Bartlett-kernel rate
        L = self.max_lags if self.max_lags is not None else int(4 * (T / 100) ** (2 / 9))
        L = min(L, T - 1)

        Xc = X - X.mean(axis=0)
        S  = Xc.T @ Xc
        for k in range(1, L + 1):
            G  = Xc[k:].T @ Xc[:-k]
            S += (1 - k / (L + 1)) * (G + G.T)
        return S / (T - 1)
