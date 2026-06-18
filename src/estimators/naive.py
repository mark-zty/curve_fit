import numpy as np
import pandas as pd

from .base import CovarianceEstimator


class NaiveEstimator(CovarianceEstimator):
    def fit(self, rates: pd.DataFrame) -> np.ndarray:
        diffs = rates.diff().dropna()
        return np.cov(diffs.values.T)
