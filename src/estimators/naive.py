import numpy as np
import pandas as pd

from .base import CovarianceEstimator


class NaiveEstimator(CovarianceEstimator):
    def fit(self, series: pd.DataFrame) -> np.ndarray:
        return np.cov(series.values.T)
