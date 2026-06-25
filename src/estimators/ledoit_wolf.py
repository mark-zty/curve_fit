import numpy as np
import pandas as pd
from sklearn.covariance import ledoit_wolf

from .base import CovarianceEstimator


class LedoitWolfEstimator(CovarianceEstimator):
    def fit(self, series: pd.DataFrame) -> np.ndarray:
        return ledoit_wolf(series.values)[0]
