import numpy as np
import pandas as pd
from sklearn.covariance import oas

from .base import CovarianceEstimator


class OASEstimator(CovarianceEstimator):
    def fit(self, series: pd.DataFrame) -> np.ndarray:
        return oas(series.values)[0]
