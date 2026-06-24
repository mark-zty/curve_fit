from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class CovarianceEstimator(ABC):
    @abstractmethod
    def fit(self, series: pd.DataFrame) -> np.ndarray:
        """
        series: wide DataFrame (Date × Tenor) of par rate levels or daily changes.
        Returns (n, n) covariance matrix.
        """
