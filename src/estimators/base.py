from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class CovarianceEstimator(ABC):
    @abstractmethod
    def fit(self, rates: pd.DataFrame) -> np.ndarray:
        """
        rates: wide DataFrame (Date × Tenor) of par rates.
        Returns (n, n) covariance matrix of daily rate changes.
        """
