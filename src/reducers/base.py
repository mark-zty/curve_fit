from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..tenor_graph import TenorGraph


@dataclass
class FactorResult:
    loadings: np.ndarray           # (n_tenors, n_factors)
    factor_labels: list[str]       # anchor tenor for each factor column
    tenors: list[str]              # tenor label for each row


class DimensionReducer(ABC):
    @abstractmethod
    def fit(self, C: np.ndarray, tenors: list[str], tenor_graph: TenorGraph) -> FactorResult:
        """
        C: (n, n) covariance matrix aligned with tenors.
        tenors: ordered tenor labels matching C's rows/cols.
        """
