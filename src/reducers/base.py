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

    def cascade(self, loadings: np.ndarray, tenor_graph: TenorGraph) -> np.ndarray:
        """Fold every illiquid tenor's risk down to the first layer, given the
        layer-ordered loading matrix. The default treats loadings as a graded
        transition and raises it to (#layers - 1) to route all intermediate layers
        through; reducers whose loadings already carry the full first-layer
        replication override this."""
        return np.linalg.matrix_power(loadings, len(tenor_graph.all_layers) - 1)
