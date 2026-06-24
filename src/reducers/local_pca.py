import numpy as np

from .base import DimensionReducer, FactorResult
from ..tenor_graph import TenorGraph


class LocalPCAReducer(DimensionReducer):
    def fit(self, C: np.ndarray, tenors: list[str], tenor_graph: TenorGraph) -> FactorResult:
        row = {t: i for i, t in enumerate(tenors)}
        order = [t for l in tenor_graph.all_layers for t in tenor_graph.tenors_in_layer(l)]
        col = {t: c for c, t in enumerate(order)}
        n = len(tenors)

        L = np.zeros((n, n))
        for t in order:
            i = row[t]
            parents = tenor_graph.predictors_of(t)
            if not parents:
                L[i, col[t]] = 1.0
                continue

            # Orthogonal (total-least-squares) regression of t on its direct
            # parents: the smallest-eigenvalue PC of the local covariance is the
            # normal w to the best-fit hyperplane, so child = -w_p / w_t on each.
            local = [row[p] for p in parents] + [i]
            w = np.linalg.eigh(C[np.ix_(local, local)])[1][:, 0]
            for p, wp in zip(parents, w[:-1]):
                L[i, col[p]] = -wp / w[-1]

        return FactorResult(
            loadings=L,
            factor_labels=order,
            tenors=tenors,
        )
