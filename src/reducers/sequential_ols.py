import numpy as np

from .base import DimensionReducer, FactorResult
from ..tenor_graph import TenorGraph


class SequentialOLSReducer(DimensionReducer):
    def fit(self, C: np.ndarray, tenors: list[str], tenor_graph: TenorGraph) -> FactorResult:
        row = {t: i for i, t in enumerate(tenors)}
        order = [t for layer in tenor_graph.all_layers for t in tenor_graph.tenors_in_layer(layer)]
        col = {t: c for c, t in enumerate(order)}
        n = len(tenors)

        L = np.zeros((n, n))
        variances = np.zeros(n)

        for t in order:
            i, c = row[t], col[t]
            L[i, c] = 1.0
            preds = tenor_graph.predictors_of(t)
            if not preds:
                variances[c] = C[i, i]
                continue
            p = [row[x] for x in preds]
            beta = np.linalg.solve(C[np.ix_(p, p)], C[p, i])
            L[i, :] += beta @ L[p, :]
            variances[c] = C[i, i] - C[i, p] @ beta

        return FactorResult(
            loadings=L,
            factor_labels=order,
            tenors=tenors,
            explained_variance=variances,
        )
