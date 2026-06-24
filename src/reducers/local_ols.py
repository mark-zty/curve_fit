import numpy as np

from .base import DimensionReducer, FactorResult
from ..tenor_graph import TenorGraph


class LocalOLSReducer(DimensionReducer):
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

            # OLS regression of t on its direct parents: beta = C_pp^{-1} C_pt.
            p = [row[x] for x in parents]
            beta = np.linalg.solve(C[np.ix_(p, p)], C[p, i])
            for x, b in zip(parents, beta):
                L[i, col[x]] = b

        return FactorResult(
            loadings=L,
            factor_labels=order,
            tenors=tenors,
        )
