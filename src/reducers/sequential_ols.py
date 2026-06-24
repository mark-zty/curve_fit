import numpy as np

from .base import DimensionReducer, FactorResult
from ..tenor_graph import TenorGraph


class SequentialOLSReducer(DimensionReducer):
    def fit(self, C: np.ndarray, tenors: list[str], tenor_graph: TenorGraph) -> FactorResult:
        row = {t: i for i, t in enumerate(tenors)}
        layers = tenor_graph.all_layers
        order = [t for l in layers for t in tenor_graph.tenors_in_layer(l)]
        col = {t: c for c, t in enumerate(order)}
        n = len(tenors)
        base = layers[0]

        L = np.zeros((n, n))

        for t in order:
            i, c, lt = row[t], col[t], tenor_graph.layer_of(t)
            if lt == base:
                L[i, c] = 1.0
                continue

            # Build t from progressively less liquid layers: regress the running
            # residual on each more-liquid layer, dropping that layer's explained
            # part before moving on. w holds the residual as a combo of raw tenors.
            w = np.zeros(n)
            w[i] = 1.0
            for k in layers:
                if k >= lt:
                    break
                layer_tenors = tenor_graph.tenors_in_layer(k)
                p = [row[x] for x in layer_tenors]
                beta = np.linalg.solve(C[np.ix_(p, p)], C[p, :] @ w)
                for x, b in zip(layer_tenors, beta):
                    L[i, col[x]] = b
                w[p] -= beta

        return FactorResult(
            loadings=L,
            factor_labels=order,
            tenors=tenors,
        )
