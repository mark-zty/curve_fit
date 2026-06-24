import numpy as np

from .base import DimensionReducer, FactorResult
from ..tenor_graph import TenorGraph


class SequentialPCAReducer(DimensionReducer):
    def fit(self, C: np.ndarray, tenors: list[str], tenor_graph: TenorGraph) -> FactorResult:
        idx = {t: i for i, t in enumerate(tenors)}
        n = len(tenors)
        all_loadings, all_labels = [], []

        active = list(range(n))     # original indices still in play
        C_res = C.copy()            # principal submatrix of C over `active`

        for layer in tenor_graph.all_layers:
            layer_tenors = tenor_graph.tenors_in_layer(layer)   # sorted by maturity
            n_L = len(layer_tenors)
            pos = {orig: i for i, orig in enumerate(active)}
            local_idx = [pos[idx[t]] for t in layer_tenors]     # layer rows within C_res

            # eigh returns eigenvalues ascending; flip to get top-n_L first
            eigenvalues, eigenvectors = np.linalg.eigh(C_res)
            P = eigenvectors[:, -n_L:][:, ::-1]         # m × n_L
            top_vals = eigenvalues[-n_L:][::-1]

            # Change of basis: invert the n_L×n_L sub-matrix at layer-tenor rows
            # so each layer tenor ends up with loading 1 on exactly one factor
            T = np.linalg.inv(P[local_idx, :])
            L = P @ T                                   # m × n_L; L[local_idx,:] = I

            # Scatter back into full-n rows; already-anchored tenors stay 0
            L_full = np.zeros((n, n_L))
            L_full[active, :] = L

            all_loadings.append(L_full)
            all_labels.extend(layer_tenors)

            # Deflate captured variance, then drop this layer's dimensions
            C_res = C_res - P @ np.diag(top_vals) @ P.T
            keep = [i for i in range(len(active)) if i not in local_idx]
            C_res = C_res[np.ix_(keep, keep)]
            active = [active[i] for i in keep]

        return FactorResult(
            loadings=np.hstack(all_loadings),
            factor_labels=all_labels,
            tenors=tenors,
        )
