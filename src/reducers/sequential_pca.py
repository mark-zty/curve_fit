import numpy as np

from .base import DimensionReducer, FactorResult
from ..tenor_graph import TenorGraph


class SequentialPCAReducer(DimensionReducer):
    def fit(self, C: np.ndarray, tenors: list[str], tenor_graph: TenorGraph) -> FactorResult:
        idx = {t: i for i, t in enumerate(tenors)}
        all_loadings, all_labels, all_variances = [], [], []
        C_res = C.copy()

        for layer in tenor_graph.all_layers:
            layer_tenors = tenor_graph.tenors_in_layer(layer)   # sorted by maturity
            layer_idx = [idx[t] for t in layer_tenors]
            n_L = len(layer_tenors)

            # eigh returns eigenvalues ascending; flip to get top-n_L first
            eigenvalues, eigenvectors = np.linalg.eigh(C_res)
            P = eigenvectors[:, -n_L:][:, ::-1]         # n × n_L
            top_vals = eigenvalues[-n_L:][::-1]

            # Change of basis: invert the n_L×n_L sub-matrix at layer-tenor rows
            # so each layer tenor ends up with loading 1 on exactly one factor
            P_L = P[layer_idx, :]                        # n_L × n_L
            T = np.linalg.inv(P_L)
            L = P @ T                                    # n × n_L; L[layer_idx,:] = I

            all_loadings.append(L)
            all_labels.extend(layer_tenors)
            all_variances.extend(top_vals.tolist())

            # Deflate: remove the variance captured by this layer's factors
            C_res = C_res - P @ np.diag(top_vals) @ P.T

        return FactorResult(
            loadings=np.hstack(all_loadings),
            factor_labels=all_labels,
            tenors=tenors,
            explained_variance=np.array(all_variances),
        )
