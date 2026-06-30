from .sequential_pca import SequentialPCAReducer
from .sequential_ols import SequentialOLSReducer
from .local_pca import LocalPCAReducer
from .local_ols import LocalOLSReducer

REGISTRY: dict[str, type] = {
    "sequential_pca": SequentialPCAReducer,
    "sequential_ols": SequentialOLSReducer,
    "local_pca": LocalPCAReducer,
    "local_ols": LocalOLSReducer,
}


def get_reducer(name: str, **kwargs):
    if name not in REGISTRY:
        raise KeyError(f"Unknown reducer '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)
