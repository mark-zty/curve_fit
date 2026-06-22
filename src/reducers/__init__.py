from .sequential_pca import SequentialPCAReducer
from .sequential_ols import SequentialOLSReducer

REGISTRY: dict[str, type] = {
    "sequential_pca": SequentialPCAReducer,
    "sequential_ols": SequentialOLSReducer,
}


def get_reducer(name: str):
    if name not in REGISTRY:
        raise KeyError(f"Unknown reducer '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name]()
