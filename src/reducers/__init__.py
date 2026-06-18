from .sequential_pca import SequentialPCAReducer

REGISTRY: dict[str, type] = {
    "sequential_pca": SequentialPCAReducer,
}


def get_reducer(name: str):
    if name not in REGISTRY:
        raise KeyError(f"Unknown reducer '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name]()
