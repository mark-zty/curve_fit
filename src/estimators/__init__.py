from .naive import NaiveEstimator

REGISTRY: dict[str, type] = {
    "naive": NaiveEstimator,
}


def get_estimator(name: str):
    if name not in REGISTRY:
        raise KeyError(f"Unknown estimator '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name]()
