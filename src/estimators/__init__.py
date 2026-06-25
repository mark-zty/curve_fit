from .naive import NaiveEstimator
from .ledoit_wolf import LedoitWolfEstimator
from .oas import OASEstimator

REGISTRY: dict[str, type] = {
    "naïve": NaiveEstimator,
    "ledoit_wolf": LedoitWolfEstimator,
    "oas": OASEstimator,
}


def get_estimator(name: str):
    if name not in REGISTRY:
        raise KeyError(f"Unknown estimator '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name]()
