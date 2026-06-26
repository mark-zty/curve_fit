from .naive import NaiveEstimator
from .ledoit_wolf import LedoitWolfEstimator
from .oas import OASEstimator
from .newey_west import NeweyWestEstimator

REGISTRY: dict[str, type] = {
    "naïve": NaiveEstimator,
    "ledoit_wolf": LedoitWolfEstimator,
    "oas": OASEstimator,
    "newey_west": NeweyWestEstimator,
}


def get_estimator(name: str):
    if name not in REGISTRY:
        raise KeyError(f"Unknown estimator '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name]()
