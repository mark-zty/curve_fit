from .naive import NaiveEstimator
from .ledoit_wolf import LedoitWolfEstimator
from .oas import OASEstimator
from .newey_west import NeweyWestEstimator
from .hjm_kernel import HJMKernelEstimator

REGISTRY: dict[str, type] = {
    "naïve": NaiveEstimator,
    "ledoit_wolf": LedoitWolfEstimator,
    "oas": OASEstimator,
    "newey_west": NeweyWestEstimator,
    "hjm_kernel": HJMKernelEstimator,
}


def get_estimator(name: str, **kwargs):
    if name not in REGISTRY:
        raise KeyError(f"Unknown estimator '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)
