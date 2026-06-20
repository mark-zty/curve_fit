import re


def tenor_to_years(tenor: str) -> float:
    m = re.fullmatch(r'(\d+)(M|Y)', tenor)
    if not m:
        raise ValueError(f"Unrecognized tenor format: {tenor!r}")
    n, unit = int(m.group(1)), m.group(2)
    return n / 12 if unit == 'M' else float(n)


class TenorGraph:
    def __init__(self, structure: dict):
        self._s = structure
        self._layers = self._infer_layers()

    def _infer_layers(self) -> dict[str, int]:
        cache: dict[str, int] = {}
        def _layer(tenor: str) -> int:
            if tenor not in cache:
                preds = self._s[tenor]["predictors"]
                cache[tenor] = 0 if not preds else max(_layer(p) for p in preds) + 1
            return cache[tenor]
        for t in self._s:
            _layer(t)
        return cache

    def layer_of(self, tenor: str) -> int:
        return self._layers[tenor]

    def tenors_in_layer(self, layer: int) -> list[str]:
        tenors = [t for t, l in self._layers.items() if l == layer]
        return sorted(tenors, key=tenor_to_years)

    @property
    def all_layers(self) -> list[int]:
        return sorted(set(self._layers.values()))

    @property
    def all_tenors(self) -> list[str]:
        return list(self._s.keys())
