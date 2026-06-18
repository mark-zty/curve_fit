TENOR_YEARS: dict[str, float] = {
    "3M": 0.25, "6M": 0.5, "9M": 0.75, "1Y": 1.0, "18M": 1.5,
    "2Y": 2.0,  "3Y": 3.0, "4Y": 4.0,  "5Y": 5.0, "7Y": 7.0,
    "10Y": 10.0, "15Y": 15.0, "20Y": 20.0, "25Y": 25.0, "30Y": 30.0,
}


class TenorGraph:
    def __init__(self, structure: dict):
        self._s = structure

    def layer_of(self, tenor: str) -> int:
        return self._s[tenor]["layer"]

    def tenors_in_layer(self, layer: int) -> list[str]:
        tenors = [t for t, v in self._s.items() if v["layer"] == layer]
        return sorted(tenors, key=lambda t: TENOR_YEARS[t])

    @property
    def all_layers(self) -> list[int]:
        return sorted({v["layer"] for v in self._s.values()})

    @property
    def all_tenors(self) -> list[str]:
        return list(self._s.keys())
