import re

import networkx as nx


def tenor_to_years(tenor: str) -> float:
    m = re.fullmatch(r'(\d+)(M|Y)', tenor)
    if not m:
        raise ValueError(f"Unrecognized tenor format: {tenor!r}")
    n, unit = int(m.group(1)), m.group(2)
    return n / 12 if unit == 'M' else float(n)


class TenorGraph:
    def __init__(self, structure: dict):
        self._s = structure
        self.graph = self._build_graph()
        self._layers: dict[str, int] | None = None

    def _build_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(self._s)
        for tenor, info in self._s.items():
            for p in info["predictors"]:
                graph.add_edge(p, tenor)
        return graph

    def _get_missing_predictors(self) -> list[str]:
        return [
            f"{tenor!r} references missing predictor {p!r}"
            for tenor, info in self._s.items()
            for p in info["predictors"]
            if p not in self._s
        ]

    def validate(self) -> tuple[bool, list[str]]:
        errors = []

        missing = self._get_missing_predictors()
        if missing:
            errors.append("Some predictors reference non-existent nodes:")
            errors.extend(f"  - {m}" for m in missing)

        if not nx.is_directed_acyclic_graph(self.graph):
            errors.append("Graph contains cycles - not a valid DAG")
            try:
                cycle = nx.find_cycle(self.graph)
                cycle_str = ' -> '.join([u for u, _ in cycle] + [cycle[0][0]])
                errors.append(f"  Cycle detected: {cycle_str}")
            except nx.NetworkXNoCycle:
                pass

        if not nx.is_weakly_connected(self.graph):
            errors.append("Graph is not weakly connected")
            components = list(nx.weakly_connected_components(self.graph))
            errors.append(f"  Found {len(components)} disconnected components:")
            for i, comp in enumerate(components, 1):
                errors.append(f"    Component {i}: {sorted(comp)}")

        self_loops = list(nx.nodes_with_selfloops(self.graph))
        if self_loops:
            errors.append(f"Self-loops detected: {self_loops}")

        return not errors, errors

    def validate_tenor_coverage(self, data_tenors: list[str]) -> list[str]:
        structure_tenors = set(self.all_tenors)
        data_tenors = set(data_tenors)
        errors = []

        missing = sorted(data_tenors - structure_tenors)
        if missing:
            errors.append(f"Tenor structure is missing tenor(s) present in data: {missing}")

        redundant = sorted(structure_tenors - data_tenors)
        if redundant:
            errors.append(f"Tenor structure has redundant tenor(s) not present in data: {redundant}")

        return errors

    def visualize(self) -> str:
        lines = []
        for layer in self.all_layers:
            lines.append(f"Layer {layer}:")
            for t in self.tenors_in_layer(layer):
                preds = self._s[t]["predictors"]
                suffix = f" <- {', '.join(preds)}" if preds else ""
                lines.append(f"  {t}{suffix}")
        return "\n".join(lines)

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
        if self._layers is None:
            self._layers = self._infer_layers()
        return self._layers[tenor]

    def predictors_of(self, tenor: str) -> list[str]:
        return list(self._s[tenor]["predictors"])

    def tenors_in_layer(self, layer: int) -> list[str]:
        if self._layers is None:
            self._layers = self._infer_layers()
        tenors = [t for t, l in self._layers.items() if l == layer]
        return sorted(tenors, key=tenor_to_years)

    @property
    def all_layers(self) -> list[int]:
        if self._layers is None:
            self._layers = self._infer_layers()
        return sorted(set(self._layers.values()))

    @property
    def all_tenors(self) -> list[str]:
        return list(self._s.keys())
