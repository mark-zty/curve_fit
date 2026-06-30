from dataclasses import dataclass


@dataclass
class Param:
    """A single tunable hyperparameter a plugin chooses to expose in the dashboard."""
    name: str
    type: type = int
    default: object = None
    min: float | None = None
    max: float | None = None
    choices: list | None = None
    label: str | None = None

    def coerce(self, raw: str | None):
        if raw is None or raw == "":
            return self.default
        try:
            v = self.type(raw)
        except (ValueError, TypeError):
            return self.default
        if self.choices is not None and v not in self.choices:
            return self.default
        if self.min is not None:
            v = max(v, self.min)
        if self.max is not None:
            v = min(v, self.max)
        return v
