from abc import ABC, abstractmethod


class AnalyticPanel(ABC):
    @abstractmethod
    def render(self, **kwargs) -> str:
        """Return an HTML <div> string for this panel."""
