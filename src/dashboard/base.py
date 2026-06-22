from abc import ABC, abstractmethod

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# Layers are shaded dark-to-light green via the Greens colormap, layer 0 darkest.
_CMAP_DARK = 0.85
_CMAP_LIGHT = 0.30
_cmap = plt.get_cmap("Greens")


def layer_color(frac: float) -> str:
    return mcolors.rgb2hex(_cmap(_CMAP_DARK - frac * (_CMAP_DARK - _CMAP_LIGHT)))


class AnalyticPanel(ABC):
    @abstractmethod
    def render(self, **kwargs) -> str:
        """Return an HTML <div> string for this panel."""
