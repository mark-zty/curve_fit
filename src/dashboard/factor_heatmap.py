import colorsys

import numpy as np
import plotly.graph_objects as go

from .base import AnalyticPanel
from ..reducers.base import FactorResult
from ..tenor_graph import TenorGraph

# Diagonal layer blocks are outlined dark-to-light purple (same hue), layer 0 darkest.
_PURPLE_HUE = 280 / 360
_LIGHTNESS_DARK = 0.20
_LIGHTNESS_LIGHT = 0.80


def _purple_shade(frac: float) -> str:
    lightness = _LIGHTNESS_DARK + frac * (_LIGHTNESS_LIGHT - _LIGHTNESS_DARK)
    r, g, b = colorsys.hls_to_rgb(_PURPLE_HUE, lightness, 1.0)
    return f"rgb({round(r * 255)},{round(g * 255)},{round(b * 255)})"


class FactorHeatmap(AnalyticPanel):
    def render(self, result: FactorResult, tenor_graph: TenorGraph, **_) -> str:
        # factor_labels are already ordered layer-then-maturity; reorder rows to match.
        order = result.factor_labels
        row_pos = {t: i for i, t in enumerate(result.tenors)}
        L = result.loadings[[row_pos[t] for t in order], :]

        layer_sizes = [len(tenor_graph.tenors_in_layer(l)) for l in tenor_graph.all_layers]
        bounds = np.cumsum([0] + layer_sizes)

        fig = go.Figure(go.Heatmap(
            z=L,
            x=order,
            y=order,
            colorscale="RdBu",
            zmid=0,
            text=np.round(L, 2),
            texttemplate="%{text}",
            colorbar=dict(title="Loading"),
        ))
        for i, n in enumerate(layer_sizes):
            start, end = bounds[i] - 0.5, bounds[i + 1] - 0.5
            frac = i / max(len(layer_sizes) - 1, 1)
            fig.add_shape(
                type="rect",
                x0=start, x1=end, y0=start, y1=end,
                line=dict(color=_purple_shade(frac), width=2),
                fillcolor="rgba(0,0,0,0)",
            )
        fig.update_layout(
            title="Factor loadings by tenor",
            xaxis_title="Factor (anchor tenor)",
            yaxis_title="Tenor",
            yaxis=dict(autorange="reversed"),
            height=600,
        )
        return fig.to_html(full_html=False, include_plotlyjs="cdn")
