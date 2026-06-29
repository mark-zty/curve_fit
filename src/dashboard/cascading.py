import numpy as np
import plotly.graph_objects as go

from .base import AnalyticPanel, layer_color
from ..reducers import get_reducer
from ..reducers.base import FactorResult
from ..tenor_graph import TenorGraph

_INTRO = (
    "Each illiquid tenor's risk is attributed entirely to the first (most liquid) layer of anchor tenors. "
    "Each row is the first-layer replication of that tenor—a hedge built from liquid instruments."
)


class CascadingPanel(AnalyticPanel):
    def render(self, result: FactorResult, tenor_graph: TenorGraph, reducer: str = "", **_) -> str:
        order = result.factor_labels
        row_pos = {t: i for i, t in enumerate(result.tenors)}
        M = result.loadings[[row_pos[t] for t in order], :]

        layers = tenor_graph.all_layers
        k = len(tenor_graph.tenors_in_layer(layers[0]))
        n = len(order)

        if n == k:
            return ("<p style='max-width:760px;color:#555;line-height:1.5'>Only one "
                    "liquidity layer — every tenor is already a driver, nothing to "
                    "cascade.</p>")

        cascaded = get_reducer(reducer).cascade(M, tenor_graph)
        Z = cascaded[k:, :k]
        cols, rows = order[:k], order[k:]

        fig = go.Figure(go.Heatmap(
            z=Z,
            x=[f"<b>{t}</b>" for t in cols],
            y=rows,
            colorscale="RdBu",
            zmid=0,
            text=np.round(Z, 2),
            texttemplate="%{text}",
            hovertemplate="anchor %{x}<br>%{y}<br>%{z:.5f}<extra></extra>",
            colorbar=dict(title="Weight"),
        ))

        # Horizontal separators between the row layer groups (layers 1..last).
        row_sizes = [len(tenor_graph.tenors_in_layer(l)) for l in layers[1:]]
        for i, b in enumerate(np.cumsum(row_sizes)[:-1], start=1):
            frac = i / max(len(layers) - 1, 1)
            fig.add_shape(
                type="line",
                x0=-0.5, x1=k - 0.5, y0=b - 0.5, y1=b - 0.5,
                line=dict(color=layer_color(frac), width=2),
            )

        fig.update_layout(
            xaxis_title="<b>First-layer anchor tenor</b>",
            yaxis_title="Tenor",
            yaxis=dict(autorange="reversed"),
            height=max(300, 36 * len(rows) + 160),
        )

        explanation = f"<p style='max-width:760px;color:#555;line-height:1.5'>{_INTRO}</p>"
        return explanation + fig.to_html(full_html=False, include_plotlyjs="cdn")
