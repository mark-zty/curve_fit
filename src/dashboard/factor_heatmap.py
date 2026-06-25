import numpy as np
import plotly.graph_objects as go

from .base import AnalyticPanel, layer_color
from ..reducers.base import FactorResult
from ..tenor_graph import TenorGraph

_INTRO = (
    "Each row is a tenor and each column an anchor tenor."
)
_DESCRIPTIONS = {
    "sequential_pca": (
        "Uses the tenor liquidity <b>layers</b>."
    ),
    "sequential_ols": (
        "Uses the tenor liquidity <b>layers</b>."
    ),
    "local_pca": (
        "Uses the tenor liquidity <b>graph</b>."
    ),
    "local_ols": (
        "Uses the tenor liquidity <b>graph</b>."
    ),
}


class FactorHeatmap(AnalyticPanel):
    def render(self, result: FactorResult, tenor_graph: TenorGraph, reducer: str = "", **_) -> str:
        # factor_labels are already ordered layer-then-maturity; reorder rows to match.
        order = result.factor_labels
        row_pos = {t: i for i, t in enumerate(result.tenors)}
        L = result.loadings[[row_pos[t] for t in order], :]

        layer_sizes = [len(tenor_graph.tenors_in_layer(l)) for l in tenor_graph.all_layers]
        bounds = np.cumsum([0] + layer_sizes)

        fig = go.Figure(go.Heatmap(
            z=L,
            x=[f"<b>{t}</b>" for t in order],
            y=order,
            colorscale="RdBu",
            zmid=0,
            text=np.round(L, 2),
            texttemplate="%{text}",
            hovertemplate="%{x}<br>%{y}<br>%{z:.5f}<extra></extra>",
            colorbar=dict(title="Loading"),
        ))
        for i, n in enumerate(layer_sizes):
            start, end = bounds[i] - 0.5, bounds[i + 1] - 0.5
            frac = i / max(len(layer_sizes) - 1, 1)
            fig.add_shape(
                type="rect",
                x0=start, x1=end, y0=start, y1=end,
                line=dict(color=layer_color(frac), width=2),
                fillcolor="rgba(0,0,0,0)",
            )
        xtitle = "Anchor Tenor"
        fig.update_layout(
            xaxis_title=f"<b>{xtitle}</b>",
            yaxis_title="Tenor",
            yaxis=dict(autorange="reversed"),
            height=600,
        )
        paras = [_INTRO, _DESCRIPTIONS.get(reducer, "")]
        explanation = "".join(
            f"<p style='max-width:760px;color:#555;line-height:1.5'>{p}</p>"
            for p in paras if p
        )
        return explanation + fig.to_html(full_html=False, include_plotlyjs="cdn")
