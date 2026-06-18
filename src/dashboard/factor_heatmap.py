import numpy as np
import plotly.graph_objects as go

from .base import AnalyticPanel
from ..reducers.base import FactorResult


class FactorHeatmap(AnalyticPanel):
    def render(self, result: FactorResult, **_) -> str:
        L = result.loadings
        fig = go.Figure(go.Heatmap(
            z=L,
            x=result.factor_labels,
            y=result.tenors,
            colorscale="RdBu",
            zmid=0,
            text=np.round(L, 2),
            texttemplate="%{text}",
            colorbar=dict(title="Loading"),
        ))
        fig.update_layout(
            title="Factor loadings by tenor",
            xaxis_title="Factor (anchor tenor)",
            yaxis_title="Tenor",
            yaxis=dict(autorange="reversed"),
            height=600,
        )
        return fig.to_html(full_html=False, include_plotlyjs="cdn")
