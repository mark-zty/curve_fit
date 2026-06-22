import base64
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from .base import AnalyticPanel, layer_color
from ..tenor_graph import TenorGraph


class TenorStructurePlot(AnalyticPanel):
    def render(self, tenor_graph: TenorGraph, **_) -> str:
        layers = tenor_graph.all_layers

        node_size = 2200

        pos, node_colors, nodes = {}, [], []
        for layer in layers:
            tenors = tenor_graph.tenors_in_layer(layer)
            color = layer_color(layer / max(len(layers) - 1, 1))
            for i, t in enumerate(tenors):
                pos[t] = (layer * 1.8, (-i + len(tenors) / 2) * 1.6)
                nodes.append(t)
                node_colors.append(color)

        fig, ax = plt.subplots(figsize=(14, 9))
        nx.draw_networkx_nodes(
            tenor_graph.graph, pos, nodelist=nodes, node_color="none",
            edgecolors=node_colors, linewidths=2.5, node_size=node_size, ax=ax,
        )
        nx.draw_networkx_labels(tenor_graph.graph, pos, font_size=12, ax=ax)
        nx.draw_networkx_edges(
            tenor_graph.graph, pos, edge_color="gray", arrows=True,
            arrowsize=20, node_size=node_size,
            connectionstyle="arc3,rad=0.15", ax=ax,
        )
        ax.set_axis_off()
        fig.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        encoded = base64.b64encode(buf.getvalue()).decode()
        return f'<img src="data:image/png;base64,{encoded}" alt="Tenor structure">'
