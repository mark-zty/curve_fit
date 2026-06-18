from pathlib import Path

from src.data_loader import load_tenor_structure, load_rates
from src.tenor_graph import TenorGraph
from src.estimators import get_estimator
from src.reducers import get_reducer
from src.dashboard import build_dashboard
from src.dashboard.factor_heatmap import FactorHeatmap

# --- config (swap strings to use a different method) ---
ESTIMATOR = "naive"
REDUCER   = "sequential_pca"
OUTPUT    = Path("dashboard.html")


def main():
    structure = load_tenor_structure("Inputs/tenor_structure.json")
    rates     = load_rates("Inputs/curve_data.csv")
    tg        = TenorGraph(structure)

    C      = get_estimator(ESTIMATOR).fit(rates)
    result = get_reducer(REDUCER).fit(C, list(rates.columns), tg)

    panels = [
        ("Factor Loadings", FactorHeatmap()),
    ]
    OUTPUT.write_text(build_dashboard(panels, result=result))
    print(f"Dashboard written to {OUTPUT}")


if __name__ == "__main__":
    main()
