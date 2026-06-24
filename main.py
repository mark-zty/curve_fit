from pathlib import Path

from flask import Flask, request

from src.data_loader import load_tenor_structure, load_rates
from src.tenor_graph import TenorGraph
from src.estimators import REGISTRY as EST_REGISTRY, get_estimator
from src.reducers import REGISTRY as RED_REGISTRY, get_reducer
from src.dashboard.factor_heatmap import FactorHeatmap
from src.dashboard.cascading import CascadingPanel
from src.dashboard.tenor_structure import TenorStructurePlot

INPUTS_DIR = Path("inputs")
SERIES = ["Change", "Level"]
app = Flask(__name__)


def _curves() -> list[str]:
    return sorted(p.name for p in INPUTS_DIR.iterdir() if p.is_dir())


def _run(curve: str, series: str, estimator: str, reducer: str):
    """Returns (result, tg, errors). result is None when the tenor structure or its data coverage is invalid."""
    base      = INPUTS_DIR / curve
    structure = load_tenor_structure(base / "tenor_structure.json")
    tg        = TenorGraph(structure)
    rates     = load_rates(base / "curve_data.csv")

    _, errors = tg.validate()
    errors += tg.validate_tenor_coverage(list(rates.columns))
    if errors:
        return None, tg, errors

    data   = rates.diff().dropna() if series == "Change" else rates
    C      = get_estimator(estimator).fit(data)
    result = get_reducer(reducer).fit(C, list(rates.columns), tg)
    return result, tg, errors


def _options(items: list[str], selected: str) -> str:
    return "".join(
        f'<option value="{v}"{"selected" if v == selected else ""}>{v}</option>'
        for v in items
    )


def _error_panel(curve: str, errors: list[str]) -> str:
    items = "".join(f"<li>{e}</li>" for e in errors)
    return f"""<div class="error">
    <h3>✗ Invalid tenor structure for {curve}</h3>
    <ul>{items}</ul>
  </div>"""


@app.route("/")
def index():
    curves     = _curves()
    estimators = list(EST_REGISTRY)
    reducers   = list(RED_REGISTRY)

    curve     = request.args.get("curve",     curves[0])
    series    = request.args.get("series",    SERIES[0])
    estimator = request.args.get("estimator", estimators[0])
    reducer   = request.args.get("reducer",   reducers[0])

    result, tg, errors = _run(curve, series, estimator, reducer)

    if result is None:
        content = _error_panel(curve, errors)
    else:
        content = f"""<section><h2>Tenor Liquidity Structure</h2>{TenorStructurePlot().render(tenor_graph=tg)}</section>
  <section><h2>Loading Matrix</h2>{FactorHeatmap().render(result=result, tenor_graph=tg, reducer=reducer)}</section>
  <section><h2>Cascading</h2>{CascadingPanel().render(result=result, tenor_graph=tg, reducer=reducer)}</section>"""

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Risk Dashboard</title>
  <style>
    body {{ font-family: sans-serif; padding: 20px; }}
    .controls {{ display: flex; gap: 24px; margin-bottom: 30px; align-items: flex-end; }}
    .controls label {{ display: flex; flex-direction: column; gap: 4px; font-size: 0.85em; color: #555; }}
    select {{ padding: 6px 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 1em; }}
    .error {{ background: #fdecea; border: 1px solid #f5c2c7; border-radius: 6px; padding: 16px 20px; color: #842029; }}
  </style>
</head>
<body>
  <h1>Risk Dashboard</h1>
  <form method="GET" class="controls">
    <label>Curve
      <select name="curve" onchange="this.form.submit()">{_options(curves, curve)}</select>
    </label>
    <label>Series
      <select name="series" onchange="this.form.submit()">{_options(SERIES, series)}</select>
    </label>
    <label>Estimator
      <select name="estimator" onchange="this.form.submit()">{_options(estimators, estimator)}</select>
    </label>
    <label>Reducer
      <select name="reducer" onchange="this.form.submit()">{_options(reducers, reducer)}</select>
    </label>
  </form>
  {content}
</body>
</html>"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, debug=True)
