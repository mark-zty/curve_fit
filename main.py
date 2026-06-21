from pathlib import Path

from flask import Flask, request

from src.data_loader import load_tenor_structure, load_rates
from src.tenor_graph import TenorGraph
from src.estimators import REGISTRY as EST_REGISTRY, get_estimator
from src.reducers import REGISTRY as RED_REGISTRY, get_reducer
from src.dashboard.factor_heatmap import FactorHeatmap

INPUTS_DIR = Path("inputs")
app = Flask(__name__)


def _curves() -> list[str]:
    return sorted(p.name for p in INPUTS_DIR.iterdir() if p.is_dir())


def _run(curve: str, estimator: str, reducer: str):
    base = INPUTS_DIR / curve
    structure = load_tenor_structure(base / "tenor_structure.json")
    rates     = load_rates(base / "curve_data.csv")
    tg        = TenorGraph(structure)
    C         = get_estimator(estimator).fit(rates)
    result    = get_reducer(reducer).fit(C, list(rates.columns), tg)
    return result, tg


def _options(items: list[str], selected: str) -> str:
    return "".join(
        f'<option value="{v}"{"selected" if v == selected else ""}>{v}</option>'
        for v in items
    )


def _validate_curve(curve: str) -> None:
    structure = load_tenor_structure(INPUTS_DIR / curve / "tenor_structure.json")
    tg = TenorGraph(structure)
    is_valid, errors = tg.validate()
    if is_valid:
        print(f"✓ {curve}: structure is valid")
        print(tg.visualize())
        tg.plot(save_path=INPUTS_DIR / curve / "tenor_structure.png")
    else:
        print(f"X {curve}: errors found")
        for error in errors:
            print(f"  - {error}")


@app.route("/")
def index():
    curves     = _curves()
    estimators = list(EST_REGISTRY)
    reducers   = list(RED_REGISTRY)

    curve     = request.args.get("curve",     curves[0])
    estimator = request.args.get("estimator", estimators[0])
    reducer   = request.args.get("reducer",   reducers[0])

    result, tg = _run(curve, estimator, reducer)
    content    = FactorHeatmap().render(result=result, tenor_graph=tg)

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
  </style>
</head>
<body>
  <h1>Risk Dashboard</h1>
  <form method="GET" class="controls">
    <label>Curve
      <select name="curve" onchange="this.form.submit()">{_options(curves, curve)}</select>
    </label>
    <label>Estimator
      <select name="estimator" onchange="this.form.submit()">{_options(estimators, estimator)}</select>
    </label>
    <label>Reducer
      <select name="reducer" onchange="this.form.submit()">{_options(reducers, reducer)}</select>
    </label>
  </form>
  <section><h2>Factor Loadings</h2>{content}</section>
</body>
</html>"""


if __name__ == "__main__":
    for curve in _curves():
        _validate_curve(curve)
    app.run(debug=True)
