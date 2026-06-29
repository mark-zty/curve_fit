from pathlib import Path

from flask import Flask, request

from src.data_loader import (
    load_tenor_structure, read_rates, pivot_rates, duplicate_quotes, incomplete_tenor_days,
)
from src.tenor_graph import TenorGraph
from src.estimators import REGISTRY as EST_REGISTRY, get_estimator
from src.reducers import REGISTRY as RED_REGISTRY, get_reducer
from src.dashboard.factor_heatmap import FactorHeatmap
from src.dashboard.cascading import CascadingPanel
from src.dashboard.tenor_structure import TenorStructurePlot

INPUTS_DIR = Path("inputs")
TRANSFORMATIONS = ["change", "level"]
app = Flask(__name__)


def _curves() -> list[str]:
    return sorted(p.name for p in INPUTS_DIR.iterdir() if p.is_dir())


def _run(curve: str, transformation: str, estimator: str, reducer: str):
    """Returns (result, tg, errors, warnings). result is None when inputs are missing
    or invalid (errors); warnings are non-blocking and the analysis still runs."""
    base    = INPUTS_DIR / curve
    missing = [f for f in ("tenor_structure.json", "curve_data.csv") if not (base / f).exists()]
    if missing:
        return None, None, [f"Missing required file: {curve}/{f}" for f in missing], []

    structure = load_tenor_structure(base / "tenor_structure.json")
    tg        = TenorGraph(structure)
    _, errors = tg.validate()
    warnings  = []
    try:
        df = read_rates(base / "curve_data.csv")
        errors   += duplicate_quotes(df)
        warnings += incomplete_tenor_days(df)
        errors   += tg.validate_tenor_coverage(list(df["Tenor"].unique()))
    except ValueError as e:
        return None, tg, errors + [str(e)], warnings
    if errors:
        return None, tg, errors, warnings

    rates  = pivot_rates(df)
    data   = rates.diff().dropna() if transformation == "change" else rates.dropna()
    C      = get_estimator(estimator).fit(data)
    result = get_reducer(reducer).fit(C, list(rates.columns), tg)
    return result, tg, errors, warnings


def _options(items: list[str], selected: str) -> str:
    return "".join(
        f'<option value="{v}"{"selected" if v == selected else ""}>{v}</option>'
        for v in items
    )


def _section(title: str, body: str, open: bool = True) -> str:
    return f"""<details class="section"{" open" if open else ""}>
    <summary>{title}</summary>
    {body}
  </details>"""


def _error_panel(curve: str, errors: list[str]) -> str:
    items = "".join(f"<li>{e}</li>" for e in errors)
    return f"""<div class="error">
    <h3>✗ Invalid input for {curve}</h3>
    <ul>{items}</ul>
  </div>"""


def _warning_panel(curve: str, warnings: list[str]) -> str:
    body = "<br>".join(w for w in warnings)
    return f"""<div class="warning">
    <h3>⚠ Warnings for {curve}</h3>
    <pre>{body}</pre>
  </div>"""


@app.route("/")
def index():
    curves     = _curves()
    estimators = list(EST_REGISTRY)
    reducers   = list(RED_REGISTRY)

    curve          = request.args.get("curve",          curves[0])
    transformation = request.args.get("transformation", TRANSFORMATIONS[0])
    estimator      = request.args.get("estimator",      estimators[0])
    reducer        = request.args.get("reducer",        reducers[0])

    result, tg, errors, warnings = _run(curve, transformation, estimator, reducer)

    if result is None:
        content = _error_panel(curve, errors)
    else:
        warning_panel = _warning_panel(curve, warnings) if warnings else ""
        content = f"""{warning_panel}
  {_section("Tenor Liquidity Structure", TenorStructurePlot().render(tenor_graph=tg), open=False)}
  {_section("Loading Matrix", FactorHeatmap().render(result=result, tenor_graph=tg, reducer=reducer))}
  {_section("Cascading Matrix", CascadingPanel().render(result=result, tenor_graph=tg, reducer=reducer))}"""

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Risk Dashboard</title>
  <style>
    body {{ font-family: sans-serif; padding: 20px; }}
    details.section {{ margin-bottom: 40px; }}
    details.section > summary {{
      list-style: none; cursor: pointer; user-select: none;
      font-size: 1.5em; font-weight: bold; display: flex; align-items: center; gap: 10px;
    }}
    details.section > summary::-webkit-details-marker {{ display: none; }}
    details.section > summary::before {{
      content: "⌃"; display: inline-block; transition: transform 0.2s; transform: rotate(180deg);
      font-size: 0.8em; color: #888;
    }}
    details.section[open] > summary::before {{ transform: rotate(0deg); }}
    .controls {{ display: flex; gap: 24px; margin-bottom: 30px; align-items: flex-end; }}
    .controls label {{ display: flex; flex-direction: column; gap: 4px; font-size: 0.85em; color: #555; }}
    select {{ padding: 6px 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 1em; }}
    .error {{ background: #fdecea; border: 1px solid #f5c2c7; border-radius: 6px; padding: 16px 20px; color: #842029; }}
    .warning {{ background: #fff3cd; border: 1px solid #ffe69c; border-radius: 6px; padding: 16px 20px; color: #664d03; margin-bottom: 24px; }}
    .warning pre {{ margin: 0; font-family: inherit; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>Risk Dashboard</h1>
  <form method="GET" class="controls">
    <label>Curve
      <select name="curve" onchange="this.form.submit()">{_options(curves, curve)}</select>
    </label>
    <label>Transformation
      <select name="transformation" onchange="this.form.submit()">{_options(TRANSFORMATIONS, transformation)}</select>
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
