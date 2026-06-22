import json
from pathlib import Path

import pandas as pd

from .tenor_graph import tenor_to_years


def load_tenor_structure(path: str | Path) -> dict:
    with open(path) as f:
        structure = json.load(f)
    for info in structure.values():
        preds = info["predictors"]
        if preds and info.get("weights") is None:
            info["weights"] = [1 / len(preds)] * len(preds)
    return structure


def load_rates(path: str | Path) -> pd.DataFrame:
    """Wide DataFrame: index=Date, columns=Tenor sorted by maturity, values=ParRate."""
    df = pd.read_csv(path, parse_dates=["Date"])
    wide = df.pivot(index="Date", columns="Tenor", values="ParRate")
    return wide[sorted(wide.columns, key=tenor_to_years)]
