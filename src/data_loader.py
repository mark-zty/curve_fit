import json
from pathlib import Path

import pandas as pd

from .tenor_graph import TENOR_YEARS


def load_tenor_structure(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_rates(path: str | Path) -> pd.DataFrame:
    """Wide DataFrame: index=Date, columns=Tenor sorted by maturity, values=ParRate."""
    df = pd.read_csv(path, parse_dates=["Date"])
    wide = df.pivot(index="Date", columns="Tenor", values="ParRate")
    return wide[sorted(wide.columns, key=lambda t: TENOR_YEARS[t])]
