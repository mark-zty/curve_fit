import json
from pathlib import Path

import pandas as pd

from .tenor_graph import tenor_to_years


def load_tenor_structure(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def read_rates(path: str | Path) -> pd.DataFrame:
    """Long-form quotes as stored on disk: one row per (Date, Tenor) with a ParRate."""
    return pd.read_csv(path, parse_dates=["Date"])


def pivot_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Long quotes -> wide DataFrame: index=Date, columns=Tenor sorted by maturity."""
    wide = df.pivot(index="Date", columns="Tenor", values="ParRate")
    return wide[sorted(wide.columns, key=tenor_to_years)]


def duplicate_quotes(df: pd.DataFrame) -> list[str]:
    """(Date, Tenor) pairs quoted more than once — these break the pivot to a wide table."""
    dup = df.duplicated(subset=["Date", "Tenor"], keep=False)
    if not dup.any():
        return []
    counts = df[dup].groupby(["Date", "Tenor"]).size()
    shown = ", ".join(f"{d.date()} {t} (x{n})" for (d, t), n in list(counts.items())[:8])
    if len(counts) > 8:
        shown += f", +{len(counts) - 8} more"
    return [f"curve_data.csv has {len(counts)} duplicated (Date, Tenor) quote(s): {shown}"]


def incomplete_tenor_days(df: pd.DataFrame) -> list[str]:
    """Dates that quote only some of the tenors seen across the data"""
    tenors = sorted(df["Tenor"].unique(), key=tenor_to_years)
    have = df.groupby("Date")["Tenor"].agg(set)
    bad = have[have.map(lambda s: len(s) < len(tenors))]
    if bad.empty:
        return []

    lines = [f"curve_data.csv has {len(bad)} day(s) with incomplete tenor coverage "
             "(some tenors quoted, some missing):"]
    for date, seen in list(bad.items())[:8]:
        missing = [t for t in tenors if t not in seen]
        shown = ", ".join(missing[:6]) + (f", +{len(missing) - 6} more" if len(missing) > 6 else "")
        lines.append(f"  - {date.date()}: missing {shown}")
    if len(bad) > 8:
        lines.append(f"  - ... and {len(bad) - 8} more day(s)")
    return lines


def load_rates(path: str | Path) -> pd.DataFrame:
    """Wide rate table for a curve. Raises ValueError on duplicate (Date, Tenor) quotes,
    which would otherwise make the pivot fail with an opaque pandas error."""
    df = read_rates(path)
    dups = duplicate_quotes(df)
    if dups:
        raise ValueError(dups[0])
    return pivot_rates(df)
