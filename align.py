"""
bioml.align — put batches of unequal length onto a common axis.

WHY THIS IS NEEDED (Week 6, mostly)
-----------------------------------
IndPenSim batches run 167-290 h: a 74% spread. Multivariate SPC, and any method
that wants a batch × time × variable array, needs the batches to be
commensurable. They are not, natively.

WHY THIS FUNCTION TAKES ARGUMENTS INSTEAD OF DOING THE OBVIOUS THING
---------------------------------------------------------------------
There is no obvious thing. The three methods below answer three different
questions and they disagree about what "the same moment in two batches" means:

    "clock"     the same elapsed hour. Hour 100 is hour 100. Assumes batches
                progress at the same rate, which is precisely what differs.
    "fraction"  the same fraction of the batch's own duration. Hour 100 of a
                200 h batch matches hour 145 of a 290 h batch. Requires knowing
                the duration in advance, so it CANNOT be used on a running
                batch — which makes it useless for live fault detection even
                though it looks best offline.
    "trim"      cut every batch to the shortest one. Throws away late-phase
                data, which for a penicillin fermentation is where the product
                is. Honest, crude, and sometimes right.

Deliberately absent: dynamic time warping. It is the standard answer in the
batch-SPC literature and it is a genuine modelling decision with its own
failure modes, so it belongs in a student's Week 6 justification rather than
behind a default here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .columns import resolve

VALID_METHODS = ("clock", "fraction", "trim")


def align_batches(
    df: pd.DataFrame,
    method: str,                       # no default — see module docstring
    reference: int | None = None,
    n_points: int | None = None,
    columns: list[str] | None = None,
    time_col: str | None = None,
    batch_col: str | None = None,
) -> pd.DataFrame:
    """
    Return a long frame in which every batch is sampled at the same set of
    alignment points.

    method    : "clock" | "fraction" | "trim". Required.
    reference : batch number whose time grid defines the target grid. Required
                for "clock" and "fraction"; ignored by "trim". Your choice of
                reference batch is a modelling choice and belongs in the report.
    n_points  : number of alignment points. Defaults to the reference batch's
                own number of samples ("clock", "fraction") or the shortest
                batch's ("trim").
    columns   : which columns to interpolate. Defaults to every numeric column
                except the identifiers.

    Interpolation is linear, in time, within each batch, with no extrapolation:
    alignment points outside a batch's own time span come back as NaN rather
    than as an invented value.
    """
    if method not in VALID_METHODS:
        raise ValueError(f"method={method!r} not in {VALID_METHODS}")

    time_col = time_col or resolve(df, "time")
    batch_col = batch_col or resolve(df, "batch")

    if columns is None:
        columns = [c for c in df.select_dtypes(include=[np.number]).columns
                   if c not in (time_col, batch_col)]

    durations = df.groupby(batch_col)[time_col].max()

    # --- build the target axis ------------------------------------------------
    if method == "trim":
        shortest = float(durations.min())
        n = n_points or int(df.groupby(batch_col).size().min())
        targets = np.linspace(float(df[time_col].min()), shortest, n)
        as_fraction = False
    else:
        if reference is None:
            raise ValueError(
                f'method="{method}" needs reference=<batch number>. Which batch '
                f'defines "normal progress" is a decision, not a detail.'
            )
        ref = df.loc[df[batch_col] == reference, time_col]
        if ref.empty:
            raise ValueError(f"reference batch {reference} not present in df")
        n = n_points or len(ref)
        targets = np.linspace(float(ref.min()), float(ref.max()), n)
        as_fraction = (method == "fraction")
        if as_fraction:
            # Express the reference grid as a fraction of the reference duration,
            # then map that same fraction onto each other batch's own duration.
            targets = (targets - targets.min()) / (targets.max() - targets.min())

    # --- resample every batch onto it ----------------------------------------
    out = []
    for b, sub in df.groupby(batch_col, sort=True):
        sub = sub.sort_values(time_col)
        t = sub[time_col].to_numpy(dtype=float)
        # For "fraction", the batch's own clock times corresponding to the
        # target fractions; for the others, the target times directly.
        t_query = targets * float(durations.loc[b]) if as_fraction else targets

        row = {batch_col: b,
               "align_point": np.arange(len(t_query)),
               time_col: t_query}
        for c in columns:
            v = sub[c].to_numpy(dtype=float)
            ok = ~np.isnan(v)                       # interpolate on observed points only
            if ok.sum() < 2:
                row[c] = np.full(len(t_query), np.nan)
                continue
            interp = np.interp(t_query, t[ok], v[ok], left=np.nan, right=np.nan)
            row[c] = interp
        out.append(pd.DataFrame(row))

    return pd.concat(out, ignore_index=True)
