"""
bioml.splits — train/test splitting and cross-validation for batch data.

DESIGN NOTE — READ THIS BEFORE USING EITHER FUNCTION
----------------------------------------------------
`how` has no default value, in either function. That is deliberate and it is
the single most important design decision in this package.

A default would be a recommendation, and a recommendation would make the Week 2
leakage lab unfalsifiable: students would "discover" the right answer by
accepting the default rather than by observing what the wrong one does. So
`how="random"` and `how="batchwise"` are exactly one line each, neither is
easier to type than the other, and neither is blessed. You choose, and you
defend the choice.

WHAT EACH STRATEGY DOES
-----------------------
random    : rows are shuffled and dealt out. Rows from one batch land in both
            train and test.
batchwise : whole batches go to one side or the other. No batch appears in
            both.
regime    : whole regimes are held out (e.g. train on recipe-driven batches,
            test on operator-controlled ones). Tests transfer, not
            interpolation.

Which of these is right depends on the question the model is supposed to
answer, and that question is not stated in the data.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut

from .columns import resolve

VALID_HOW = ("random", "batchwise", "regime")


def _check_how(how: str) -> str:
    """Reject anything that is not one of the three named strategies."""
    if how not in VALID_HOW:
        raise ValueError(
            f"how={how!r} is not a splitting strategy. Choose one of {VALID_HOW} "
            f"and be ready to say why."
        )
    return how


def split(
    df: pd.DataFrame,
    how: str,                      # no default, on purpose — see module docstring
    test_size: float = 0.3,
    seed: int = 0,
    group_col: str | None = None,
    regime_col: str | None = None,
    test_regimes: Sequence | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split `df` into (train, test).

    df          : a frame with one row per observation, including a batch column.
    how         : "random" | "batchwise" | "regime". Required.
    test_size   : fraction of ROWS (how="random") or of BATCHES (how="batchwise").
                  Ignored for how="regime", where the held-out set is named.
    seed        : random seed. Same seed gives the same split every run, so a
                  number you report is reproducible by whoever reads your report.
    group_col   : batch column name. Resolved automatically if None.
    regime_col  : required for how="regime" — the column holding the regime label
                  (e.g. Control_ref or PAT_ref).
    test_regimes: required for how="regime" — the regime value(s) to hold out.

    Returns two frames with the original columns and a fresh RangeIndex.
    """
    _check_how(how)
    rng = np.random.default_rng(seed)

    if how == "random":
        # Deal rows out at random, ignoring which batch each row came from.
        # Consecutive rows within a batch are 0.2 h apart and are near-copies
        # of each other, so this puts near-copies on both sides of the split.
        n_test = int(round(len(df) * test_size))
        perm = rng.permutation(len(df))
        test_idx, train_idx = perm[:n_test], perm[n_test:]
        return (df.iloc[np.sort(train_idx)].reset_index(drop=True),
                df.iloc[np.sort(test_idx)].reset_index(drop=True))

    group_col = group_col or resolve(df, "batch")

    if how == "batchwise":
        # Whole batches move together. The unit of randomisation is the batch,
        # because the batch is the unit that was independently run.
        batches = np.sort(df[group_col].unique())
        n_test = max(1, int(round(len(batches) * test_size)))
        test_batches = set(rng.permutation(batches)[:n_test].tolist())
        mask = df[group_col].isin(test_batches)
        return (df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True))

    # how == "regime"
    if regime_col is None or test_regimes is None:
        raise ValueError(
            'how="regime" needs regime_col= and test_regimes=, e.g. '
            'regime_col=ctrl, test_regimes=[1]. Naming the held-out regime is '
            'the point of this strategy; there is no sensible default.'
        )
    mask = df[regime_col].isin(list(test_regimes))
    if not mask.any():
        raise ValueError(f"no rows have {regime_col} in {list(test_regimes)}")
    if mask.all():
        raise ValueError(f"every row has {regime_col} in {list(test_regimes)} — "
                         f"that leaves no training data")
    return (df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True))


def cv_score(
    model,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    groups: pd.Series | np.ndarray | None,
    how: str,                      # no default, same reason as split()
    n_splits: int = 5,
    scoring=r2_score,
) -> np.ndarray:
    """
    Cross-validated scores, one per fold.

    model   : any unfitted sklearn-style estimator. It is cloned per fold, so
              the object you pass in is never fitted and cannot leak state
              between folds.
    X, y    : feature matrix and target, already aligned and free of NaN.
    groups  : the batch number for each row. Required for "batchwise" and
              "regime"; ignored for "random".
    how     : "random" | "batchwise" | "regime". Required.
    n_splits: folds. Ignored for "regime", which does leave-one-regime-out.
    scoring : callable(y_true, y_pred) -> float. Defaults to R².

    Returns an array of fold scores. Report the mean AND the spread: with
    n ≈ 100 batches the fold-to-fold variation is usually larger than the
    difference between two models, and a mean alone hides that.
    """
    _check_how(how)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if how == "random":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        folds = splitter.split(X)
    else:
        if groups is None:
            raise ValueError(f'how="{how}" needs groups= (the batch number per row)')
        groups = np.asarray(groups)
        if how == "batchwise":
            # GroupKFold keeps every batch entirely within one fold.
            n = min(n_splits, len(np.unique(groups)))
            folds = GroupKFold(n_splits=n).split(X, y, groups)
        else:  # regime: groups holds the regime label, not the batch number
            folds = LeaveOneGroupOut().split(X, y, groups)

    scores = []
    for train_idx, test_idx in folds:
        est = clone(model)                      # fresh, unfitted copy each fold
        est.fit(X[train_idx], y[train_idx])
        scores.append(scoring(y[test_idx], est.predict(X[test_idx])))
    return np.array(scores)
