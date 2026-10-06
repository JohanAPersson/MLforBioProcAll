# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Week 2 — Validation and the leakage lab (REFERENCE)
#
# **Instructor reference. Correct, verified, not distributed.**
#
# Unlike Week 1, this notebook carries **no planted bug**. It does not need one:
# the wrong answer is available as a one-line argument change, and the student
# is asked to run both and explain the difference. The patch that generates the
# student copy removes only the `REFERENCE-ONLY` verification cell at the end.
#
# ## The claim being tested
#
# > A model of penicillin titer, validated by splitting rows at random, reports
# > an R² that has nothing to do with how it will perform on the next batch.
#
# You are not told this. You produce both numbers and account for the gap.
#
# ## The single decision of the week
#
# `bioml.split(df, how=...)` has no default. `"random"` and `"batchwise"` are
# each one word. Nothing in the API prefers either. Choosing, and being able to
# say why, is the whole deliverable.

# %%
# --- Setup -------------------------------------------------------------------
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import bioml

DATA_DIR = Path("fixture")
bioml.set_data_dir(DATA_DIR)
print(bioml.about())

manifest = bioml.load_manifest()
if manifest.get("synthetic"):
    print("\n*** SYNTHETIC FIXTURE — code verification only, not IndPenSim ***\n")

WEEK2_BATCHES = range(1, 31)          # same 30 recipe-driven batches as Week 1
proc = bioml.load_process_vars(batches=WEEK2_BATCHES)
roles = bioml.load_column_roles()

col = bioml.resolve_many(proc, ["time", "batch", "p_offline"])

# %% [markdown]
# ## 1. Build the feature matrix
#
# Two decisions are made here, and both are stated rather than buried.
#
# **Which rows.** The target is an offline assay observed on roughly 2% of rows.
# A row without an assay has no target, so the modelling rows are the assay
# rows. There are about 20 per batch — roughly 600 in total, from 30 batches.
#
# **Which columns.** Only `online_measurable`. That excludes `simulator_state`
# (`P:g/L` is the answer; `S:g/L` is not measured at this scale), the other
# offline assays, the design metadata, the identifiers, and anything with zero
# variance. What is left is what a plant would actually hand you.

# %%
# Start from the role table, not from a hand-typed list. A hand-typed list is
# something you have to keep in sync with the data; this is not.
online = roles.loc[roles.role == "online_measurable", "column"].tolist()
feature_cols = [c for c in online if c in proc.columns]

# Drop anything that does not vary within this subset — a constant column
# contributes nothing and makes StandardScaler divide by zero.
nunique = proc[feature_cols].nunique(dropna=True)
constant = nunique[nunique <= 1].index.tolist()
feature_cols = [c for c in feature_cols if c not in constant]

# Elapsed process time: included on purpose. A plant soft sensor does know how
# long the batch has been running, so excluding it would model a sensor nobody
# operates. It is a decision, and it belongs in your report as one.
feature_cols = [col["time"]] + feature_cols

# Keep only the rows where the target was actually measured.
model_df = proc.loc[proc[col["p_offline"]].notna(),
                    [col["batch"]] + feature_cols + [col["p_offline"]]].dropna()
model_df = model_df.reset_index(drop=True)

X_cols = feature_cols
y_col = col["p_offline"]

print(f"{len(model_df)} modelling rows from {model_df[col['batch']].nunique()} batches")
print(f"{len(X_cols)} features, target = {y_col}")
print(f"\n{len(model_df) / model_df[col['batch']].nunique():.0f} rows per batch "
      f"— but the number of INDEPENDENT experiments is "
      f"{model_df[col['batch']].nunique()}, not {len(model_df)}.")

# Assert rather than trust: if a simulator_state column ever reaches the feature
# list, every number below becomes meaningless and nothing else would catch it.
forbidden = roles.loc[roles.role.isin(["simulator_state", "design_metadata"]),
                      "column"].tolist()
leaked = [c for c in X_cols if c in forbidden]
assert not leaked, f"forbidden columns in the feature matrix: {leaked}"
print("\nno simulator_state or design_metadata column reached the feature matrix")

# %% [markdown]
# ## 2. One model, two validation protocols
#
# The model is held fixed so that the only thing that changes between the two
# numbers is how the data was divided.

# %%
def make_model():
    """
    Standardise, then ridge regression.

    StandardScaler: the columns are in wildly different units (kelvin, litres
    per hour, grams per minute), and a penalised regression that is not scaled
    penalises the large-unit columns less. Fitted inside the pipeline so it is
    refitted on each training set and never sees the test set.

    Ridge rather than ordinary least squares: the process variables are strongly
    collinear (every feed, and volume, and time, all rise together), which makes
    an unpenalised fit numerically unstable and its coefficients unreadable.
    alpha is fixed here so that Week 2 varies exactly one thing; tuning it is a
    Week 4 problem and needs a nested loop to do honestly.
    """
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


results = {}
for how in ["random", "batchwise"]:
    # The ONLY line that differs between the two protocols.
    train, test = bioml.split(model_df, how=how, test_size=0.3, seed=0)

    model = make_model()
    model.fit(train[X_cols], train[y_col])
    pred = model.predict(test[X_cols])

    ss_res = float(np.sum((test[y_col] - pred) ** 2))
    ss_tot = float(np.sum((test[y_col] - test[y_col].mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean((test[y_col] - pred) ** 2)))

    # How many batches appear on BOTH sides of the split. For "random" this is
    # every batch; for "batchwise" it is zero by construction. This number is
    # the mechanism, stated as a number.
    shared = len(set(train[col["batch"]]) & set(test[col["batch"]]))

    results[how] = {"r2": r2, "rmse": rmse, "shared_batches": shared,
                    "n_train": len(train), "n_test": len(test),
                    "y_true": test[y_col].to_numpy(), "y_pred": pred}

    print(f"how={how:<10} R² = {r2:6.3f}   RMSE = {rmse:5.2f} g/L   "
          f"batches in both train and test: {shared}")

gap = results["random"]["r2"] - results["batchwise"]["r2"]
print(f"\nR² gap: {gap:.3f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
for ax, how in zip(axes, ["random", "batchwise"]):
    r = results[how]
    bioml.parity_plot(r["y_true"], r["y_pred"], ax=ax,
                      title=f'how="{how}"   R² = {r["r2"]:.3f}')
fig.tight_layout()

# Read the two panels together. The left one is not a better model. It is the
# same model, scored on rows whose neighbours it was trained on.

# %% [markdown]
# ## 3. Cross-validation, same distinction
#
# A single split is one draw. Cross-validation gives the spread, and the spread
# matters more than the mean when n = 30 batches.

# %%
X = model_df[X_cols].to_numpy(dtype=float)
y = model_df[y_col].to_numpy(dtype=float)
groups = model_df[col["batch"]].to_numpy()

for how in ["random", "batchwise"]:
    scores = bioml.cv_score(make_model(), X, y, groups=groups, how=how, n_splits=5)
    print(f'how="{how}":  R² per fold '
          f'{np.array2string(scores, precision=3, floatmode="fixed")}')
    print(f'            mean {scores.mean():6.3f}   sd {scores.std():.3f}\n')

# Report the mean AND the spread. With 30 batches the fold-to-fold variation is
# frequently larger than the difference between two candidate models, and a
# table of means alone hides that from your reader and from you.

# %% [markdown]
# ## 4. Read the source
#
# **Deliverable.** Open `bioml/splits.py`, read `split()`, and write two
# sentences explaining why `how="random"` inflates the score on this data.
#
# A complete answer names the unit of independence. Consecutive rows in one
# batch are 0.2 h apart, from the same inoculum, the same feed profile and the
# same vessel; they are repeated observations of one experiment, not thirty
# separate ones. A random split puts near-copies on both sides, so the test set
# measures interpolation between points the model has already seen rather than
# performance on a batch it has not.
#
# The number to quote alongside it is `shared_batches` above: it is the
# mechanism expressed as a count.

# %%
# A third strategy exists and is worth knowing about now, though Week 6 is where
# it earns its keep: hold out a whole control REGIME, not just whole batches.
# Weeks 1-2 use only recipe-driven batches, so there is no second regime here to
# hold out — the call below is the shape of it, deliberately left commented.
#
#   ctrl = bioml.resolve(proc, "control_ref")
#   train, test = bioml.split(all_batches, how="regime",
#                             regime_col=ctrl, test_regimes=[1])
#
# Each of the three answers a different question: interpolation within a batch,
# generalisation to a new batch, transfer to a new way of running the plant.
# Only the second and third are questions a plant ever asks.

# %% [markdown]
# ---
# ## REFERENCE-ONLY — verification
#
# Removed from the student copy by the patch.
#
# **Calibration status.** The thresholds below are provisional. They were set
# against the synthetic fixture, whose correlation structure is a toy. Re-run
# this notebook on the real IndPenSim Parquet and reset them to what that
# produces before Week 2 is taught. Until then the directional check is the
# real test; the magnitude check is a placeholder.

# %%
# Thresholds. TODO: reset against the real data before the term starts.
MIN_GAP = 0.05           # provisional
MIN_RANDOM_R2 = 0.50     # provisional


def run_checks(results: dict, model_df: pd.DataFrame, X_cols: list,
               roles: pd.DataFrame, col: dict) -> None:
    failures: list[str] = []

    def check(ok: bool, msg: str) -> None:
        print(f"  {'PASS' if ok else 'FAIL'}  {msg}")
        if not ok:
            failures.append(msg)

    print("STRUCTURE")
    check(len(model_df) > 300, f"enough labelled rows ({len(model_df)})")
    check(model_df[col["batch"]].nunique() == 30, "30 batches represented")
    forbidden = roles.loc[roles.role.isin(["simulator_state", "design_metadata"]),
                          "column"].tolist()
    check(not [c for c in X_cols if c in forbidden],
          "no simulator_state or design_metadata column is a feature")

    print("\nSPLIT MECHANICS (these hold on any dataset)")
    check(results["batchwise"]["shared_batches"] == 0,
          "batchwise split shares no batch between train and test")
    check(results["random"]["shared_batches"] > 0,
          "random split does share batches — that is the mechanism")

    print("\nPHENOMENON (provisional thresholds — recalibrate on real data)")
    check(results["random"]["r2"] > results["batchwise"]["r2"],
          "random-split R² exceeds batchwise-split R² (direction)")
    gap = results["random"]["r2"] - results["batchwise"]["r2"]
    check(gap > MIN_GAP, f"the gap is large enough to be the lesson "
                         f"({gap:.3f} > {MIN_GAP})")
    check(results["random"]["r2"] > MIN_RANDOM_R2,
          f"the optimistic number looks convincing on its own "
          f"({results['random']['r2']:.3f} > {MIN_RANDOM_R2})")

    print()
    if failures:
        raise AssertionError(f"{len(failures)} check(s) failed: {failures}")
    print("ALL CHECKS PASSED")


run_checks(results, model_df, X_cols, roles, col)
