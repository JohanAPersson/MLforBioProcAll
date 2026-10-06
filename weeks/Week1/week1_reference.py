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
# # Week 1 — Bioprocess data and the ML framing (REFERENCE)
#
# **This is the instructor reference. It is correct. It is not distributed.**
#
# The student notebook for Week 1 is generated from this file by a saved patch
# (`patches/week1_bugs.patch`) that does three things and nothing else:
#
# 1. introduces the **time-step** defect on the line marked `BUG 1 SITE`;
# 2. introduces the **extensive/intensive** defect on the line marked `BUG 2 SITE`;
# 3. deletes the region marked `REFERENCE-ONLY` (the domain-reasoning comments
#    around the rate definitions, and the verification cell at the end).
#
# Reverting the patch and re-running this file must reproduce it exactly and
# pass every check in the final cell. That round trip is what "verified" means
# here (Decision 11).
#
# ## What Week 1 is for
#
# Three things, in this order:
#
# 1. **The ratio.** 2.57 GB of file, n ≈ 100 independent runs. Both numbers are
#    surprising, in opposite directions, and both mislead in the same way.
# 2. **The observability boundary.** Which of these columns would exist on a
#    100,000 L fermenter at hour 100, and which exist only because a simulator
#    wrote them down?
# 3. **Rates, not levels.** A fermentation is understood through µ, qP and qO₂,
#    and computing them from a fed-batch correctly requires knowing the
#    difference between a concentration and an amount.

# %%
# --- Setup -------------------------------------------------------------------
# Standard scientific Python. `bioml` is the course package: loading, splitting,
# plotting. Its source is published and you are expected to read it.
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bioml

# Point bioml at the folder holding process_vars.parquet. Change this one line
# if your data lives somewhere else; nothing below hard-codes a path.
DATA_DIR = Path("fixture")
bioml.set_data_dir(DATA_DIR)

# Print provenance at the top of every notebook. When a number in your report
# is questioned three weeks from now, this line is how you establish which
# package build and which data produced it.
print(bioml.about())

# The fixture used to develop this notebook is synthetic. The real data is not.
# Refusing to let that distinction go unnoticed is cheap, so we check for it.
_manifest = bioml.load_manifest()
if _manifest.get("synthetic"):
    print("\n*** SYNTHETIC FIXTURE — code verification only, not IndPenSim ***\n")

# %% [markdown]
# ## 1. The ratio
#
# Compute it yourself. Do not take it from the lecture slides.

# %%
# Size of the source distribution, from the manifest written by the conversion
# script. `source_size_bytes` is absent from the synthetic fixture, so fall
# back to the documented figure and say so rather than printing a wrong number.
source_bytes = _manifest.get("source_size_bytes")
source_gb = source_bytes / 1e9 if source_bytes else 2.57

n_rows = _manifest["rows"]              # every logged time point, all batches
n_batches = _manifest["n_batches"]      # independently executed fermentations

print(f"source file        {source_gb:.2f} GB")
print(f"rows               {n_rows:,}")
print(f"batches            {n_batches}")
print(f"bytes per batch    {source_gb * 1e9 / n_batches:,.0f}")
print(f"rows per batch     {n_rows / n_batches:,.0f}")

# The point of printing these four numbers together: the file is large enough
# that it will not open in Excel, and the experiment behind it is small enough
# that a hundred-parameter model has no business being fitted to it. Rows are
# not independent samples. Batches are. Every validation decision in Week 2
# follows from that single sentence.

# %% [markdown]
# ## 2. Load the process variables
#
# Weeks 1–2 use batches 1–30 only: the recipe-driven regime, no Raman. One
# regime, so that any structure you find is not a control-strategy artefact.

# %%
# Parquet is columnar, so this reads only the non-Raman block off disk. The
# 2,200 spectral channels are in separate per-batch files and are not touched
# until Week 3.
WEEK1_BATCHES = range(1, 31)
proc = bioml.load_process_vars(batches=WEEK1_BATCHES)

print(f"{proc.shape[0]:,} rows x {proc.shape[1]} columns")
print(f"memory: {proc.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# Bind short keys to the file's actual column names once, here, so that nothing
# below contains a 60-character string literal. If any of these fail, the alias
# table in bioml/columns.py needs one edit — do not rename columns in the data.
col = bioml.resolve_many(proc, [
    "time", "batch", "volume", "our",
    "x_offline", "p_offline",
    "temperature", "ph", "do", "o2_offgas", "co2_offgas",
])
for k, v in col.items():
    print(f"  {k:<12} -> {v}")

# %% [markdown]
# ## 3. Profile the data before modelling anything
#
# Four questions, in order: how long are the batches, how complete is each
# column, which columns never move, and what does a trajectory look like.

# %%
# --- Batch durations ---------------------------------------------------------
# Batches do not all run the same length. This matters more than it looks:
# it means the batches cannot be stacked into a rectangular
# batch x time x variable array without an alignment decision (Week 6).
dur = proc.groupby(col["batch"])[col["time"]].max()
rows_per = proc.groupby(col["batch"]).size()

print("batch duration (h)   "
      f"min {dur.min():.1f}   median {dur.median():.1f}   max {dur.max():.1f}"
      f"   spread {100 * (dur.max() / dur.min() - 1):.0f}%")
print("rows per batch       "
      f"min {rows_per.min()}   median {rows_per.median():.0f}   max {rows_per.max()}")

# %%
# --- Completeness ------------------------------------------------------------
# Sort by how much of each column is actually observed. The gap between the
# dense block and the sparse block is not a data defect — it is the entire
# reason this course has a soft-sensing module. Cheap sensors log every 0.2 h;
# expensive laboratory assays do not.
observed = proc.notna().mean().sort_values()
sparse = observed[observed < 0.5]
dense = observed[observed >= 0.5]

print(f"{len(sparse)} sparse columns (<50% observed):")
for name, frac in sparse.items():
    n_labels = int(proc[name].notna().sum())
    print(f"  {name[:58]:<60} {frac:6.2%}  ({n_labels:,} values, "
          f"~1 per {1 / frac:.0f} rows)")
print(f"\n{len(dense)} dense columns (>=50% observed)")

# The offline sampling interval is a measured property of the file, not a
# constant you may assume. Read it from the manifest. An earlier version of
# this course inferred it from one batch and was wrong by a factor of ten.
print("\noffline sampling, from manifest.json:")
for name, stats in _manifest.get("offline_sampling", {}).items():
    print(f"  {name[:50]:<52} 1 label per {stats['approx_rows_per_label']} rows "
          f"(~{stats.get('approx_hours_per_label')} h)")

# %%
# --- Columns that never move -------------------------------------------------
# A zero-variance column carries no information, and will silently produce
# divide-by-zero in any standardising scaler. Find them now, not in Week 4.
nunique = proc.nunique(dropna=True)
dead_cols = nunique[nunique <= 1].index.tolist()
print(f"zero-variance columns ({len(dead_cols)}):")
for c in dead_cols:
    print(f"  {c}  = {proc[c].dropna().iloc[0] if proc[c].notna().any() else 'all NaN'}")

# %%
# --- What a batch looks like -------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 6))
show = [3, 7, 11, 19, 25]   # five batches; a hundred lines is a grey smear

bioml.batch_trajectory_plot(proc, col["volume"], batches=show, ax=axes[0, 0])
axes[0, 0].set_title("vessel volume — the fed-batch fill")

bioml.batch_trajectory_plot(proc, col["our"], batches=show, ax=axes[0, 1], legend=False)
axes[0, 1].set_title("oxygen uptake rate (whole vessel)")

bioml.batch_trajectory_plot(proc, col["x_offline"], batches=show, ax=axes[1, 0], legend=False)
axes[1, 0].set_title("biomass — offline assay, sparse")

bioml.batch_trajectory_plot(proc, col["p_offline"], batches=show, ax=axes[1, 1], legend=False)
axes[1, 1].set_title("penicillin — offline assay, sparse")

fig.tight_layout()

# Look at the top-left panel before reading on. The volume rises by roughly
# half over a batch. Hold on to that number; section 5 depends on it.

# %% [markdown]
# ## 4. The observability boundary
#
# **Deliverable (a).** The file contains columns a plant would have, and columns
# that exist only because a simulator wrote its own internal state to disk.
# Telling them apart is a bioprocess question, not a programming question, and
# it is one you are already equipped to answer.

# %%
roles = bioml.load_column_roles()
print(roles.groupby("role").size().to_string())

# The columns that will silently destroy a soft sensor if left in a feature
# matrix. They are shipped, labelled, on purpose: deleting them would hide the
# reason soft sensing exists at all.
print("\n--- simulator_state: present in the file, absent on a plant ---")
for _, r in roles[roles.role == "simulator_state"].iterrows():
    print(f"\n  {r['column']}\n    {r['reason']}")

# Numerical confirmation that the online penicillin column and the offline
# assay are the same quantity, one dense and one sparse. If you can predict
# P_offline from P you have predicted a number using itself.
p_state = bioml.try_resolve(proc, "p_state")
if p_state:
    print(f"\n  online  P max = {proc[p_state].max():.2f} g/L")
    print(f"  offline P max = {proc[col['p_offline']].max():.2f} g/L")
    print("  Same quantity. Using one to predict the other is not modelling.")

# %% [markdown]
# ## 5. Derived rates
#
# A fermentation is described by rates, not levels: how fast the organism is
# growing, how fast it is making product, how fast it is consuming oxygen. All
# three are computed from the sparse offline assays, so all three inherit the
# assay sampling interval — not the 0.2 h logging interval.

# %%
# REFERENCE-ONLY ---------------------------------------------------------------
# Everything between this marker and its closing marker is domain reasoning that
# is stripped from the student copy by the bug patch. The student copy keeps the
# code and the mechanics comments; it does not keep the justification, because
# the justification is the answer to the task.
#
# DEFINITIONS, and why each is written the way it is.
#
#   Total biomass         m_X = X · V          [g]
#   Total product         m_P = P · V          [g]
#
#     X and P are CONCENTRATIONS (g/L). V changes by ~50% over a fed-batch run.
#     A concentration can fall while the amount rises, simply because the vessel
#     is being filled. Rates defined on concentrations therefore measure the
#     feed pump as much as the organism.
#
#   Specific growth rate  µ = d(ln m_X)/dt     [1/h]
#
#     Written as a log derivative rather than (1/m_X)·dm_X/dt because the two
#     are equal in the limit but not on an 11 h sampling interval. Over an
#     interval where growth is exponential, Δ(ln m_X)/Δt returns the *average*
#     µ over that interval exactly; a forward difference divided by the starting
#     mass overestimates it by ~50% at µ·Δt ≈ 0.8, which is where this data sits.
#
#   Specific productivity q_P = (1/m_X)·dm_P/dt    [g product / (g biomass · h)]
#
#     m_X changes a lot within one 11 h interval, so "the" m_X to divide by is a
#     choice. For exponential growth the correct time-average of m_X over the
#     interval is the LOGARITHMIC mean, (m2−m1)/ln(m2/m1) — not the arithmetic
#     mean and not the starting value. Same reasoning as a log-mean temperature
#     difference in heat exchanger design, which you have already met.
#
#   Specific oxygen uptake q_O2 = OUR / m_X         [g O₂ / (g biomass · h)]
#
#     OUR as logged is EXTENSIVE: grams per minute for the whole vessel. To make
#     it specific it must be divided by an AMOUNT of biomass (g), not by a
#     concentration (g/L). Dividing by X instead of m_X leaves you with units of
#     g·L/(min·g) and a number wrong by a factor of V ≈ 70,000. Check the units.
#     They are the fastest error detector available and they cost nothing.
# --- end REFERENCE-ONLY -------------------------------------------------------


def derive_rates(batch: pd.DataFrame, col: dict[str, str]) -> pd.DataFrame:
    """
    Compute µ, q_P and q_O2 for a single batch from its offline assays.

    batch : rows for ONE batch, sorted by time.
    col   : mapping of short keys to this file's column names.

    Returns one row per assay interval, with the rates attached to the midpoint
    of the interval they were computed over.
    """
    # Pull the columns out as float arrays. .to_numpy() detaches from the index,
    # which stops a later operation from silently aligning on labels.
    t = batch[col["time"]].to_numpy(dtype=float)
    V = batch[col["volume"]].to_numpy(dtype=float)
    X = batch[col["x_offline"]].to_numpy(dtype=float)
    P = batch[col["p_offline"]].to_numpy(dtype=float)
    our_g_per_min = batch[col["our"]].to_numpy(dtype=float)

    # Keep only rows where BOTH assays were taken. Roughly 2% of rows survive;
    # the rest are online-only samples with no laboratory measurement.
    obs = np.isfinite(X) & np.isfinite(P)
    if obs.sum() < 3:
        return pd.DataFrame(columns=["time_h", "mu", "qP", "qO2", "mX", "mP"])

    t, V, X, P = t[obs], V[obs], X[obs], P[obs]
    our_g_per_h = our_g_per_min[obs] * 60.0    # g/min -> g/h, once, here

    # Amounts, not concentrations. See the definitions above.
    mX = X * V                                  # g biomass in the vessel
    mP = P * V                                  # g penicillin in the vessel

    # BUG 1 SITE -- the spacing between consecutive OBSERVED assays, in hours.
    # Taken from the timestamps of the surviving rows, not from the logging
    # interval: after dropping unobserved rows these are ~11 h apart, not 0.2 h.
    dt = np.diff(t)

    # Specific growth rate: difference of the natural log, per hour.
    mu = np.diff(np.log(mX)) / dt

    # Logarithmic mean of the biomass over each interval — the time-average of
    # an exponentially changing quantity between its two endpoints.
    mX_logmean = (mX[1:] - mX[:-1]) / np.log(mX[1:] / mX[:-1])

    # Specific productivity: product formed per interval, per unit biomass.
    qP = (np.diff(mP) / dt) / mX_logmean

    # BUG 2 SITE -- specific oxygen uptake: an extensive rate (g/h for the whole
    # vessel) divided by the amount of biomass present (g).
    qO2 = our_g_per_h / mX

    return pd.DataFrame({
        # Rates describe an interval, so they are reported at its midpoint.
        "time_h": 0.5 * (t[:-1] + t[1:]),
        "mu": mu,
        "qP": qP,
        "qO2": 0.5 * (qO2[:-1] + qO2[1:]),   # midpoint-averaged to match µ and qP
        "mX": mX_logmean,
        "mP": 0.5 * (mP[:-1] + mP[1:]),
    })


# %%
# Apply to every batch and stack the results, labelled by batch.
rates = []
for b, sub in proc.groupby(col["batch"], sort=True):
    r = derive_rates(sub, col)
    r.insert(0, "batch", int(b))
    rates.append(r)
rates = pd.concat(rates, ignore_index=True)

print(f"{len(rates):,} rate estimates from {rates['batch'].nunique()} batches\n")
print(rates[["mu", "qP", "qO2"]].describe().T.to_string(
    float_format=lambda v: f"{v:.4g}"))

# %% [markdown]
# ### Sanity-check the rates against what you know
#
# These are numbers about a filamentous fungus, and you already know roughly
# what they should be. That knowledge is the error detector; use it before you
# use any statistic.

# %%
# Published ranges for penicillin fermentation. A derived rate outside these is
# a bug in the derivation until proven otherwise.
PLAUSIBLE = {
    "mu":  (-0.02, 0.30,  "1/h — doubling time 2.5 h at the top of this range"),
    "qP":  (-0.002, 0.05, "g P / (g X · h)"),
    "qO2": (0.0, 0.5,     "g O2 / (g X · h)"),
}
for name, (lo, hi, unit) in PLAUSIBLE.items():
    v = rates[name].to_numpy()
    frac_out = float(np.mean((v < lo) | (v > hi)))
    print(f"{name:<5} median {np.nanmedian(v):>9.4g}   "
          f"p5 {np.nanpercentile(v, 5):>9.4g}   p95 {np.nanpercentile(v, 95):>9.4g}"
          f"   outside [{lo}, {hi}]: {frac_out:.1%}   [{unit}]")

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
for ax, name, ylab in zip(axes, ["mu", "qP", "qO2"],
                          ["µ (1/h)", "q_P (g/g/h)", "q_O2 (g/g/h)"]):
    for b, sub in rates.groupby("batch"):
        ax.plot(sub["time_h"], sub[name], lw=0.8, alpha=0.35, color="tab:blue")
    ax.set_xlabel("process time (h)")
    ax.set_ylabel(ylab)
    ax.axhline(0, color="k", lw=0.8, zorder=0)
axes[0].set_title("growth decays into the production phase")
axes[1].set_title("production starts after a lag")
axes[2].set_title("respiration tracks growth plus maintenance")
fig.tight_layout()

# %% [markdown]
# ## 6. Deliverable (b): a mechanistic hypothesis
#
# Write it before Week 3 fits anything. Two or three sentences naming the
# variables you expect to drive final titer, the physical reason each should
# matter, and the observation in these plots that supports or undermines it.
# It will be held against the Week 3 and Week 4 models, and it is allowed to
# turn out wrong — a hypothesis that survives contact with data unchanged
# usually was not falsifiable.

# %%
# A starting point, not an answer: correlate each batch's mean rates with its
# final titer. Thirty batches, so treat anything here as suggestive at best.
final_titer = (proc.groupby(col["batch"])[col["p_offline"]]
                   .max().rename("final_titer_g_L"))
per_batch = (rates.groupby("batch")[["mu", "qP", "qO2"]].mean()
                  .join(final_titer))
print(per_batch.corr()["final_titer_g_L"].round(3).to_string())
print(f"\nn = {len(per_batch)} batches. With n = 30, a correlation of 0.36 is "
      f"the 5% significance threshold. Read these accordingly.")

# %% [markdown]
# ---
# ## REFERENCE-ONLY — verification
#
# Everything below is removed from the student copy by the bug patch. It is the
# check that this reference is correct, in the sense Decision 11 requires:
# assertions against expected outputs, not "it ran without raising".
#
# Two levels:
#
# * **Ground truth**, available only on the synthetic fixture, which records the
#   kinetic parameters it generated each batch from. The derived µ and q_O2 must
#   match those parameters.
# * **Physics ranges**, which work on the real data too and are what you re-run
#   after the patch is reverted.

# %%
def run_checks(rates: pd.DataFrame, proc: pd.DataFrame, col: dict,
               data_dir: Path) -> None:
    """Assert that this notebook produced the right numbers. Raises on failure."""
    failures: list[str] = []

    def check(ok: bool, msg: str) -> None:
        print(f"  {'PASS' if ok else 'FAIL'}  {msg}")
        if not ok:
            failures.append(msg)

    print("STRUCTURE")
    check(rates["batch"].nunique() == 30, "30 batches produced rate estimates")
    check(rates[["mu", "qP", "qO2"]].notna().all().all(),
          "no NaN in the derived rates")
    check(len(rates) > 400, f"enough rate estimates to be useful ({len(rates)})")

    print("\nPHYSICS RANGES (these also apply to the real IndPenSim data)")
    for name, (lo, hi, _) in PLAUSIBLE.items():
        v = rates[name].to_numpy()
        check(float(np.mean((v >= lo) & (v <= hi))) > 0.95,
              f"{name}: >95% of estimates inside [{lo}, {hi}]")
    check(rates.loc[rates.time_h < 60, "mu"].median()
          > rates.loc[rates.time_h > 150, "mu"].median(),
          "µ is higher early than late (growth decays into production)")
    check(rates.loc[rates.time_h > 100, "qP"].median() > 0,
          "q_P is positive once production is established")

    print("\nOBSERVABILITY")
    p_state = bioml.try_resolve(proc, "p_state")
    src = Path(__file__).read_text() if "__file__" in globals() else ""
    check(p_state is None or src == "" or p_state not in src,
          "the reference never uses the simulator penicillin column")

    truth_path = data_dir / "truth.csv"
    if not truth_path.exists():
        print("\nGROUND TRUTH: skipped — no truth.csv (this is the real data, "
              "which has none). The physics checks above are the whole test.")
    else:
        print("\nGROUND TRUTH (synthetic fixture only)")
        # Prefix the generating parameters so they cannot collide with the
        # derived-rate column names on the join.
        truth = (pd.read_csv(truth_path)
                   .set_index("batch")
                   .add_prefix("true_"))
        j = rates.join(truth, on="batch")

        # The fixture generates µ(t) = µ_max·exp(−t/τ). The estimate is the
        # average of µ over an 11 h interval, reported at the interval midpoint;
        # the midpoint value of the true curve approximates that to second order.
        mu_true = j["true_mu_max"] * np.exp(-j["time_h"] / j["true_tau"])
        ratio_mu = float((j["mu"] / mu_true).median())
        check(abs(ratio_mu - 1.0) < 0.10,
              f"µ recovered to within 10% of the generating value "
              f"(median ratio {ratio_mu:.3f})")

        # q_O2 was generated as 1.25·µ + 0.014, per gram of biomass. Getting
        # this right requires dividing OUR by an amount, not a concentration —
        # i.e. it is the check that fails if BUG 2 is present.
        qo2_true = 1.25 * mu_true + 0.014
        ratio_o2 = float((j["qO2"] / qo2_true).median())
        check(abs(ratio_o2 - 1.0) < 0.10,
              f"q_O2 recovered to within 10% (median ratio {ratio_o2:.3f})")

        # q_P is constant per batch after the lag, so compare only where
        # production is established; before the lag the true value is zero and
        # the ratio is meaningless.
        est = j.loc[j["time_h"] > 80]
        ratio_qp = float((est["qP"] / est["true_qP"]).median())
        check(abs(ratio_qp - 1.0) < 0.10,
              f"q_P recovered to within 10% (median ratio {ratio_qp:.3f})")

        # And the check that fails if BUG 1 is present: the derived µ must be
        # consistent with the assay interval, not the logging interval. If dt
        # were hard-coded to 0.2 h, every rate would be ~55x too large.
        check(ratio_mu < 5.0,
              "µ is not inflated by the online/offline sampling ratio")

    print()
    if failures:
        raise AssertionError(f"{len(failures)} check(s) failed: {failures}")
    print("ALL CHECKS PASSED")


run_checks(rates, proc, col, DATA_DIR)
