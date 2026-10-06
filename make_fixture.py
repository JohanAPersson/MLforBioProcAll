#!/usr/bin/env python3
"""
make_fixture.py — generate a small synthetic dataset shaped exactly like the
output of prepare_indpensim.py.

WHAT THIS IS FOR
----------------
Verifying code, not teaching. It exists so that the Week 1 and Week 2 reference
notebooks can be run end to end, and their checks evaluated, on a machine that
does not have the 2.57 GB IndPenSim CSV. That is the only purpose.

It also does something the real data cannot: it writes `truth.csv`, holding the
exact kinetic parameters used to generate each batch. The Week 1 reference can
therefore check that its derived specific growth rate equals the specific
growth rate the data was built from — a real test with a right answer, rather
than "the number looks about right".

WHAT THIS IS NOT
----------------
It is NOT IndPenSim and it must never be distributed to students. The kinetics
are a toy: exponential-decay growth, constant specific productivity, no
substrate limitation, no pH/DO coupling, no fault physics. Any conclusion drawn
from it about penicillin fermentation is a conclusion about this script.

Every file it writes is stamped `"synthetic": true` in manifest.json, and the
reference notebooks print a warning when they see that flag.

USAGE
    python tools/make_fixture.py --outdir fixture/
    python tools/make_fixture.py --outdir fixture/ --batches 12 --raman-channels 60
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Column names as they appear in the distributed IndPenSim CSV (post-repair).
# If the real file disagrees, fix bioml/columns.py's ALIASES — this dict is
# only here so the fixture exercises the same resolver the real data does.
C = {
    "time":        "Time (h)",
    "batch":       "Batch ID",
    "batch_ref":   "Batch reference(Batch_ref:Batch ref)",
    "control_ref": "0 - Recipe driven 1 - Operator controlled(Control_ref:Control ref)",
    "pat_ref":     "PAT_ref",
    "fault_flag":  "Fault flag",
    "fault_ref":   "Fault reference(Fault_ref:Fault ref)",
    "volume":      "Vessel Volume(V:L)",
    "weight":      "Vessel Weight(Wt:Kg)",
    "temperature": "Temperature(T:K)",
    "ph":          "pH(pH:pH)",
    "do":          "Dissolved oxygen concentration(DO2:mg/L)",
    "o2_offgas":   "Oxygen in percent in off-gas(O2:O2  (%))",
    "co2_offgas":  "carbon dioxide percent in off-gas(CO2outgas:%)",
    "our":         "Oxygen Uptake Rate(OUR:(g min^{-1}))",
    "cer":         "Carbon evolution rate(CER:g/h)",
    "heat":        "Generated heat(Q:kJ)",
    "pressure":    "Air head pressure(pressure:bar)",
    "rpm":         "Agitator RPM(RPM:RPM)",
    "aeration":    "Aeration rate(Fg:L/h)",
    "feed_sugar":  "Sugar feed rate(Fs:L/h)",
    "feed_paa":    "PAA flow(Fpaa:PAA flow (L/h))",
    "feed_oil":    "Oil flow(Foil:L/hr)",
    "feed_water":  "Water for injection/dilution(Fw:L/h)",
    "feed_acid":   "Acid flow rate(Fa:L/h)",
    "feed_base":   "Base flow rate(Fb:L/h)",
    "nh3_shots":   "Ammonia shots(NH3_shots:kgs)",
    "p_state":     "Penicillin concentration(P:g/L)",
    "s_state":     "Substrate concentration(S:g/L)",
    "p_offline":   "Offline Penicillin concentration(P_offline:P(g L^{-1}))",
    "x_offline":   "Offline Biomass concentratio(X_offline:X(g L^{-1}))",
    "viscosity":   "Viscosity(Viscosity_offline:centPoise)",
    "nh3_offline": "NH_3 concentration off-line(NH3_offline:NH3 (g L^{-1}))",
    "paa_offline": "PAA concentration offline(PAA_offline:PAA (g L^{-1}))",
}

DT = 0.2              # online sampling interval, hours — matches IndPenSim
ASSAY_EVERY_H = 11.0  # offline assay interval; ~98% missing, matches Decision 8c


def regime_of(b: int) -> tuple[int, int]:
    """(Control_ref, PAT_ref) for batch b, matching the real file's structure."""
    if 31 <= b <= 60:
        return 1, 1        # operator controlled, no Raman
    if 61 <= b <= 90:
        return 0, 2        # recipe driven, Raman recording
    return 0, 1            # 1-30 and 91-100: recipe driven, no Raman


def make_batch(b: int, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """
    Build one batch, and return it alongside the kinetic parameters used.

    The generative model, stated so the checks have something to check against:

        mu(t)   = mu_max * exp(-t / tau)              specific growth rate, 1/h
        mX(t)   = mX0 * exp(mu_max*tau*(1 - exp(-t/tau)))     total biomass, g
        dmP/dt  = qP * mX  for t > t_lag              total penicillin, g
        V(t)    = V0 + Fv * t                         fed-batch fill, L

    Concentrations are mass / volume, so X and P both fall relative to their
    total masses as the vessel fills. That is the whole point: a fixture where
    volume was constant could not exercise the extensive/intensive distinction.
    """
    duration = float(rng.uniform(167.0, 290.0))
    t = np.arange(DT, duration + DT / 2, DT)
    n = len(t)

    # --- kinetics (the ground truth the checks verify against) --------------
    mu_max = float(rng.uniform(0.055, 0.085))     # 1/h, plausible for P. chrysogenum
    # Total log-growth over the batch, chosen so final biomass lands near
    # 30-45 g/L. tau follows from it, rather than being drawn independently:
    # drawing both freely puts X two orders of magnitude off.
    log_growth = float(rng.uniform(2.7, 3.2))
    tau = log_growth / mu_max                     # h, decay of growth into production
    qP = float(rng.uniform(0.0035, 0.0065))       # g penicillin / (g biomass · h)
    t_lag = float(rng.uniform(18.0, 30.0))        # h before product forms

    V0 = float(rng.uniform(57_000, 62_000))
    Fv = float(rng.uniform(75.0, 105.0))          # L/h net fill rate
    V = V0 + Fv * t                               # ends 71k-92k L on a 100k vessel

    mX0 = float(rng.uniform(1.1e5, 1.5e5))        # g total biomass, ~2 g/L inoculum
    mu_true = mu_max * np.exp(-t / tau)
    mX = mX0 * np.exp(mu_max * tau * (1.0 - np.exp(-t / tau)))

    producing = (t > t_lag).astype(float)
    mP = np.concatenate([[0.0], np.cumsum(qP * mX[:-1] * producing[:-1] * DT)])

    X = mX / V                                    # g/L
    P = mP / V                                    # g/L
    S = 15.0 * np.exp(-t / 90.0) + rng.normal(0, 0.05, n)   # simulator state only

    # --- oxygen: an EXTENSIVE rate, in g/min, exactly as IndPenSim ships it -
    # Growth-associated plus maintenance, both proportional to total biomass.
    our_g_per_h = (1.25 * mu_true + 0.014) * mX
    our = our_g_per_h / 60.0                      # g/min
    cer = 0.95 * our_g_per_h                      # g/h

    # --- controlled and measured process signals ---------------------------
    T = 298.0 + rng.normal(0, 0.25, n)
    pH = 6.5 + rng.normal(0, 0.04, n)
    DO = np.clip(8.0 - 5.0 * (mX / mX.max()) + rng.normal(0, 0.15, n), 0.2, 20.0)
    o2 = np.clip(0.2095 - 0.035 * (our_g_per_h / our_g_per_h.max())
                 + rng.normal(0, 3e-4, n), 0.10, 0.25)
    co2 = np.clip(0.004 + 0.030 * (cer / cer.max()) + rng.normal(0, 3e-4, n), 0.0, 0.2)

    df = pd.DataFrame({
        C["time"]: t,
        C["batch"]: b,
        C["batch_ref"]: b,
        C["control_ref"]: regime_of(b)[0],
        C["pat_ref"]: regime_of(b)[1],
        C["volume"]: V,
        C["weight"]: V * 1.02,
        C["temperature"]: T,
        C["ph"]: pH,
        C["do"]: DO,
        C["o2_offgas"]: o2,
        C["co2_offgas"]: co2,
        C["our"]: our,
        C["cer"]: cer,
        C["heat"]: 1.6e3 * our_g_per_h,
        C["pressure"]: 1.1 + rng.normal(0, 0.01, n),
        C["rpm"]: 100.0,                       # zero variance — "dead" column
        C["aeration"]: 62_000.0 + rng.normal(0, 200, n),
        C["feed_sugar"]: np.clip(Fv * 0.55 * (1 - np.exp(-t / 20)), 0, None),
        C["feed_paa"]: np.clip(4.0 + 0.01 * t, 0, None),
        C["feed_oil"]: np.clip(Fv * 0.12, 0, None) + np.zeros(n),
        C["feed_water"]: np.clip(Fv * 0.30, 0, None) + np.zeros(n),
        C["feed_acid"]: np.abs(rng.normal(0.4, 0.1, n)),
        C["feed_base"]: np.abs(rng.normal(1.1, 0.2, n)),
        C["nh3_shots"]: 0.0,                   # zero variance — "dead" column
        C["p_state"]: P,                       # SIMULATOR STATE, not a sensor
        C["s_state"]: S,                       # SIMULATOR STATE, not a sensor
    })

    # --- offline assays: sparse, and noisy, as laboratory assays are --------
    assay_stride = int(round(ASSAY_EVERY_H / DT))
    idx = np.arange(int(round(4.0 / DT)), n, assay_stride)   # first assay ~4 h in
    for key, series, noise in (("x_offline", X, 0.02),
                               ("p_offline", P, 0.02),
                               ("viscosity", 20 + 80 * (mX / mX.max()), 0.03),
                               ("nh3_offline", 2.0 + 0.5 * np.sin(t / 40), 0.05),
                               ("paa_offline", 1.2 + 0.3 * np.cos(t / 60), 0.05)):
        col = np.full(n, np.nan)
        col[idx] = series[idx] * (1 + rng.normal(0, noise, len(idx)))
        df[C[key]] = col

    # --- faults: batches 91-100 only, and invisible in the design labels ----
    fault = np.zeros(n)
    fault_ref = 0
    if b >= 91:
        fault_ref = int(rng.integers(1, 4))
        onset = int(rng.uniform(0.35, 0.65) * n)
        fault[onset:] = 1.0
        # A fault has to be visible in a signal, or Week 6 has nothing to find.
        df.loc[onset:, C["ph"]] += np.linspace(0, 0.45, n - onset)
        df.loc[onset:, C["do"]] *= np.linspace(1.0, 0.55, n - onset)
    df[C["fault_flag"]] = fault
    df[C["fault_ref"]] = float(fault_ref)

    truth = {"batch": b, "mu_max": mu_max, "tau": tau, "qP": qP, "t_lag": t_lag,
             "V0": V0, "Fv": Fv, "mX0": mX0, "duration_h": duration,
             "assay_interval_h": ASSAY_EVERY_H}
    return df, truth


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--outdir", type=Path, default=Path("fixture"))
    ap.add_argument("--batches", type=int, default=100)
    ap.add_argument("--raman-channels", type=int, default=80,
                    help="reduced from the real 2,200 to keep the fixture small")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--withhold-fault-labels", action="store_true")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    a.outdir.mkdir(parents=True, exist_ok=True)
    raman_dir = a.outdir / "raman"
    raman_dir.mkdir(exist_ok=True)

    frames, truths = [], []
    for b in range(1, a.batches + 1):
        df, truth = make_batch(b, rng)
        frames.append(df)
        truths.append(truth)

        # Raman only for batches 61-90 (PAT_ref == 2), as in the real file.
        if regime_of(b)[1] == 2:
            wn = np.linspace(2400, 201, a.raman_channels)
            base = np.exp(-((wn[None, :] - 1600) ** 2) / (2 * 260 ** 2))
            pen = np.exp(-((wn[None, :] - 995) ** 2) / (2 * 55 ** 2))
            # Spectrum = broth background + a penicillin band scaling with titer.
            sig = (4.0e3 * base
                   + 9.0e2 * pen * df[C["p_state"]].to_numpy()[:, None]
                   + rng.normal(0, 25, (len(df), a.raman_channels)))
            spec = pd.DataFrame(sig.astype("float32"),
                                columns=[f"{w:.0f}" for w in wn])
            spec.insert(0, C["time"], df[C["time"]].to_numpy())
            spec.to_parquet(raman_dir / f"batch_{b:03d}.parquet", index=False)

    proc = pd.concat(frames, ignore_index=True)

    # dtypes mirror prepare_indpensim.py: float32 everywhere except Time (h).
    for col in proc.columns:
        if col == C["time"]:
            proc[col] = proc[col].astype("float64")
        elif col == C["batch"]:
            proc[col] = proc[col].astype("int16")
        else:
            proc[col] = proc[col].astype("float32")

    # --- column_roles.csv ---------------------------------------------------
    roles = []
    role_map = {
        C["time"]: ("identifier", "process time within the batch"),
        C["batch"]: ("identifier", "batch number — the independent unit"),
        C["batch_ref"]: ("identifier", "redundant copy of Batch ID"),
        C["control_ref"]: ("design_metadata", "control regime, constant within a batch"),
        C["pat_ref"]: ("design_metadata", "Raman regime, constant within a batch"),
        C["fault_flag"]: ("design_metadata", "ANSWER KEY for Week 6 — not a feature"),
        C["fault_ref"]: ("design_metadata", "which fault was injected"),
        C["p_state"]: ("simulator_state",
                       "SIMULATOR GROUND TRUTH for titer; no online penicillin "
                       "probe exists on a 100,000 L fermenter"),
        C["s_state"]: ("simulator_state",
                       "SIMULATOR STATE; routine online glucose is not standard "
                       "at production scale"),
        C["rpm"]: ("dead", "zero variance across the whole file"),
        C["nh3_shots"]: ("dead", "zero variance across the whole file"),
    }
    for col in proc.columns:
        if col in role_map:
            role, reason = role_map[col]
        elif "offline" in col.lower() or "Viscosity" in col:
            role, reason = "offline_assay", "sparse laboratory measurement"
        else:
            role, reason = "online_measurable", "available from a plant sensor"
        roles.append({"column": col, "role": role, "reason": reason})
    roles.append({"column": f"<{a.raman_channels} Raman channels: 2400..201 cm-1>",
                  "role": "raman", "reason": "spectral channels, stored per batch"})
    pd.DataFrame(roles).to_csv(a.outdir / "column_roles.csv", index=False)

    # --- fault labels -------------------------------------------------------
    if a.withhold_fault_labels:
        cols = [C["batch"], C["time"], C["fault_flag"], C["fault_ref"]]
        proc[cols].to_parquet(a.outdir / "fault_labels.parquet", index=False)
        proc = proc.drop(columns=[C["fault_flag"], C["fault_ref"]])

    proc.to_parquet(a.outdir / "process_vars.parquet", index=False,
                    compression="snappy")
    pd.DataFrame(truths).to_csv(a.outdir / "truth.csv", index=False)

    # --- manifest, same shape as prepare_indpensim.py writes ---------------
    sizes = proc.groupby(C["batch"]).size()
    offline = {c: {"fraction_observed": round(float(proc[c].notna().mean()), 5)}
               for c in proc.columns if "offline" in c.lower() or "Viscosity" in c}
    for c, s in offline.items():
        f = s["fraction_observed"]
        s["approx_rows_per_label"] = round(1 / f, 1) if f else None
        s["approx_hours_per_label"] = round(DT / f, 2) if f else None

    manifest = {
        "synthetic": True,
        "WARNING": ("Generated by tools/make_fixture.py for CODE VERIFICATION "
                    "ONLY. This is not IndPenSim. Do not distribute, do not "
                    "draw any bioprocess conclusion from it."),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_dataset": "SYNTHETIC fixture imitating IndPenSim",
        "rows": int(len(proc)),
        "n_batches": int(proc[C["batch"]].nunique()),
        "rows_per_batch": {"min": int(sizes.min()), "median": int(sizes.median()),
                           "max": int(sizes.max())},
        "sample_interval_h": DT,
        "n_raman_channels": a.raman_channels,
        "offline_sampling": offline,
        "fault_labels_withheld": bool(a.withhold_fault_labels),
    }
    (a.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"fixture written to {a.outdir.resolve()}")
    print(f"  {len(proc):,} rows x {proc.shape[1]} columns, "
          f"{proc[C['batch']].nunique()} batches")
    print(f"  process_vars.parquet  "
          f"{(a.outdir / 'process_vars.parquet').stat().st_size / 1e6:.1f} MB")
    print(f"  raman/                {len(list(raman_dir.glob('*.parquet')))} files")
    print("  truth.csv             kinetic parameters used per batch")
    print("\n  SYNTHETIC. Verification only. Never distribute this to students.")


if __name__ == "__main__":
    main()
