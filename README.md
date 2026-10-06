# ML for Bioprocesses — Weeks 1–2 build

Stage 1 of the Decision 11 workflow: **verified reference code first, student
copies second.** Nothing in this tree is a student-facing file. The bugged
notebooks are not here on purpose.

```
bioml/                     the frozen course package (Decision 9)
  __init__.py              public API, __version__
  columns.py               short keys -> IndPenSim's actual column names
  data.py                  load_process_vars / load_raman / roles / manifest
  splits.py                split(how=) and cv_score(how=) — no default
  plots.py                 parity, residual, batch trajectory
  align.py                 align_batches(method=, reference=)
reference/
  week1_reference.py       Week 1, correct, fully commented, self-checking
  week2_reference.py       Week 2, correct, fully commented, self-checking
tasks/
  week1_task.md            student task sheet
  week2_task.md            student task sheet
tools/
  make_fixture.py          synthetic IndPenSim-shaped data, for verification only
```

## Running it

```bash
pip install pandas pyarrow scikit-learn matplotlib
export PYTHONPATH=.

python tools/make_fixture.py --outdir fixture
python reference/week1_reference.py     # ends in ALL CHECKS PASSED or raises
python reference/week2_reference.py
```

Point at the real data by editing the single `DATA_DIR` line at the top of each
reference, or by setting `BIOML_DATA`.

## Why the references are `.py` and not `.ipynb`

They are [jupytext](https://jupytext.readthedocs.io) percent-format scripts:
`# %%` starts a cell, `# %% [markdown]` a text cell. They run as plain Python
*and* open as notebooks.

```bash
pip install jupytext
jupytext --to ipynb reference/week1_reference.py
```

This matters for Decision 11 point 4. A patch against `.ipynb` JSON is not
reviewable and `patch -R` does not cleanly recover the original — cell ids,
execution counts and stored outputs all churn. A patch against percent-format
`.py` is a dozen readable lines, and reverting it is a genuine mechanical check
rather than a ritual. The `.ipynb` students receive is generated from the
patched `.py`, never edited by hand.

## The verification loop

Each reference ends in a `run_checks()` cell that raises on failure. Two levels:

- **Physics and structure** — plausible ranges for µ, q_P, q_O₂; batch counts;
  the assertion that no `simulator_state` column reached a feature matrix.
  These work on the real data.
- **Ground truth** — only on the synthetic fixture, which writes `truth.csv`
  holding the kinetic parameters each batch was generated from. The derived
  rates are checked against those parameters directly.

Current status on the fixture: µ, q_P and q_O₂ all recovered to within 0.6% of
the generating values.

## What has been verified, and what has not

**Verified.** Both references run end to end and pass every check. Both planted
defects were confirmed to run silently and to break the checks — the time-step
defect inflates µ and q_P by 55.3×, the extensive/intensive defect inflates
q_O₂ by 7.0×10⁴. Neither raises an exception.

**Not verified — requires the real IndPenSim Parquet:**

1. **Column names.** `bioml/columns.py` binds by regex, and the patterns were
   written against the names in `prepare_indpensim.py` plus the published
   dataset description. Run `bioml.audit(df)` on the real file once; every
   unresolved key is a one-line fix in `ALIASES`.
2. **The offline assay interval.** The fixture uses 11 h. The real figure comes
   from `manifest.json` and both references read it from there rather than
   assuming it — but the Week 1 time-step defect's *magnitude* is
   assay_interval / 0.2, so the task sheet's expected answer depends on it.
3. **The Week 2 R² collapse.** On the fixture the gap is 0.90 → 0.81. The
   fixture's batches are near-identical by construction, so this understates
   what the real data should show. `MIN_GAP` and `MIN_RANDOM_R2` in
   `week2_reference.py` are placeholders marked TODO. Reset them to what the
   real data produces before Week 2 is taught.

## The fixture is not IndPenSim

`tools/make_fixture.py` generates toy kinetics — exponential-decay growth,
constant specific productivity, no substrate limitation, no fault physics. It
exists so this code can be verified without the 2.57 GB CSV. Every file it
writes is stamped `"synthetic": true` and both references print a warning when
they see that flag. It must never be distributed.

## Data citation

Goldrick, S., Stefan, A., Lovett, D., Montague, G., Lennox, B. (2015). *The
development of an industrial-scale fed-batch fermentation simulation.*
Mendeley Data, https://data.mendeley.com/datasets/pdnjz7zz5x/2 — CC BY 4.0
