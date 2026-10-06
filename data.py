"""
bioml.data — loaders for the distributed IndPenSim Parquet files.

These functions read what `prepare_indpensim.py` wrote. They do no cleaning,
no imputation, no feature selection and no reshaping. Anything that looks like
a modelling choice is deliberately absent — see Decision 9's exclusion list.

DATA LAYOUT EXPECTED
--------------------
    <data_dir>/
        process_vars.parquet    all 100 batches, all non-Raman columns
        column_roles.csv        one row per column: column, role, reason
        manifest.json           what the conversion found in the source file
        raman/batch_001.parquet ... batch_100.parquet
        fault_labels.parquet    only if --withhold-fault-labels was used

Set the directory once, with BIOML_DATA in the environment or set_data_dir().
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .columns import resolve

# Module-level default. Overridden by set_data_dir() or the BIOML_DATA env var.
_DATA_DIR = Path(os.environ.get("BIOML_DATA", "data"))


def set_data_dir(path: str | Path) -> Path:
    """Point bioml at the folder holding process_vars.parquet. Returns the path."""
    global _DATA_DIR
    _DATA_DIR = Path(path)
    if not _DATA_DIR.exists():
        raise FileNotFoundError(f"data directory does not exist: {_DATA_DIR.resolve()}")
    return _DATA_DIR


def data_dir() -> Path:
    """The folder bioml is currently reading from."""
    return _DATA_DIR


# ---------------------------------------------------------------------------
# Process variables
# ---------------------------------------------------------------------------

def load_process_vars(
    batches: Sequence[int] | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Load the non-Raman block.

    batches : iterable of batch numbers (1-100), or None for all 100.
    columns : actual column names to read, or None for all. Parquet is
              columnar, so naming columns here reads only those columns off
              disk — this is the whole reason the course does not ship CSV.

    Rows are returned sorted by (batch, time). No index is set: a RangeIndex
    keeps `groupby` and `iloc` behaving the way a novice expects, and an
    accidental index-alignment join is a bug class this cohort cannot debug.
    """
    path = _DATA_DIR / "process_vars.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run prepare_indpensim.py, or call "
            f"bioml.set_data_dir(...) to point at the right folder."
        )

    # If the caller names columns, make sure the identifiers come along —
    # a frame without Batch ID cannot be split correctly, and returning one
    # would invite exactly the leakage the course is about.
    read_cols = None
    if columns is not None:
        head = pd.read_parquet(path, engine="pyarrow").head(0)
        ident = [resolve(head, "batch"), resolve(head, "time")]
        read_cols = ident + [c for c in columns if c not in ident]

    df = pd.read_parquet(path, columns=read_cols, engine="pyarrow")

    batch_col = resolve(df, "batch")
    time_col = resolve(df, "time")

    if batches is not None:
        wanted = [int(b) for b in batches]
        df = df[df[batch_col].isin(wanted)]
        missing = sorted(set(wanted) - set(df[batch_col].unique().tolist()))
        if missing:
            raise ValueError(f"batches not present in the file: {missing}")

    return df.sort_values([batch_col, time_col]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Raman spectra
# ---------------------------------------------------------------------------

def load_raman(batches: Iterable[int]) -> dict[int, pd.DataFrame]:
    """
    Load Raman spectra, ONE FRAME PER BATCH.

    Returns {batch_number: DataFrame}. Each frame has a Time column and 2,200
    wavenumber columns; it does NOT have a Batch ID column, because the batch
    number is the dictionary key.

    This deliberately does not concatenate (Decision 10). Stacking the frames
    for batches 61-90, labelling each row with its batch, and joining the
    result to the ~98%-sparse offline assay is the Week 3 lab. A
    `load_raman_concatenated()` here would delete that lab.

    Note that not every batch has spectra: only batches 61-90 were run with
    the spectrometer recording (PAT_ref = 2). Asking for batch 12 raises.
    """
    raman_dir = _DATA_DIR / "raman"
    if not raman_dir.exists():
        raise FileNotFoundError(f"{raman_dir} not found.")

    out: dict[int, pd.DataFrame] = {}
    for b in batches:
        b = int(b)
        path = raman_dir / f"batch_{b:03d}.parquet"
        if not path.exists():
            available = sorted(int(p.stem.split("_")[1]) for p in raman_dir.glob("*.parquet"))
            raise FileNotFoundError(
                f"no Raman file for batch {b}. Available: "
                f"{available[0]}-{available[-1]} ({len(available)} batches)."
            )
        out[b] = pd.read_parquet(path, engine="pyarrow")
    return out


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def load_column_roles() -> pd.DataFrame:
    """
    Load the column-role table: one row per column, with role and reason.

    Roles: identifier, design_metadata, simulator_state, offline_assay,
    online_measurable, raman, dead.

    Read this before building any feature matrix. `simulator_state` columns are
    present in the file and absent on a plant; using one as a feature is not a
    bug that any validation protocol will catch.
    """
    path = _DATA_DIR / "column_roles.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    return pd.read_csv(path)


def load_manifest() -> dict:
    """
    Load manifest.json — what the conversion measured in the source file.

    Use it instead of hard-coding numbers. The offline sampling interval in
    particular is a property of the data, not a constant you may assume; an
    earlier version of this course assumed it and was wrong by a factor of ten.
    """
    path = _DATA_DIR / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    return json.loads(path.read_text())


def load_fault_labels() -> pd.DataFrame:
    """
    Load the withheld Week 6 answer key, if the conversion withheld it.

    Raises if the labels were not withheld — in that case `Fault flag` is
    still sitting in process_vars.parquet and you did not need this function.
    """
    path = _DATA_DIR / "fault_labels.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Either the conversion was run without "
            f"--withhold-fault-labels (so Fault flag is in process_vars.parquet), "
            f"or you are pointed at the wrong data directory."
        )
    return pd.read_parquet(path, engine="pyarrow")
