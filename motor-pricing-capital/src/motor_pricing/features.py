"""Feature engineering shared by the GLM and the gradient-boosting challenger.

`build_model_frame` produces the banded rating factors used by both models, so the
transparent GLM and the ML challenger see exactly the same information and the
comparison is fair. `build_design` turns those factors into a one-hot design matrix
for the in-house GLM; the ML model consumes the banded frame directly with native
categorical handling.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .data import clean_frequency

# Categorical + numeric factors used by both models.
CATEGORICAL = ["Area", "VehPowerGroup", "VehAgeBand", "DrivAgeBand",
               "BonusMalusBand", "VehBrand", "VehGas", "Region"]
NUMERIC = ["LogDensity"]
# Continuous versions the ML model can use directly (trees handle raw numerics).
ML_FEATURES = ["Area", "VehPowerGroup", "VehAgeBand", "DrivAgeCapped",
               "BonusMalusCapped", "VehBrand", "VehGas", "LogDensity", "Region"]
ML_CATEGORICAL = ["Area", "VehPowerGroup", "VehAgeBand", "VehBrand", "VehGas", "Region"]


def build_model_frame(freq_raw: pd.DataFrame) -> pd.DataFrame:
    d = clean_frequency(freq_raw)
    d["VehPowerGroup"] = np.where(d["VehPower"] >= 9, "9+", d["VehPower"].astype(str))
    d["VehAgeBand"] = pd.cut(d["VehAge"], [-1, 0, 10, np.inf],
                             labels=["0", "1-10", "11+"]).astype(str)
    d["DrivAgeBand"] = pd.cut(d["DrivAge"], [17, 20, 25, 30, 40, 50, 70, np.inf],
                              labels=["18-20", "21-25", "26-30", "31-40",
                                      "41-50", "51-70", "71+"]).astype(str)
    d["BonusMalusBand"] = pd.cut(np.minimum(d["BonusMalus"], 150),
                                 [0, 50, 60, 70, 90, 110, np.inf],
                                 labels=["50", "51-60", "61-70", "71-90",
                                         "91-110", "111+"]).astype(str)
    d["DrivAgeCapped"] = np.minimum(d["DrivAge"], 90)
    d["BonusMalusCapped"] = np.minimum(d["BonusMalus"], 150)
    d["LogDensity"] = np.log(d["Density"])
    return d


def build_design(df: pd.DataFrame, encoder=None):
    """One-hot encode CATEGORICAL (dropping a reference level) + NUMERIC + intercept.

    The category set is learned once from the *training* data and stored in the
    returned `encoder`; passing that encoder back reproduces the identical columns
    on any other data. This matters for correctness: the test design must use every
    category learned in training (not just the reference), and a category absent
    from a given split still gets its column (all zeros), so train and test always
    share the same columns. An unseen category at scoring time maps to the reference
    (all its indicator columns are zero).

    Returns (X, names, encoder) where encoder = {"categories": {factor: [levels]},
    "ref": {factor: ref_level}}.
    """
    if encoder is None:
        categories = {f: sorted(df[f].dropna().unique().tolist()) for f in CATEGORICAL}
        ref = {f: categories[f][0] for f in CATEGORICAL}
        encoder = {"categories": categories, "ref": ref}
    categories, ref = encoder["categories"], encoder["ref"]

    cols, names = [np.ones(len(df))], ["Intercept"]
    for f in CATEGORICAL:
        for lv in categories[f]:
            if lv == ref[f]:
                continue
            cols.append((df[f] == lv).astype(float).values); names.append(f"{f}={lv}")
    for c in NUMERIC:
        cols.append(df[c].values.astype(float)); names.append(c)
    return np.column_stack(cols), names, encoder


def build_severity_frame(freq_raw: pd.DataFrame, sev: pd.DataFrame) -> pd.DataFrame:
    """One row per individual claim carrying the policy's rating factors."""
    base = build_model_frame(freq_raw)
    keep = ["IDpol"] + CATEGORICAL + NUMERIC + ML_FEATURES
    keep = list(dict.fromkeys(keep))
    merged = sev.merge(base[keep], on="IDpol", how="inner")
    return merged[merged["ClaimAmount"] > 0].copy()
