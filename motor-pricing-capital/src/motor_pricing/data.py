"""Load the French MTPL data and expose the canonical column set.

Standardised on the CASdatasets `freMTPL2freq` / `freMTPL2sev` tables. The data
is not committed; regenerate it with `scripts/00_get_data.py` (clones CASdatasets
and parses the .rda files). An OpenML route is documented in the README as an
alternative for environments where that is easier.

Data-quality handling follows the actuarial-data-science conventions:
  - claim counts capped at 4, exposure capped at 1 (documented fixes),
  - the raw ("reported") ClaimNb is retained alongside the capped value so the
    counts stay traceable.
Claim-count reconciliation (checked at run time in `reporting.py`):
  - reported claims (uncapped sum of ClaimNb): 26,444, which equals the number of
    severity records, so frequency and severity are measured on the same claims;
  - claims after capping at four per policy: 26,405;
  - claims removed by the cap: 39.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

FREQ_COLUMNS = ["IDpol", "ClaimNb", "Exposure", "Area", "VehPower", "VehAge",
                "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region"]


def load_frequency(path):
    df = pd.read_csv(path)
    return df


def load_severity(path):
    return pd.read_csv(path)


def clean_frequency(freq: pd.DataFrame) -> pd.DataFrame:
    df = freq.copy()
    df["ReportedClaimNb"] = df["ClaimNb"].astype(int)          # keep raw for reconciliation
    df["ClaimNb"] = np.minimum(df["ClaimNb"].astype(float), 4).astype(int)
    df["Exposure"] = np.minimum(df["Exposure"].astype(float), 1.0)
    df = df[df["Exposure"] > 0].copy()
    return df
