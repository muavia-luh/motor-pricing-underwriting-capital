"""Deterministic policy-level train/test split, applied once and reused by every
model so the GLM and the ML challenger are compared on the identical test set."""
from __future__ import annotations
import numpy as np
import pandas as pd


def train_test_split(frame: pd.DataFrame, test_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(frame))
    cut = int((1 - test_frac) * len(frame))
    tr = frame.iloc[idx[:cut]].reset_index(drop=True)
    te = frame.iloc[idx[cut:]].reset_index(drop=True)
    return tr, te


def grouped_train_test_split(frame: pd.DataFrame, group_col="IDpol",
                             test_frac=0.2, seed=42):
    """Split so that all rows sharing a group (e.g. all claims on one policy) fall
    entirely in train or entirely in test. Prevents leakage when a policy has more
    than one claim in the severity data."""
    groups = frame[group_col].unique()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(groups))
    cut = int((1 - test_frac) * len(groups))
    train_groups = set(groups[perm[:cut]])
    mask = frame[group_col].isin(train_groups).values
    tr = frame.iloc[mask].reset_index(drop=True)
    te = frame.iloc[~mask].reset_index(drop=True)
    return tr, te
