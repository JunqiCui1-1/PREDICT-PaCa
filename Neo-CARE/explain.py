"""
Step 8 · Explainability & Clinical Usability for DRIFT-NAT-X

What this does (rebuilds a light-weight DR engine, then explains it):
- Rebuild: propensities π, stabilized weights, per-arm outcome models Mhat, recommendations
- Global: policy-level permutation importance (risk-based)
- Variable effects: PD curves, Δ-risk curves, and a 3D Δ-risk surface
- Surrogate: a compact policy tree (CART) with leaf coverage/consistency
- Individual: patient cards (per-arm risks + feasibility + counterfactual visual hints)
- Diagnostics: calibration (Brier/ECE), overlap/weights, Love plot (SMD), ESS by arm
- Subgroups: small forest plot vs fixed-best arm

All figures saved at 400 dpi.

Usage (see scripts/driftnatx_explain.py):
    python scripts/driftnatx_explain.py --input /path/to/Chort_total.csv --output ./outputs
"""

from __future__ import annotations
import os, re, json
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.calibration import calibration_curve
import yaml

from .io_utils import read_csv_any, norm_name, find_col_by_names
from .preprocessing import is_id_like, to01, split_feature_types, build_preprocessor
from .propensities import prop_factorized, prop_multinomial, overlap_score, stabilized_weights
from .outcomes import aipw_values
from .policy import simplex_grid, fit_mds_quantiles, eval_policy_vector, guardrail_decision
from .plotting import savefig

# ---------- Matplotlib defaults ----------
matplotlib.rcParams.update({"font.size": 11, "axes.labelsize": 11, "axes.titlesize": 12})

# ---------- Constants (kept aligned with Step 6/7) ----------
NAT_CANDS = ["Neoadjuvant Therapy", "Neoadjuvant_therapy", "NAT", "Neoadjuvant"]
REGIMEN_COLS = {
    "FFX": ["Folfirinox", "FOLFOXIRI", "FolfiriNox"],
    "GemCap": ["Gemcitabine Capecitabine", "GemCap", "Gemcitabine+Capecitabine"],
    "CRT": ["Chemoradiotherapy", "Chemo-Radiotherapy", "Chemoradiation"],
}
TIMING_BIN_COLS = ["Less Than 4 Weeks", "Weeks 4-6", "Greater Than 6 Weeks"]
MDS_CANDS = ["Minimum Days To Surgery", "min_days_to_surgery", "Days To Surgery"]

OUTCOME_CANDS = {
    "Fistula": ["Fistula"],
    "Infection": ["Infection"],
    "Delayed Gastric Emptying": ["Delayed Gastrric Emptying", "Delayed Gastric Emptying"],
    "Death 90 Days": ["Death 90 Days", "Death 90 Day", "Death90Days"],
}
COMP_NAME = "Composite (Any of 4 Endpoints)"
BASE_OUTS = ["Fistula", "Infection", "Delayed Gastric Emptying", "Death 90 Days"]

ARM_LIST = [f"{r}@{t}" for r in ["FFX","GemCap","CRT"] for t in ["<4w","4-6w",">6w"]]
ARM2IDX  = {a:i for i,a in enumerate(ARM_LIST)}
reg_map = {"FFX":0,"GemCap":1,"CRT":2}
t_map   = {"<4w":0,"4-6w":1,">6w":2}
start_week = {"<4w":0.0, "4-6w":4.0, ">6w":6.0}

def format_arm_label(a: str) -> str:
    r, t = a.split("@")
    r = {"GemCap":"GC", "FFX":"FFX", "CRT":"CRT"}[r]
    t = t.replace("Less Than 4 Weeks","<4w").replace("Weeks 4-6","4-6w").replace("Greater Than 6 Weeks",">6w")
    return f"{r}@{t}"

# ---------- Config ----------
class ExplainConfig:
    """Config for Step 8 explainability pipeline."""
    def __init__(
        self,
        random_state: int = 42,
        pi_clip: float = 1e-3,
        weight_trunc_pct: float = 0.99,
        max_cat_unique: int = 50,
        weight_step: float = 0.05,
        reg_pref: float = 0.01,
        margin_thresh: float = 0.01,
        overlap_thresh: float = 0.10,
        mds_feas_q: float = 0.25,
        penalty: float = 0.05,
        t_r_default: Dict[str, float] = None,
        buffer_r_default: Dict[str, float] = None,
        estimate_durations: bool = True,
        dpi: int = 400,
        n_patient_cards: int = 20,
    ):
        self.random_state = random_state
        self.pi_clip = pi_clip
        self.weight_trunc_pct = weight_trunc_pct
        self.max_cat_unique = max_cat_unique
        self.weight_step = weight_step
        self.reg_pref = reg_pref
        self.margin_thresh = margin_thresh
        self.overlap_thresh = overlap_thresh
        self.mds_feas_q = mds_feas_q
        self.penalty = penalty
        self.t_r_default =
