"""
 Multi-arm DR-learner + 3D visualization for neoadjuvant timing.

Population: NAT==1
Treatment (A): timing with 3 arms <4w / 4–6w / >6w (encoded 0/1/2)
Outcomes: Fistula, Infection, Delayed Gastric Emptying, Death 90 Days, Composite(any)
Method: Pairwise DR-learner (doubly-robust pseudo-outcomes) with GBMs
Figures: 3D τ surfaces, per-arm risk surfaces, 3D scatter, τ histograms, ICE-like curves
All figures saved at 400 dpi.

Usage (see scripts/driftnatx_hte3d.py):
    python scripts/driftnatx_hte3d.py --input /path/to/Chort_total.csv --output ./outputs
"""

from __future__ import annotations
import os, json, inspect
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
import yaml

from .io_utils import read_csv_any, norm_name, find_col_by_names
from .preprocessing import is_id_like, to01, split_feature_types, build_preprocessor

# --- Matplotlib defaults (safe fonts) ---
matplotlib.rcParams.update({"font.size": 11, "axes.labelsize": 11, "axes.titlesize": 12})

# -------------------- Defaults & constants --------------------
NAT_CANDS = ["Neoadjuvant Therapy", "Neoadjuvant_therapy", "NAT", "Neoadjuvant"]
TIMING_BIN_COLS = ["Less Than 4 Weeks", "Weeks 4-6", "Greater Than 6 Weeks"]
MDS_CANDS = ["Minimum Days To Surgery", "min_days_to_surgery", "Days To Surgery"]

OUTCOME_CANDIDATES = {
    "Fistula": ["Fistula"],
    "Infection": ["Infection"],
    "Delayed Gastric Emptying": ["Delayed Gastric Emptying", "Delayed Gastrric Emptying"],
    "Death 90 Days": ["Death 90 Days", "Death 90 Day", "Death90Days"]
}
COMP_NAME = "Composite (Any of 4 Endpoints)"
ARM_LABELS = ["<4w", "4-6w", ">6w"]
A_MAP = {"<4w": 0, "4-6w": 1, ">6w": 2}
PAIR_ORDER = [(0, 2), (0, 1), (1, 2)]
PAIR_NAMES = {(0, 2): "<4w vs >6w", (0, 1): "<4w vs 4-6w", (1, 2): "4-6w vs >6w"}

# -------------------- Config --------------------
class HTE3DConfig:
    """Config for Step 7 HTE + 3D surfaces."""
    def __init__(
        self,
        random_state: int = 42,
        pi_clip: float = 1e-3,
        max_cat_unique: int = 50,
        grid_n: int = 30,
        ice_n: int = 60,
        feature_select: str = "mi_then_variance",  # ["mi_then_variance", "variance"]
        dpi: int = 400,
    ):
        self.random_state = random_state
        self.pi_clip = pi_clip
        self.max_cat_unique = max_cat_unique
        self.grid_n = grid_n
        self.ice_n = ice_n
        self.feature_select = feature_select
        self.dpi = dpi

# -------------------- Helpers --------------------
def _savefig(fig, path: str, dpi: int):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def _resolve_outcomes(df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, np.ndarray]]:
    """Return mapping from outcome name -> column, and binary arrays."""
    def resolve_outcome_col(df_, cands):
        col = find_col_by_names(df_.columns, cands)
        if col is None:
            raise ValueError(f"Could not find outcome column from candidates: {cands}")
        return col

    res_y = {k: resolve_outcome_col(df, v) for k, v in OUTCOME_CANDIDATES.items()}
    comp = (
        to01(df[res_y["Fistula"]]) |
        to01(df[res_y["Infection"]]) |
        to01(df[res_y["Delayed Gastric Emptying"]]) |
        to01(df[res_y["Death 90 Days"]])
    ).astype(int)
    y_map = {
        "Fistula": to01(df[res_y["Fistula"]]),
        "Infection": to01(df[res_y["Infection"]]),
        "Delayed Gastric Emptying": to01(df[res_y["Delayed Gastric Emptying"]]),
        "Death 90 Days": to01(df[res_y["Death 90 Days"]]),
        COMP_NAME: comp,
    }
    return res_y, y_map

def _derive_timing_series(df: pd.DataFrame) -> pd.Series:
    cols_ok = [c for c in TIMING_BIN_COLS if c in df.columns]
    if len(cols_ok) == 3:
        def pick(row):
            if row["Less Than 4 Weeks"] == 1: return "<4w"
            if row["Weeks 4-6"] == 1: return "4-6w"
            if row["Greater Than 6 Weeks"] == 1: return ">6w"
            return np.nan
        return df.apply(pick, axis=1)
    mds_col = find_col_by_names(df.columns, MDS_CANDS)
    if mds_col is not None:
        def by_days(v):
            try:
                w = float(v) / 7.0
                if w < 4: return "<4w"
                elif w <= 6: return "4-6w"
                else: return ">6w"
            except Exception:
                return np.nan
        return df[mds_col].map(by_days)
    raise ValueError("Could not find timing bin columns nor MDS column.")

def _fit_propensity(X: np.ndarray, A_idx: np.ndarray, seed: int, clip: float) -> np.ndarray:
    clf = LogisticRegression(multi_class="multinomial", solver="lbfgs",
                             random_state=seed, max_iter=2000)
    clf.fit(X, A_idx)
    pi = clf.predict_proba(X)
    return np.clip(pi, clip, 1.0 - clip)

def _fit_outcome_per_arm(y: np.ndarray, A_idx: np.ndarray, s: int, X: np.ndarray, seed: int) -> Tuple[np.ndarray, GradientBoostingClassifier]:
    mask = (A_idx == s)
    clf = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=seed
    )
    clf.fit(X[mask], y[mask])
    p = clf.predict_proba(X)[:, 1]
    return np.clip(p, 1e-6, 1-1e-6), clf

def _dr_pseudo(y, A_idx, pi_hat, m_hat, s, t):
    # ψ_{s,t}(X) = [ 1{A=s}*(Y-m_s)/π_s - 1{A=t}*(Y-m_t)/π_t ] + (m_s - m_t)
    return ((A_idx == s) * (y - m_hat[:, s]) / pi_hat[:, s]
            - (A_idx == t) * (y - m_hat[:, t]) / pi_hat[:, t]) + (m_hat[:, s] - m_hat[:, t])

def _learn_tau(pseudo: np.ndarray, X: np.ndarray, seed: int) -> GradientBoostingRegressor:
    reg = GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.8, random_state=seed
    )
    reg.fit(X, pseudo)
    return reg

def _pick_2d_features_for_surfaces(df: pd.DataFrame, num_cols: List[str], tau_vec: np.ndarray,
                                   strategy: str = "mi_then_variance") -> Tuple[str, str]:
    """Pick two continuous features for 3D surfaces."""
    # candidate numeric columns with enough granularity
    candidates = [c for c in num_cols if df[c].dropna().nunique() >= 10]
    if len(candidates) >= 2:
        if strategy == "mi_then_variance":
            try:
                Xtmp = np.column_stack([df[c].astype(float).fillna(df[c].median()) for c in candidates])
                mi = mutual_info_regression(Xtmp, tau_vec, random_state=0)
                order = np.argsort(mi)[::-1]
                # fall back to variance if MI ties/degeneracy
                f1 = candidates[order[0]]
                # choose the next with highest variance among the rest
                rest = [candidates[i] for i in order[1:]]
                if len(rest) > 0:
                    f2 = max(rest, key=lambda c: float(df[c].astype(float).var()))
                    return f1, f2
            except Exception:
                pass
        # variance-only fallback
        var_sorted = sorted(candidates, key=lambda c: float(df[c].astype(float).var()), reverse=True)
        return var_sorted[0], var_sorted[1]
    # ultimate fallback: take first two available columns (num or cat)
    pool = (num_cols + [c for c in df.columns if c not in num_cols])[:2]
    if len(pool) < 2:
        raise ValueError("Not enough features to draw 3D surfaces.")
    return pool[0], pool[1]

def _grid_values(col: pd.Series, n: int) -> np.ndarray:
    lo = col.astype(float).quantile(0.05)
    hi = col.astype(float).quantile(0.95)
    return np.linspace(lo, hi, n)

def _build_base_row(df: pd.DataFrame) -> Dict[str, object]:
    base_vals = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            base_vals[c] = s.astype(float).median()
        else:
            base_vals[c] = s.mode().iloc[0] if s.notna().any() else ""
    return base_vals

# -------------------- Main entry --------------------
def run_hte3d(input_csv: str, output_dir: str, cfg: HTE3DConfig, config_yaml: str | None = None) -> Dict:
    """Run Step 7 HTE + 3D visualizations. Returns a metadata dict."""
    # YAML override
    if config_yaml:
        with open(config_yaml, "r") as f:
            y = yaml.safe_load(f) or {}
        for k, v in y.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(output_dir, f"hte3d_{ts}")
    os.makedirs(root, exist_ok=True)

    # ---------- Load & filter NAT == 1 ----------
    df = read_csv_any(input_csv)
    nat_col = find_col_by_names(df.columns, NAT_CANDS)
    if nat_col is None:
        raise ValueError("Could not find 'Neoadjuvant Therapy' column (or aliases).")
    df = df.loc[to01(df[nat_col]) == 1].reset_index(drop=True)

    # ---------- Treatment (timing) ----------
    df["timing_bin"] = _derive_timing_series(df)
    df = df.loc[df["timing_bin"].isin(ARM_LABELS)].reset_index(drop=True)
    A_idx = df["timing_bin"].map(A_MAP).astype(int).values

    # ---------- Outcomes & composite ----------
    res_y, Y = _resolve_outcomes(df)
    y = Y[COMP_NAME].astype(int)

    # ---------- Covariates (exclude leakage/time-bins/IDs/MDS/etc.) ----------
    exclude_norm = {
        "survival days", "death 365 days",
        "weeks 4 6", "less than 4 weeks", "greater than 6 weeks",
        "minimum days to surgery", "min days to surgery", "days to surgery",
        "folfirinox", "gemcitabine capecitabine", "chemoradiotherapy",
        "neoadjuvant therapy", "nat", "neoadjuvant",
        "timing bin", "arm9"
    }
    user_exclude = {"mean arterial pressure", "hematocrit"}

    covar_cols: List[str] = []
    for c in df.columns:
        if c in res_y.values() or c == COMP_NAME: continue
        if is_id_like(c): continue
        n = norm_name(c)
        if n in exclude_norm: continue
        if n in {"less than 4 weeks", "weeks 4 6", "greater than 6 weeks"}: continue
        if n in {"minimum days to surgery", "min days to surgery", "days to surgery"}: continue
        if c == nat_col: continue
        if n in user_exclude: continue
        covar_cols.append(c)
    covar_cols = sorted(covar_cols)

    X_raw = df[covar_cols].copy()
    num_cols, cat_cols = split_feature_types(X_raw, cfg.max_cat_unique)
    pre_X = build_preprocessor(num_cols, cat_cols)
    X_enc = pre_X.fit_transform(X_raw)
    Xdense = X_enc.toarray() if hasattr(X_enc, "toarray") else X_enc

    # ---------- Multi-arm DR engine ----------
    pi_hat = _fit_propensity(Xdense, A_idx, cfg.random_state, cfg.pi_clip)

    m_hat = np.zeros((len(df), 3))
    arm_clfs: List[GradientBoostingClassifier] = []
    for s in range(3):
        p, clf = _fit_outcome_per_arm(y, A_idx, s, Xdense, cfg.random_state)
        m_hat[:, s] = p
        arm_clfs.append(clf)

    # Pairwise DR pseudo-outcomes and τ models
    TAU_REG: Dict[Tuple[int, int], GradientBoostingRegressor] = {}
    TAU_VAL: Dict[Tuple[int, int], np.ndarray] = {}
    for (s, t) in PAIR_ORDER:
        psi = _dr_pseudo(y, A_idx, pi_hat, m_hat, s, t)
        reg = _learn_tau(psi, Xdense, cfg.random_state)
        TAU_REG[(s, t)] = reg
        TAU_VAL[(s, t)] = reg.predict(Xdense)

    # ---------- Save τ CSV and histograms ----------
    tau_df = pd.DataFrame({
        "tau_<4w_vs_>6w": TAU_VAL[(0, 2)],
        "tau_<4w_vs_4-6w": TAU_VAL[(0, 1)],
        "tau_4-6w_vs_>6w": TAU_VAL[(1, 2)],
    })
    tau_df.to_csv(os.path.join(root, "tau_pairwise_DR.csv"), index=False)

    for (s, t) in PAIR_ORDER:
        name = PAIR_NAMES[(s, t)]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(TAU_VAL[(s, t)], bins=40)
        ax.set_title(f"DR-learner τ distribution: {name}")
        ax.set_xlabel("τ (positive favors first treatment)")
        ax.set_ylabel("Count")
        _savefig(fig, os.path.join(root, f"tau_hist_{s}_{t}.png"), dpi=cfg.dpi)

    # ---------- Pick 2D features for 3D surfaces ----------
    f1, f2 = _pick_2d_features_for_surfaces(df, num_cols, TAU_VAL[(0, 2)], cfg.feature_select)

    # ---------- Build grid on (f1, f2) and predict surfaces ----------
    base_vals = _build_base_row(X_raw)
    v1 = _grid_values(df[f1], cfg.grid_n)
    v2 = _grid_values(df[f2], cfg.grid_n)
    V1, V2 = np.meshgrid(v1, v2)

    rows = []
    for i in range(cfg.grid_n):
        for j in range(cfg.grid_n):
            r = base_vals.copy()
            r[f1] = V1[i, j]
            r[f2] = V2[i, j]
            rows.append(r)
    grid_df = pd.DataFrame(rows)[X_raw.columns]
    Xg = pre_X.transform(grid_df)
    Xgd = Xg.toarray() if hasattr(Xg, "toarray") else Xg

    # per-arm risks on grid
    M0 = arm_clfs[0].predict_proba(Xgd)[:, 1].reshape(cfg.grid_n, cfg.grid_n)
    M1 = arm_clfs[1].predict_proba(Xgd)[:, 1].reshape(cfg.grid_n, cfg.grid_n)
    M2 = arm_clfs[2].predict_proba(Xgd)[:, 1].reshape(cfg.grid_n, cfg.grid_n)

    # τ surfaces on grid (focus on <4w vs >6w and <4w vs 4-6w)
    T02 = TAU_REG[(0, 2)].predict(Xgd).reshape(cfg.grid_n, cfg.grid_n)
    T01 = TAU_REG[(0, 1)].predict(Xgd).reshape(cfg.grid_n, cfg.grid_n)

    # ---- 3D τ surface: <4w vs >6w ----
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(V1, V2, T02, linewidth=0, antialiased=True)
    ax.set_xlabel(f1); ax.set_ylabel(f2); ax.set_zlabel("τ (<4w − >6w)")
    ax.set_title("HTE surface: τ <4w vs >6w")
    _savefig(fig, os.path.join(root, "3D_tau_surface_early_vs_late.png"), dpi=cfg.dpi)

    # ---- 3D per-arm risk surfaces ----
    for mat, name in zip([M0, M1, M2], ["<4w", "4-6w", ">6w"]):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(V1, V2, mat, linewidth=0, antialiased=True)
        ax.set_xlabel(f1); ax.set_ylabel(f2); ax.set_zlabel("Event risk")
        ax.set_title(f"Risk surface m_s(X): {name}")
        fname = f"3D_risk_surface_{name.replace('>','gt').replace('<','lt').replace(' ','_')}.png"
        _savefig(fig, os.path.join(root, fname), dpi=cfg.dpi)

    # ---- 3D scatter: τ <4w vs >6w on real patients ----
    mask = df[f1].notna() & df[f2].notna()
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(df.loc[mask, f1].astype(float), df.loc[mask, f2].astype(float),
               TAU_VAL[(0, 2)][mask], s=8)
    ax.set_xlabel(f1); ax.set_ylabel(f2); ax.set_zlabel("τ (<4w − >6w)")
    ax.set_title("HTE 3D scatter: patients")
    _savefig(fig, os.path.join(root, "3D_tau_scatter_patients.png"), dpi=cfg.dpi)

    # ---- ICE-like τ curves for f1 and f2 ----
    def ice_curve(var: str, pair=(0, 2)):
        xs = _grid_values(df[var], cfg.ice_n)
        rows = []
        for x in xs:
            r = base_vals.copy()
            r[var] = x
            rows.append(r)
        grid = pd.DataFrame(rows)[X_raw.columns]
        Xg_ = pre_X.transform(grid)
        Xgd_ = Xg_.toarray() if hasattr(Xg_, "toarray") else Xg_
        tau = TAU_REG[pair].predict(Xgd_)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xs, tau, "-")
        ax.set_xlabel(var); ax.set_ylabel(f"τ ({PAIR_NAMES[pair]})")
        ax.set_title(f"ICE-like τ vs {var} (others at median/mode)")
        _savefig(fig, os.path.join(root, f"ICE_tau_{var.replace(' ','_')}.png"), dpi=cfg.dpi)

    ice_curve(f1, pair=(0, 2))
    ice_curve(f2, pair=(0, 2))

    # ---- τ-based simple recommendation ----
    rec = []
    for i in range(len(df)):
        if TAU_VAL[(0, 2)][i] > 0 and TAU_VAL[(0, 1)][i] > 0:
            rec.append("<4w")
        elif TAU_VAL[(1, 2)][i] > 0:
            rec.append("4-6w")
        else:
            rec.append(">6w")

    out_df = pd.DataFrame({
        "observed_timing": df["timing_bin"].values,
        "tau_<4w_vs_>6w": TAU_VAL[(0, 2)],
        "tau_<4w_vs_4-6w": TAU_VAL[(0, 1)],
        "tau_4-6w_vs_>6w": TAU_VAL[(1, 2)],
        "tau_based_recommendation": rec,
    })
    out_df.to_csv(os.path.join(
