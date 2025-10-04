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
        self.t_r_default = t_r_default or {"FFX": 12.0, "GemCap": 12.0, "CRT": 6.0}
        self.buffer_r_default = buffer_r_default or {"FFX": 2.0, "GemCap": 2.0, "CRT": 3.0}
        self.estimate_durations = estimate_durations
        self.dpi = dpi
        self.n_patient_cards = n_patient_cards

# ---------- Small helpers ----------
def _get_feature_names_from_ct(ct, input_features):
    """Try extracting user-friendly feature names from a ColumnTransformer."""
    try:
        names = ct.get_feature_names_out()
        return [str(n) for n in names]
    except Exception:
        pass
    try:
        out = []
        for name, trans, cols in getattr(ct, "transformers_", []):
            if name == "remainder" and trans == "drop":
                continue
            last = list(trans.named_steps.values())[-1] if hasattr(trans, "named_steps") else trans
            if hasattr(last, "get_feature_names_out"):
                sub = last.get_feature_names_out(cols)
                out.extend([str(x) for x in sub])
            else:
                out.extend([str(c) for c in cols])
        if out:
            return out
    except Exception:
        pass
    d = input_features.shape[1] if hasattr(input_features, "shape") else len(input_features)
    return [f"f{i}" for i in range(d)]

def _wrap_long(nm: str, maxlen: int = 26) -> str:
    s = str(nm)
    if len(s) <= maxlen:
        return s
    parts = re.split(r"([ _])", s)
    lines, cur = [], ""
    for p in parts:
        if len(cur) + len(p) <= maxlen:
            cur += p
        else:
            lines.append(cur)
            cur = p.lstrip()
    if cur:
        lines.append(cur)
    return "\n".join(lines[:4])

def _brier_score(y_true, p): return float(np.mean((p - y_true) ** 2))

def _ece(y_true, p, M: int = 10):
    bins = np.linspace(0, 1, M+1)
    idx = np.digitize(p, bins) - 1
    ece = 0.0
    n = len(y_true)
    for m in range(M):
        msk = (idx == m)
        if msk.sum() == 0: continue
        conf = p[msk].mean()
        acc  = y_true[msk].mean()
        ece += (msk.sum()/n) * abs(acc - conf)
    return float(ece)

def _smd_numeric(x, g, w=None):
    """Average absolute standardized mean difference across all arm pairs."""
    arms = np.unique(g)
    pairs = []
    for i in range(len(arms)):
        for j in range(i+1, len(arms)):
            gi, gj = arms[i], arms[j]
            ix, jx = (g == gi), (g == gj)
            if w is None:
                m1, m2 = x[ix].mean(), x[jx].mean()
                v1, v2 = x[ix].var(ddof=1), x[jx].var(ddof=1)
            else:
                mw = lambda a, wa: np.sum(wa * a) / np.sum(wa)
                vw = lambda a, wa: np.sum(wa * (a - mw(a, wa)) ** 2) / np.sum(wa)
                m1, m2 = mw(x[ix], w[ix]), mw(x[jx], w[jx])
                v1, v2 = vw(x[ix], w[ix]), vw(x[jx], w[jx])
            pooled = np.sqrt((v1 + v2) / 2.0 + 1e-12)
            if pooled > 0:
                pairs.append(abs((m1 - m2) / pooled))
    return float(np.mean(pairs)) if pairs else 0.0

# ---------- Main entry ----------
def run_explainability(input_csv: str, output_dir: str, cfg: ExplainConfig, config_yaml: str | None = None) -> Dict:
    """Run Step 8 explainability pipeline. Returns metadata dict."""
    # YAML override
    if config_yaml:
        with open(config_yaml, "r") as f:
            y = yaml.safe_load(f) or {}
        for k, v in y.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(output_dir, f"explain_{ts}")
    os.makedirs(root, exist_ok=True)

    # ---------- Load & NAT==1 ----------
    df = read_csv_any(input_csv)
    nat_col = find_col_by_names(df.columns, NAT_CANDS)
    if nat_col is None:
        raise ValueError("Could not find 'Neoadjuvant Therapy' column (or aliases).")
    df = df.loc[to01(df[nat_col]) == 1].reset_index(drop=True)

    # ---------- Regimen ----------
    def pick_regimen_label(row) -> str | float:
        hits = []
        for lab, cands in REGIMEN_COLS.items():
            col = find_col_by_names(df.columns, cands)
            if col is not None and row.get(col, 0) == 1:
                hits.append(lab)
        return hits[0] if hits else np.nan

    df["regimen"] = df.apply(pick_regimen_label, axis=1)
    df = df.loc[df["regimen"].isin(["FFX","GemCap","CRT"])].reset_index(drop=True)

    # ---------- Timing ----------
    def derive_timing_series(df_: pd.DataFrame) -> pd.Series:
        cols_ok = [c for c in TIMING_BIN_COLS if c in df_.columns]
        if len(cols_ok) == 3:
            def pick(row):
                if row["Less Than 4 Weeks"] == 1: return "<4w"
                if row["Weeks 4-6"] == 1: return "4-6w"
                if row["Greater Than 6 Weeks"] == 1: return ">6w"
                return np.nan
            return df_.apply(pick, axis=1)
        mds_col = find_col_by_names(df_.columns, MDS_CANDS)
        if mds_col is not None:
            def by_days(v):
                try:
                    w = float(v) / 7.0
                    if w < 4: return "<4w"
                    elif w <= 6: return "4-6w"
                    else: return ">6w"
                except Exception:
                    return np.nan
            return df_[mds_col].map(by_days)
        raise ValueError("Could not find timing bin columns nor MDS column.")
    df["timing_bin"] = derive_timing_series(df)
    df = df.loc[df["timing_bin"].isin(["<4w","4-6w",">6w"])].reset_index(drop=True)

    A_idx = df.apply(lambda r: ARM2IDX[f"{r['regimen']}@{r['timing_bin']}"], axis=1).astype(int).values
    R_idx = df["regimen"].map(reg_map).astype(int).values
    T_idx = df["timing_bin"].map(t_map).astype(int).values

    # ---------- Outcomes & composite ----------
    def resolve_outcome_col(df_, cands):
        col = find_col_by_names(df_.columns, cands)
        if col is None:
            raise ValueError(f"Could not find outcome column from candidates: {cands}")
        return col

    res_y = {k: resolve_outcome_col(df, v) for k, v in OUTCOME_CANDS.items()}
    df[COMP_NAME] = (
        to01(df[res_y["Fistula"]]) |
        to01(df[res_y["Infection"]]) |
        to01(df[res_y["Delayed Gastric Emptying"]]) |
        to01(df[res_y["Death 90 Days"]])
    ).astype(int)
    all_outs = BASE_OUTS + [COMP_NAME]
    Y = {name: (df[COMP_NAME].values.astype(int) if name == COMP_NAME else to01(df[res_y[name]])) for name in all_outs}

    # ---------- Covariates ----------
    exclude_norm = {
        "survival days", "death 365 days",
        "weeks 4 6", "less than 4 weeks", "greater than 6 weeks",
        "minimum days to surgery", "min days to surgery", "days to surgery",
        "folfirinox", "gemcitabine capecitabine", "chemoradiotherapy",
        "neoadjuvant therapy", "nat", "neoadjuvant",
        "regimen", "timing bin", "arm9",
        "mean arterial pressure", "hematocrit",
    }
    covar_cols: List[str] = []
    for c in df.columns:
        if c in res_y.values() or c == COMP_NAME: continue
        if is_id_like(c): continue
        n = norm_name(c)
        if n in exclude_norm: continue
        if n in {"less than 4 weeks", "weeks 4 6", "greater than 6 weeks"}: continue
        if n in {"minimum days to surgery", "min days to surgery", "days to surgery"}: continue
        if c == nat_col: continue
        covar_cols.append(c)
    covar_cols = sorted(covar_cols)

    X_raw = df[covar_cols].copy()
    num_cols, cat_cols = split_feature_types(X_raw, cfg.max_cat_unique)
    pre_X = build_preprocessor(num_cols, cat_cols)
    X_enc = pre_X.fit_transform(X_raw)
    Xdense = X_enc.toarray() if hasattr(X_enc, "toarray") else X_enc

    # ---------- Propensities & weights ----------
    pi_fact = prop_factorized(Xdense, R_idx, T_idx, random_state=cfg.random_state, pi_clip=cfg.pi_clip)
    pi_multi = prop_multinomial(Xdense, A_idx, random_state=cfg.random_state, pi_clip=cfg.pi_clip)
    pi_arm = pi_fact if overlap_score(pi_fact) < overlap_score(pi_multi) else pi_multi
    sw_arm = stabilized_weights(A_idx, pi_arm, trunc_pct=cfg.weight_trunc_pct)

    # ---------- Outcome models (per arm) ----------
    def _fit_outcome_arm(y, A_idx, s, X):
        mask = (A_idx == s)
        clf = GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.8, random_state=cfg.random_state
        )
        clf.fit(X[mask], y[mask])
        p = clf.predict_proba(X)[:, 1]
        return np.clip(p, 1e-6, 1-1e-6), clf

    Mhat: Dict[str, np.ndarray] = {}
    CLF_OUT: Dict[str, List[GradientBoostingClassifier]] = {}
    for out in all_outs:
        yy = Y[out]
        mat = np.zeros((len(yy), 9))
        clfs: List[GradientBoostingClassifier] = []
        for s in range(9):
            ps, clf = _fit_outcome_arm(yy, A_idx, s, Xdense)
            mat[:, s] = ps
            clfs.append(clf)
        Mhat[out] = mat
        CLF_OUT[out] = clfs

    # ---------- Composite weighting & feasibility ----------
    R_tensor = np.stack([Mhat[o] for o in BASE_OUTS], axis=-1)  # (n,9,4)
    mds_weeks = fit_mds_quantiles(df, covar_cols, find_col_by_names(df.columns, MDS_CANDS), random_state=cfg.random_state)
    mds_q = mds_weeks[cfg.mds_feas_q]

    T_R = dict(cfg.t_r_default)
    BUFFER_R = dict(cfg.buffer_r_default)
    if cfg.estimate_durations:
        start_mid = {"<4w":2.0, "4-6w":5.0, ">6w":7.0}
        for r in ["FFX","GemCap","CRT"]:
            maskR = (df["regimen"] == r)
            diffs = []
            for t in ["<4w","4-6w",">6w"]:
                maskRT = maskR & (df["timing_bin"] == t)
                if maskRT.sum() == 0: continue
                diffs.extend(list(mds_q[maskRT] - start_mid[t]))
            if len(diffs) > 0:
                v = np.median(np.clip(diffs, 2.0, 24.0))
                T_R[r] = max(4.0, float(v - BUFFER_R.get(r, 2.0)))

    FEAS = np.zeros((len(df), 9), dtype=bool)
    for r in ["FFX","GemCap","CRT"]:
        for t in ["<4w","4-6w",">6w"]:
            s = ARM2IDX[f"{r}@{t}"]
            FEAS[:, s] = (start_week[t] + T_R[r] + BUFFER_R[r]) <= mds_q

    # weight search
    W_grid = simplex_grid(cfg.weight_step)
    w0 = np.ones(4) / 4.0

    def _evaluate_recommended(y_true, per_arm_pred):
        rec_arm = per_arm_pred.argmin(axis=1)
        risk = eval_policy_vector(y_true, A_idx, pi_arm, sw_arm, rec_arm, Mhat[COMP_NAME])
        return risk, rec_arm

    best_w, best_loss, best_rec = None, np.inf, None
    for w in W_grid:
        comp = (R_tensor * w.reshape(1, 1, -1)).sum(axis=-1)
        comp_pen = comp + (~FEAS) * cfg.penalty
        risk, rec = _evaluate_recommended(Y[COMP_NAME], comp_pen)
        loss = risk + cfg.reg_pref * np.sum((w - w0) ** 2)
        if loss < best_loss:
            best_loss, best_w, best_rec = loss, w.copy(), rec.copy()

    comp_unpen = (R_tensor * best_w.reshape(1, 1, -1)).sum(axis=-1)
    earliest_order = [ARM2IDX[f"{r}@<4w"] for r in ["FFX","GemCap","CRT"]] + \
                     [ARM2IDX[f"{r}@4-6w"] for r in ["FFX","GemCap","CRT"]] + \
                     [ARM2IDX[f"{r}@>6w"] for r in ["FFX","GemCap","CRT"]]
    rec_dnx = np.array([
        guardrail_decision(comp_unpen[i, :], FEAS[i, :], pi_arm[i, :],
                           margin=cfg.margin_thresh, penalty=cfg.penalty, arm_order=earliest_order)
        for i in range(len(df))
    ], dtype=int)
    risk_dnx = eval_policy_vector(Y[COMP_NAME], A_idx, pi_arm, sw_arm, rec_dnx, Mhat[COMP_NAME])

    # fixed-best arm baseline
    arm_means = aipw_values(Y[COMP_NAME], A_idx, pi_arm, Mhat[COMP_NAME], sw_arm)
    fixed_best = int(np.argmin(arm_means))
    risk_fixed = eval_policy_vector(Y[COMP_NAME], A_idx, pi_arm, sw_arm,
                                    np.full(len(df), fixed_best, int), Mhat[COMP_NAME])

    # ---------- (A) Policy-level permutation importance ----------
    np.random.seed(cfg.random_state)
    perm_rows = []
    for c in covar_cols:
        df_perm = X_raw.copy()
        df_perm[c] = np.random.permutation(df_perm[c].values)
        Xp = pre_X.transform(df_perm)
        Xpd = Xp.toarray() if hasattr(Xp, "toarray") else Xp
        # recompute per-arm risks with trained COMP models
        R_perm = []
        for o in BASE_OUTS:
            # reuse COMP models for simplicity of the global policy signal; you can swap to o-specific
            # but keeping aligned with the light-weight approach
            clfs = CLF_OUT[o]
            mat = np.zeros((len(df_perm), 9))
            for s in range(9):
                mat[:, s] = np.clip(clfs[s].predict_proba(Xpd)[:, 1], 1e-6, 1-1e-6)
            R_perm.append(mat)
        R_perm = np.stack(R_perm, axis=-1)
        comp_perm = (R_perm * best_w.reshape(1, 1, -1)).sum(axis=-1)
        comp_perm_pen = comp_perm + (~FEAS) * cfg.penalty
        _, recp = _evaluate_recommended(Y[COMP_NAME], comp_perm_pen)
        riskp = eval_policy_vector(Y[COMP_NAME], A_idx, pi_arm, sw_arm, recp, Mhat[COMP_NAME])
        perm_rows.append({"feature": c, "delta_risk": float(riskp - risk_dnx)})

    pi_tab = pd.DataFrame(perm_rows).sort_values("delta_risk", ascending=False)
    topK = min(20, len(pi_tab))
    pi_tab.head(topK).to_csv(os.path.join(root, "PolicyPermutationImportance_top.csv"), index=False)

    fig = plt.figure(figsize=(8, max(5, 0.35 * topK)))
    ax = fig.add_subplot(111)
    top = pi_tab.head(topK)
    ax.barh(top["feature"][::-1], top["delta_risk"][::-1])
    ax.set_xlabel("Increase in AIPW composite risk after permutation")
    ax.set_title("Policy-level Permutation Importance (Top)")
    savefig(fig, os.path.join(root, "PolicyPermutationImportance.png"))

    # ---------- (B) Variable effects: PD / Δ-Risk & 3D Δ-Risk ----------
    # choose two highest-variance continuous covariates
    num_candidates = [c for c in num_cols if df[c].dropna().nunique() >= 10]
    if len(num_candidates) >= 2:
        var_sorted = sorted(num_candidates, key=lambda c: df[c].astype(float).var(), reverse=True)
        f1, f2 = var_sorted[0], var_sorted[1]
    else:
        pool = num_cols[:2] if len(num_cols) >= 2 else (num_cols + cat_cols)[:2]
        f1, f2 = pool[0], pool[1]

    def _grid_pd_delta(f1, f2, ngrid=30):
        base = {}
        for c in X_raw.columns:
            s = X_raw[c]
            base[c] = s.median() if pd.api.types.is_numeric_dtype(s) else (s.mode().iloc[0] if s.notna().any() else "")
        v1 = np.linspace(df[f1].astype(float).quantile(0.05), df[f1].astype(float).quantile(0.95), ngrid)
        v2 = np.linspace(df[f2].astype(float).quantile(0.05), df[f2].astype(float).quantile(0.95), ngrid)
        V1, V2 = np.meshgrid(v1, v2)
        rows = []
        for i in range(ngrid):
            for j in range(ngrid):
                r = base.copy(); r[f1] = V1[i, j]; r[f2] = V2[i, j]
                rows.append(r)
        grid = pd.DataFrame(rows)[X_raw.columns]
        Xg = pre_X.transform(grid)
        Xgd = Xg.toarray() if hasattr(Xg, "toarray") else Xg
        clfs = CLF_OUT[COMP_NAME]
        m = [np.clip(clfs[s].predict_proba(Xgd)[:, 1], 1e-6, 1-1e-6).reshape(ngrid, ngrid) for s in range(9)]
        idx_early = [ARM2IDX[f"{r}@<4w"] for r in ["FFX","GemCap","CRT"]]
        idx_late  = [ARM2IDX[f"{r}@>6w"] for r in ["FFX","GemCap","CRT"]]
        M_early = np.mean([m[s] for s in idx_early], axis=0)
        M_late  = np.mean([m[s] for s in idx_late ], axis=0)
        Delta = M_early - M_late
        return V1, V2, Delta

    V1, V2, Delta = _grid_pd_delta(f1, f2, ngrid=30)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(V1, V2, Delta, linewidth=0, antialiased=True)
    ax.set_xlabel(f1); ax.set_ylabel(f2); ax.set_zlabel("Δ-Risk (early - late)")
    ax.set_title("3D Δ-Risk surface (<4w vs >6w)")
    savefig(fig, os.path.join(root, "3D_DeltaRisk_surface.png"))

    def _pd_curves(var: str, n: int = 60):
        base = {}
        for c in X_raw.columns:
            s = X_raw[c]
            base[c] = s.median() if pd.api.types.is_numeric_dtype(s) else (s.mode().iloc[0] if s.notna().any() else "")
        xs = np.linspace(df[var].astype(float).quantile(0.05), df[var].astype(float).quantile(0.95), n)
        rows = []
        for x in xs:
            r = base.copy(); r[var] = x
            rows.append(r)
        grid = pd.DataFrame(rows)[X_raw.columns]
        Xg = pre_X.transform(grid); Xgd = Xg.toarray() if hasattr(Xg, "toarray") else Xg
        clfs = CLF_OUT[COMP_NAME]
        idx_e = [ARM2IDX[f"{r}@<4w"] for r in ["FFX","GemCap","CRT"]]
        idx_m = [ARM2IDX[f"{r}@4-6w"] for r in ["FFX","GemCap","CRT"]]
        idx_l = [ARM2IDX[f"{r}@>6w"] for r in ["FFX","GemCap","CRT"]]
        r_e = np.mean([clfs[s].predict_proba(Xgd)[:, 1] for s in idx_e], axis=0)
        r_m = np.mean([clfs[s].predict_proba(Xgd)[:, 1] for s in idx_m], axis=0)
        r_l = np.mean([clfs[s].predict_proba(Xgd)[:, 1] for s in idx_l], axis=0)
        # risk PD
        fig = plt.figure(); ax = fig.add_subplot(111)
        ax.plot(xs, r_e, label="<4w"); ax.plot(xs, r_m, label="4-6w"); ax.plot(xs, r_l, label=">6w")
        ax.set_xlabel(var); ax.set_ylabel("Event risk"); ax.set_title(f"PD curves by timing — {var}"); ax.legend()
        savefig(fig, os.path.join(root, f"PD_risk_{var.replace(' ','_')}.png"))
        # Δ-risk curve
        fig = plt.figure(); ax = fig.add_subplot(111)
        ax.plot(xs, r_e - r_l, "-")
        ax.set_xlabel(var); ax.set_ylabel("Δ-Risk (early - late)"); ax.set_title(f"Δ-Risk vs {var}")
        savefig(fig, os.path.join(root, f"DeltaRisk_{var.replace(' ','_')}.png"))

    _pd_curves(f1); _pd_curves(f2)

    # ---------- (C) Surrogate policy tree ----------
    rec_labels = np.array([format_arm_label(ARM_LIST[i]).replace("@", " ") for i in rec_dnx])
    tree_clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.05, random_state=cfg.random_state)
    tree_clf.fit(Xdense, rec_labels)

    raw_names = _get_feature_names_from_ct(pre_X, Xdense)
    pretty_names = []
    for n in raw_names:
        n2 = re.sub(r"^(num__|cat__|onehot__)", "", str(n))
        n2 = n2.replace("_", " ")
        pretty_names.append(_wrap_long(n2, maxlen=26))

    fig, ax = plt.subplots(figsize=(24, 14))
    plot_tree(
        tree_clf,
        filled=True,
        rounded=True,
        feature_names=pretty_names,
        class_names=np.unique(rec_labels),
        fontsize=10,
        impurity=True,
        proportion=False,
        precision=2,
    )
    plt.title("Policy surrogate tree (CART)", fontsize=14)
    savefig(fig, os.path.join(root, "PolicyTree.png"))

    leaf_id = tree_clf.apply(Xdense)
    leaves = np.unique(leaf_id)
    rows = []
    for lid in leaves:
        m = (leaf_id == lid)
        cov = float(m.mean())
        vc = pd.Series(rec_labels[m]).value_counts(normalize=True)
        maj = float(vc.iloc[0])
        top = str(vc.index[0])
        rows.append({"leaf": int(lid), "coverage": cov, "majority_arm": top, "majority_rate": maj})
    pd.DataFrame(rows).to_csv(os.path.join(root, "PolicyTree_leaves.csv"), index=False)

    # ---------- (D) Individual patient cards ----------
    n_cards = min(cfg.n_patient_cards, len(df))
    card_dir = os.path.join(root, "patient_cards"); os.makedirs(card_dir, exist_ok=True)
    for i in range(n_cards):
        risks = Mhat[COMP_NAME][i, :]
        feas  = FEAS[i, :].astype(int)
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)
        xlbl = [format_arm_label(a) for a in ARM_LIST]
        ax.bar(xlbl, risks, color=["C0" if j != rec_dnx[i] else "C3" for j in range(9)])
        for j in range(9):
            if feas[j] == 0:
                ax.text(j, risks[j] + 0.01, "×", ha="center", va="bottom", fontsize=10)
        plt.xticks(rotation=35, ha="right")
        ax.set_ylabel("Event risk"); ax.set_title(f"Patient #{i+1} — risks & feasibility (× = infeasible)")
        savefig(fig, os.path.join(card_dir, f"patient_{i+1}_risks.png"))

    pd.DataFrame({
        "patient_index": np.arange(len(df)) + 1,
        "recommended_arm": [ARM_LIST[a] for a in rec_dnx]
    }).to_csv(os.path.join(root, "patient_recommendations.csv"), index=False)

    # ---------- (E) Diagnostics ----------
    # Calibration (timing-averaged across regimens)
    cal_dir = os.path.join(root, "calibration"); os.makedirs(cal_dir, exist_ok=True)
    timing_groups = {
        "<4w": [ARM2IDX[f"{r}@<4w"] for r in ["FFX","GemCap","CRT"]],
        "4-6w": [ARM2IDX[f"{r}@4-6w"] for r in ["FFX","GemCap","CRT"]],
        ">6w": [ARM2IDX[f"{r}@>6w"] for r in ["FFX","GemCap","CRT"]],
    }
    for t_label, idxs in timing_groups.items():
        p = np.mean([Mhat[COMP_NAME][:, s] for s in idxs], axis=0)
        y_true = Y[COMP_NAME]
        prob_true, prob_pred = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
        fig = plt.figure(); ax = fig.add_subplot(111)
        ax.plot(prob_pred, prob_true, "o-"); ax.plot([0, 1], [0, 1], "--")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Observed"); ax.set_title(f"Calibration — {t_label}")
        savefig(fig, os.path.join(cal_dir, f"calibration_{t_label}.png"))
        with open(os.path.join(cal_dir, f"calibration_{t_label}_metrics.json"), "w") as f:
            json.dump({"Brier": _brier_score(y_true, p), "ECE": _ece(y_true, p, M=10)}, f, indent=2)

    # Overlap / max propensity
    fig = plt.figure(); ax = fig.add_subplot(111)
    ax.hist(np.max(pi_arm, axis=1), bins=30)
    ax.set_xlabel("max π(A|X)"); ax.set_ylabel("Count"); ax.set_title("Overlap diagnostic: max propensity")
    savefig(fig, os.path.join(root, "diag_overlap_maxpi.png"))

    # Weight histograms
    wei_dir = os.path.join(root, "weights"); os.makedirs(wei_dir, exist_ok=True)
    for s, a in enumerate(ARM_LIST):
        fig = plt.figure(); ax = fig.add_subplot(111)
        ax.hist(sw_arm[:, s], bins=30)
        ax.set_xlabel("weight"); ax.set_ylabel("Count"); ax.set_title(f"Weights — {format_arm_label(a)}")
        savefig(fig, os.path.join(wei_dir, f"weights_{s}.png"))

    # Love plot (numeric only): SMD before vs after weighting
    num_only = [c for c in num_cols]
    g = A_idx.copy()
    w_eff = np.zeros(len(df))
    for s in range(9):
        w_eff[g == s] = sw_arm[g == s, s]
    smd_before, smd_after = [], []
    for c in num_only:
        xv = df[c].astype(float).values
        smd_before.append(_smd_numeric(xv, g, w=None))
        smd_after.append(_smd_numeric(xv, g, w=w_eff))
    love = pd.DataFrame({"feature": num_only, "SMD_before": smd_before, "SMD_after": smd_after})
    love.to_csv(os.path.join(root, "LovePlot_numeric.csv"), index=False)
    show = love.sort_values("SMD_before", ascending=False).head(30)
    fig = plt.figure(figsize=(8, max(6, 0.3 * len(show))))
    ax = fig.add_subplot(111)
    yt = np.arange(len(show))
    ax.plot(show["SMD_before"].values, yt, "o-", label="Before")
    ax.plot(show["SMD_after"].values, yt, "o-", label="After")
    ax.set_yticks(yt); ax.set_yticklabels(show["feature"].values)
    ax.set_xlabel("Absolute SMD (avg pairwise)"); ax.legend(); ax.set_title("Love plot (numeric top 30)")
    savefig(fig, os.path.join(root, "LovePlot_numeric_top30.png"))

    # ESS by arm
    ESS = []
    for s in range(9):
        w = sw_arm[:, s]
        ESS.append((np.sum(w) ** 2) / np.sum(w ** 2))
    pd.DataFrame({"arm": ARM_LIST, "ESS": ESS}).to_csv(os.path.join(root, "ESS_by_arm.csv"), index=False)

    # ---------- (F) Subgroup forest (illustrative) ----------
    sub_dir = os.path.join(root, "subgroup"); os.makedirs(sub_dir, exist_ok=True)
    candidate_vars = [
        "Age","Sex","Diabetes Mellitus","Coronary Artery Disease","End Tidal CO2",
        "Hemoglobin","Potassium","Sodium","Total Protein","White Blood Cell Count"
    ]
    sub_vars = [v for v in candidate_vars if v in df.columns]

    def _aipw_policy_diff(mask: np.ndarray, rec_arm: np.ndarray) -> float:
        y_ = Y[COMP_NAME][mask]; a = A_idx[mask]
        pi = pi_arm[mask]; sw = sw_arm[mask]; mhat = Mhat[COMP_NAME][mask]
        r  = eval_policy_vector(y_, a, pi, sw, rec_arm[mask], mhat)
        rf = eval_policy_vector(y_, a, pi, sw, np.full(mask.sum(), fixed_best, int), mhat)
        return float(r - rf)

    effects = []
    for v in sub_vars:
        s = df[v]
        if pd.api.types.is_numeric_dtype(s):
            thr = np.nanmedian(s.astype(float))
            g1 = (s.astype(float) <= thr); g2 = (s.astype(float) > thr)
            if g1.sum() > 20 and g2.sum() > 20:
                effects.append({"subgroup": f"{v} ≤ median", "diff": _aipw_policy_diff(g1, rec_dnx)})
                effects.append({"subgroup": f"{v} > median", "diff": _aipw_policy_diff(g2, rec_dnx)})
        else:
            vc = s.value_counts()
            for val, cnt in vc.items():
                if cnt < 20: continue
                m = (s == val)
                effects.append({"subgroup": f"{v} = {val}", "diff": _aipw_policy_diff(m, rec_dnx)})

    eff = pd.DataFrame(effects)
    eff.to_csv(os.path.join(sub_dir, "subgroup_effects.csv"), index=False)
    fig = plt.figure(figsize=(8, max(6, 0.35 * len(eff))))
    ax = fig.add_subplot(111)
    ax.plot(eff["diff"].values, np.arange(len(eff)), "o")
    ax.axvline(0, color="gray", ls="--")
    ax.set_yticks(np.arange(len(eff))); ax.set_yticklabels(eff["subgroup"].values)
    ax.set_xlabel("Policy − FixedBest (AIPW composite risk)"); ax.set_title("Subgroup effect (lower is better)")
    savefig(fig, os.path.join(sub_dir, "subgroup_forest.png"))

    # ---------- Summary ----------
    summary = {
        "model": "DRIFT-NAT-X (Explainability)",
        "best_weights": {k: float(v) for k, v in zip(BASE_OUTS, best_w)},
        "composite_aipw_risk": {"DRIFT_NAT_X": float(risk_dnx), "Fixed_best": float(risk_fixed)},
        "top_permutation_importance": pi_tab.head(topK).to_dict(orient="records"),
        "surface_features": [f1, f2],
        "durations_weeks": {"T_R": T_R, "BUFFER_R": BUFFER_R},
        "mds_quantile_q": cfg.mds_feas_q,
        "root_dir": root,
    }
    with open(os.path.join(root, "Explainability_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Step 8 explainability outputs at: {root}")
    return summary
