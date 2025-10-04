import os, json, math
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import yaml

from .config import Config
from .io_utils import read_csv_any, norm_name, find_col_by_names
from .preprocessing import is_id_like, to01, split_feature_types, build_preprocessor
from .propensities import prop_factorized, prop_multinomial, overlap_score, stabilized_weights
from .outcomes import fit_outcome_model_arm, aipw_values
from .policy import simplex_grid, fit_mds_quantiles, eval_policy_vector, guardrail_decision
from .plotting import savefig, bar_with_ci

log = logging.getLogger("driftnatx")
log.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
log.addHandler(ch)

NAT_CANDS = ["Neoadjuvant Therapy", "Neoadjuvant_therapy", "NAT", "Neoadjuvant"]
REGIMEN_COLS = {
    "FFX": ["Folfirinox", "FOLFOXIRI", "FolfiriNox"],
    "GemCap": ["Gemcitabine Capecitabine", "GemCap", "Gemcitabine+Capecitabine"],
    "CRT": ["Chemoradiotherapy", "Chemo-Radiotherapy", "Chemoradiation"]
}
TIMING_BIN_COLS = ["Less Than 4 Weeks", "Weeks 4-6", "Greater Than 6 Weeks"]
MDS_CANDS = ["Minimum Days To Surgery", "min_days_to_surgery", "Days To Surgery"]

OUTCOME_CANDIDATES = {
    "Fistula": ["Fistula"],
    "Infection": ["Infection"],
    "Delayed Gastric Emptying": ["Delayed Gastrric Emptying", "Delayed Gastric Emptying"],
    "Death 90 Days": ["Death 90 Days", "Death 90 Day", "Death90Days"]
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

def run_pipeline(input_csv: str, output_dir: str, cfg: Config, config_yaml: str | None = None) -> Dict:
    if config_yaml:
        with open(config_yaml, "r") as f:
            y = yaml.safe_load(f)
        for k, v in (y or {}).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(output_dir, f"driftnatx_{ts}")
    os.makedirs(root, exist_ok=True)
    per_dir = os.path.join(root, "per_regimen"); os.makedirs(per_dir, exist_ok=True)
    wei_dir = os.path.join(root, "weights"); os.makedirs(wei_dir, exist_ok=True)

    # ---------------------- Load & filter ----------------------
    df = read_csv_any(input_csv)
    nat_col = find_col_by_names(df.columns, NAT_CANDS)
    if nat_col is None:
        raise ValueError("Could not find 'Neoadjuvant Therapy' column (or aliases).")
    df = df.loc[to01(df[nat_col]) == 1].reset_index(drop=True)
    log.info(f"NAT==1 rows: {len(df)}")

    def pick_regimen_label(row) -> str | float:
        hits = []
        for lab, cands in REGIMEN_COLS.items():
            col = find_col_by_names(df.columns, cands)
            if col is not None and row.get(col, 0) == 1:
                hits.append(lab)
        return hits[0] if hits else np.nan

    df["regimen"] = df.apply(pick_regimen_label, axis=1)
    df = df.loc[df["regimen"].isin(["FFX","GemCap","CRT"])].reset_index(drop=True)

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
                    w = float(v)/7.0
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

    # ---------------------- Outcomes & composite ----------------------
    def resolve_outcome_col(df_, cands):
        col = find_col_by_names(df_.columns, cands)
        if col is None:
            raise ValueError(f"Could not find outcome column from candidates: {cands}")
        return col

    res_y = {k: resolve_outcome_col(df, v) for k, v in OUTCOME_CANDIDATES.items()}
    df[COMP_NAME] = (
        to01(df[res_y["Fistula"]]) |
        to01(df[res_y["Infection"]]) |
        to01(df[res_y["Delayed Gastric Emptying"]]) |
        to01(df[res_y["Death 90 Days"]])
    ).astype(int)
    all_outs = BASE_OUTS + [COMP_NAME]
    Y = {name: (df[COMP_NAME].values.astype(int) if name == COMP_NAME else to01(df[res_y[name]])) for name in all_outs}

    # ---------------------- Covariates ----------------------
    exclude_norm = set(cfg.user_exclude) | {
        "survival days", "death 365 days",
        "weeks 4 6", "less than 4 weeks", "greater than 6 weeks",
        "minimum days to surgery", "min days to surgery", "days to surgery",
        "folfirinox", "gemcitabine capecitabine", "chemoradiotherapy",
        "neoadjuvant therapy", "neoadjuvant therapy", "nat", "neoadjuvant",
        "regimen", "timing bin", "arm9"
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
    covar_cols = sorted(c for c in covar_cols if norm_name(c) not in set(cfg.user_exclude))
    log.info("Included covariates: %d", len(covar_cols))

    X_raw = df[covar_cols].copy()
    num_cols, cat_cols = split_feature_types(X_raw, cfg.max_cat_unique)
    pre_X = build_preprocessor(num_cols, cat_cols)
    X_enc = pre_X.fit_transform(X_raw)
    Xdense = X_enc.toarray() if hasattr(X_enc, "toarray") else X_enc

    # ---------------------- Propensities ----------------------
    pi_fact = prop_factorized(Xdense, R_idx, T_idx, random_state=cfg.random_state, pi_clip=cfg.pi_clip)
    pi_multi = prop_multinomial(Xdense, A_idx, random_state=cfg.random_state, pi_clip=cfg.pi_clip)
    pi_arm = pi_fact if overlap_score(pi_fact) < overlap_score(pi_multi) else pi_multi
    sw_arm = stabilized_weights(A_idx, pi_arm, trunc_pct=cfg.weight_trunc_pct)

    # ---------------------- Outcome models ----------------------
    Mhat: Dict[str, np.ndarray] = {}
    AIPW: Dict[str, np.ndarray] = {}
    for out in all_outs:
        y = Y[out]
        m_mat = np.zeros((len(y), 9))
        for s in range(9):
            m_mat[:, s] = fit_outcome_model_arm(
                y=y, A_idx=A_idx, s=s, X_full=Xdense, Xdense=Xdense, R_idx=R_idx, T_idx=T_idx,
                random_state=cfg.random_state
            )
        Mhat[out] = m_mat
        AIPW[out] = aipw_values(y, A_idx, pi_arm, m_mat, sw_arm)

    # ---------------------- Per-regimen timing plots ----------------------
    from .plotting import bar_with_ci
    for out in all_outs:
        for r in ["FFX","GemCap","CRT"]:
            idxs = [ARM2IDX[f"{r}@{t}"] for t in ["<4w","4-6w",">6w"]]
            vals = AIPW[out][idxs]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            bar_with_ci(ax, ["<4w","4-6w",">6w"], list(vals), list(vals-0.0), list(vals+0.0), rotate=15)
            ax.set_title(f"{r}: AIPW event risk by timing — {out}")
            ax.set_ylabel("Event probability")
            savefig(fig, os.path.join(per_dir, f"{r}_timing_{norm_name(out)}.png"))

    # ---------------------- Feasibility (MDS) ----------------------
    mds_col = find_col_by_names(df.columns, MDS_CANDS)
    if mds_col is None:
        raise ValueError("Could not find a Minimum Days To Surgery column.")
    mds_weeks = fit_mds_quantiles(df, covar_cols, mds_col, random_state=cfg.random_state)
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
    for i, r in enumerate(["FFX","GemCap","CRT"]):
        for j, t in enumerate(["<4w","4-6w",">6w"]):
            s = ARM2IDX[f"{r}@{t}"]
            FEAS[:, s] = (start_week[t] + T_R[r] + BUFFER_R[r]) <= mds_q

    # ---------------------- Weight search & guardrail ----------------------
    W_grid = simplex_grid(cfg.weight_step)
    w0 = np.ones(4) / 4.0
    R_tensor = np.stack([Mhat[o][:, :] for o in BASE_OUTS], axis=-1)  # (n,9,4)
    y_comp = Y[COMP_NAME]

    def evaluate_recommended_aipw(y_true, A_idx, pi_arm, sw_arm, per_arm_pred):
        rec_arm = per_arm_pred.argmin(axis=1)
        risk = eval_policy_vector(y_true, A_idx, pi_arm, sw_arm, rec_arm, Mhat[COMP_NAME])
        return risk, rec_arm

    best_w, best_loss, best_rec = None, np.inf, None
    for w in W_grid:
        comp = (R_tensor * w.reshape(1, 1, -1)).sum(axis=-1)
        comp_pen = comp + (~FEAS) * cfg.penalty
        risk, rec = evaluate_recommended_aipw(y_comp, A_idx, pi_arm, sw_arm, comp_pen)
        loss = risk + cfg.reg_pref * np.sum((w - w0) ** 2)
        if loss < best_loss:
            best_loss, best_w, best_rec = loss, w.copy(), rec.copy()

    comp_unpen = (R_tensor * best_w.reshape(1, 1, -1)).sum(axis=-1)
    earliest_order = [ARM2IDX[f"{r}@<4w"] for r in ["FFX","GemCap","CRT"]] + \
                     [ARM2IDX[f"{r}@4-6w"] for r in ["FFX","GemCap","CRT"]] + \
                     [ARM2IDX[f"{r}@>6w"] for r in ["FFX","GemCap","CRT"]]
    rec_arm_dnx = np.array([
        guardrail_decision(comp_unpen[i, :], FEAS[i, :], pi_arm[i, :],
                           margin=cfg.margin_thresh, penalty=cfg.penalty, arm_order=earliest_order)
        for i in range(len(df))
    ], dtype=int)
    risk_dnx = eval_policy_vector(y_comp, A_idx, pi_arm, sw_arm, rec_arm_dnx, Mhat[COMP_NAME])

    fixed_best_arm = int(np.argmin(aipw_values(y_comp, A_idx, pi_arm, Mhat[COMP_NAME], sw_arm)))
    risk_fixed = eval_policy_vector(y_comp, A_idx, pi_arm, sw_arm,
                                    np.full(len(df), fixed_best_arm, int), Mhat[COMP_NAME])

    def earliest_feasible_row(feas_row):
        for arm in earliest_order:
            if feas_row[arm]:
                return arm
        return int(np.argmax(feas_row.astype(int)))

    rec_earliest = np.array([earliest_feasible_row(FEAS[i, :]) for i in range(len(df))], dtype=int)
    risk_earliest = eval_policy_vector(y_comp, A_idx, pi_arm, sw_arm, rec_earliest, Mhat[COMP_NAME])

    # ---------------------- Bootstrap policy risks (fixed-model) ----------------------
    rng = np.random.RandomState(cfg.random_state)

    def bootstrap_policy_risk(policy_rec_arm):
        vals = []
        n = len(policy_rec_arm)
        for _ in range(cfg.boot_b):
            idx = rng.choice(n, n, replace=True)
            r = eval_policy_vector(y_comp[idx], A_idx[idx], pi_arm[idx], sw_arm[idx],
                                   policy_rec_arm[idx], Mhat[COMP_NAME][idx])
            vals.append(r)
        return float(np.mean(vals)), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))

    mean_dnx, lo_dnx, hi_dnx = bootstrap_policy_risk(rec_arm_dnx)
    mean_fix, lo_fix, hi_fix = bootstrap_policy_risk(np.full(len(df), fixed_best_arm, int))
    mean_eft, lo_eft, hi_eft = bootstrap_policy_risk(rec_earliest)

    # ---------------------- Plots & tables ----------------------
    tab = pd.DataFrame({
        "Model": ["DRIFT-NAT-X (ours)", "Fixed best joint arm", "Earliest feasible"],
        "Risk":  [mean_dnx, mean_fix, mean_eft],
        "lo":    [lo_dnx,   lo_fix,   lo_eft],
        "hi":    [hi_dnx,   hi_fix,   hi_eft]
    })
    tab.to_csv(os.path.join(root, "DNX_vs_baselines_with_CI.csv"), index=False)

    fig = plt.figure(figsize=(7.5, 5.5))
    ax = fig.add_subplot(111)
    bar_with_ci(ax, tab["Model"].tolist(), tab["Risk"].tolist(), tab["lo"].tolist(), tab["hi"].tolist(), rotate=15)
    ax.set_ylabel("Event probability (AIPW, composite)")
    ax.set_title("DRIFT-NAT-X vs Baselines (Composite, ±95% CI)")
    savefig(fig, os.path.join(root, "DNX_vs_baselines_with_CI.png"))

    arm_means = aipw_values(Y[COMP_NAME], A_idx, pi_arm, Mhat[COMP_NAME], sw_arm)
    labels = [format_arm_label(a) for a in ARM_LIST]
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    bar_with_ci(ax, labels, list(arm_means), list(arm_means-0.0), list(arm_means+0.0), rotate=35)
    ax.set_title("Per-arm AIPW risk — Composite")
    ax.set_ylabel("Event probability")
    savefig(fig, os.path.join(root, "PerArm_AIPW_risk_composite.png"))

    rec_labels = [format_arm_label(ARM_LIST[k]) for k in rec_arm_dnx]
    vc = pd.Series(rec_labels).value_counts()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.bar(vc.index.astype(str), vc.values)
    ax.set_title("DRIFT-NAT-X recommended arm distribution")
    ax.set_ylabel("N")
    plt.xticks(rotation=35, ha="right")
    savefig(fig, os.path.join(root, "DNX_recommendation_distribution.png"))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(np.max(pi_arm, axis=1), bins=30)
    ax.set_title("Overlap diagnostic: histogram of max π(A|X)")
    ax.set_xlabel("max π(A|X)"); ax.set_ylabel("Count")
    savefig(fig, os.path.join(root, "diag_overlap_maxpi.png"))

    for s, a in enumerate(ARM_LIST):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(stabilized_weights(A_idx, pi_arm)[:, s], bins=30)
        ax.set_title(f"Stabilized truncated weights — {format_arm_label(a)}")
        ax.set_xlabel("weight"); ax.set_ylabel("Count")
        savefig(fig, os.path.join(wei_dir, f"weights_{s}_{format_arm_label(a)}.png"))

    heat = []
    for r in ["FFX","GemCap","CRT"]:
        row = []
        for t in ["<4w","4-6w",">6w"]:
            s = ARM2IDX[f"{r}@{t}"]
            row.append(FEAS[:, s].mean())
        heat.append(row)
    heat = np.array(heat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(heat, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(["<4w","4-6w",">6w"])
    ax.set_yticks(range(3)); ax.set_yticklabels(["FFX","GC","CRT"])
    ax.set_title(f"Feasibility heatmap (mean feasibility by arm, q={cfg.mds_feas_q:.2f})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig(fig, os.path.join(root, "feasibility_heatmap.png"))

    N_SHOW = min(cfg.n_show_gantt, len(df))
    fig = plt.figure(figsize=(10, 0.6*N_SHOW + 2))
    ax = fig.add_subplot(111)
    for i in range(N_SHOW):
        for q in [0.10, 0.50, 0.90]:
            w = mds_weeks[q][i]
            ax.plot([w, w], [i-0.3, i+0.3])
        arm = ARM_LIST[rec_arm_dnx[i]]
        r, t = arm.split("@")
        start = start_week[t]
        ax.plot([start, start + T_R[r] + BUFFER_R[r]], [i, i])
    ax.set_yticks(np.arange(N_SHOW)); ax.set_yticklabels([f"pt{i+1}" for i in range(N_SHOW)])
    ax.set_xlabel("Weeks since diagnosis")
    ax.set_title("Gantt: p10/p50/p90 MDS and DNX NAT+buffer bar")
    savefig(fig, os.path.join(root, "DNX_Gantt_firstN.png"))

    w_df = pd.DataFrame({"Outcome": BASE_OUTS, "Weight": best_w})
    w_df.to_csv(os.path.join(root, "DNX_learned_weights.csv"), index=False)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(BASE_OUTS, best_w)
    ax.set_ylim(0, 1)
    ax.set_title("DRIFT-NAT-X learned outcome weights")
    ax.set_ylabel("Weight")
    savefig(fig, os.path.join(root, "DNX_learned_weights.png"))

    pd.DataFrame({
        "DNX_recommended_arm": [ARM_LIST[k] for k in rec_arm_dnx],
        "DNX_recommended_arm_short": [format_arm_label(ARM_LIST[k]) for k in rec_arm_dnx]
    }).to_csv(os.path.join(root, "DNX_recommended_arms.csv"), index=False)

    meta = {
        "model": "DRIFT-NAT-X (improved)",
        "arms": ARM_LIST,
        "short_labels": [format_arm_label(a) for a in ARM_LIST],
        "learned_weights": {k: float(v) for k, v in zip(BASE_OUTS, best_w)},
        "composite_aipw_risk": {
            "DRIFT_NAT_X": float(mean_dnx),
            "Fixed_best_arm": float(mean_fix),
            "Earliest_feasible": float(mean_eft)
        },
        "CI_95": {
            "DRIFT_NAT_X": [float(lo_dnx), float(hi_dnx)],
            "Fixed_best_arm": [float(lo_fix), float(hi_fix)],
            "Earliest_feasible": [float(lo_eft), float(hi_eft)],
        },
        "durations_weeks": {"T_R": T_R, "BUFFER_R": BUFFER_R},
        "mds_quantile_q": cfg.mds_feas_q,
        "params": {
            "weight_step": cfg.weight_step, "reg_pref": cfg.reg_pref,
            "margin_thresh": cfg.margin_thresh, "overlap_thresh": cfg.overlap_thresh,
            "penalty": cfg.penalty, "boot_b": cfg.boot_b
        },
        "root_dir": root
    }
    with open(os.path.join(root, "DNX_summary.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    pd.DataFrame({"included_covar": covar_cols}).to_csv(os.path.join(root, "Z_included_covariates.csv"), index=False)
    log.info("DONE. Outputs at: %s", root)
    return meta
