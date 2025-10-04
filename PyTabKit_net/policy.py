from typing import Dict, Tuple
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

def simplex_grid(step=0.05) -> np.ndarray:
    vals = np.arange(0.0, 1.0 + 1e-9, step)
    ws = []
    for a in vals:
        for b in vals:
            for c in vals:
                d = 1.0 - (a+b+c)
                if d < -1e-9:
                    continue
                if d < 0:
                    d = 0.0
                if abs(a+b+c+d - 1.0) <= 1e-9:
                    ws.append([a, b, c, d])
    return np.array(ws)

def fit_mds_quantiles(df, covar_cols, mds_col_name, random_state=42) -> Dict[float, np.ndarray]:
    y_mds = df[mds_col_name].astype(float).values
    Xb = df[covar_cols].copy()
    from .preprocessing import split_feature_types, build_preprocessor
    nb, cb = split_feature_types(Xb, 50)
    pre_b = build_preprocessor(nb, cb)
    Xb_enc = pre_b.fit_transform(Xb)
    preds = {}
    for q in [0.10, 0.25, 0.50, 0.90]:
        gbr = GradientBoostingRegressor(loss="quantile", alpha=q, random_state=random_state)
        gbr.fit(Xb_enc, y_mds)
        preds[q] = gbr.predict(Xb_enc) / 7.0  # convert days to weeks
    return preds

def eval_policy_vector(y_true, A_idx, pi_arm, sw_arm, rec_arm, mhat_comp):
    numer_tot, denom_tot = 0.0, 0.0
    for s in range(9):
        mask = (rec_arm == s)
        if mask.sum() == 0:
            continue
        m = mhat_comp[:, s]
        numer_tot += np.sum((m + ((A_idx == s) * (y_true - m) / pi_arm[:, s])) * sw_arm[:, s] * mask)
        denom_tot += np.sum(sw_arm[:, s] * mask)
    denom_tot = max(denom_tot, 1e-12)
    return numer_tot / denom_tot

def guardrail_decision(r_vec, feas_vec, pi_vec, margin=0.01, penalty=0.05, arm_order=None) -> int:
    r_pen = r_vec + (~feas_vec)*penalty
    order = np.argsort(r_pen)
    best, second = order[0], order[1]
    gap = r_pen[second] - r_pen[best]
    if (gap < margin) or (pi_vec.max() < 0.10):
        if arm_order is None:
            arm_order = list(range(9))
        for arm in arm_order:
            if feas_vec[arm]:
                return arm
    return int(best)
