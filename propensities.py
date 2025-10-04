import numpy as np
from sklearn.linear_model import LogisticRegression

def truncate_weights(w, pct=0.99):
    hi = np.quantile(w, pct)
    lo = np.quantile(w, 1.0 - pct)
    return np.clip(w, lo, hi)

def prop_factorized(X, R_idx, T_idx, random_state=42, pi_clip=1e-3):
    clf_R = LogisticRegression(multi_class="multinomial", solver="lbfgs",
                               random_state=random_state, max_iter=2000)
    clf_R.fit(X, R_idx)
    pi_R = clf_R.predict_proba(X)
    pi_T_given_R = np.zeros((len(R_idx), 3, 3))
    for ridx in range(3):
        mask = (R_idx == ridx)
        clf_T = LogisticRegression(multi_class="multinomial", solver="lbfgs",
                                   random_state=random_state, max_iter=2000)
        clf_T.fit(X[mask], T_idx[mask])
        pi_T_given_R[:, ridx, :] = clf_T.predict_proba(X)
    pi = np.zeros((len(R_idx), 9))
    for i, _ in enumerate(["FFX","GemCap","CRT"]):
        for j, _ in enumerate(["<4w","4-6w",">6w"]):
            arm = i*3 + j
            pi[:, arm] = pi_R[:, i] * pi_T_given_R[:, i, j]
    return np.clip(pi, pi_clip, 1.0 - pi_clip)

def prop_multinomial(X, A_idx, random_state=42, pi_clip=1e-3):
    clf = LogisticRegression(multi_class="multinomial", solver="lbfgs",
                             random_state=random_state, max_iter=2000)
    clf.fit(X, A_idx)
    pi = clf.predict_proba(X)
    return np.clip(pi, pi_clip, 1.0 - pi_clip)

def overlap_score(pi):
    return float(np.mean(np.max(pi, axis=1)))

def stabilized_weights(A_idx, pi_arm, trunc_pct=0.99):
    p_marg = np.bincount(A_idx, minlength=9) / float(len(A_idx))
    sw_arm = p_marg / pi_arm
    for s in range(9):
        sw_arm[:, s] = truncate_weights(sw_arm[:, s], pct=trunc_pct)
    return sw_arm
