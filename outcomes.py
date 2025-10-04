import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

def fit_outcome_model_arm(y, A_idx, s, X_full, Xdense, R_idx, T_idx, random_state=42,
                          min_n=30, min_events=5):
    mask = (A_idx == s)
    n = int(mask.sum())
    if n >= min_n and y[mask].sum() >= min_events and (n - y[mask].sum()) >= min_events:
        clf = GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.8, random_state=random_state
        )
        clf.fit(X_full[mask], y[mask])
        p = clf.predict_proba(X_full)[:, 1]
        return np.clip(p, 1e-6, 1-1e-6)
    # pooled fallback (by regimen + timing dummies)
    ridx = s // 3
    maskR = (R_idx == ridx)
    T_onehot = np.zeros((len(y), 3)); T_onehot[np.arange(len(y)), T_idx] = 1
    Z = np.hstack([Xdense, T_onehot])
    clf = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=3,
        subsample=0.8, random_state=random_state
    )
    clf.fit(Z[maskR], y[maskR])
    p = clf.predict_proba(Z)[:, 1]
    return np.clip(p, 1e-6, 1-1e-6)

def aipw_values(y, A_idx, pi_arm, m_hat_all, sw_arm):
    vals = []
    for s in range(9):
        m = m_hat_all[:, s]
        numer = (m + ((A_idx == s) * (y - m) / pi_arm[:, s])) * sw_arm[:, s]
        denom = sw_arm[:, s]
        vals.append(numer.sum() / denom.sum())
    return np.array(vals)
