from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import inspect

ID_KEYS = {"id","subject_id","patient_id","mrn","patientunitstayid"}

def is_id_like(col: str) -> bool:
    lc = col.lower()
    return (lc in ID_KEYS) or lc.endswith("_id")

def to01(series: pd.Series) -> np.ndarray:
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        vals = s.fillna(0).astype(float)
        uniq = np.unique(vals[~np.isnan(vals)])
        if set(uniq.astype(int)) <= {0, 1}:
            return vals.astype(int)
        return (vals > 0).astype(int)
    ss = s.astype(str).str.strip().str.lower()
    return ss.isin({"1","true","t","yes","y"}).astype(int)

def split_feature_types(dfX: pd.DataFrame, max_cat_unique: int) -> Tuple[List[str], List[str]]:
    num_cols, cat_cols = [], []
    for c in dfX.columns:
        s = dfX[c]
        if pd.api.types.is_numeric_dtype(s):
            if (not np.issubdtype(s.dtype, np.floating)) and s.dropna().nunique() <= max_cat_unique:
                cat_cols.append(c)
            else:
                num_cols.append(c)
        else:
            if s.astype(str).nunique() <= max_cat_unique:
                cat_cols.append(c)
    return num_cols, cat_cols

def make_ohe_kwargs():
    kw = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        kw["sparse_output"] = True
    else:
        kw["sparse"] = True
    return kw

def build_preprocessor(num_cols, cat_cols) -> ColumnTransformer:
    ohe_kwargs = make_ohe_kwargs()
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(**ohe_kwargs)),
    ])
    return ColumnTransformer(
        transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
        sparse_threshold=1.0
    )
