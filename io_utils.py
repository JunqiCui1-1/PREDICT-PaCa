import re
import pandas as pd
from typing import List

def read_csv_any(path: str) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "gbk", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            pass
    return pd.read_csv(path, low_memory=False)

def norm_name(s: str) -> str:
    s = str(s).lower().strip()
    s = s.replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s

def find_col_by_names(df_cols, candidates: List[str]):
    cnorms = [norm_name(c) for c in df_cols]
    for cand in candidates:
        n = norm_name(cand)
        if n in cnorms:
            return df_cols[cnorms.index(n)]
    return None
