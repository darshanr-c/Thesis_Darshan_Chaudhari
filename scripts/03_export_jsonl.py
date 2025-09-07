# scripts/03_export_jsonl.py
import json
import ast
import numpy as np
import pandas as pd
from pathlib import Path
import yaml


# Load config & paths
CFG = yaml.safe_load(open("configs/path_column.yaml"))

COMM_PARQ   = Path(CFG["paths"]["commentary_parquet"])
SC_PARQ     = Path(CFG["paths"]["scorecards_parquet"])
COMM_JSONL  = Path(CFG["paths"]["commentary_jsonl"])
SC_JSONL    = Path(CFG["paths"]["scorecards_jsonl"])
MANIFEST    = Path(CFG["paths"]["manifest_jsonl"])
# NOTE: reports_jsonl is created by you manually with good summaries; not written here.


# Helpers: coerce to JSON-safe Python types
def to_py(x):
    """Convert numpy/pandas types to pure Python for json.dumps."""
    if isinstance(x, (np.integer,)):        return int(x)
    if isinstance(x, (np.floating,)):
        if np.isnan(x):                     return None
        return float(x)
    if isinstance(x, (np.bool_,)):          return bool(x)
    if isinstance(x, (np.ndarray,)):        return x.tolist()
    # pandas/np datetime-like
    if hasattr(x, "isoformat"):
        try:                                return x.isoformat()
        except Exception:                   return str(x)
    return x

def ensure_list_of_str(x):
    """Ensure value is a list[str]. Accepts list/ndarray/JSON-string."""
    if isinstance(x, list):
        return [str(i) for i in x]
    if isinstance(x, np.ndarray):
        return [str(i) for i in x.tolist()]
    if isinstance(x, str):
        # try JSON first
        try:
            obj = json.loads(x)
            if isinstance(obj, list):
                return [str(i) for i in obj]
        except Exception:
            pass
        return [x] if x else []
    return []

def ensure_list_of_dicts(x):
    """Ensure value is a list[dict]. Accepts list/ndarray/JSON/repr-string."""
    if isinstance(x, list):
        out = []
        for item in x:
            if isinstance(item, dict):
                out.append({k: to_py(v) for k, v in item.items()})
            else:
                out.append(to_py(item))
        return out
    if isinstance(x, np.ndarray):
        return ensure_list_of_dicts(x.tolist())
    if isinstance(x, str):
        # JSON then literal_eval as fallback
        try:
            obj = json.loads(x)
            if isinstance(obj, list):
                return ensure_list_of_dicts(obj)
        except Exception:
            try:
                obj = ast.literal_eval(x)
                if isinstance(obj, list):
                    return ensure_list_of_dicts(obj)
            except Exception:
                return []
    return []

def is_na(v):
    """Robust NaN/None/empty detection for mixed dtypes."""
    # Empty lists/arrays are NA for our purposes
    if isinstance(v, (list, np.ndarray)):
        return len(v) == 0
    try:
        return pd.isna(v)
    except Exception:
        return v is None


# Load inputs
comm = pd.read_parquet(COMM_PARQ)
sc   = pd.read_parquet(SC_PARQ)

# Keep only matches present in BOTH
common_ids = sorted(set(comm["match_id"]).intersection(set(sc["match_id"])))
comm = comm[comm["match_id"].isin(common_ids)].reset_index(drop=True)
sc   = sc[sc["match_id"].isin(common_ids)].reset_index(drop=True)


# Write commentary.jsonl
COMM_JSONL.parent.mkdir(parents=True, exist_ok=True)
with open(COMM_JSONL, "w", encoding="utf-8") as f:
    for _, r in comm.iterrows():
        chunks = ensure_list_of_str(r.get("commentary_chunks", []))
        obj = {
            "match_id": int(to_py(r["match_id"])),
            "commentary_chunks": chunks,
            "commentary_full": str(r.get("commentary_full", ""))
        }
        f.write(json.dumps(obj, ensure_ascii=False, default=to_py) + "\n")


# Write scorecards.jsonl
keep_cols = [
    "match_id","team1","team2","date","venue","toss_winner","toss_decision",
    "winner","result","result_margin",
    "first_innings_runs","first_innings_wkts",
    "second_innings_runs","second_innings_wkts",
    "top_batters","top_bowlers"
]
# Filter to existing columns only (handles slight schema differences)
keep_cols = [c for c in keep_cols if c in sc.columns]
sc_use = sc[keep_cols].copy()

with open(SC_JSONL, "w", encoding="utf-8") as f:
    for _, r in sc_use.iterrows():
        stats = {}
        for k in keep_cols:
            if k == "match_id":
                continue
            v = r[k]
            if is_na(v):
                continue
            if k in ("top_batters", "top_bowlers"):
                stats[k] = ensure_list_of_dicts(v)
            else:
                stats[k] = to_py(v)

        obj = {
            "match_id": int(to_py(r["match_id"])),
            "stats": stats
        }
        f.write(json.dumps(obj, ensure_ascii=False, default=to_py) + "\n")


# Write manifest of match_ids
with open(MANIFEST, "w", encoding="utf-8") as f:
    for mid in common_ids:
        f.write(json.dumps({"match_id": int(to_py(mid))}, ensure_ascii=False) + "\n")

print(f"✓ Wrote {COMM_JSONL} ({len(comm)})")
print(f"✓ Wrote {SC_JSONL} ({len(sc_use)})")
print(f"✓ Wrote {MANIFEST} ({len(common_ids)})")
print("Reminder: create data_processed/reports.jsonl with your good summaries for training.")