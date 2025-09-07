import pandas as pd
import yaml
from pathlib import Path

CFG = yaml.safe_load(open("configs/path_column.yaml"))
XLSX = Path(CFG["paths"]["raw_excel"])
OUT  = Path(CFG["paths"]["commentary_parquet"])

mid = CFG["keys"]["match_id_xlsx"]       # "matchID"
C   = CFG["commentary_columns"]
inn, ov, ball, text = C["innings"], C["over"], C["ball"], C["text"]

df = pd.read_excel(XLSX)
need = [mid, inn, ov, ball, text]
missing = [c for c in need if c not in df.columns]
if missing:
    raise ValueError(f"Missing in Excel: {missing}")

def clean(s):
    if not isinstance(s,str): return ""
    s = s.replace("\u00a0"," ").strip()
    bad = ("End of over", "Drinks", "Strategic timeout")
    return "" if (not s or any(s.startswith(b) for b in bad)) else s

df["cmt"] = df[text].apply(clean)
df = df[df["cmt"].str.len()>0].copy()

for c in [inn, ov, ball]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=[mid, inn, ov, ball])
df = df.sort_values([mid, inn, ov, ball])

def make_chunks(lines, max_words, overlap):
    chunks=[]; cur=[]; wcount=0
    for line in lines:
        w=len(line.split())
        if cur and (wcount+w>max_words):
            chunks.append(" ".join(cur))
            if overlap>0:
                tail=" ".join(" ".join(cur).split()[-overlap:])
                cur=[tail]; wcount=len(tail.split())
            else:
                cur=[]; wcount=0
        cur.append(line); wcount+=w
    if cur: chunks.append(" ".join(cur))
    return chunks

MAXW = CFG["chunking"]["max_words"]
OVLP = CFG["chunking"]["overlap_words"]

rows=[]
for mid_val, g in df.groupby(mid):
    lines = g["cmt"].tolist()
    chunks = make_chunks(lines, MAXW, OVLP)
    rows.append({
        "match_id": int(mid_val),
        "n_lines": len(lines),
        "commentary_chunks": chunks,
        "commentary_full": " ".join(lines)   # optional
    })

comm = pd.DataFrame(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
comm.to_parquet(OUT, index=False)
print("commentary_per_match.parquet:", len(comm))