import pandas as pd 
import yaml
from pathlib import Path

CFG = yaml.safe_load(open("configs/path_column.yaml"))
DELIV = Path(CFG["paths"]["deliveries_csv"])
MATCH = Path(CFG["paths"]["matches_csv"])
OUT   = Path(CFG["paths"]["scorecards_parquet"])

mid_matches = CFG["keys"]["match_id_in_matches"]          # "id"
mid_deliv   = CFG["keys"]["match_id_in_deliveries"]       # "match_id"

deliveries = pd.read_csv(DELIV)
matches    = pd.read_csv(MATCH)

# Filter matches to 2024 only
matches_24 = matches[matches["season"]=="2024"].copy()
matches_24 = matches_24.rename(columns={mid_matches: "match_id"})

# Per-ball glue
deliveries["runs_total"] = deliveries.get("total_runs", deliveries.get("runs",0))
deliveries["is_wicket"]  = deliveries.get("is_wicket", 0)

# Innings aggregates
inn = deliveries.groupby([mid_deliv, "inning"]).agg(
    runs=("runs_total","sum"),
    wkts=("is_wicket","sum")
).reset_index().rename(columns={mid_deliv:"match_id"})

def pick(df, inn_no):
    s = df[df["inning"]==inn_no]
    return (int(s["runs"].sum()) if len(s) else None,
            int(s["wkts"].sum()) if len(s) else None)

rows=[]
for mid, g in inn.groupby("match_id"):
    r1,w1 = pick(g,1); r2,w2 = pick(g,2)
    rows.append({
        "match_id": mid,
        "first_innings_runs": r1, "first_innings_wkts": w1,
        "second_innings_runs": r2, "second_innings_wkts": w2
    })
base = pd.DataFrame(rows)

# Top batters
bat = deliveries.groupby([mid_deliv,"batter"]).agg(
    runs=("batsman_runs","sum"),
    balls=("batter","count")
).reset_index().rename(columns={mid_deliv:"match_id"})
top_bat = (bat.sort_values(["match_id","runs","balls"], ascending=[True,False,True])
             .groupby("match_id")
             .apply(lambda g: g.head(3)[["batter","runs","balls"]].to_dict("records"))
             .reset_index(name="top_batters"))

# Top bowlers (simple)
bowl = deliveries.groupby([mid_deliv,"bowler"]).agg(
    runs_conceded=("runs_total","sum"),
    balls=("bowler","count"),
    wkts=("is_wicket","sum")
).reset_index().rename(columns={mid_deliv:"match_id"})
bowl["overs"] = bowl["balls"].apply(lambda b: f"{b//6}.{b%6}")
top_bowl = (bowl.sort_values(["match_id","wkts","runs_conceded"], ascending=[True,False,True])
              .groupby("match_id")
              .apply(lambda g: g.head(3)[["bowler","wkts","runs_conceded","overs"]].to_dict("records"))
              .reset_index(name="top_bowlers"))

meta_cols = [c for c in ["match_id","team1","team2","date","venue","toss_winner",
                         "toss_decision","winner","result","result_margin"] if c in matches_24.columns]
meta = matches_24[meta_cols].drop_duplicates("match_id")

scorecards = (meta.merge(base, on="match_id", how="left")
                   .merge(top_bat, on="match_id", how="left")
                   .merge(top_bowl, on="match_id", how="left"))

OUT.parent.mkdir(parents=True, exist_ok=True)
scorecards.to_parquet(OUT, index=False)
print("scorecards_per_match.parquet:", len(scorecards))