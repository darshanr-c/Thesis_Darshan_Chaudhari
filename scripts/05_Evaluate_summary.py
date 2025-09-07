# scripts/evaluate_reports.py
# Minimal scorer: factual presence checks + ROUGE-L (pure Python, no extra deps)

import json
import argparse
import re

def load_jsonl(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            data[o["match_id"]] = o
    return data

def lcs_len(a_tokens, b_tokens):
    # classic DP (O(n*m)), fine for short texts
    n, m = len(a_tokens), len(b_tokens)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        ai = a_tokens[i-1]
        for j in range(1, m+1):
            if ai == b_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[n][m]

def rouge_l(pred, ref):
    # whitespace tokenization
    p = pred.split()
    r = ref.split()
    if not p or not r:
        return {"r": 0.0, "p": 0.0, "f": 0.0}
    lcs = lcs_len(p, r)
    rec = lcs / len(r)
    prec = lcs / len(p)
    if rec + prec == 0:
        f = 0.0
    else:
        beta = (len(r)/len(p)) if len(p) > 0 else 1.0
        f = (1 + beta**2) * (prec * rec) / (rec + beta**2 * prec + 1e-12)
    return {"r": rec, "p": prec, "f": f}

def contains_literal(text, value):
    if not value:
        return True  # treat missing fact as not-required
    # case-insensitive substring match, collapse spaces
    t = re.sub(r"\s+", " ", text).lower()
    v = re.sub(r"\s+", " ", str(value)).lower()
    return v in t

def compute_loser(stats):
    t1, t2, w = stats.get("team1",""), stats.get("team2",""), stats.get("winner","")
    if not w: return ""
    return t2 if w == t1 else t1

def fact_checks(pred_text, stats):
    winner = stats.get("winner","")
    margin = str(stats.get("result_margin",""))
    venue  = stats.get("venue","")
    toss_winner  = stats.get("toss_winner","")
    toss_decision= stats.get("toss_decision","")
    loser = compute_loser(stats)

    checks = {
        "has_winner": contains_literal(pred_text, winner),
        "has_margin": contains_literal(pred_text, margin),
        "has_venue":  contains_literal(pred_text, venue),
        "has_toss":   contains_literal(pred_text, toss_winner) and contains_literal(pred_text, toss_decision),
        "has_loser":  contains_literal(pred_text, loser) if loser else True
    }
    checks["all_pass"] = all(checks.values())
    return checks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated_path", required=True, help="Path to a text file containing the generated report")
    ap.add_argument("--scorecards", default="data_processed/scorecards.jsonl")
    ap.add_argument("--reports",    default="data_processed/reports.jsonl")
    ap.add_argument("--match_id", type=int, required=True)
    args = ap.parse_args()

    # load inputs
    with open(args.generated_path, "r", encoding="utf-8") as f:
        pred = f.read().strip()

    sc  = load_jsonl(args.scorecards)
    rpt = load_jsonl(args.reports)

    if args.match_id not in sc:
        raise SystemExit(f"match_id {args.match_id} not found in scorecards.")
    if args.match_id not in rpt:
        raise SystemExit(f"match_id {args.match_id} not found in reports (gold).")

    stats = sc[args.match_id]["stats"]
    gold  = rpt[args.match_id]["report_text"]

    facts = fact_checks(pred, stats)
    rl    = rouge_l(pred, gold)

    print("=== FACT CHECKS ===")
    for k,v in facts.items():
        print(f"{k}: {v}")
    print("\n=== ROUGE-L ===")
    print(f"recall={rl['r']:.3f}  precision={rl['p']:.3f}  F1={rl['f']:.3f}")

if __name__ == "__main__":
    main()