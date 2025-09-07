import os, json, argparse, multiprocessing, re
from typing import List, Dict, Any
from llama_cpp import Llama

# --- minimal set of scorecard fields to keep prompt short & factual ---
ESSENTIAL_KEYS = {
    "team1","team2","winner","result","result_margin","venue","date",
    "toss_winner","toss_decision",
    "first_innings_runs","first_innings_wkts",
    "second_innings_runs","second_innings_wkts",
}

KEYWORDS_HIGH = ("wicket", "out", "caught", "bowled", "lbw", "review", "drs", "six", "four", "over", "powerplay", "death", "target", "needed")

def compute_loser(stats: dict) -> str:
    t1, t2, w = stats.get("team1",""), stats.get("team2",""), stats.get("winner","")
    if not w: return ""
    return t2 if w == t1 else t1

def build_facts(stats_c: dict) -> str:
    loser = compute_loser(stats_c)
    fi = f'{stats_c.get("first_innings_runs")}/{stats_c.get("first_innings_wkts")}' \
         if stats_c.get("first_innings_runs") is not None and stats_c.get("first_innings_wkts") is not None else "NA"
    si = f'{stats_c.get("second_innings_runs")}/{stats_c.get("second_innings_wkts")}' \
         if stats_c.get("second_innings_runs") is not None and stats_c.get("second_innings_wkts") is not None else "NA"
    facts = {
        "winner": stats_c.get("winner",""),
        "loser": loser,
        "result": stats_c.get("result",""),
        "result_margin": stats_c.get("result_margin",""),
        "venue": stats_c.get("venue",""),
        "date": stats_c.get("date",""),
        "toss_winner": stats_c.get("toss_winner",""),
        "toss_decision": stats_c.get("toss_decision",""),
        "team1": stats_c.get("team1",""),
        "team2": stats_c.get("team2",""),
        "first_innings": fi,
        "second_innings": si
    }
    return json.dumps(facts, ensure_ascii=False)

def load_jsonl(path: str) -> dict:
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            data[o["match_id"]] = o
    return data

def compact_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    s = {k: stats.get(k) for k in ESSENTIAL_KEYS if k in stats}
    if "top_batters" in stats:
        s["top_batters"] = stats["top_batters"][:2]
    if "top_bowlers" in stats:
        s["top_bowlers"] = stats["top_bowlers"][:2]
    # drop null/empty
    return {k: v for k, v in s.items() if v not in (None, "", [], {})}

def pick_chunks(chunks: List[str], want: int) -> List[str]:
    """Keyword-prioritized head+tail pick to keep high-signal lines."""
    if len(chunks) <= want:
        return chunks

    # score lines by keyword hits (case-insensitive)
    def score(s: str) -> int:
        t = s.lower()
        return sum(1 for kw in KEYWORDS_HIGH if kw in t)

    scored = [(score(c), i, c) for i, c in enumerate(chunks)]
    scored.sort(key=lambda x: (x[0], -x[1]))   # prefer later lines on tie to capture finish
    # take half from best keyworded, half from timeline edges
    k = max(1, want // 2)
    best_kw = [c for _, _, c in scored[::-1][:k]]  # top by score
    half = want - len(best_kw)
    head_tail = chunks[: half // 2] + chunks[-(half - half // 2):] if half > 0 else []
    # preserve order: head, best_kw (in original order), tail
    def orig_order(seq): 
        pos = {c:i for i,c in enumerate(chunks)}
        return sorted(seq, key=lambda x: pos.get(x, 10**9))
    merged = orig_order(head_tail + best_kw)
    # ensure unique, preserve order
    seen = set(); final=[]
    for c in merged:
        if c not in seen:
            seen.add(c); final.append(c)
    return final[:want]

def system_prompt() -> str:
    return (
        "You are a cricket Expert. "
        "NEVER contradict provided FACTS. "
        "If a detail is not in FACTS or EXCERPTS, OMIT it. "
        "Use concise, neutral language. "
        "Avoid invented players, scores, or events."
    )

def user_prefix(stats_c: Dict[str, Any]) -> str:
    facts = build_facts(stats_c)
    rules = (
        "Write a concise IPL match summary with exactly these 5 sections:\n"
        "1) Summary Opening (winner, margin, context)\n"
        "2) Turning Events (crucial moments)\n"
        "3) Standout Performances (one batter, one bowler)\n"
        "4) Short Details (toss, DRS, injuries, pitch)\n"
        "5) Summary Closing (implication/next match)\n\n"
        "Rules:\n"
        "- 180–220 words total.\n"
        "- Copy names/numbers EXACTLY from FACTS when used.\n"
        "- Do NOT invent facts beyond FACTS/EXCERPTS.\n"
        "- REQUIRED opening sentence template (fill with FACTS):\n"
        "  \"{winner} beat {loser} by {result_margin} at {venue}.\"\n"
    )
    return (
        f"### FACTS (copy exactly when referenced)\n{facts}\n\n"
        f"### TASK\n{rules}\n"
        "### COMMENTARY EXCERPTS (chronological)\n"
    )

def token_count(llm: Llama, text: str) -> int:
    return len(llm.tokenize(text.encode("utf-8")))

def fit_to_budget_chat(llm: Llama, sys: str, user_pref: str, chunks_all: List[str],
                       n_ctx: int, max_new: int, buffer_tokens: int,
                       min_chunks: int) -> str:
    chs = chunks_all[:]
    def build_user(chs_: List[str]) -> str:
        return user_pref + "\n\n".join(chs_) + "\n\nREPORT:"
    while True:
        user_text = build_user(chs)
        joined = f"<s>system: {sys}\nuser: {user_text}"
        need = token_count(llm, joined) + max_new + buffer_tokens
        if need <= n_ctx:
            return user_text
        if len(chs) > min_chunks:
            # drop middle line to save tokens while keeping start/end flow
            mid = len(chs)//2
            chs.pop(mid)
        else:
            # final fallback: drop commentary completely, rely on scorecard
            return user_pref + "\n\nREPORT:"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ABSOLUTE path to .gguf")
    ap.add_argument("--commentary", default="data_processed/commentary.jsonl")
    ap.add_argument("--scorecards", default="data_processed/scorecards.jsonl")
    ap.add_argument("--match_id", type=int, required=True)
    ap.add_argument("--n_ctx", type=int, default=1536)         # safe for M1; raise to 2048 if it fits
    ap.add_argument("--n_gpu_layers", type=int, default=24)    # reduce if OOM; increase if it fits
    ap.add_argument("--max_chunks", type=int, default=6)
    ap.add_argument("--min_chunks", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--budget_buffer", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.15)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=40)
    args = ap.parse_args()

    # Load data
    comm = load_jsonl(args.commentary)
    sc   = load_jsonl(args.scorecards)
    if args.match_id not in comm or args.match_id not in sc:
        raise SystemExit(f"match_id {args.match_id} not found in both files.")
    chunks_all = comm[args.match_id]["commentary_chunks"]
    stats_c = compact_stats(sc[args.match_id]["stats"])

    # Init LLM first (to use llm.tokenize for budgeting)
    n_threads = min(8, max(1, (multiprocessing.cpu_count() or 8)))
    model_path = os.path.expanduser(os.path.expandvars(args.model))
    llm = Llama(
        model_path=model_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=n_threads,
        verbose=False
    )

    # Build messages with token budgeting
    sys = system_prompt()
    selected = pick_chunks(chunks_all, args.max_chunks)
    user_pref = user_prefix(stats_c)
    user_msg = fit_to_budget_chat(
        llm=llm, sys=sys, user_pref=user_pref, chunks_all=selected,
        n_ctx=args.n_ctx, max_new=args.max_new_tokens, buffer_tokens=args.budget_buffer,
        min_chunks=args.min_chunks
    )

    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": user_msg}
    ]

    # Generate via chat API (uses Llama-2 chat template from GGUF)
    resp = llm.create_chat_completion(
        messages=messages,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        repeat_penalty=1.1,
        stream=False
    )
    
    def passes_fact_check(text: str, stats: dict) -> bool:
        # must contain winner and margin literally
        w = stats.get("winner","")
        m = str(stats.get("result_margin",""))
        return (w in text) and (m in text)

    text = resp["choices"][0]["message"]["content"].strip()

    # Retry up to 2 times if fact check fails or text is too short
    attempt = 1
    while (not passes_fact_check(text, stats_c) or len(text.split()) < 120) and attempt <= 2:
        resp = llm.create_chat_completion(
            messages=[
                {"role":"system","content": sys},
                {"role":"user","content": user_msg},
                {"role":"assistant","content":"(previous attempt contradicted FACTS / too short)"},
                {"role":"user","content":"Regenerate the report. Follow the REQUIRED opening sentence exactly and keep to 180–220 words."}
        ],
        temperature=max(0.05, args.temperature - 0.05*attempt),
        top_p=0.85,
        top_k=30,
        max_tokens=args.max_new_tokens,
        repeat_penalty=1.15,
        stream=False
    )
    text = resp["choices"][0]["message"]["content"].strip()
    attempt += 1
    
    print("\n=== GENERATED REPORT ===\n")
    print(text)

if __name__ == "__main__":
    main()