"""Aggregate an experiment folder across repeats into Markdown tables.

    .venv/bin/python scripts/analyze_matrix.py results/matrix   # writes results/matrix/summary.md

Groups runs by (planner, config, model); reports mean and min-max over repeats.
For P2 it also scores the SLM's goal translation against the episode's true goal.
"""

import glob
import json
import os
import statistics as st
import sys
from collections import defaultdict

ORDER = {"classical_only": 0, "slm_only": 1, "slm_classical": 2}
NAMES = {"classical_only": "P1 classical", "slm_only": "P0 SLM-only", "slm_classical": "P2 SLM+classical"}


def load(root):
    runs = []
    for d in glob.glob(os.path.join(root, "*", "*")):
        if not os.path.exists(os.path.join(d, "summary.json")):
            continue
        cfg = json.load(open(os.path.join(d, "config.json")))
        recs = [json.loads(l) for l in open(os.path.join(d, "records.jsonl"))]
        runs.append((cfg, json.load(open(os.path.join(d, "summary.json"))), recs))
    return runs


def fmt(vals, pct=False, nd=0):
    if not vals:
        return "-"
    m = st.mean(vals)
    f = (lambda v: f"{100 * v:.0f}%") if pct else (lambda v: f"{v:.{nd}f}")
    return f(m) if len(vals) == 1 else f"{f(m)} ({f(min(vals))}–{f(max(vals))})"


def goal_accuracy(recs):
    """P2: episodes whose SLM goal equals the true goal (as sets)."""
    hit = n = 0
    for r in recs:
        path = os.path.join(r["_dir"], "episodes", str(r["episode_id"]), "plan.json")
        calls = json.load(open(path)).get("planner_calls", [])
        slm = next((c for c in calls if "slm_goal" in c), None)
        n += 1
        hit += bool(slm) and set(slm["slm_goal"]) == set(r["goal"])
    return hit / n if n else None


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "results/matrix"
    groups = defaultdict(list)
    for cfg, summ, recs in load(root):
        for r in recs:
            r["_dir"] = os.path.join(root, cfg["benchmark"], cfg["run_id"])
        groups[(cfg["planner"], cfg.get("config"), cfg.get("model") or "-")].append((cfg, summ, recs))

    lines = [f"# Matrix summary: `{root}`", "",
             "Mean over repeats, with (min–max) when there is more than one repeat.", "",
             "| Planner | Config | Model | Repeats | Episodes | Success | Sub-goals | Valid plans | "
             "Avg sim steps | Tokens in / out | Plan latency s | Repaired | Top failures |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(groups, key=lambda k: (ORDER.get(k[0], 9), k[2] != "qwen3:4b-instruct", k[2], k[1] or "")):
        runs = groups[key]
        S = [s for _, s, _ in runs]
        codes = defaultdict(int)
        for s in S:
            for c, n in s.get("failure_codes", {}).items():
                codes[c] += n
        top = ", ".join(f"{c} {n / len(S):.1f}" for c, n in sorted(codes.items(), key=lambda x: -x[1])[:3])
        lines.append(
            f"| {NAMES.get(key[0], key[0])} | {key[1]} | {key[2]} | {len(runs)} | {S[0]['episodes']} | "
            f"{fmt([s['success_rate'] for s in S], pct=True)} | {fmt([s['subgoal_rate'] for s in S], pct=True)} | "
            f"{fmt([s['plan_valid_rate'] for s in S], pct=True)} | {fmt([s['avg_sim_steps'] for s in S])} | "
            f"{fmt([s['avg_prompt_tokens'] for s in S])} / {fmt([s['avg_completion_tokens'] for s in S])} | "
            f"{fmt([s['avg_planning_latency_s'] for s in S], nd=2)} | "
            f"{fmt([s.get('repaired_rate', 0) for s in S], pct=True)} | {top or '-'} |")

    p2 = [(k, runs) for k, runs in groups.items() if k[0] == "slm_classical"]
    if p2:
        lines += ["", "## P2: does the SLM translate the instruction into the right goal?", "",
                  "| Model | Exact goal (episodes) |", "|---|---|"]
        for k, runs in sorted(p2, key=lambda x: x[0][2]):
            acc = [goal_accuracy(recs) for _, _, recs in runs]
            lines.append(f"| {k[2]} | {fmt([a for a in acc if a is not None], pct=True)} |")

    out = os.path.join(root, "summary.md")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
