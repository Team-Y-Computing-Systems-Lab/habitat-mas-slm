"""E9 analysis: conditions A-E per episode set, paired against A (docs/e9_design.md §6).

    .venv/bin/python scripts/analyze_e9.py results/e9            # writes results/e9/summary.md
    .venv/bin/python scripts/analyze_e9.py results/e9/pilot      # the pilot gate

Paired tests on the same episodes: Wilcoxon signed-rank for makespan (sim steps), planning time
and expanded states; exact McNemar for success.
"""

import glob
import json
import os
import statistics as st
import sys
from math import comb

from scipy.stats import wilcoxon

ORDER = ["none", "hand", "slm_raw", "slm_verified", "slm_hint"]
LABEL = {"none": "A plain", "hand": "B hand", "slm_raw": "C SLM raw", "slm_verified": "D SLM verified",
         "slm_hint": "E SLM hint"}


def load(root):
    runs = {}
    for d in glob.glob(os.path.join(root, "*", "*")):
        if not os.path.exists(os.path.join(d, "summary.json")):
            continue
        cfg = json.load(open(os.path.join(d, "config.json")))
        if not cfg.get("aid"):
            continue
        recs = {}
        for line in open(os.path.join(d, "records.jsonl")):
            r = json.loads(line)
            plan = json.load(open(os.path.join(d, "episodes", str(r["episode_id"]), "plan.json")))
            call = next((c for c in reversed(plan["planner_calls"]) if "expanded" in c), {})
            r["_expanded"], r["_aid"] = call.get("expanded"), call.get("aid") or {}
            recs[r["episode_id"]] = r
        runs[(cfg["benchmark"], cfg["aid"], cfg.get("aid_model") or "-")] = recs
    return runs


def mcnemar(a, b):
    x = sum(a[e] and not b[e] for e in a)
    y = sum(b[e] and not a[e] for e in a)
    k = x + y
    p = min(1.0, 2 * sum(comb(k, i) for i in range(min(x, y) + 1)) / 2 ** k) if k else 1.0
    return x, y, p


def paired_w(a, b):
    d = [x - y for x, y in zip(a, b)]
    if not any(d):
        return 1.0
    return float(wilcoxon(a, b).pvalue)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "results/e9"
    runs = load(root)
    lines = [f"# E9 summary: `{root}`", "",
             "Paired against A (plain) on the same episodes. Makespan = sim steps until every robot is done.", ""]
    for bench in sorted({k[0] for k in runs}):
        base = runs.get((bench, "none", "-"))
        if base is None:
            continue
        lines += [f"## {bench}", "",
                  "| Condition | Model | Eps | Success | Sub-goals | Makespan (Δ vs A, p) | Waits / wait steps | "
                  "Busy fails | Plan s (p) | Expanded | Excl. correct / false / missed | Landmarks true / false | "
                  "Macros used / fallbacks | Tokens in/out |",
                  "|" + "---|" * 14]
        for (b, aid, model), recs in sorted(runs.items(), key=lambda kv: (ORDER.index(kv[0][1]), kv[0][2])):
            if b != bench:
                continue
            eps = sorted(set(recs) & set(base), key=int)
            m = [recs[e]["metrics"] for e in eps]
            mb = [base[e]["metrics"] for e in eps]
            span, span_b = [x["sim_steps"] for x in m], [x["sim_steps"] for x in mb]
            lat, lat_b = [x["planning_latency_s"] for x in m], [x["planning_latency_s"] for x in mb]
            itf = [x["interference"] for x in m]
            x, y, p_s = mcnemar({e: recs[e]["metrics"]["success"] for e in eps},
                                {e: base[e]["metrics"]["success"] for e in eps})
            e9 = [recs[e].get("e9") or {} for e in eps]
            cnt = lambda k: sum(len(z.get(k, [])) for z in e9)  # noqa: E731
            exp = [recs[e]["_expanded"] for e in eps if recs[e]["_expanded"] is not None]
            aidl = [recs[e]["_aid"] for e in eps]
            tok = [(z.get("tokens_in", 0), z.get("tokens_out", 0)) for z in e9]
            delta = st.mean(a - c for a, c in zip(span, span_b))
            lines.append(
                f"| {LABEL[aid]} | {model} | {len(eps)} | {100 * st.mean(r['success'] for r in m):.0f}% "
                f"({x}/{y}, p={p_s:.2f}) | {100 * st.mean(r['subgoal_rate'] for r in m):.0f}% | "
                f"{st.mean(span):.0f} ({delta:+.0f}, p={paired_w(span, span_b):.2f}) | "
                f"{st.mean(i['waits'] for i in itf):.2f} / {st.mean(i['wait_steps'] for i in itf):.0f} | "
                f"{sum(i['busy_failures'] for i in itf)} | {st.mean(lat):.1f} (p={paired_w(lat, lat_b):.2f}) | "
                f"{st.mean(exp):.0f} | "
                + (f"{cnt('exclusive_correct')} / {cnt('exclusive_false')} / {cnt('exclusive_missed')} | "
                   f"{cnt('landmarks_true')} / {cnt('landmarks_false')} | " if aid.startswith("slm") else "- | - | ")
                + f"{sum(a.get('macros_used', 0) for a in aidl)} / {sum(a.get('macro_fallbacks', 0) for a in aidl)} | "
                + (f"{st.mean(t[0] for t in tok):.0f} / {st.mean(t[1] for t in tok):.0f} |" if aid.startswith("slm") else "- |"))
        lines.append("")
    out = os.path.join(root, "summary.md")
    open(out, "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
