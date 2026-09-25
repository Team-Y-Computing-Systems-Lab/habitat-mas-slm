"""HTML reports for stored runs. Open the files in a browser (or VS Code's preview).

  results/<benchmark>/<run>/report.html   one run: every episode's plan, trace, video
  results/index.html                      all runs side by side (from index.csv)

    .venv/bin/python -m mas.eval.report [results_dir]     # regenerate everything
"""

import csv
import html
import json
import os
import sys
from typing import List

CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--muted:#6e6e73;--line:#e5e5ea;--card:#f7f7f9;--ok:#1a7f37;--bad:#c62828}
@media (prefers-color-scheme:dark){:root{--bg:#161618;--fg:#f2f2f7;--muted:#a1a1a6;--line:#2c2c2e;--card:#1f1f22;--ok:#3fb950;--bad:#f85149}}
body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1200px;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:0}.muted{color:var(--muted)}
table{border-collapse:collapse;width:100%;margin:8px 0}th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600}td.num{text-align:right;font-variant-numeric:tabular-nums}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:16px 0}
.row{display:flex;gap:16px;flex-wrap:wrap}.row>div{flex:1 1 420px;min-width:0}
video,img{max-width:100%;border-radius:6px}.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}
code{font-size:12px}.kpi{display:inline-block;margin:6px 18px 6px 0}.kpi b{display:block;font-size:20px}
"""


def _badge(ok: bool, yes="success", no="failed") -> str:
    return f'<span class="{"ok" if ok else "bad"}">{yes if ok else no}</span>'


def _e(x) -> str:
    return html.escape(str(x))


def run_report(run_dir: str) -> str:
    config = json.load(open(os.path.join(run_dir, "config.json")))
    records = [json.loads(line) for line in open(os.path.join(run_dir, "records.jsonl"))]
    summary_path = os.path.join(run_dir, "summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}

    kpis = [("success", summary.get("success_rate")), ("sub-goals", summary.get("subgoal_rate")),
            ("valid plans", summary.get("plan_valid_rate")), ("sim steps", summary.get("avg_sim_steps")),
            ("tokens in/out", f"{summary.get('avg_prompt_tokens')}/{summary.get('avg_completion_tokens')}"),
            ("plan latency s", summary.get("avg_planning_latency_s"))]
    out = [f"<!doctype html><meta charset=utf-8><title>{_e(config['run_id'])}</title><style>{CSS}</style>",
           f"<h1>{_e(config['planner'])} · {_e(config.get('config', config.get('prompt', 'base')))} · {_e(config.get('model') or 'no SLM')}</h1>",
           f"<div class=muted>{_e(config['benchmark'])} · {len(records)} episodes · fleet "
           f"{_e(', '.join(config['fleet']))} · recovery: {_e(config.get('recovery'))} · {_e(config['started'])}</div>",
           "<div>" + "".join(f'<span class=kpi><b>{_e(v)}</b><span class=muted>{_e(k)}</span></span>'
                             for k, v in kpis) + "</div>"]
    if summary.get("failure_codes"):
        out.append("<div class=muted>failures: " + ", ".join(
            f"{_e(k)} ×{v}" for k, v in summary["failure_codes"].items()) + "</div>")

    for r in records:
        m = r["metrics"]
        plan_rows = "".join(
            f"<tr><td>{_e(robot)}</td><td>" + "<br>".join(
                f"<code>{_e(s['skill'])}({_e(', '.join(map(str, s['args'].values())))})</code>" for s in steps)
            + "</td></tr>" for robot, steps in r["plan"].items())
        trace_rows = "".join(
            f"<tr><td>{_e(t.get('robot'))}</td><td><code>{_e(t.get('skill'))}"
            f"({_e(', '.join(map(str, (t.get('args') or {}).values())))})</code></td>"
            f"<td>{_badge(t.get('status') == 'ok', t.get('code'), t.get('code'))}</td>"
            f"<td class=num>{_e(t.get('sim_steps'))}</td><td class=muted>{_e(t.get('message', ''))[:90]}</td></tr>"
            for t in r["trace"])
        goals = "".join(f"<li>{_badge(ok, '✓', '✗')} <code>{_e(g)}</code></li>" for g, ok in m["goal_met"].items())
        video = (f'<video controls preload=metadata src="{_e(r["video"])}"></video>'
                 if r.get("video") else "<div class=muted>no video</div>")
        errors = f"<div class=bad>plan errors: {_e('; '.join(r['plan_errors']))}</div>" if r["plan_errors"] else ""
        out.append(f"""<div class=card>
<h2>Episode {_e(r['episode_id'])} · {_badge(m['success'])} <span class=muted>
sub-goals {m['subgoal_rate']:.0%} · {m['sim_steps']} sim steps · {m['prompt_tokens']}+{m['completion_tokens']} tokens ·
plan {m['planning_latency_s']}s · execution {r['execution_wall_s']}s</span></h2>
<p>{_e(r['instruction'])}</p>
<div class=row><div>{video}</div><div>
<b>Goal</b><ul>{goals}</ul>{errors}
<b>Plan</b><table>{plan_rows}</table>
</div></div>
<b>Execution</b><table><tr><th>robot</th><th>skill</th><th>result</th><th>steps</th><th>message</th></tr>{trace_rows}</table>
</div>""")
    path = os.path.join(run_dir, "report.html")
    open(path, "w").write("\n".join(out))
    return path


def index_report(results_dir: str) -> str:
    rows: List[dict] = []
    index = os.path.join(results_dir, "index.csv")
    if os.path.exists(index):
        rows = list(csv.DictReader(open(index)))
    cols = ["benchmark", "planner", "config", "model", "repeat", "episodes", "success_rate", "subgoal_rate", "plan_valid_rate",
            "avg_sim_steps", "avg_prompt_tokens", "avg_completion_tokens", "avg_planning_latency_s"]
    body = "".join(
        "<tr>" + "".join(f"<td class={'num' if c not in ('benchmark', 'planner', 'config', 'model') else ''}>{_e(r.get(c, ''))}</td>"
                         for c in cols)
        + f"<td><a href=\"{_e(r['run_dir'])}/report.html\">report</a></td></tr>" for r in rows)
    doc = (f"<!doctype html><meta charset=utf-8><title>MAS results</title><style>{CSS}</style>"
           f"<h1>Results</h1><div class=muted>one row per run; planners: slm_only (P0), classical_only (P1), "
           f"slm_classical (P2)</div><table><tr>{''.join(f'<th>{c}</th>' for c in cols)}<th></th></tr>{body}</table>")
    path = os.path.join(results_dir, "index.html")
    open(path, "w").write(doc)
    return path


def rebuild_index(results_dir: str) -> None:
    """Rewrite index.csv from the run directories on disk (skips _superseded/)."""
    rows = []
    for root, dirs, files in os.walk(results_dir):
        dirs[:] = [d for d in dirs if not d.startswith("_")]
        if "summary.json" in files and "config.json" in files:
            cfg = json.load(open(os.path.join(root, "config.json")))
            summ = json.load(open(os.path.join(root, "summary.json")))
            rows.append({"run_id": cfg["run_id"], "benchmark": cfg["benchmark"], "planner": cfg["planner"],
                         "config": cfg.get("config", cfg.get("prompt", "base")), "model": cfg.get("model") or "-",
                         "repeat": cfg.get("repeat"),
                         **{k: v for k, v in summ.items() if k != "failure_codes"},
                         "run_dir": os.path.relpath(root, results_dir)})
    rows.sort(key=lambda r: r["run_id"].split("__")[-1])
    if rows:
        with open(os.path.join(results_dir, "index.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)


def main():
    results_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "results")
    for root, dirs, files in os.walk(results_dir):
        dirs[:] = [d for d in dirs if not d.startswith("_")]
        if "records.jsonl" in files:
            print(run_report(root))
    rebuild_index(results_dir)
    print(index_report(results_dir))


if __name__ == "__main__":
    main()
