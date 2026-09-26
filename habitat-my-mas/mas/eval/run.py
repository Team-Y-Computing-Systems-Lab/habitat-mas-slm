"""Run a planner on benchmark episodes and store results + videos.

    conda activate habitat-mas
    python -m mas.sim_host.server --benchmark hssd_fetch_stretch &
    .venv/bin/python -m mas.eval.run --planner slm_only --prompt reach_v2 --normalize --repair --episodes 5
    .venv/bin/python -m mas.eval.run --planner classical_only --episodes 5
    .venv/bin/python -m mas.eval.run --planner slm_classical --model qwen3:4b-instruct --episodes 5

Layout (one directory per run, same format for every planner mode):

  results/<benchmark>/<planner>-<config>__<model>__<timestamp>[__r<repeat>]/
    config.json                 how the run was made
    records.jsonl               one JSON record per episode (schema: docs/results.md)
    summary.json                aggregate metrics
    episodes/<id>/plan.json     plan + raw SLM output and prompt
    episodes/<id>/video.mp4     top-down map + robot cameras
    episodes/<id>/video_final.png
  results/index.csv             one row per run, for comparing modes
"""

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
from contextlib import AsyncExitStack
from datetime import datetime
from typing import Dict, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mas.planners.base import Plan
from mas.planners.classical import ClassicalOnlyPlanner, SLMClassicalPlanner
from mas.planners.repair import check, normalize
from mas.planners.slm_only import PROMPT_VARIANTS, SLMOnlyPlanner
from mas.sim_host.client import SimClient
from mas.slm.client import DEFAULT_URL, SLMClient

from .instructions import explicit
from .report import index_report, rebuild_index, run_report

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EPISODE_WALL_LIMIT_S = 900


# ------------------------------------------------------------------ MCP fleet

async def connect_fleet(stack: AsyncExitStack, robot_ids: List[str], sim_url: str):
    """One MCP server per robot; discover each robot's tools and capabilities."""
    sessions, robots = {}, []
    for rid in robot_ids:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mas.mcp_servers.robot_server", "--robot", rid, "--sim-url", sim_url],
            cwd=PROJECT_ROOT,
        )
        read, write = await stack.enter_async_context(stdio_client(params))
        s = await stack.enter_async_context(ClientSession(read, write))
        await s.initialize()
        tools = (await s.list_tools()).tools
        caps = json.loads((await s.read_resource(f"robot://{rid}/capabilities")).contents[0].text)
        sessions[rid] = s
        robots.append({
            "id": rid,
            "capabilities": caps,
            "tools": [{"name": t.name, "description": t.description or "",
                       "input_schema": t.input_schema, "meta": t.meta} for t in tools],
        })
    return sessions, robots


def tool_result(call) -> dict:
    if call.structured_content is not None:
        sc = call.structured_content
        return sc.get("result", sc)
    try:
        return json.loads(call.content[0].text)
    except Exception:
        return {"status": "failed", "code": "BAD_TOOL_RESULT", "message": str(call.content), "steps": 0}


# ------------------------------------------------------------------ execution

async def execute(plan: Plan, sessions: Dict[str, ClientSession], t0: float) -> List[dict]:
    """Run each robot's steps in order, robots in parallel. P0: no recovery,
    a robot stops at its first failed step."""
    trace: List[dict] = []

    async def run_robot(robot: str, steps):
        for i, step in enumerate(steps):
            entry = {"robot": robot, "index": i, "skill": step.skill, "args": step.args,
                     "t_start": round(time.time() - t0, 2)}
            if robot not in sessions:
                res = {"status": "failed", "code": "INVALID_ROBOT", "message": robot, "steps": 0}
            else:
                try:
                    res = tool_result(await sessions[robot].call_tool(step.skill, step.args))
                except Exception as e:  # unknown tool, bad args, ...
                    res = {"status": "failed", "code": "INVALID_STEP", "message": str(e), "steps": 0}
            entry.update(status=res.get("status"), code=res.get("code"), message=res.get("message"),
                         sim_steps=res.get("steps", 0), t_end=round(time.time() - t0, 2))
            trace.append(entry)
            print(f"    {robot:10s} {step.skill}({', '.join(map(str, step.args.values()))}) "
                  f"-> {entry['code']} [{entry['sim_steps']} steps]", flush=True)
            if entry["status"] != "ok":
                break

    await asyncio.wait_for(
        asyncio.gather(*(run_robot(r, s) for r, s in plan.steps.items())),
        timeout=EPISODE_WALL_LIMIT_S)
    return trace


# -------------------------------------------------------------------- metrics

def score(goal: List[str], final: dict, plan: Plan, trace: List[dict]) -> dict:
    facts = set(final["facts"])
    met = {g["fact"]: g["met"] for g in final["goal_status"]}  # habitat's geometric check
    slm_calls = plan.planner_calls
    return {
        "success": bool(final["metrics"].get("pddl_success", False)),   # habitat's check (geometry)
        "success_symbolic": all(g in facts for g in goal),              # our tracked facts
        "subgoal_rate": sum(met.values()) / max(1, len(met)),
        "goal_met": met,
        "plan_parse_ok": plan.parse_ok,
        "plan_valid": plan.parse_ok and not plan.errors,
        "plan_steps": plan.num_steps(),
        "steps_ok": sum(e["status"] == "ok" for e in trace),
        "steps_failed": sum(e["status"] != "ok" for e in trace),
        "failure_codes": [e["code"] for e in trace if e["status"] != "ok"],
        "sim_steps": final["step"],
        "prompt_tokens": sum(c.get("prompt_tokens", 0) for c in slm_calls),
        "completion_tokens": sum(c.get("completion_tokens", 0) for c in slm_calls),
        "planning_latency_s": round(sum(c.get("latency_s", 0) for c in slm_calls), 2),
    }


def summarize(records: List[dict]) -> dict:
    n = len(records)
    if not n:
        return {"episodes": 0}
    avg = lambda k: round(sum(r["metrics"][k] for r in records) / n, 3)  # noqa: E731
    codes: Dict[str, int] = {}
    for r in records:
        for c in r["metrics"]["failure_codes"]:
            codes[c] = codes.get(c, 0) + 1
    return {
        "episodes": n,
        "success_rate": avg("success"),
        "success_symbolic_rate": avg("success_symbolic"),
        "subgoal_rate": avg("subgoal_rate"),
        "plan_valid_rate": avg("plan_valid"),
        "avg_plan_steps": avg("plan_steps"),
        "avg_sim_steps": avg("sim_steps"),
        "avg_prompt_tokens": avg("prompt_tokens"),
        "avg_completion_tokens": avg("completion_tokens"),
        "avg_planning_latency_s": avg("planning_latency_s"),
        "avg_execution_wall_s": round(sum(r["execution_wall_s"] for r in records) / n, 2),
        "repaired_rate": round(sum(bool(r["metrics"].get("repaired")) for r in records) / n, 3),
        "avg_plan_rewrites": round(sum(r["metrics"].get("plan_rewrites", 0) for r in records) / n, 2),
        "failure_codes": codes,
    }


# ------------------------------------------------------------------------ run

def make_planner(args):
    if args.planner == "slm_only":
        return SLMOnlyPlanner(SLMClient(args.model, base_url=args.slm_url), prompt=args.prompt)
    if args.planner == "classical_only":
        return ClassicalOnlyPlanner(engine=args.engine)
    return SLMClassicalPlanner(SLMClient(args.model, base_url=args.slm_url), engine=args.engine)


def run_tag(args) -> str:
    """Short name of the configuration, used in run ids and the index."""
    if args.planner == "slm_only":
        return "+".join([args.prompt] + (["norm"] if args.normalize else []) + (["repair"] if args.repair else []))
    return args.engine


def feasibility(sim: SimClient, benchmark: str, ep: str, robot_ids: List[str], cache_dir: str) -> dict:
    """Simulation reach check for every robot/object pair, cached per episode and fleet."""
    path = os.path.join(cache_dir, benchmark, f"{ep}__{'-'.join(robot_ids)}.json")
    if os.path.exists(path):
        return {**json.load(open(path)), "cached": True}
    t0 = time.time()
    res = sim.call("feasibility", episode_id=ep, active=robot_ids)
    assert res["episode_id"] == str(ep), f"feasibility ran on episode {res['episode_id']}, not {ep}"
    res["wall_s"] = round(time.time() - t0, 2)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(res, open(path, "w"), indent=2)
    return {**res, "cached": False}


async def main_async(args):
    sim = SimClient(args.sim_url, timeout=3600)
    info = sim.info()
    episodes = args.episode_ids or sim.episodes()[: args.episodes]
    robot_ids = [r["id"] for r in info["pool"]]
    uses_slm = args.planner != "classical_only"
    needs_feas = args.planner != "slm_only" or args.repair
    tag = run_tag(args)
    model_tag = re.sub(r"[^A-Za-z0-9.]+", "-", args.model) if uses_slm else "none"
    if args.skip_existing:  # resume a matrix: a finished run with this config/model/repeat exists
        import glob as _glob
        pattern = os.path.join(os.path.abspath(args.out), info["benchmark"],
                               f"{args.planner}-{tag}__{model_tag}__*")
        for d in _glob.glob(pattern):
            if os.path.exists(os.path.join(d, "summary.json")):
                cfg = json.load(open(os.path.join(d, "config.json")))
                if cfg.get("repeat") == args.repeat and cfg.get("episodes") == episodes:
                    print(f"[skip] already done: {d}")
                    return
    run_id = f"{args.planner}-{tag}__{model_tag}__{datetime.now():%Y%m%d-%H%M%S}" + (
        f"__r{args.repeat}" if args.repeat is not None else "")
    run_dir = os.path.join(os.path.abspath(args.out), info["benchmark"], run_id)
    os.makedirs(os.path.join(run_dir, "episodes"), exist_ok=True)
    config = {"run_id": run_id, "benchmark": info["benchmark"], "planner": args.planner, "config": tag,
              "prompt": args.prompt if args.planner == "slm_only" else None,
              "normalize": args.normalize, "repair": args.repair,
              "engine": args.engine if args.planner != "slm_only" else None,
              "model": args.model if uses_slm else None, "slm_url": args.slm_url if uses_slm else None,
              "repeat": args.repeat, "episodes": episodes, "fleet": robot_ids,
              "instruction_level": "explicit", "recovery": "none", "feasibility_check": needs_feas,
              "video": not args.no_video, "started": datetime.now().isoformat(timespec="seconds")}
    json.dump(config, open(os.path.join(run_dir, "config.json"), "w"), indent=2)
    print(f"[run] {run_id} -> {run_dir}")

    planner = make_planner(args)
    records = []
    async with AsyncExitStack() as stack:
        sessions, robots = await connect_fleet(stack, robot_ids, args.sim_url)
        object_names: List[str] = []
        for ep in episodes:
            ep_dir = os.path.join(run_dir, "episodes", str(ep))
            os.makedirs(ep_dir, exist_ok=True)
            feas = None
            if needs_feas:
                feas = feasibility(sim, info["benchmark"], ep, robot_ids, args.feasibility_cache)
            state = sim.reset(episode_id=ep, active=robot_ids)
            goal = state["goal"]
            object_names = [o["name"] for o in state["objects"]]
            instruction = explicit(goal)
            print(f"\n[ep {ep}] {instruction}", flush=True)
            ctx = {"goal": goal, "feasibility": feas["matrix"] if feas else None}

            try:
                plan = planner.plan(instruction, robots, state, ctx)
            except Exception as e:  # a planner bug must not end the whole run
                plan = Plan(parse_ok=False, errors=[f"planner error: {e!r}"[:500]])
            pipeline = {"rewrites": [], "check_before": None, "check_after": None, "repaired": False}
            if args.planner == "slm_only" and plan.parse_ok:
                if args.normalize:
                    pipeline["rewrites"] += normalize(plan, object_names)
                if args.repair:
                    problems = check(plan, state, goal, ctx["feasibility"])
                    pipeline["check_before"] = problems
                    if problems:
                        plan = planner.repair(plan, problems, instruction, robots, state)
                        pipeline["repaired"] = True
                        if args.normalize and plan.parse_ok:
                            pipeline["rewrites"] += normalize(plan, object_names)
                        pipeline["check_after"] = check(plan, state, goal, ctx["feasibility"])
            json.dump({**plan.to_dict(), "pipeline": pipeline, "feasibility": feas},
                      open(os.path.join(ep_dir, "plan.json"), "w"), indent=2)
            print(f"  plan: {plan.num_steps()} steps, valid={plan.parse_ok and not plan.errors}"
                  + (f" rewrites={len(pipeline['rewrites'])}" if pipeline["rewrites"] else "")
                  + (f" repaired ({len(pipeline['check_before'])} problems -> {len(pipeline['check_after'] or [])})"
                     if pipeline["repaired"] else "")
                  + (f" errors={plan.errors}" if plan.errors else ""), flush=True)

            video = None
            record_video = not args.no_video
            if record_video:
                sim.start_recording(os.path.join(ep_dir, "video.mp4"), every=args.video_every)
            t0 = time.time()
            try:
                trace = await execute(plan, sessions, t0)
            except asyncio.TimeoutError:
                trace = [{"robot": "*", "status": "failed", "code": "EPISODE_WALL_LIMIT", "sim_steps": 0}]
            wall = round(time.time() - t0, 2)
            if record_video:
                video = sim.stop_recording()
            final = sim.state()

            metrics = score(goal, final, plan, trace)
            metrics.update(
                plan_rewrites=len(pipeline["rewrites"]),
                repaired=pipeline["repaired"],
                check_problems_before=len(pipeline["check_before"] or []),
                check_problems_after=len(pipeline["check_after"] or []) if pipeline["repaired"] else None,
                feasibility_sim_steps=feas["sim_steps"] if feas else 0,
                feasibility_wall_s=(feas.get("wall_s", 0) if feas and not feas.get("cached") else 0),
            )
            record = {
                "run_id": run_id, "benchmark": info["benchmark"], "episode_id": ep,
                "planner": args.planner, "config": tag, "model": config["model"], "repeat": args.repeat,
                "fleet": robot_ids, "instruction": instruction, "goal": goal,
                "initial_facts": state["facts"], "final_facts": final["facts"],
                "plan": {r: [vars(s) for s in steps] for r, steps in plan.steps.items()},
                "plan_errors": plan.errors, "pipeline": pipeline,
                "feasibility": feas["matrix"] if feas else None,
                "trace": trace, "execution_wall_s": wall, "metrics": metrics,
                "video": os.path.relpath(video["video"], run_dir) if video else None,
            }
            records.append(record)
            with open(os.path.join(run_dir, "records.jsonl"), "a") as f:
                f.write(json.dumps(record) + "\n")
            print(f"  -> success={metrics['success']} subgoals={metrics['subgoal_rate']:.2f} "
                  f"sim_steps={metrics['sim_steps']} tokens={metrics['prompt_tokens']}+"
                  f"{metrics['completion_tokens']} plan_latency={metrics['planning_latency_s']}s", flush=True)

    summary = summarize(records)
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w"), indent=2)
    rebuild_index(os.path.abspath(args.out))
    print(f"\n[summary] {json.dumps(summary, indent=2)}")
    print(f"[report] {run_report(run_dir)}\n[report] {index_report(os.path.abspath(args.out))}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--planner", default="slm_only", choices=["slm_only", "classical_only", "slm_classical"])
    p.add_argument("--model", default="qwen3:4b-instruct", help="Ollama model (SLM planners)")
    p.add_argument("--prompt", default="reach_v2", choices=PROMPT_VARIANTS,
                   help="slm_only prompt variant (docs/findings.md)")
    p.add_argument("--normalize", action="store_true",
                   help="slm_only: rewrite navigate_to(place)+pick(obj) into navigate_to(obj)")
    p.add_argument("--repair", action="store_true",
                   help="slm_only: check the plan (incl. simulated reach) and give the SLM one repair round")
    p.add_argument("--engine", default="fast-downward-opt", choices=["fast-downward", "fast-downward-opt", "pyperplan"],
                   help="classical planning engine (unified-planning)")
    p.add_argument("--slm-url", default=DEFAULT_URL)
    p.add_argument("--sim-url", default="http://127.0.0.1:8765")
    p.add_argument("--episodes", type=int, default=5, help="first N episodes of the benchmark")
    p.add_argument("--episode-ids", nargs="*", help="explicit episode ids (overrides --episodes)")
    p.add_argument("--repeat", type=int, help="repeat index, recorded in the run id and records")
    p.add_argument("--out", default=os.path.join(PROJECT_ROOT, "results"))
    p.add_argument("--feasibility-cache", default=os.path.join(PROJECT_ROOT, "results", "feasibility"))
    p.add_argument("--skip-existing", action="store_true",
                   help="exit if a finished run with the same config, model, repeat and episodes exists")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--video-every", type=int, default=2, help="record every Nth sim step")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
