"""How well does an arm-height range predict whether a robot can pick / place an object?

    conda activate habitat-mas
    python scripts/calibrate_reach.py --benchmark hssd_fetch_stretch --episodes 10

For every (episode, object, robot) it records the height-range prediction
(forward kinematics on the robot's official URDF), then really navigates and
picks (and, if that worked, places) and records the outcome. Writes
results/reach_calibration/<benchmark>.jsonl.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from mas.sim_host.host import SimHost  # noqa: E402
from mas.sim_host.skills import SkillFailure, make_runner  # noqa: E402


def run_skill(host, robot, skill, **args):
    try:
        r = make_runner(host, robot, skill, args)
        res = r.immediate_result()
        while res is None:
            host.step({robot: r.action()})
            res = r.tick()
        return res["code"]
    except SkillFailure as e:
        return e.code


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="hssd_fetch_stretch")
    p.add_argument("--episodes", type=int, default=10)
    args = p.parse_args()
    host = SimHost(args.benchmark, ["habitat.environment.max_episode_steps=5000"])
    os.makedirs("results/reach_calibration", exist_ok=True)
    out = open(f"results/reach_calibration/{args.benchmark}.jsonl", "w")
    for ep in host.episodes()[: args.episodes]:
        st = host.reset(ep)
        for o in st["objects"]:
            for rid in [r["id"] for r in host.robots]:
                host.reset(ep)
                reach = host.arm_reach(rid)
                h, gh = o["height_m"], o["goal_height_m"]
                rec = {"episode": ep, "object": o["name"], "robot": rid, "height_m": h, "goal_height_m": gh,
                       "pred_pick": bool(reach and reach["min_height_m"] <= h <= reach["max_height_m"]),
                       "pred_place": bool(reach and reach["min_height_m"] <= gh <= reach["max_height_m"])}
                run_skill(host, rid, "navigate_to", target=o["name"])
                rec["actual_pick"] = run_skill(host, rid, "pick", object=o["name"])
                rec["actual_place"] = None
                if rec["actual_pick"] == "OK":
                    run_skill(host, rid, "navigate_to", target=o["goal_receptacle"])
                    rec["actual_place"] = run_skill(host, rid, "place", object=o["name"], place=o["goal_receptacle"])
                out.write(json.dumps(rec) + "\n")
                out.flush()
                print(f"ep {ep:5s} {o['name']:22s} {rid:10s} h={h:.2f} pred={rec['pred_pick']!s:5s} -> pick {rec['actual_pick']}"
                      f" | goal h={gh:.2f} pred={rec['pred_place']!s:5s} -> place {rec['actual_place']}", flush=True)


if __name__ == "__main__":
    main()
