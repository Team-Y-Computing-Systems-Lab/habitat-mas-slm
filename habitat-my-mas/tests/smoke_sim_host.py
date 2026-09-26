"""Smoke test for the sim host robot pool. Needs a running server:

    conda activate habitat-mas
    python -m mas.sim_host.server --benchmark hssd_fetch_stretch &
    python tests/smoke_sim_host.py            # any Python >= 3.8
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mas.sim_host.client import SimClient, SimHostError  # noqa: E402

PARK_Y = -100.0


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        sys.exit(1)


def main():
    c = SimClient()
    info = c.info()
    pool = [r["id"] for r in info["pool"]]
    print("pool:", pool, "| episodes:", info["num_episodes"])
    check(len(pool) >= 2, "pool has at least two robots")
    keep, park = pool[0], pool[1]

    st = c.reset(episode_id=c.episodes()[0], active=[keep])
    robots = {r["id"]: r for r in st["robots"]}
    print("objects:", [(o["name"], o["start_receptacle"], "->", o["goal_receptacle"]) for o in st["objects"]])
    check(robots[keep]["active"], f"{keep} active after reset")
    check(not robots[park]["active"] and robots[park]["pos"][1] < PARK_Y + 1, f"{park} parked below the scene")

    for _ in range(5):
        out = c.step()
    robots = {r["id"]: r for r in out["robots"]}
    check(robots[park]["pos"][1] < PARK_Y + 1, "parked robot stays parked after 5 steps")

    try:
        c.step({park: {"name": "base_velocity", "args": {"base_vel": [0.0, 0.0]}}})
        check(False, "action on a parked robot is rejected")
    except SimHostError as e:
        check(e.code == "ROBOT_INACTIVE", f"action on a parked robot is rejected ({e.code})")

    joined = c.activate(park)
    check(joined["active"] and joined["pos"][1] > PARK_Y + 1, f"{park} joins at {joined['pos']}")
    left = c.deactivate(keep)
    check(not left["active"] and left["pos"][1] < PARK_Y + 1, f"{keep} leaves the fleet")
    prev_step = c.state()["step"]
    out = c.step({park: {"name": "base_velocity", "args": {"base_vel": [0.0, 0.0]}}})
    check(out["step"] == prev_step + 1 and not out["episode_over"], "joined robot can act")
    back = c.activate(keep)
    check(back["active"], f"{keep} rejoins at its last pose {back['pos']}")
    print("\nall checks passed")


if __name__ == "__main__":
    main()
