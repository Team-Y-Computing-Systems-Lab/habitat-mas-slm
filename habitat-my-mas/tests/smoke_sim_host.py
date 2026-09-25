"""Smoke test for the sim host robot pool. Needs a running server:

    conda activate habitat
    python -m mas.sim_host.server --benchmark replica_pool4 &
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

    fleet0 = pool[:2]
    st = c.reset(episode_id=c.episodes()[0], active=fleet0)
    robots = {r["id"]: r for r in st["robots"]}
    print("objects:", [(o["name"], o["start_receptacle"], "->", o["goal_receptacle"]) for o in st["objects"]])
    check(all(robots[r]["active"] for r in fleet0), f"fleet {fleet0} active after reset")
    parked = [r for r in pool if r not in fleet0]
    check(all(not robots[r]["active"] and robots[r]["pos"][1] < PARK_Y + 1 for r in parked),
          f"{parked} parked below the scene")

    for _ in range(5):
        out = c.step()
    robots = {r["id"]: r for r in out["robots"]}
    check(all(robots[r]["pos"][1] < PARK_Y + 1 for r in parked), "parked robots stay parked after 5 steps")

    try:
        c.step({parked[0]: {"name": "wait", "args": {"wait": [1.0]}}})
        check(False, "action on a parked robot is rejected")
    except SimHostError as e:
        check(e.code == "ROBOT_INACTIVE", f"action on a parked robot is rejected ({e.code})")

    # a robot joins and another leaves mid-episode
    joined = c.activate(parked[0])
    check(joined["active"] and joined["pos"][1] > PARK_Y + 1, f"{parked[0]} joins at {joined['pos']}")
    left = c.deactivate(fleet0[1])
    check(not left["active"] and left["pos"][1] < PARK_Y + 1, f"{fleet0[1]} leaves the fleet")
    prev_step = c.state()["step"]
    out = c.step({parked[0]: {"name": "wait", "args": {"wait": [1.0]}}})
    check(out["step"] == prev_step + 1 and not out["episode_over"], "joined robot can act")

    back = c.activate(fleet0[1])
    check(back["active"], f"{fleet0[1]} rejoins at its last pose {back['pos']}")
    print("\nall checks passed")


if __name__ == "__main__":
    main()
