"""E8: harder episodes from official ones, by promoting clutter objects to targets.

Every official HSSD episode (hab3-episodes) already places 8 YCB objects, of which 2 are
targets. For N in {4, 6, 8} we keep the 2 official targets and promote N-2 of the other
6 objects, each with a new goal spot sampled on a different receptacle of the same scene
with habitat-lab's own receptacle sampler and `snap_down` (the method the official episode
generator uses). Goal spots are kept only if they are 0.3-1.2 m above the floor, have
navigable floor within 1 m, and are at least 0.3 m from other objects and goals.
A promoted object whose official start is out of that height range (on the floor, or too
high) also gets a new start spot sampled the same way; these are listed in
`info.moved_starts`. Objects already in range are promoted first.

The task spec (PDDL goal) for N objects is written next to the episodes.

    conda activate habitat-mas
    python -m mas.datasets.scale_episodes --n 4 6 8 --episode-ids $(cat scripts/e7_episodes.txt)
"""

import argparse
import copy
import gzip
import json
import os
import random

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "generated", "e8")

H_MIN, H_MAX, NAV_DIST, GOAL_SEP, TRIES = 0.3, 1.2, 1.0, 0.3, 400


def task_spec(n: int, robots: int = 2) -> str:
    """Official `multi_agent_tidy_house` spec generalised to n objects."""
    lines = ["# multi_agent_tidy_house with %d target objects (generated, E8)" % n, "objects:"]
    for i in range(n):
        lines += [f"  - name: any_targets|{i}", "    expr_type: movable_entity_type",
                  f"  - name: TARGET_any_targets|{i}", "    expr_type: goal_entity_type"]
    for r in range(robots):
        lines += [f"  - name: robot_{r}", "    expr_type: robot_entity_type"]
    lines += ["", "goal:", "  expr_type: AND", "  sub_exprs:"]
    lines += [f"    - at(any_targets|{i},TARGET_any_targets|{i})" for i in range(n)]
    lines += [f"    - not_holding(robot_{r})" for r in range(robots)]
    lines += ["", "stage_goals: {}", "solution: []", ""]
    return "\n".join(lines)


def spec_path(n: int) -> str:
    return os.path.join(OUT_DIR, f"tidy_house_n{n}.yaml")


def data_path(n: int) -> str:
    return os.path.join(OUT_DIR, f"hssd_n{n}.json.gz")


def _handles(ep: dict) -> list:
    """Instance handles of ep['rigid_objs'] in order (habitat appends :0000, :0001, ...)."""
    seen, out = {}, []
    for cfg, _ in ep["rigid_objs"]:
        base = cfg.replace(".object_config.json", "")
        k = seen.get(base, 0)
        seen[base] = k + 1
        out.append(f"{base}_:{k:04d}")
    return out


def promote(env, ep_json: dict, n: int, rng: random.Random) -> dict:
    """Return a copy of `ep_json` with n targets, or None if not enough goal spots were found."""
    from habitat.datasets.rearrange.samplers.receptacle import find_receptacles
    from habitat.sims.habitat_simulator.sim_utilities import snap_down

    sim = env.sim
    rom = sim.get_rigid_object_manager()
    floor = float(np.median([np.array(sim.get_agent_data(i).articulated_agent.base_pos)[1]
                             for i in range(sim.num_articulated_agents)]))
    recs = [r for r in find_receptacles(sim) if r.parent_object_handle]
    labels = ep_json["info"]["object_labels"]
    goals = [np.array(t)[:3, 3] for t in ep_json["targets"].values()]
    out = copy.deepcopy(ep_json)

    def sample_spot(obj, avoid_parent, taken):
        """(transform, receptacle parent handle) of a reachable spot on another receptacle."""
        saved = obj.transformation
        for _ in range(TRIES):
            rec = rng.choice(recs)
            if rec.parent_object_handle == avoid_parent:
                continue
            p = rec.sample_uniform_global(sim, 1.0)
            if not (H_MIN <= p[1] - floor <= H_MAX):
                continue
            nav = np.array(sim.pathfinder.snap_point(p))
            if np.isnan(nav).any() or np.linalg.norm((nav - np.array(p))[[0, 2]]) > NAV_DIST \
                    or abs(nav[1] - floor) > 0.3:
                continue
            if any(np.linalg.norm(np.array(p) - g) < GOAL_SEP for g in taken):
                continue
            parent = rom.get_object_by_handle(rec.parent_object_handle) \
                or sim.get_articulated_object_manager().get_object_by_handle(rec.parent_object_handle)
            obj.translation = p + np.array([0, 0.08, 0])
            ok = snap_down(sim, obj, [parent.object_id] if parent else None)
            # snap_down can fall through to a lower surface; keep only spots on the sampled one
            if ok and abs(obj.translation[1] - p[1]) <= 0.15 and H_MIN <= obj.translation[1] - floor <= H_MAX:
                spot = (obj.transformation, rec.parent_object_handle)
                obj.transformation = saved
                return spot
            obj.transformation = saved
        return None

    handles = _handles(ep_json)
    candidates = [h for h in handles if h not in labels]
    rng.shuffle(candidates)
    # objects already within reach first; the rest get a new start spot (recorded in info)
    candidates.sort(key=lambda h: not H_MIN <= rom.get_object_by_handle(h).translation[1] - floor <= H_MAX)
    starts = [np.array(rom.get_object_by_handle(h).translation) for h in handles]
    for handle in candidates:
        if len(out["info"]["object_labels"]) >= n:
            break
        obj = rom.get_object_by_handle(handle)
        start_rec = ep_json["name_to_receptacle"].get(handle, "")
        start_parent = start_rec.split("|")[0] if start_rec else None
        moved = None
        if not H_MIN <= obj.translation[1] - floor <= H_MAX or start_parent is None:
            moved = sample_spot(obj, start_parent, goals + starts)
            if moved is None:
                continue
        goal = sample_spot(obj, moved[1] if moved else start_parent, goals + starts)
        if goal is None:
            continue
        if moved:
            obj.transformation = moved[0]  # later samples must avoid it
            k = handles.index(handle)
            out["rigid_objs"][k][1] = [list(map(float, row)) for row in np.array(moved[0])]
            start_parent = moved[1]
            out["info"].setdefault("moved_starts", []).append(handle)
        i = len(out["info"]["object_labels"])
        out["info"]["object_labels"][handle] = f"any_targets|{i}"
        out["targets"][handle] = [list(map(float, row)) for row in np.array(goal[0])]
        out["target_receptacles"].append([start_parent, None])
        out["goal_receptacles"].append([goal[1], None])
        goals.append(np.array(goal[0])[:3, 3])
    if len(out["info"]["object_labels"]) < n:
        return None
    # habitat renumbers episode ids on load; the sim host looks episodes up by this instead
    out["info"]["source_episode_id"] = ep_json["episode_id"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--episode-ids", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from mas.sim_host.host import SimHost

    os.makedirs(OUT_DIR, exist_ok=True)
    src = json.load(gzip.open(os.path.join("data", "datasets", "hssd", "rearrange", "val",
                                           "social_rearrange.json.gz")))
    by_id = {e["episode_id"]: e for e in src["episodes"]}
    host = SimHost("hssd_fetch_stretch")
    made = {n: [] for n in args.n}
    for eid in args.episode_ids:
        host.reset(eid)
        for n in args.n:
            ep = promote(host.env, by_id[eid], n, random.Random(f"{args.seed}-{eid}-{n}"))
            print(f"episode {eid} n={n}: {'ok' if ep else 'not enough goal spots'}", flush=True)
            if ep:
                made[n].append(ep)
    for n, eps in made.items():
        with open(spec_path(n), "w") as f:
            f.write(task_spec(n))
        with gzip.open(data_path(n), "wt") as f:
            json.dump({**{k: v for k, v in src.items() if k != "episodes"}, "episodes": eps}, f)
        print(f"n={n}: {len(eps)} episodes -> {data_path(n)}")


if __name__ == "__main__":
    main()
