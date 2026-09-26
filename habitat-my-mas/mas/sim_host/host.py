"""SimHost: owns one habitat-lab Env and manages a pool of robots.

Fleet changes (design.md §4a): Habitat fixes the set of articulated agents when
the env is built, so every robot the episode might use is spawned up front.
Robots outside the fleet are *parked* far below the scene and receive only
`wait` actions. Activating a robot moves it back onto the navmesh.
"""

from typing import Any, Dict, List, Optional

import numpy as np

import habitat
from habitat.config.default import get_config

from mas.skills.catalog import skills_for

from .benchmarks import BENCHMARKS, EXTRA_OVERRIDES, ROBOT_PREFIX
from .naming import NameTable

PARK_Y = -100.0
PARK_SPACING = 10.0


class SimHostError(Exception):
    """Error with a structured code the executor can act on (design.md §6)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    try:  # magnum vectors index like sequences but lack __iter__
        return [float(x[i]) for i in range(len(x))]
    except TypeError:
        return str(x)


class SimHost:
    def __init__(self, benchmark: str, overrides: Optional[List[str]] = None):
        if benchmark not in BENCHMARKS:
            raise SimHostError("UNKNOWN_BENCHMARK", f"choose from {sorted(BENCHMARKS)}")
        self.benchmark = benchmark
        overrides = EXTRA_OVERRIDES.get(benchmark, []) + (overrides or [])
        self.config = get_config(BENCHMARKS[benchmark], overrides)
        self.env = habitat.Env(config=self.config)

        sim_cfg = self.config.habitat.simulator
        self.robots: List[Dict[str, Any]] = []
        counts: Dict[str, int] = {}
        for idx, agent_name in enumerate(sim_cfg.agents_order):
            rtype = sim_cfg.agents[agent_name].articulated_agent_type
            prefix = ROBOT_PREFIX.get(rtype, rtype.lower())
            n = counts.get(prefix, 0)
            counts[prefix] = n + 1
            self.robots.append({"id": f"{prefix}_{n}", "type": rtype, "agent_idx": idx})
        self._by_id = {r["id"]: r for r in self.robots}

        self._active: Dict[str, bool] = {}
        self._saved_pose: Dict[str, tuple] = {}
        self._names = NameTable()
        self._step_count = 0
        self._last_obs: Dict[str, Any] = {}
        # symbolic state for the planner, updated by skills (see skills.py)
        self._robot_place: Dict[str, str] = {}
        self._obj_place: Dict[str, Optional[str]] = {}
        self._observed: set = set()
        self.labels: Dict[str, str] = {}  # robot -> current skill, shown in videos
        self._recorder = None

    # ------------------------------------------------------------------ episodes

    def episodes(self) -> List[str]:
        return [ep.episode_id for ep in self.env.episodes]

    def reset(self, episode_id: Optional[str] = None,
              active: Optional[List[str]] = None) -> Dict[str, Any]:
        """Load an episode. `active` lists robot ids in the fleet (default: all)."""
        if episode_id is not None:
            matches = [ep for ep in self.env.episodes if ep.episode_id == str(episode_id)]
            if not matches:
                raise SimHostError("UNKNOWN_EPISODE", f"episode {episode_id} not in dataset")
            self.env.current_episode = matches[0]
        self._last_obs = self.env.reset()
        self._clear_action_caches()
        self._names = NameTable()
        self._step_count = 0
        self.labels = {}
        self._index_episode()

        active = set(active) if active is not None else set(self._by_id)
        unknown = active - set(self._by_id)
        if unknown:
            raise SimHostError("UNKNOWN_ROBOT", f"not in pool: {sorted(unknown)}")
        for r in self.robots:
            self._saved_pose[r["id"]] = self._pose(r["id"])
            self._active[r["id"]] = True
            if r["id"] not in active:
                self.deactivate(r["id"])
        return self.state()

    def _clear_action_caches(self) -> None:
        """Habitat's nav actions cache approach points and "already arrived" flags
        per episode id; resetting the *same* episode (e.g. after feasibility
        trials) must not inherit them."""
        for name, act in self.env.task.actions.items():
            if name.endswith("oracle_nav_action"):
                for attr, val in (("_targets", {}), ("prev_nav_done", False),
                                  ("prev_match_target_id", None), ("skill_done", False), ("ep_id", None)):
                    if hasattr(act, attr):
                        setattr(act, attr, val)

    # ---------------------------------------------------------------- the fleet

    def _agent(self, robot_id: str):
        if robot_id not in self._by_id:
            raise SimHostError("UNKNOWN_ROBOT", robot_id)
        return self.env.sim.agents_mgr[self._by_id[robot_id]["agent_idx"]]

    def _pose(self, robot_id: str) -> tuple:
        art = self._agent(robot_id).articulated_agent
        return np.array(art.base_pos), float(art.base_rot)

    def deactivate(self, robot_id: str) -> Dict[str, Any]:
        """Remove a robot from the fleet by parking it outside the scene."""
        if not self._active.get(robot_id, False):
            return self.robot(robot_id)
        agent = self._agent(robot_id)
        if agent.grasp_mgr.is_grasped:
            held = self.robot(robot_id)["holding"]
            agent.grasp_mgr.desnap()  # a leaving robot drops what it holds
            self._obj_place[held] = None  # dropped somewhere: unknown place
        self._saved_pose[robot_id] = self._pose(robot_id)
        idx = self._by_id[robot_id]["agent_idx"]
        agent.articulated_agent.base_pos = np.array([idx * PARK_SPACING, PARK_Y, 0.0])
        self._active[robot_id] = False
        return self.robot(robot_id)

    def activate(self, robot_id: str, pos: Optional[List[float]] = None,
                 yaw: Optional[float] = None) -> Dict[str, Any]:
        """Add a robot to the fleet, at `pos` or where it was last active."""
        if self._active.get(robot_id, False):
            return self.robot(robot_id)
        saved_pos, saved_yaw = self._saved_pose[robot_id]
        target = np.array(pos if pos is not None else saved_pos, dtype=np.float32)
        snapped = np.array(self.env.sim.pathfinder.snap_point(target))
        if np.isnan(snapped).any():
            raise SimHostError("NOT_NAVIGABLE", f"no navmesh point near {target.tolist()}")
        art = self._agent(robot_id).articulated_agent
        art.base_pos = snapped  # base_pos is at navmesh height; the offset is internal
        art.base_rot = yaw if yaw is not None else saved_yaw
        self._active[robot_id] = True
        return self.robot(robot_id)

    def robot(self, robot_id: str) -> Dict[str, Any]:
        agent = self._agent(robot_id)
        pos, yaw = self._pose(robot_id)
        held = None
        if agent.grasp_mgr.is_grasped:
            obj = self.env.sim.get_rigid_object_manager().get_object_by_id(agent.grasp_mgr.snap_idx)
            held = self._names.name(obj.handle)
        return {**self._by_id[robot_id], "active": self._active.get(robot_id, False),
                "pos": _jsonable(pos), "yaw": yaw, "holding": held}

    def fleet(self) -> List[Dict[str, Any]]:
        return [self.robot(r["id"]) for r in self.robots]

    # ------------------------------------------------- names <-> PDDL entities

    def _index_episode(self) -> None:
        """Map readable object/place names to habitat's PDDL entities.

        Habitat names task objects "any_targets|i" and their goal locations
        "TARGET_any_targets|i". We expose objects by name (bowl_0) and places by
        receptacle name (table_02_0); several goal locations can share one
        receptacle.
        """
        ep = self.env.current_episode
        self._entities = self.env.task.pddl_problem.get_ordered_entities_list()
        self._entity_idx = {e.name: i for i, e in enumerate(self._entities)}
        labels = ep.info.get("object_labels", {})
        self._objects = []  # i -> {"name", "handle", "start", "goal"}
        for handle, label in sorted(labels.items(), key=lambda kv: int(kv[1].split("|")[1])):
            i = int(label.split("|")[1])
            start = ep.target_receptacles[i][0] if i < len(ep.target_receptacles) else None
            goal = ep.goal_receptacles[i][0] if i < len(ep.goal_receptacles) else None
            self._objects.append({
                "name": self._names.name(handle),
                "handle": handle,
                "start": self._names.name(start) if start else None,
                "goal": self._names.name(goal) if goal else None,
            })
        self._obj_place = {o["name"]: o["start"] for o in self._objects}
        self._delivered: set = set()  # objects placed on their goal receptacle
        self._robot_place = {r["id"]: f"start_{r['id']}" for r in self.robots}
        self._observed = set()

    def resolve_object(self, name: str) -> int:
        for i, o in enumerate(self._objects):
            if o["name"] == name:
                return self._entity_idx[f"any_targets|{i}"]
        raise SimHostError("UNKNOWN_OBJECT", f"{name}; objects: {[o['name'] for o in self._objects]}")

    def places(self) -> List[str]:
        names = [o["start"] for o in self._objects] + [o["goal"] for o in self._objects]
        return sorted({n for n in names if n})

    def resolve_goal_place(self, place: str, obj: Optional[str] = None) -> int:
        """Goal-location entity on `place`, preferring the one meant for `obj`."""
        matches = [i for i, o in enumerate(self._objects) if o["goal"] == place]
        if not matches:
            raise SimHostError("UNKNOWN_PLACE", f"{place} is not a placement location; "
                               f"choose from {sorted({o['goal'] for o in self._objects if o['goal']})}")
        for i in matches:
            if self._objects[i]["name"] == obj:
                return self._entity_idx[f"TARGET_any_targets|{i}"]
        return self._entity_idx[f"TARGET_any_targets|{matches[0]}"]

    def resolve_nav_target(self, robot_id: str, target: str) -> int:
        """Entity to drive to for an object or place name.

        Receptacles are large, so a place name is resolved by intent: a robot
        holding an object goes to that object's goal spot on the place; an
        empty-handed robot goes to an object waiting there to be moved.
        """
        if any(o["name"] == target for o in self._objects):
            return self.resolve_object(target)
        held = self.robot(robot_id)["holding"]
        goal_here = any(o["goal"] == target for o in self._objects)
        if held and goal_here:
            return self.resolve_goal_place(target, held)
        waiting = [i for i, o in enumerate(self._objects)
                   if self._obj_place.get(o["name"]) == target and o["name"] != held
                   and o["name"] not in self._delivered]
        # prefer objects that still need moving (their goal is elsewhere, or not yet placed)
        waiting.sort(key=lambda i: self._objects[i]["goal"] == target)
        if waiting:
            return self._entity_idx[f"any_targets|{waiting[0]}"]
        if goal_here:
            return self.resolve_goal_place(target, held)
        if target in self.places():
            raise SimHostError("NO_NAV_POINT", f"no task object left on {target} to navigate to")
        raise SimHostError("UNKNOWN_PLACE", f"{target}; places: {self.places()}, "
                           f"objects: {[o['name'] for o in self._objects]}")

    def entity_pos(self, entity_idx: int) -> List[float]:
        pos = self.env.task.pddl_problem.sim_info.get_entity_pos(self._entities[entity_idx])
        return _jsonable(pos)

    # ------------------------------------------ symbolic state (for the planner)

    def set_robot_place(self, robot_id: str, target: str) -> None:
        # navigating to an object puts the robot at that object's place
        self._robot_place[robot_id] = self._obj_place.get(target) or target

    def on_pick(self, robot_id: str, obj: str) -> None:
        self._obj_place[obj] = None
        self._delivered.discard(obj)

    def on_place(self, robot_id: str, obj: str, place: str) -> None:
        self._obj_place[obj] = place
        if any(o["name"] == obj and o["goal"] == place for o in self._objects):
            self._delivered.add(obj)

    def on_look(self, robot_id: str) -> None:
        self._observed.add(self._robot_place[robot_id])

    def visible_objects(self, robot_id: str) -> List[str]:
        idx = self._by_id[robot_id]["agent_idx"]
        detected = self._last_obs.get(f"agent_{idx}_detected_objects")
        if detected is None:
            return []
        start = self.config.habitat.simulator.object_ids_start
        rom = self.env.sim.get_rigid_object_manager()
        seen = set()
        for sem_id in np.asarray(detected).ravel().tolist():
            obj = rom.get_object_by_id(int(sem_id) - start)
            if obj is not None:
                seen.add(obj.handle)
        return [o["name"] for o in self._objects if o["handle"] in seen]

    def facts(self) -> List[str]:
        """Current symbolic state as PDDL facts over readable names."""
        out = []
        for r in self.robots:
            rid = r["id"]
            if not self._active.get(rid):
                continue
            out.append(f"(robot-at {rid} {self._robot_place[rid]})")
            held = self.robot(rid)["holding"]
            if held:
                out.append(f"(holding {rid} {held})")
            elif any(s["name"] == "pick" for s in self.skills(rid)):
                out.append(f"(hand-empty {rid})")
        for obj, place in self._obj_place.items():
            if place:
                out.append(f"(obj-at {obj} {place})")
        out += [f"(observed {p})" for p in sorted(self._observed)]
        return out

    def task_goal(self) -> List[str]:
        """The episode's PDDL goal, rewritten over readable names."""
        names = {f"any_targets|{i}": o["name"] for i, o in enumerate(self._objects)}
        names.update({f"TARGET_any_targets|{i}": o["goal"] for i, o in enumerate(self._objects)})
        names.update({f"agent_{r['agent_idx']}": r["id"] for r in self.robots})
        pred_map = {"at": "obj-at", "robot_at": "robot-at", "not_holding": "hand-empty"}

        def walk(expr):
            if hasattr(expr, "sub_exprs"):
                for sub in expr.sub_exprs:
                    yield from walk(sub)
            else:
                args = [names.get(e.name, e.name) for e in expr._arg_values]
                if expr.name == "robot_at":  # habitat order is (place, robot)
                    args = args[::-1]
                yield f"({pred_map.get(expr.name, expr.name)} {' '.join(args)})"

        return list(walk(self.env.task.pddl_problem.goal))

    def goal_status(self) -> List[Dict[str, Any]]:
        """Each goal fact and whether it holds, checked by habitat on the sim geometry."""
        pp = self.env.task.pddl_problem
        readable = self.task_goal()

        def leaves(expr):
            if hasattr(expr, "sub_exprs"):
                for sub in expr.sub_exprs:
                    yield from leaves(sub)
            else:
                yield expr

        return [{"fact": fact, "met": bool(pp.is_expr_true(expr))}
                for fact, expr in zip(readable, leaves(pp.goal))]

    def arm_reach(self, robot_id: str) -> Optional[Dict[str, float]]:
        """Heights (m above the floor) the end effector can reach, from forward
        kinematics over random joint configurations within the joint limits."""
        rtype = self._by_id[robot_id]["type"]
        if not hasattr(self, "_reach_cache"):
            self._reach_cache: Dict[str, Optional[Dict[str, float]]] = {}
        if rtype in self._reach_cache:
            return self._reach_cache[rtype]
        k = self._by_id[robot_id]["agent_idx"]
        act = self.env.task.actions.get(f"agent_{k}_arm_pick_action")
        if act is None:
            self._reach_cache[rtype] = None
            return None
        ik = act.arm_ctrlr._ik_helper
        if ik is None:  # created on the first episode reset
            self.reset()
            ik = act.arm_ctrlr._ik_helper
        lo, hi = (np.array(v) for v in ik.get_joint_limits())
        art = self._agent(robot_id).articulated_agent
        T, floor = art.base_transformation, float(np.array(art.base_pos)[1])
        rng = np.random.default_rng(0)
        heights = []
        for _ in range(2000):
            q = lo + rng.random(len(lo)) * (hi - lo)
            heights.append(float(T.transform_point(np.array(ik.calc_fk(q), dtype=np.float32))[1]) - floor)
        ik.set_arm_state(np.array(art.arm_joint_pos))  # leave the IK model as the arm is
        reach = {"min_height_m": round(max(0.0, min(heights)), 2), "max_height_m": round(max(heights), 2)}
        self._reach_cache[rtype] = reach
        return reach

    def feasibility(self, episode_id: Optional[str] = None,
                    active: Optional[List[str]] = None) -> Dict[str, Any]:
        """Furniture-aware reach check by simulation: for every arm robot and
        task object, try navigate+pick (and, if that works, navigate+place on
        the goal) in a trial rollout, then reset the episode. This stands in
        for a motion planner's feasibility check; the geometric IK check
        (`check_reach`) proved unreliable here (docs/findings.md).

        Returns {"matrix": {robot: {object: {"pick": code, "place": code|None}}},
                 "sim_steps": steps spent on trials}. The episode is left freshly reset.
        """
        from .skills import SkillFailure, make_runner  # skills imports this module

        if self._recorder is not None:
            raise SimHostError("SIM_BUSY", "stop recording before a feasibility check")
        if episode_id is not None:
            self.reset(episode_id, active=active)
        ep = self.env.current_episode.episode_id
        active = active or [r["id"] for r in self.robots if self._active.get(r["id"])]
        steps = 0

        def run(rid, skill, **args):
            nonlocal steps
            try:
                runner = make_runner(self, rid, skill, args)
                res = runner.immediate_result()
                while res is None:
                    self.step({rid: runner.action()})
                    res = runner.tick()
                steps += runner.steps
                return res["code"]
            except SkillFailure as e:
                return e.code

        matrix: Dict[str, Dict[str, Dict[str, Optional[str]]]] = {}
        for rid in active:
            if not any(s["name"] == "pick" for s in self.skills(rid)):
                continue
            for o in list(self._objects):
                self.reset(ep, active=active)
                run(rid, "navigate_to", target=o["name"])
                pick = run(rid, "pick", object=o["name"])
                place = None
                if pick == "OK" and o["goal"]:
                    run(rid, "navigate_to", target=o["goal"])
                    place = run(rid, "place", object=o["name"], place=o["goal"])
                matrix.setdefault(rid, {})[o["name"]] = {"pick": pick, "place": place}
        self.reset(ep, active=active)
        return {"episode_id": ep, "matrix": matrix, "sim_steps": steps}

    def check_reach(self, robot_id: str, object: str, place: Optional[str] = None,
                    tol: float = 0.15) -> Dict[str, Any]:
        """Can this robot's arm get to the object (or, with `place`, to the
        object's goal spot on that place)? Accounts for the furniture around it.

        The robot is put at the approach pose its own navigation would use,
        facing the target; inverse kinematics solves for the arm; the result
        is reachable when the gripper ends within `tol` (the magic-grasp
        distance) of the target and the arm in that pose does not intersect
        the scene. Everything is restored afterwards; no sim step is taken.
        """
        k = self._by_id[robot_id]["agent_idx"]
        act = self.env.task.actions.get(f"agent_{k}_arm_pick_action")
        if act is None:
            return {"reachable": False, "reason": "NO_ARM"}
        entity = self.resolve_goal_place(place, object) if place else self.resolve_object(object)
        target = np.array(self.entity_pos(entity), dtype=np.float32)
        nav = self.env.task.actions[f"agent_{k}_oracle_nav_action"]
        ik = act.arm_ctrlr._ik_helper
        art = self._agent(robot_id).articulated_agent
        saved = (np.array(art.base_pos), float(art.base_rot), np.array(art.arm_joint_pos))
        try:
            approach, _ = nav._get_target_for_idx(entity)
            art.base_pos = np.array(approach, dtype=np.float32)
            # face the target: try both yaw conventions, keep the one that points at it
            rel = target - np.array(art.base_pos)
            best = None
            for yaw in (np.arctan2(-rel[2], rel[0]), np.arctan2(rel[2], rel[0])):
                art.base_rot = float(yaw)
                fwd = np.array(art.base_transformation.transform_vector(np.array([1.0, 0, 0], np.float32)))
                cos = float(np.dot(fwd[[0, 2]], rel[[0, 2]])
                            / (np.linalg.norm(fwd[[0, 2]]) * np.linalg.norm(rel[[0, 2]]) + 1e-9))
                if best is None or cos > best[0]:
                    best = (cos, float(yaw))
            art.base_rot = best[1]

            T = art.base_transformation
            local = np.array(T.inverted().transform_point(target))
            lo, hi = (np.array(v) for v in ik.get_joint_limits())
            q = np.array(saved[2])
            for _ in range(5):  # restart IK from its own answer; one call may stop short
                ik.set_arm_state(q)
                q = np.clip(np.array(ik.calc_ik(local))[: len(lo)], lo, hi)
            residual = float(np.linalg.norm(np.array(ik.calc_fk(q)) - local))

            # arm-scene intersection in that pose
            art.arm_joint_pos = q
            self.env.sim.perform_discrete_collision_detection()
            robot_sim_id = art.sim_obj.object_id
            target_handle = self._names.handle(object)
            rom = self.env.sim.get_rigid_object_manager()
            target_obj = rom.get_object_by_handle(target_handle) if target_handle else None
            ignore = {robot_sim_id} | ({target_obj.object_id} if target_obj is not None else set())
            arm_links = set(art.params.arm_joints) | set(getattr(art.params, "ee_links", []))
            hits = 0
            for c in self.env.sim.get_physics_contact_points():
                for me, other, link in ((c.object_id_a, c.object_id_b, c.link_id_a),
                                        (c.object_id_b, c.object_id_a, c.link_id_b)):
                    if me == robot_sim_id and other not in ignore and link in arm_links \
                            and c.contact_distance < -0.01:
                        hits += 1
        finally:
            art.base_pos, art.base_rot = saved[0], saved[1]
            art.arm_joint_pos = saved[2]
            ik.set_arm_state(saved[2])
        reachable = residual <= tol and hits == 0
        reason = "OK" if reachable else ("COLLISION" if residual <= tol else "TOO_FAR")
        return {"reachable": reachable, "reason": reason, "residual_m": round(residual, 3),
                "arm_contacts": hits, "target_height_m": self._height(target)}

    def _height(self, pos) -> float:
        """Height above the floor below `pos`."""
        floor = np.array(self.env.sim.pathfinder.snap_point(np.array(pos, dtype=np.float32)))
        base = floor[1] if not np.isnan(floor).any() else 0.0
        return round(float(pos[1]) - float(base), 2)

    def skills(self, robot_id: str) -> List[dict]:
        prefix = f"agent_{self._by_id[robot_id]['agent_idx']}_"
        low = [k[len(prefix):] for k in self.env.action_space.spaces if k.startswith(prefix)]
        return skills_for(low)

    # -------------------------------------------------------------------- state

    def state(self) -> Dict[str, Any]:
        ep = self.env.current_episode
        rom = self.env.sim.get_rigid_object_manager()
        objects = []
        for o in self._objects:
            obj = rom.get_object_by_handle(o["handle"])
            pos = _jsonable(obj.translation) if obj is not None else None
            goal_idx = self._entity_idx.get(f"TARGET_any_targets|{self._objects.index(o)}")
            objects.append({
                "name": o["name"],
                "handle": o["handle"],
                "pos": pos,
                "height_m": self._height(pos) if pos else None,
                "goal_height_m": self._height(self.entity_pos(goal_idx)) if goal_idx is not None else None,
                "goal_pos": self.entity_pos(goal_idx) if goal_idx is not None else None,
                "start_receptacle": o["start"],
                "goal_receptacle": o["goal"],
            })
        return {
            "benchmark": self.benchmark,
            "episode_id": ep.episode_id,
            "scene_id": ep.scene_id,
            "step": self._step_count,
            "episode_over": self.env.episode_over,
            "robots": self.fleet(),
            "objects": objects,
            "places": self.places(),
            "facts": self.facts(),
            "goal": self.task_goal(),
            "goal_status": self.goal_status(),
            "metrics": _jsonable(self.env.get_metrics()),
        }

    # --------------------------------------------------------------------- step

    def step(self, actions: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Advance the sim one step.

        actions: {robot_id: {"name": "oracle_nav_action", "args": {"oracle_nav_action": [3]}}}
        Action and arg names are given without the "agent_<i>_" prefix.
        Robots with no action (and all parked robots) wait.
        """
        actions = actions or {}
        names, args = [], {}
        for r in self.robots:
            rid, prefix = r["id"], f"agent_{r['agent_idx']}_"
            act = actions.get(rid)
            if act is not None and not self._active[rid]:
                raise SimHostError("ROBOT_INACTIVE", f"{rid} is not in the fleet")
            if act is None:
                act = {"name": "wait", "args": {"wait": [1.0]}}
            full = prefix + act["name"]
            if full not in self.env.action_space.spaces:
                raise SimHostError("UNKNOWN_ACTION", f"{rid} has no action {act['name']}")
            names.append(full)
            # EMOS ends the episode when all rearrange_stop args are 1, which is
            # vacuously true when none are sent, so always send an explicit 0.
            if prefix + "rearrange_stop" in self.env.action_space.spaces:
                args[prefix + "rearrange_stop"] = np.array([0.0], dtype=np.float32)
            for k, v in act.get("args", {}).items():
                args[prefix + k] = np.array(v, dtype=np.float32)
        if self.env.episode_over:
            raise SimHostError("EPISODE_OVER", "call reset first")
        # habitat takes the observations from the *last* action's return value;
        # arm pick/place return an ndarray, so they must not come last
        returns_array = ("arm_pick_action", "arm_place_action")
        names.sort(key=lambda n: 0 if n.endswith(returns_array) else 1)
        if names[-1].endswith(returns_array):
            names.append(names[-1].split("arm_")[0] + "wait")  # no-op, returns None
            args[names[-1]] = np.array([1.0], dtype=np.float32)
        self._last_obs = self.env.step({"action": tuple(names), "action_args": args})
        self._step_count += 1
        if self._recorder is not None:
            self._recorder.capture(self._step_count)
        return {"step": self._step_count, "episode_over": self.env.episode_over,
                "robots": self.fleet(), "metrics": _jsonable(self.env.get_metrics())}

    # ---------------------------------------------------------------- recording

    def start_recording(self, path: str, every: int = 2, fps: int = 15) -> Dict[str, Any]:
        from .recorder import Recorder  # imageio/cv2 only needed when recording
        if self._recorder is not None:
            self._recorder.close()
        self._recorder = Recorder(self, path, every=every, fps=fps)
        self._recorder.capture(0)
        return {"recording": path}

    def stop_recording(self) -> Dict[str, Any]:
        if self._recorder is None:
            raise SimHostError("NOT_RECORDING", "call start_recording first")
        out, self._recorder = self._recorder.close(), None
        return out

    def info(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "pool": self.robots,
            "num_episodes": len(self.env.episodes),
            "actions": sorted(self.env.action_space.spaces.keys()),
        }
