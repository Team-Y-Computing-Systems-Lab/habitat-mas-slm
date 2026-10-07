"""Skill runners: turn one catalog skill into many low-level sim steps.

Each runner is driven by the scheduler, one sim step at a time, so several
robots can run skills in parallel. A runner never raises for task-level
problems; it returns a structured result the executor can recover from
(design.md §6):

  {"status": "ok" | "failed", "code": "...", "message": "...", "steps": n}

Navigation uses habitat's official OracleNavAction. Arm skills (pick, place,
reset_arm) are our own controller: turn the base so the arm can reach, move
the end effector a few centimetres per step with IK on the robot's official
URDF (ik.py), then grasp or release with habitat's grasp manager.
"""

from typing import Any, Dict, Optional

import numpy as np

from mas.skills.catalog import SKILLS

from .host import IDLE, SimHost, SimHostError

GRASP_DIST = 0.15       # m, end effector to object for a grasp (habitat's magic-grasp distance)
PLACE_THRESH = 0.15     # m, held object to its goal spot before release
EE_STEP = 0.04          # m, end-effector motion per step
ARM_STALL_STEPS = 25    # steps without 3 mm of progress: out of reach
JOINT_STEP = 0.15       # rad (or m for prismatic joints) per step when returning to rest
TURN_TOL = np.radians(3)
NAV_STALL_STEPS = 60    # base not moving this long (after it moved): navigation is over
NAV_NEAR_M = 1.0        # ...and counts as arrived if this close to the target
REACH_OK = 0.12         # m, IK residual below which a base pose counts as reaching the target
                        #    (a grasp succeeds within GRASP_DIST = 0.15 m)
APPROACH_DISTS = (0.45, 0.6, 0.75, 0.9)  # m from the target, for alternative base poses
MAX_ATTEMPTS = 3        # base poses to try before OUT_OF_REACH


class SkillFailure(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class SkillRunner:
    max_steps = 200

    def __init__(self, host: SimHost, robot_id: str):
        self.host = host
        self.robot_id = robot_id
        self.idx = host._by_id[robot_id]["agent_idx"]
        self.steps = 0

    # overridden by skills
    def immediate_result(self) -> Optional[dict]:
        """A result that needs no sim steps (e.g. already there)."""
        return None

    def action(self) -> Dict[str, Any]:
        raise NotImplementedError

    def after_step(self) -> Optional[dict]:
        """Return a result when done, else None."""
        raise NotImplementedError

    # helpers
    @property
    def agent(self):
        return self.host.env.sim.agents_mgr[self.idx]

    def task_action(self, name: str):
        return self.host.env.task.actions[f"agent_{self.idx}_{name}"]

    def ok(self, message: str = "", **extra) -> dict:
        return {"status": "ok", "code": "OK", "message": message, "steps": self.steps, **extra}

    def tick(self) -> Optional[dict]:
        self.steps += 1
        result = self.after_step()
        if result is None and self.steps >= self.max_steps:
            raise SkillFailure("TIMEOUT", f"not done after {self.steps} steps")
        return result


class NavigateTo(SkillRunner):
    max_steps = 1500

    def __init__(self, host, robot_id, target: str):
        super().__init__(host, robot_id)
        self.target = target
        self.entity = host.resolve_nav_target(robot_id, target)
        self._last_pos, self._still, self._moved = None, 0, False

    def action(self):
        return {"name": "oracle_nav_action", "args": {"oracle_nav_action": [self.entity + 1]}}

    def after_step(self):
        nav = self.task_action("oracle_nav_action")
        if nav.skill_done:
            self.host.set_robot_place(self.robot_id, self.target)
            return self.ok(f"at {self.target}")
        # habitat's nav can livelock: blocked short of its approach point, it
        # keeps turning in place and never reports done
        # (the initial turn-on-the-spot is not a stall: count only after moving)
        pos = np.array(self.agent.articulated_agent.base_pos)
        if self._last_pos is None:
            self._last_pos = pos
        elif np.linalg.norm((pos - self._last_pos)[[0, 2]]) < 0.01:
            self._still += 1 if self._moved else 0
        else:
            self._still, self._last_pos, self._moved = 0, pos, True
        if self._still >= NAV_STALL_STEPS:
            _, target = nav._targets.get(self.entity, (None, None))
            dist = float(np.linalg.norm((np.array(target) - pos)[[0, 2]])) if target is not None else 1e9
            if dist <= NAV_NEAR_M:
                self.host.set_robot_place(self.robot_id, self.target)
                return self.ok(f"near {self.target} ({dist:.2f} m; approach point blocked)")
            raise SkillFailure("NAV_STUCK", f"stopped {dist:.2f} m from {self.target}")
        return None


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


class ArmSkill(SkillRunner):
    """plan -> [drive] -> orient -> reach -> (subclass: grasp/release) -> rest.

    plan:  with the IK model (no sim steps), find base poses from which the arm
           reaches the target: first the current spot at any heading, then
           floor positions 0.45-0.9 m around the target. If none is predicted
           to reach, the best heading on the current spot is still tried.
    drive: official coordinate navigation to the chosen position.
    If the arm still stalls, retract and try the next pose (MAX_ATTEMPTS).
    """
    max_steps = 1500

    def __init__(self, host, robot_id):
        super().__init__(host, robot_id)
        self.phase = "plan"
        self.poses: list = []
        self.attempt = 0
        self.yaw_goal: Optional[float] = None
        self.turn_still = 0
        self.best = np.inf
        self.stall = 0
        self.after_rest: Optional[str] = None

    # subclasses
    def target_world(self) -> np.ndarray:
        raise NotImplementedError

    def contact(self) -> bool:
        """At the target: grasp or release. Return True when done."""
        raise NotImplementedError

    def contact_distance(self) -> float:
        return float(np.linalg.norm(self.host.ee_world(self.robot_id) - self.target_world()))

    # -- planning base poses with the IK model --------------------------------
    def _residual(self, pos, yaw: float) -> float:
        art = self.agent.articulated_agent
        ik = self.host.arm(self.robot_id)
        saved = (np.array(art.base_pos), float(art.base_rot))
        art.base_pos, art.base_rot = np.asarray(pos, dtype=np.float32), float(yaw)
        local = self.host.to_local(self.robot_id, self.target_world())
        res = float(np.linalg.norm(ik.fk(ik.ik(local, iterations=25)) - local))
        art.base_pos, art.base_rot = saved
        return res

    def _plan_poses(self) -> list:
        import habitat_sim
        art = self.agent.articulated_agent
        pf = self.host.env.sim.pathfinder
        here, yaw0 = np.array(art.base_pos), float(art.base_rot)
        target = self.target_world()
        poses = []
        scored = [(self._residual(here, yaw0 + d), abs(_wrap(d)), yaw0 + d) for d in np.radians(np.arange(0, 360, 15))]
        stay = [x for x in scored if x[0] < REACH_OK]
        if stay:  # smallest turn on the spot
            poses.append({"pos": here, "yaw": min(stay, key=lambda x: x[1])[2], "dist": 0.0})
        island = pf.get_island(here)
        for d in APPROACH_DISTS:
            for a in np.radians(np.arange(0, 360, 30)):
                p = np.array([target[0] + d * np.cos(a), here[1], target[2] + d * np.sin(a)], dtype=np.float32)
                s = np.array(pf.snap_point(p))
                if np.isnan(s).any() or np.linalg.norm((s - p)[[0, 2]]) > 0.15 or pf.get_island(s) != island:
                    continue
                path = habitat_sim.ShortestPath()
                path.requested_start, path.requested_end = here, s
                if not pf.find_path(path):
                    continue
                ok = [(self._residual(s, y), y) for y in np.radians(np.arange(0, 360, 45))]
                res, yaw = min(ok)
                if res < REACH_OK:
                    poses.append({"pos": s, "yaw": float(yaw), "dist": float(path.geodesic_distance)})
        poses.sort(key=lambda x: x["dist"])
        if not poses:  # nothing predicted to reach: still try the best heading on the spot
            poses.append({"pos": here, "yaw": min(scored)[2], "dist": 0.0})
        return poses

    def _next_pose(self) -> None:
        pose = self.poses.pop(0)
        self.attempt += 1
        self.yaw_goal, self._best_err, self.turn_still = pose["yaw"], np.inf, 0
        self.best, self.stall = np.inf, 0
        if pose["dist"] > 0:
            self.phase, self.drive_to, self.drive_steps, self._drive_prev = "drive", pose["pos"], 0, None
        else:
            self.phase = "orient"

    # -- stepping ---------------------------------------------------------------
    def action(self):
        if self.phase == "plan":
            self.poses = self._plan_poses()
            self._next_pose()
        if self.phase == "drive":
            return {"name": "oracle_nav_coord_action",
                    "args": {"oracle_nav_coord_action": list(map(float, self.drive_to)),
                             "oracle_nav_lookat_action": list(map(float, self.target_world())), "mode": [0]}}
        if self.phase == "orient":
            err = _wrap(self.yaw_goal - float(self.agent.articulated_agent.base_rot))
            # stop turning when aligned, or when the heading stops improving
            # (the base is blocked and jitters in place)
            if abs(err) > TURN_TOL and self.turn_still < 20:
                return {"name": "base_velocity", "args": {"base_vel": [0.0, float(np.clip(3 * err, -1, 1))]}}
            self.phase = "reach"
        if self.phase == "reach":
            ik = self.host.arm(self.robot_id)
            ee = ik.fk()
            goal = self.host.to_local(self.robot_id, self.target_world())
            delta = goal - ee
            n = float(np.linalg.norm(delta))
            self.host.set_arm(self.robot_id, ik.ik(ee + delta * min(1.0, EE_STEP / max(n, 1e-6))))
        elif self.phase == "rest":
            cur = np.array(self.agent.articulated_agent.arm_joint_pos)
            self.host.set_arm(self.robot_id, cur + np.clip(self._rest() - cur, -JOINT_STEP, JOINT_STEP))
        return IDLE

    def _rest(self) -> np.ndarray:
        # the start pose can sit slightly outside joint limits habitat enforces
        lo, hi = self.host.arm(self.robot_id).limits()
        return np.clip(self.host._rest_q[self.robot_id], lo, hi)

    def after_step(self):
        art = self.agent.articulated_agent
        if self.phase == "drive":
            self.drive_steps += 1
            pos = np.array(art.base_pos)
            moved = self._drive_prev is None or np.linalg.norm((pos - self._drive_prev)[[0, 2]]) > 0.005
            self._drive_still = 0 if moved else getattr(self, "_drive_still", 0) + 1
            self._drive_prev = pos
            close = np.linalg.norm((pos - self.drive_to)[[0, 2]]) < 0.15
            if self.task_action("oracle_nav_coord_action").skill_done or close \
                    or self._drive_still > 40 or self.drive_steps > 600:
                self.phase = "orient"
            return None
        if self.phase == "orient":
            err = abs(_wrap(self.yaw_goal - float(art.base_rot)))
            if err < self._best_err - np.radians(0.5):
                self._best_err, self.turn_still = err, 0
            else:
                self.turn_still += 1
            return None
        if self.phase == "rest":
            cur = np.array(art.arm_joint_pos)
            moved = getattr(self, "_rest_prev", None) is None or np.abs(cur - self._rest_prev).max() > 1e-4
            self._rest_prev = cur
            if np.abs(cur - self._rest()).max() < 0.01 or not moved:
                self._rest_prev = None
                if self.after_rest == "retry":  # retracted; go to the next pose
                    self.after_rest = None
                    self._next_pose()
                    return None
                return self.ok(self.done_message)
            return None
        d = self.contact_distance()
        if self.contact():
            self.phase = "rest"
            return None
        if d < self.best - 0.003:
            self.best, self.stall = d, 0
        else:
            self.stall += 1
        if self.stall >= ARM_STALL_STEPS:
            if self.poses and self.attempt < MAX_ATTEMPTS:
                self.phase, self.after_rest = "rest", "retry"
                return None
            raise SkillFailure("OUT_OF_REACH", f"arm stopped {d:.2f} m from the target after {self.attempt} pose(s)")
        return None


class Pick(ArmSkill):
    def __init__(self, host, robot_id, object: str):
        super().__init__(host, robot_id)
        self.object = object
        self.handle = host._names.handle(object)
        host.resolve_object(object)  # raises UNKNOWN_OBJECT for a bad name
        self.done_message = f"holding {object}"

    def _obj(self):
        return self.host.env.sim.get_rigid_object_manager().get_object_by_handle(self.handle)

    def immediate_result(self):
        if self.agent.grasp_mgr.is_grasped:
            raise SkillFailure("HAND_FULL", f"{self.robot_id} is already holding {self.host.robot(self.robot_id)['holding']}")
        return None

    def target_world(self):
        return np.array(self._obj().translation)

    def contact(self) -> bool:
        if self.contact_distance() >= GRASP_DIST:
            return False
        self.agent.grasp_mgr.snap_to_obj(self._obj().object_id)
        self.host.on_pick(self.robot_id, self.object)
        return True


class Place(ArmSkill):
    def __init__(self, host, robot_id, object: str, place: str):
        super().__init__(host, robot_id)
        self.object, self.place = object, place
        self.goal = np.array(host.entity_pos(host.resolve_goal_place(place, object)))
        self.done_message = f"{object} on {place}"

    def immediate_result(self):
        held = self.host.robot(self.robot_id)["holding"]
        if held != self.object:
            raise SkillFailure("NOT_HOLDING", f"{self.robot_id} holds {held}, not {self.object}")
        return None

    def target_world(self):
        return self.goal

    def contact_distance(self) -> float:
        obj = self.host.env.sim.get_rigid_object_manager().get_object_by_handle(self.host._names.handle(self.object))
        return float(np.linalg.norm(np.array(obj.translation) - self.goal))

    def contact(self) -> bool:
        if self.contact_distance() >= PLACE_THRESH:
            return False
        self.agent.grasp_mgr.desnap()
        self.host.on_place(self.robot_id, self.object, self.place)
        return True


class ResetArm(ArmSkill):
    def __init__(self, host, robot_id):
        super().__init__(host, robot_id)
        self.phase = "rest"  # no planning or reaching
        self.done_message = "arm at rest"


class Look(SkillRunner):
    max_steps = 1

    def action(self):
        return IDLE

    def after_step(self):
        seen = self.host.visible_objects(self.robot_id)
        self.host.on_look(self.robot_id)
        return self.ok(f"sees {seen}" if seen else "sees no task objects", visible=seen)


class Wait(SkillRunner):
    def __init__(self, host, robot_id, steps: int = 1):
        super().__init__(host, robot_id)
        self.n = max(1, min(int(steps), 200))

    def action(self):
        return IDLE

    def after_step(self):
        return self.ok(f"waited {self.steps} steps") if self.steps >= self.n else None


RUNNERS = {
    "navigate_to": NavigateTo,
    "pick": Pick,
    "place": Place,
    "look": Look,
    "reset_arm": ResetArm,
    "wait": Wait,
}
assert set(RUNNERS) == set(SKILLS), "every catalog skill needs a runner"


def make_runner(host: SimHost, robot_id: str, skill: str, args: Dict[str, Any]) -> SkillRunner:
    if skill not in RUNNERS:
        raise SkillFailure("UNKNOWN_SKILL", skill)
    if skill not in {s["name"] for s in host.skills(robot_id)}:
        raise SkillFailure("SKILL_UNAVAILABLE", f"{robot_id} cannot {skill}")
    try:
        return RUNNERS[skill](host, robot_id, **args)
    except TypeError as e:
        raise SkillFailure("BAD_ARGS", str(e))
    except SimHostError as e:
        raise SkillFailure(e.code, str(e))
