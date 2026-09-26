"""Skill runners: turn one catalog skill into many low-level sim steps.

Each runner is driven by the scheduler, one sim step at a time, so several
robots can run skills in parallel. A runner never raises for task-level
problems; it returns a structured result the executor can recover from
(design.md §6):

  {"status": "ok" | "failed", "code": "...", "message": "...", "steps": n}
"""

from typing import Any, Dict, Optional

import numpy as np

from mas.skills.catalog import SKILLS

from .host import SimHost, SimHostError

PLACE_THRESH = 0.15   # m, held object to goal location before release
STALL_EPS = 1e-3      # m, end-effector movement counted as "not moving"
STALL_STEPS = 25      # consecutive stalled steps before giving up
NAV_STALL_STEPS = 60  # base not moving this long (after it moved): navigation is over
NAV_NEAR_M = 1.0      # ...and counts as arrived if this close to the target


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


class _ArmReset:
    """Shared tail of pick/place: move the arm back to rest."""

    def __init__(self):
        self.prev_joints = None
        self.still = 0

    def action(self):
        return {"name": "arm_reset_action", "args": {"arm_reset_action": [1.0]}}

    def done(self, agent) -> bool:
        joints = np.array(agent.articulated_agent.arm_joint_pos)
        if self.prev_joints is not None and np.abs(joints - self.prev_joints).max() < 1e-3:
            self.still += 1
        else:
            self.still = 0
        self.prev_joints = joints
        return self.still >= 3


class _Stall:
    def __init__(self):
        self.prev = None
        self.count = 0

    def update(self, pos) -> bool:
        pos = np.array(pos)
        if self.prev is not None and np.linalg.norm(pos - self.prev) < STALL_EPS:
            self.count += 1
        else:
            self.count = 0
        self.prev = pos
        return self.count >= STALL_STEPS


class NavigateTo(SkillRunner):
    max_steps = 1500

    def __init__(self, host, robot_id, target: str):
        super().__init__(host, robot_id)
        self.target = target
        self.entity = host.resolve_nav_target(robot_id, target)
        self._last_pos, self._still, self._moved = None, 0, False

    def immediate_result(self):
        nav = self.task_action("oracle_nav_action")
        # the nav action ignores a repeated request for a target it already reached
        if getattr(nav, "prev_nav_done", False) and getattr(nav, "prev_match_target_id", None) == self.entity:
            self.host.set_robot_place(self.robot_id, self.target)
            return self.ok(f"already at {self.target}")
        return None

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


class Pick(SkillRunner):
    max_steps = 400

    def __init__(self, host, robot_id, object: str):
        super().__init__(host, robot_id)
        self.object = object
        self.entity = host.resolve_object(object)
        self.handle = host._names.handle(object)
        self.stall = _Stall()
        self.reset = None

    def immediate_result(self):
        if self.agent.grasp_mgr.is_grasped:
            raise SkillFailure("HAND_FULL", f"{self.robot_id} is already holding {self.host.robot(self.robot_id)['holding']}")
        return None

    def action(self):
        if self.reset:
            return self.reset.action()
        return {"name": "arm_pick_action",
                "args": {"arm_pick_action": [self.entity, 1.0], "grip_pick_action": [1.0]}}

    def after_step(self):
        gm = self.agent.grasp_mgr
        if self.reset:
            return self.ok(f"holding {self.object}") if self.reset.done(self.agent) else None
        if gm.is_grasped:
            held = self.host.env.sim.get_rigid_object_manager().get_object_by_id(gm.snap_idx)
            if held.handle != self.handle:
                gm.desnap()
                raise SkillFailure("WRONG_OBJECT", f"grasped {self.host._names.name(held.handle)} instead")
            self.host.on_pick(self.robot_id, self.object)
            self.reset = _ArmReset()
            return None
        if self.stall.update(self.agent.articulated_agent.ee_transform().translation):
            raise SkillFailure("OUT_OF_REACH", f"arm stopped before reaching {self.object}")
        return None


class Place(SkillRunner):
    max_steps = 400

    def __init__(self, host, robot_id, object: str, place: str):
        super().__init__(host, robot_id)
        self.object, self.place = object, place
        self.entity = host.resolve_goal_place(place, object)
        self.goal_pos = np.array(host.entity_pos(self.entity))
        self.stall = _Stall()
        self.releasing = False
        self.reset = None

    def immediate_result(self):
        held = self.host.robot(self.robot_id)["holding"]
        if held != self.object:
            raise SkillFailure("NOT_HOLDING", f"{self.robot_id} holds {held}, not {self.object}")
        return None

    def action(self):
        if self.reset:
            return self.reset.action()
        if self.releasing:
            return {"name": "arm_place_action",
                    "args": {"arm_place_action": [self.entity, 0.0], "grip_place_action": [-1.0]}}
        return {"name": "arm_place_action",
                "args": {"arm_place_action": [self.entity, 2.0], "grip_place_action": [1.0]}}

    def after_step(self):
        if self.reset:
            return self.ok(f"{self.object} on {self.place}") if self.reset.done(self.agent) else None
        if self.releasing:
            if not self.agent.grasp_mgr.is_grasped:
                self.host.on_place(self.robot_id, self.object, self.place)
                self.reset = _ArmReset()
            return None
        obj = self.host.env.sim.get_rigid_object_manager().get_object_by_handle(
            self.host._names.handle(self.object))
        if np.linalg.norm(np.array(obj.translation) - self.goal_pos) < PLACE_THRESH:
            self.releasing = True
            return None
        if self.stall.update(self.agent.articulated_agent.ee_transform().translation):
            raise SkillFailure("OUT_OF_REACH", f"arm stopped before reaching {self.place}")
        return None


class Look(SkillRunner):
    max_steps = 1

    def action(self):
        return {"name": "wait", "args": {"wait": [1.0]}}

    def after_step(self):
        seen = self.host.visible_objects(self.robot_id)
        self.host.on_look(self.robot_id)
        return self.ok(f"sees {seen}" if seen else "sees no task objects", visible=seen)


class ResetArm(SkillRunner):
    max_steps = 80

    def __init__(self, host, robot_id):
        super().__init__(host, robot_id)
        self.reset = _ArmReset()

    def action(self):
        return self.reset.action()

    def after_step(self):
        return self.ok("arm at rest") if self.reset.done(self.agent) else None


class Wait(SkillRunner):
    def __init__(self, host, robot_id, steps: int = 1):
        super().__init__(host, robot_id)
        self.n = max(1, min(int(steps), 200))

    def action(self):
        return {"name": "wait", "args": {"wait": [1.0]}}

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
