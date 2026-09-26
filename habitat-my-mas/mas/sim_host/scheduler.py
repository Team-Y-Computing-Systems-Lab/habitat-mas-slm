"""Runs every robot's current skill together, one sim step at a time.

All simulator calls happen on the thread that calls `run_forever()` (the
renderer's GL context belongs to it). HTTP handler threads only `submit()`
requests and wait for replies.

`run_skill` requests stay open until the skill finishes, so each robot's MCP
server can block on its own skill while other robots keep moving
(decentralized execution, design.md §3).
"""

import queue
import threading
import traceback
from typing import Any, Dict, Tuple

from .host import SimHost, SimHostError
from .skills import SkillFailure, SkillRunner, make_runner

# methods callable directly on the host between sim steps
HOST_METHODS = {"info", "episodes", "reset", "state", "fleet", "robot",
                "activate", "deactivate", "step", "skills", "facts", "task_goal", "goal_status", "arm_reach", "check_reach", "feasibility",
                "start_recording", "stop_recording"}


class _Request:
    def __init__(self, method: str, params: Dict[str, Any]):
        self.method, self.params = method, params
        self.done = threading.Event()
        self.reply: Dict[str, Any] = {}

    def finish(self, **reply):
        self.reply = reply
        self.done.set()


class Scheduler:
    def __init__(self, host: SimHost):
        self.host = host
        self.inbox: "queue.Queue[_Request]" = queue.Queue()
        self.active: Dict[str, Tuple[SkillRunner, _Request, str]] = {}

    def submit(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        req = _Request(method, params)
        self.inbox.put(req)
        req.done.wait()
        return req.reply

    # ------------------------------------------------------------ sim thread

    def run_forever(self):
        while True:
            try:
                # block only when idle; otherwise keep stepping
                req = self.inbox.get(timeout=None if not self.active else 0)
                self._handle(req)
                while not self.inbox.empty():
                    self._handle(self.inbox.get_nowait())
            except queue.Empty:
                pass
            if self.active:
                self._step()

    def _handle(self, req: _Request):
        try:
            if req.method == "run_skill":
                return self._start_skill(req, **req.params)
            if req.method == "active_skills":
                return req.finish(result={rid: name for rid, (_, _, name) in self.active.items()})
            if req.method not in HOST_METHODS:
                raise SimHostError("UNKNOWN_METHOD", req.method)
            if req.method in ("step", "reset", "feasibility") and self.active:
                raise SimHostError("SIM_BUSY", f"skills running: {sorted(self.active)}")
            result = getattr(self.host, req.method)(**req.params)
            if req.method == "deactivate":
                self._fail(req.params["robot_id"], "ROBOT_LOST", "robot left the fleet")
            req.finish(result=result)
        except SimHostError as e:
            req.finish(error={"code": e.code, "message": str(e)})
        except Exception as e:
            traceback.print_exc()
            req.finish(error={"code": "INTERNAL", "message": repr(e)})

    def _start_skill(self, req: _Request, robot_id: str, skill: str, args: Dict[str, Any] = None):
        if robot_id not in self.host._by_id:
            return req.finish(result=self._failed("UNKNOWN_ROBOT", robot_id))
        if not self.host._active.get(robot_id):
            return req.finish(result=self._failed("ROBOT_INACTIVE", f"{robot_id} is not in the fleet"))
        if robot_id in self.active:
            return req.finish(result=self._failed("ROBOT_BUSY", f"{robot_id} is running {self.active[robot_id][2]}"))
        if self.host.env.episode_over:
            return req.finish(result=self._failed("EPISODE_OVER", "reset the episode"))
        try:
            runner = make_runner(self.host, robot_id, skill, args or {})
            result = runner.immediate_result()
        except SkillFailure as e:
            return req.finish(result=self._failed(e.code, str(e)))
        if result is not None:
            return req.finish(result=self._with_robot(result, robot_id))
        self.active[robot_id] = (runner, req, skill)
        shown = ", ".join(str(v) for v in (args or {}).values())
        self.host.labels[robot_id] = f"{skill}({shown})"

    def _step(self):
        actions = {rid: runner.action() for rid, (runner, _, _) in self.active.items()}
        try:
            self.host.step(actions)
        except SimHostError as e:
            for rid in list(self.active):
                self._fail(rid, e.code, str(e))
            return
        except Exception as e:  # never let a sim error kill the server
            traceback.print_exc()
            for rid in list(self.active):
                self._fail(rid, "INTERNAL", repr(e))
            return
        for rid in list(self.active):
            runner, req, _ = self.active[rid]
            try:
                result = runner.tick()
            except SkillFailure as e:
                result = self._failed(e.code, str(e), runner.steps)
            except Exception as e:
                traceback.print_exc()
                result = self._failed("INTERNAL", repr(e), runner.steps)
            if result is not None:
                del self.active[rid]
                self.host.labels[rid] = f"{self.host.labels.get(rid, '')} {result['code']}"
                req.finish(result=self._with_robot(result, rid))
        if self.host.env.episode_over:
            for rid in list(self.active):
                self._fail(rid, "EPISODE_OVER", "episode ended while the skill was running")

    def _fail(self, robot_id: str, code: str, message: str):
        if robot_id in self.active:
            runner, req, _ = self.active.pop(robot_id)
            req.finish(result=self._with_robot(self._failed(code, message, runner.steps), robot_id))

    @staticmethod
    def _failed(code: str, message: str, steps: int = 0) -> Dict[str, Any]:
        return {"status": "failed", "code": code, "message": message, "steps": steps}

    def _with_robot(self, result: Dict[str, Any], robot_id: str) -> Dict[str, Any]:
        try:
            result["robot"] = self.host.robot(robot_id)
        except Exception:
            pass
        return result
