"""Standard-library client for the sim host. Safe to import from any Python >= 3.8."""

import json
import urllib.request
from typing import Any, Dict, List, Optional


class SimHostError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


class SimClient:
    def __init__(self, url: str = "http://127.0.0.1:8765", timeout: float = 300.0):
        self.url = url.rstrip("/") + "/rpc"
        self.timeout = timeout

    def call(self, method: str, **params) -> Any:
        body = json.dumps({"method": method, "params": params}).encode()
        req = urllib.request.Request(self.url, body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                reply = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            reply = json.loads(e.read())
        if "error" in reply:
            raise SimHostError(reply["error"]["code"], reply["error"]["message"])
        return reply["result"]

    def info(self) -> Dict[str, Any]:
        return self.call("info")

    def episodes(self) -> List[str]:
        return self.call("episodes")

    def reset(self, episode_id: Optional[str] = None, active: Optional[List[str]] = None):
        return self.call("reset", episode_id=episode_id, active=active)

    def state(self) -> Dict[str, Any]:
        return self.call("state")

    def fleet(self) -> List[Dict[str, Any]]:
        return self.call("fleet")

    def activate(self, robot_id: str, pos=None, yaw=None):
        return self.call("activate", robot_id=robot_id, pos=pos, yaw=yaw)

    def deactivate(self, robot_id: str):
        return self.call("deactivate", robot_id=robot_id)

    def step(self, actions: Optional[Dict[str, Dict[str, Any]]] = None):
        return self.call("step", actions=actions)

    def skills(self, robot_id: str) -> List[Dict[str, Any]]:
        return self.call("skills", robot_id=robot_id)

    def robot(self, robot_id: str) -> Dict[str, Any]:
        return self.call("robot", robot_id=robot_id)

    def start_recording(self, path: str, every: int = 2, fps: int = 15):
        return self.call("start_recording", path=path, every=every, fps=fps)

    def stop_recording(self) -> Dict[str, Any]:
        return self.call("stop_recording")

    def run_skill(self, robot_id: str, skill: str, **args) -> Dict[str, Any]:
        """Blocks until the skill finishes. Returns {"status", "code", "message", "steps", "robot"}."""
        return self.call("run_skill", robot_id=robot_id, skill=skill, args=args)
