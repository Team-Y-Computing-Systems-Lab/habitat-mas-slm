"""One MCP server per robot: its skills as tools, its capabilities as resources.

    .venv/bin/python -m mas.mcp_servers.robot_server --robot fetch_0 [--sim-url http://127.0.0.1:8765]

Normally an MCP client (the coordinator) launches this over stdio, one process
per robot in the fleet. Tools are registered from what the robot can actually
do in the sim (mas.skills.catalog), so a drone never advertises pick/place.

Every tool returns the sim host's structured result:
  {"status": "ok"|"failed", "code": ..., "message": ..., "steps": n, "robot": {...}}
Failures are results, not protocol errors, so the executor can recover from them.

Each tool's `meta` carries its PDDL action schema, so the classical planner
can build its domain from `tools/list` alone.
"""

import argparse
import json
from typing import Any, Dict

import anyio
from mcp.server.mcpserver import MCPServer

from mas.sim_host.client import SimClient, SimHostError
from mas.skills.robots import ROBOT_TRAITS


def build_server(robot_id: str, sim: SimClient) -> MCPServer:
    info = sim.info()
    entry = next((r for r in info["pool"] if r["id"] == robot_id), None)
    if entry is None:
        raise SystemExit(f"{robot_id} not in sim pool {[r['id'] for r in info['pool']]}")
    skills = {s["name"]: s for s in sim.skills(robot_id)}
    traits = ROBOT_TRAITS.get(entry["type"], {})
    reach = sim.call("arm_reach", robot_id=robot_id)  # from URDF forward kinematics

    server = MCPServer(
        name=f"robot-{robot_id}",
        instructions=(
            f"Skills of robot {robot_id} ({entry['type']}, {traits.get('mobility', 'unknown')} mobility). "
            f"Read robot://{robot_id}/capabilities before assigning it work."
        ),
    )

    async def run(skill: str, **args) -> Dict[str, Any]:
        try:
            return await anyio.to_thread.run_sync(lambda: sim.run_skill(robot_id, skill, **args))
        except SimHostError as e:
            return {"status": "failed", "code": e.code, "message": str(e), "steps": 0}
        except OSError as e:  # connection refused/reset: the sim host is down
            return {"status": "failed", "code": "SIM_UNREACHABLE", "message": repr(e), "steps": 0}

    # One function per catalog skill; typed parameters become the tool's input schema.
    async def navigate_to(target: str) -> Dict[str, Any]:
        return await run("navigate_to", target=target)

    async def pick(object: str) -> Dict[str, Any]:
        return await run("pick", object=object)

    async def place(object: str, place: str) -> Dict[str, Any]:
        return await run("place", object=object, place=place)

    async def look() -> Dict[str, Any]:
        return await run("look")

    async def reset_arm() -> Dict[str, Any]:
        return await run("reset_arm")

    async def wait(steps: int = 10) -> Dict[str, Any]:
        return await run("wait", steps=steps)

    impls = {"navigate_to": navigate_to, "pick": pick, "place": place,
             "look": look, "reset_arm": reset_arm, "wait": wait}

    for name, skill in skills.items():
        desc = skill["description"]
        if skill["params"]:
            desc += " Args: " + "; ".join(f"{k}: {v}" for k, v in skill["params"].items()) + "."
        if skill["pddl"]:
            desc += f" PDDL pre: {skill['pddl']['precondition']} eff: {skill['pddl']['effect']}"
        server.add_tool(
            impls[name],
            name=name,
            description=desc,
            meta={"robot": robot_id, "robot_type": entry["type"], "pddl": skill["pddl"],
                  "typical_steps": skill["typical_steps"]},
        )

    @server.tool(description="Current pose, whether the robot is in the fleet, and what it holds.")
    async def status() -> Dict[str, Any]:
        return await anyio.to_thread.run_sync(lambda: sim.robot(robot_id))

    @server.resource(f"robot://{robot_id}/capabilities", mime_type="application/json",
                     description="What this robot can do: mobility, arm, and skills.")
    def capabilities() -> str:
        return json.dumps({
            "id": robot_id,
            "type": entry["type"],
            **traits,
            "skills": sorted(skills),
            "arm_reach": reach,  # {"min_height_m", "max_height_m"} or None without an arm
        })

    @server.resource(f"robot://{robot_id}/skills", mime_type="application/json",
                     description="Full skill descriptors including PDDL action schemas.")
    def skill_descriptors() -> str:
        return json.dumps(list(skills.values()))

    return server


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot", required=True)
    p.add_argument("--sim-url", default="http://127.0.0.1:8765")
    args = p.parse_args()
    build_server(args.robot, SimClient(args.sim_url, timeout=3600)).run("stdio")


if __name__ == "__main__":
    main()
