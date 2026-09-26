"""End-to-end MCP test: spawn robot MCP servers over stdio and drive the sim.

    conda activate habitat-mas
    python -m mas.sim_host.server --benchmark hssd_fetch_stretch &
    .venv/bin/python tests/smoke_mcp_robots.py
"""

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

from mas.sim_host.client import SimClient  # noqa: E402


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        sys.exit(1)


async def connect(stack: AsyncExitStack, robot_id: str) -> ClientSession:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mas.mcp_servers.robot_server", "--robot", robot_id],
        cwd=ROOT,
    )
    read, write = await stack.enter_async_context(stdio_client(params))
    session = await stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    return session


def result_of(call) -> dict:
    if call.structured_content is not None:
        sc = call.structured_content
        return sc.get("result", sc)
    return json.loads(call.content[0].text)


async def main():
    sim = SimClient()
    st = sim.reset(episode_id=sim.episodes()[0], active=["fetch_0", "stretch_0"])
    obj = st["objects"][0]

    async with AsyncExitStack() as stack:
        fetch = await connect(stack, "fetch_0")
        stretch = await connect(stack, "stretch_0")

        tools = {t.name: t for t in (await fetch.list_tools()).tools}
        print("fetch tools:", sorted(tools))
        check({"navigate_to", "pick", "place"} <= set(tools), "fetch advertises navigate_to/pick/place")
        check(tools["pick"].meta["pddl"]["precondition"].startswith("(and (robot-at"),
              "tool meta carries the PDDL schema")
        print("  pick schema:", json.dumps(tools["pick"].input_schema["properties"]))

        caps = await stretch.read_resource("robot://stretch_0/capabilities")
        caps = json.loads(caps.contents[0].text)
        print("stretch capabilities:", caps)
        check(caps["mobility"] == "wheeled" and caps["arm_reach"]["max_height_m"] > 1.0,
              "capability resource reports mobility and arm reach")

        async def call(session, tool, **args):
            r = result_of(await session.call_tool(tool, args))
            print(f"  {tool}({args}) -> {r['status']} {r['code']} steps={r['steps']} {r['message']}", flush=True)
            return r

        # fetch pick-and-place through MCP while stretch looks in parallel
        async def fetch_job():
            ok = [(await call(fetch, "navigate_to", target=obj["name"]))["status"],
                  (await call(fetch, "pick", object=obj["name"]))["status"],
                  (await call(fetch, "navigate_to", target=obj["goal_receptacle"]))["status"],
                  (await call(fetch, "place", object=obj["name"], place=obj["goal_receptacle"]))["status"]]
            return ok

        fetch_ok, look = await asyncio.gather(fetch_job(), call(stretch, "look"))
        check(all(s == "ok" for s in fetch_ok), f"fetch moved {obj['name']} to {obj['goal_receptacle']} via MCP")
        check(look["status"] == "ok", "stretch look ran in parallel")

        bad = await call(fetch, "place", object=obj["name"], place=obj["goal_receptacle"])
        check(bad["status"] == "failed" and bad["code"] == "NOT_HOLDING",
              "a failed precondition returns a structured failure")

    facts = sim.state()["facts"]
    check(f"(obj-at {obj['name']} {obj['goal_receptacle']})" in facts, "symbolic facts updated")
    print("\nall checks passed")


if __name__ == "__main__":
    asyncio.run(main())
