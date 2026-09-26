"""P0: the SLM writes the whole multi-robot plan in one call. No classical planner.

The SLM sees the same things any planner would:
  - the instruction
  - each robot's capabilities and skills, as discovered through MCP
  - the current world state (which object is on which place, where robots are)
and returns per-robot step lists that the robots execute in parallel.

Prompt variants (docs/findings.md explains why each exists):
  base         original prompt: skills and places, no geometry
  reach        + each robot's arm reach and each object's / goal spot's height
  reach_alloc  + reach info, and the SLM must fill an allocation table
                 (who can reach what) before writing the plan
  reach_v2     + reach info, each object written as "now on X -> goes to Y",
                 and a fixed four-step template per object
  reach_hint   reach_v2 + the system compares heights and tells the SLM which
                 robots can handle each object (arithmetic moved out of the SLM)
"""

import json
from typing import Dict, List

from mas.slm.client import SLMClient, parse_json

from .base import Plan, Step, validate

PROMPT_VARIANTS = ("base", "reach", "reach_alloc", "reach_v2", "reach_hint")

RULES = """Rules:
- Each robot runs its own list in order. Different robots run at the same time.
- Use only the skills listed for that robot, with exactly the listed arguments.
- To pick an object a robot must first navigate_to that object.
- To place a held object a robot must first navigate_to the place.
- A robot can hold one object at a time.
- Never give the same object to two robots.
- Robots not needed may get an empty list."""

REACH_RULES = """
- A robot can pick an object only if the object's height is within that robot's arm reach.
- A robot can place an object only if the goal spot's height is within its arm reach.
- Give each object to a robot that can reach BOTH its height and its goal spot height.
  If only one robot can, that robot must do it, even if it then does several objects in a row."""

FORMAT = """
Answer with JSON only, in this form:
{"reasoning": "<one or two sentences>",
 "plan": {"<robot_id>": [{"skill": "<skill name>", "args": {"<arg>": "<value>"}}]}}"""

FORMAT_ALLOC = """
First fill "allocation": one row per object to move, listing which robots can reach both
its height and its goal spot height, and which robot you assign. Then write the plan so it
follows the allocation. For each assigned object the steps are:
navigate_to(object), pick(object), navigate_to(goal place), place(object, goal place).

Answer with JSON only, in this form:
{"allocation": [{"object": "<name>", "height_m": 0.0, "goal_place": "<place>", "goal_height_m": 0.0,
                 "can_reach": ["<robot_id>"], "assigned_to": "<robot_id>"}],
 "plan": {"<robot_id>": [{"skill": "<skill name>", "args": {"<arg>": "<value>"}}]}}"""

TEMPLATE_RULES = """
- To move object X to place P, a robot does exactly these four steps:
  navigate_to(X), pick(X), navigate_to(P), place(X, P).
  navigate_to the OBJECT itself before pick, never its goal place.
- Split the objects between robots only when both can reach them."""

HEAD = "You are the task planner for a team of household robots.\nAssign work to the robots and write each robot's list of skill calls.\n\n"

SYSTEM = {
    "base": HEAD + RULES + "\n" + FORMAT,
    "reach": HEAD + RULES + REACH_RULES + "\n" + FORMAT,
    "reach_alloc": HEAD + RULES + REACH_RULES + "\n" + FORMAT_ALLOC,
    "reach_v2": HEAD + RULES + REACH_RULES + TEMPLATE_RULES + "\n" + FORMAT,
    "reach_hint": HEAD + RULES + REACH_RULES + TEMPLATE_RULES + "\n" + FORMAT,
}


def can_handle(robots: List[dict], obj: dict) -> List[str]:
    """Robots whose arm reaches both the object's height and its goal spot height."""
    out = []
    for r in robots:
        ar = r["capabilities"].get("arm_reach")
        if ar and all(ar["min_height_m"] <= h <= ar["max_height_m"]
                      for h in (obj["height_m"], obj["goal_height_m"])):
            out.append(r["id"])
    return out


def describe_robots(robots: List[dict], with_reach: bool = False) -> str:
    lines = []
    for r in robots:
        caps = r["capabilities"]
        reach = ""
        if with_reach:
            ar = caps.get("arm_reach")
            reach = (f", arm reaches heights {ar['min_height_m']:.2f}-{ar['max_height_m']:.2f} m"
                     if ar else ", cannot pick or place")
        lines.append(f"- {r['id']} ({caps.get('type')}, {caps.get('mobility')} mobility, "
                     f"arm: {caps.get('arm') or 'none'}{reach})")
        for t in r["tools"]:
            if t["name"] == "status":
                continue
            args = ", ".join(t["input_schema"].get("properties", {}))
            desc = t["description"].split(" PDDL pre:")[0]  # the SLM gets prose, not PDDL
            lines.append(f"    {t['name']}({args}): {desc}")
    return "\n".join(lines)


def describe_tasks(state: dict, robots: List[dict], hint: bool) -> str:
    """One line per object: where it is now, where it goes, heights (reach_v2 / reach_hint)."""
    lines = []
    for o in state["objects"]:
        line = (f"- {o['name']}: now on {state_place(state, o['name'])} at height {o['height_m']:.2f} m"
                f" -> goes to {o['goal_receptacle']} (goal spot height {o['goal_height_m']:.2f} m)")
        if hint:
            who = can_handle(robots, o)
            line += f"; can be handled by: {', '.join(who) if who else 'no robot'}"
        lines.append(line)
    robots_txt = [f"- {r['id']} is at its start position, holding {r['holding'] or 'nothing'}"
                  for r in state["robots"] if r["active"]]
    return ("Objects (heights above the floor):\n" + "\n".join(lines)
            + "\n\nRobots:\n" + "\n".join(robots_txt))


def state_place(state: dict, obj: str) -> str:
    for f in state["facts"]:
        parts = f.strip("()").split()
        if parts[0] == "obj-at" and parts[1] == obj:
            return parts[2]
    return "unknown"


def describe_world(state: dict, with_heights: bool = False) -> str:
    on = {}
    for f in state["facts"]:
        parts = f.strip("()").split()
        if parts[0] == "obj-at":
            on.setdefault(parts[2], []).append(parts[1])
    lines = [f"- {place}: {', '.join(on.get(place, [])) or 'nothing'}" for place in state["places"]]
    robots = [f"- {r['id']} is at its start position, holding {r['holding'] or 'nothing'}"
              for r in state["robots"] if r["active"]]
    text = "Places and the task objects on them:\n" + "\n".join(lines)
    if with_heights:
        rows = [f"- {o['name']}: at height {o['height_m']:.2f} m; its goal spot on "
                f"{o['goal_receptacle']} is at height {o['goal_height_m']:.2f} m"
                for o in state["objects"] if o.get("height_m") is not None]
        text += "\n\nObject heights (above the floor):\n" + "\n".join(rows)
    return text + "\n\nRobots:\n" + "\n".join(robots)


class SLMOnlyPlanner:
    mode = "slm_only"

    def __init__(self, slm: SLMClient, prompt: str = "reach_v2"):
        if prompt not in PROMPT_VARIANTS:
            raise ValueError(f"prompt must be one of {PROMPT_VARIANTS}")
        self.slm = slm
        self.prompt = prompt

    def _user(self, instruction: str, robots: List[dict], state: dict) -> str:
        geo = self.prompt != "base"
        if self.prompt in ("reach_v2", "reach_hint"):
            world = describe_tasks(state, robots, hint=self.prompt == "reach_hint")
        else:
            world = describe_world(state, with_heights=geo)
        return (f"Robots and their skills:\n{describe_robots(robots, with_reach=geo)}\n\n"
                f"{world}\n\nInstruction: {instruction}")

    def _ask(self, user: str, robots: List[dict], out: Plan) -> Plan:
        """One SLM call; parse into `out.steps` (replacing them) and validate."""
        system = SYSTEM[self.prompt]
        call = self.slm.chat(system, user)
        out.planner_calls.append({**call.to_dict(), "system": system, "user": user})
        out.raw = call.output
        data = parse_json(call.output)
        if not isinstance(data, dict) or not isinstance(data.get("plan"), dict):
            out.parse_ok = False
            out.errors = ["output is not JSON with a 'plan' object"]
            return out
        out.parse_ok = True
        out.steps = {
            robot: [Step(skill=str(s.get("skill")), args=s.get("args") or {})
                    for s in (steps or []) if isinstance(s, dict)]
            for robot, steps in data["plan"].items()
        }
        tools: Dict[str, Dict[str, dict]] = {
            r["id"]: {t["name"]: t["input_schema"] for t in r["tools"]} for r in robots}
        out.errors = validate(out, tools)
        return out

    def plan(self, instruction: str, robots: List[dict], state: dict, ctx: dict = None) -> Plan:
        return self._ask(self._user(instruction, robots, state), robots, Plan())

    def repair(self, plan: Plan, problems: List[str], instruction: str,
               robots: List[dict], state: dict) -> Plan:
        """Second and last SLM call: the previous plan plus the checker's findings."""
        previous = json.dumps({r: [{"skill": s.skill, "args": s.args} for s in steps]
                               for r, steps in plan.steps.items()})
        user = (self._user(instruction, robots, state)
                + f"\n\nYour previous plan:\n{previous}\n\nA checker found these problems:\n"
                + "\n".join(f"- {p}" for p in problems)
                + "\n\nWrite a corrected plan that fixes every problem, in the same JSON format.")
        return self._ask(user, robots, plan)
