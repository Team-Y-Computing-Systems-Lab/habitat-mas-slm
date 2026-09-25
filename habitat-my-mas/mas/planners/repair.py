"""Deterministic plan fixes and checks applied to SLM plans (docs/findings.md F2, F5).

normalize(plan)  rewrites `navigate_to(<place>)` followed by `pick(<object>)`
                 into `navigate_to(<object>)`: the SLM almost never navigates to
                 the object itself, and a place name is ambiguous when several
                 objects sit on it.
check(plan, ...) walks each robot's step list over the symbolic state and
                 lists concrete problems (wrong order, holding two things, an
                 object the robot cannot reach, goals never achieved). The
                 errors are written for the SLM, which gets one round to repair.
"""

from typing import Dict, List, Optional

from .base import Plan, Step


def normalize(plan: Plan, object_names: List[str]) -> List[str]:
    """Rewrite navigation targets in place; return a description of each change."""
    changes = []
    objs = set(object_names)
    for robot, steps in plan.steps.items():
        for i in range(len(steps) - 1):
            nav, nxt = steps[i], steps[i + 1]
            if nav.skill == "navigate_to" and nxt.skill == "pick":
                target, obj = nav.args.get("target"), nxt.args.get("object")
                if obj in objs and target != obj:
                    changes.append(f"{robot} step {i}: navigate_to({target}) -> navigate_to({obj})")
                    steps[i] = Step("navigate_to", {"target": obj})
    return changes


def check(plan: Plan, state: dict, goal: List[str],
          feasibility: Optional[Dict[str, Dict[str, dict]]] = None) -> List[str]:
    """Problems with the plan, phrased as instructions for fixing it."""
    errors = []
    where = {}  # object -> place it currently rests on
    for f in state["facts"]:
        p = f.strip("()").split()
        if p[0] == "obj-at":
            where[p[1]] = p[2]
    owner: Dict[str, str] = {}
    delivered: Dict[str, str] = {}
    held_at_end: Dict[str, Optional[str]] = {}

    for robot, steps in plan.steps.items():
        loc, holding = None, None
        for i, s in enumerate(steps):
            tag = f"{robot} step {i + 1} {s.skill}({', '.join(map(str, s.args.values()))})"
            if s.skill == "navigate_to":
                loc = s.args.get("target")
            elif s.skill == "pick":
                obj = s.args.get("object")
                if holding:
                    errors.append(f"{tag}: {robot} is still holding {holding}; place it before picking another object")
                if loc not in (obj, where.get(obj)):
                    errors.append(f"{tag}: {robot} is not at {obj}; add navigate_to({obj}) right before this pick")
                if obj in owner and owner[obj] != robot:
                    errors.append(f"{tag}: {obj} is also given to {owner[obj]}; give each object to one robot")
                owner[obj] = robot
                row = (feasibility or {}).get(robot, {}).get(obj)
                if feasibility is not None and (row is None or row["pick"] != "OK"):
                    errors.append(f"{tag}: {robot} cannot pick {obj} (its arm cannot reach it there); "
                                  f"give {obj} to a robot that can")
                holding = obj
            elif s.skill == "place":
                obj, place = s.args.get("object"), s.args.get("place")
                if holding != obj:
                    errors.append(f"{tag}: {robot} is holding {holding or 'nothing'}, not {obj}")
                if loc != place:
                    errors.append(f"{tag}: {robot} is not at {place}; add navigate_to({place}) right before this place")
                row = (feasibility or {}).get(robot, {}).get(obj)
                if feasibility is not None and row is not None and row["pick"] == "OK" and row["place"] != "OK":
                    errors.append(f"{tag}: {robot} cannot place {obj} on {place} (the goal spot is out of its "
                                  f"reach); give {obj} to a robot that can")
                holding = None
                delivered[obj] = place
        held_at_end[robot] = holding

    for g in goal:
        p = g.strip("()").split()
        if p[0] == "obj-at" and delivered.get(p[1]) != p[2]:
            got = delivered.get(p[1])
            errors.append(f"goal not reached: {p[1]} must end on {p[2]}"
                          + (f", but the plan puts it on {got}" if got else ", but no robot places it there"))
        if p[0] == "hand-empty" and held_at_end.get(p[1]):
            errors.append(f"goal not reached: {p[1]} ends holding {held_at_end[p[1]]}")
    return errors
