"""Skill catalog: what each skill does, what it needs, and its PDDL model.

Pure Python (3.8+) with no habitat import, so the sim host (Python 3.9) and the
MCP servers / planner (Python 3.10+) share one definition.

A robot gets a skill when it has every low-level habitat action the skill
`requires`. Each skill carries a PDDL action schema written against DOMAIN
below; the planner builds its domain from the skills the current fleet
advertises (design.md §4b), so a robot without an arm never contributes
pick/place.

Locations are receptacles (tables, cabinets, ...) named as in the sim host
state, plus each robot's start and each object's current spot. The PDDL type is
`location` because unified-planning does not allow a type named like an action
(`place`). Each schema's `cost` is the action cost the planner minimizes.
"""

from typing import Dict, List

DOMAIN_TYPES = ["robot", "object", "location"]

DOMAIN_PREDICATES = [
    "(robot-at ?r - robot ?p - location)",
    "(obj-at ?o - object ?p - location)",
    "(holding ?r - robot ?o - object)",
    "(hand-empty ?r - robot)",
    "(observed ?p - location)",
    # capability facts, from each robot's reach check (MCP tool check_reach)
    "(can-pick ?r - robot ?o - object)",
    "(can-place ?r - robot ?o - object ?p - location)",
]

DOMAIN_FUNCTIONS = [
    "(nav-cost ?r - robot ?from - location ?to - location)",  # estimated sim steps
]


def _skill(name, description, params, requires, pddl=None, typical_steps=None):
    return {
        "name": name,
        "description": description,
        "params": params,            # {param_name: description}; all strings
        "requires": requires,        # low-level habitat actions the robot must have
        "pddl": pddl,                # None: utility skill, not used by the planner
        "typical_steps": typical_steps,
    }


SKILLS: Dict[str, dict] = {
    s["name"]: s
    for s in [
        _skill(
            "navigate_to",
            "Drive, walk or fly to a place (receptacle) or to an object. "
            "Uses the robot's own mobility: a drone flies, a legged robot can "
            "take stairs, a wheeled robot cannot.",
            {"target": "place name (e.g. table_02_0) or object name (e.g. bowl_0)"},
            ["oracle_nav_action"],
            pddl={
                "parameters": "(?r - robot ?from - location ?to - location)",
                "precondition": "(robot-at ?r ?from)",
                "effect": "(and (robot-at ?r ?to) (not (robot-at ?r ?from)))",
                "cost": "(nav-cost ?r ?from ?to)",
            },
            typical_steps=150,
        ),
        _skill(
            "pick",
            "Pick up an object. The robot must be at the object's place, have "
            "an empty gripper, and the object must be inside its arm workspace. "
            "The arm is reset afterwards.",
            {"object": "object name, e.g. bowl_0"},
            ["oracle_nav_action", "arm_pick_action", "arm_reset_action"],
            pddl={
                "parameters": "(?r - robot ?o - object ?p - location)",
                "precondition": "(and (robot-at ?r ?p) (obj-at ?o ?p) (hand-empty ?r) (can-pick ?r ?o))",
                "effect": "(and (holding ?r ?o) (not (obj-at ?o ?p)) (not (hand-empty ?r)))",
                "cost": "80",
            },
            typical_steps=80,
        ),
        _skill(
            "place",
            "Put the held object on a place. The robot must be at that place "
            "and the place must be inside its arm workspace. The arm is reset "
            "afterwards.",
            {"object": "the held object's name", "place": "place name, e.g. table_02_0"},
            ["oracle_nav_action", "arm_place_action", "arm_reset_action"],
            pddl={
                "parameters": "(?r - robot ?o - object ?p - location)",
                "precondition": "(and (robot-at ?r ?p) (holding ?r ?o) (can-place ?r ?o ?p))",
                "effect": "(and (obj-at ?o ?p) (hand-empty ?r) (not (holding ?r ?o)))",
                "cost": "80",
            },
            typical_steps=80,
        ),
        _skill(
            "look",
            "Report which task objects the robot's cameras can currently see.",
            {},
            ["wait"],
            pddl={
                "parameters": "(?r - robot ?p - location)",
                "precondition": "(robot-at ?r ?p)",
                "effect": "(observed ?p)",
                "cost": "1",
            },
            typical_steps=1,
        ),
        _skill(
            "reset_arm",
            "Move the arm back to its resting pose. pick and place already do "
            "this, so it is only needed for recovery.",
            {},
            ["arm_reset_action"],
            typical_steps=30,
        ),
        _skill(
            "wait",
            "Do nothing for a number of simulation steps.",
            {"steps": "number of steps, 1-200"},
            ["wait"],
            typical_steps=None,
        ),
    ]
}


def skills_for(low_level_actions: List[str]) -> List[dict]:
    """Skills a robot can run given the habitat actions it has."""
    have = set(low_level_actions)
    return [s for s in SKILLS.values() if set(s["requires"]) <= have]
