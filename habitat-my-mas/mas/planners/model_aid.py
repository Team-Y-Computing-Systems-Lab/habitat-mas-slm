"""E9: an SLM proposes extra structure for the planner's model; code verifies it.

The SLM never adds actions or effects. Round 1 handles three kinds of proposal
(docs/e9_design.md, docs/findings.md 2026-10-29):

  exclusive  receptacles where only one robot can work at a time. Used by the
             timeline scheduler (schedule()) that runs after the planner.
             Verified against the simulator (SimHost.receptacle_capacity).
  landmarks  "(holding <robot> <object>)": that robot must be the one moving the
             object. Used as an allocation constraint. Verified by deleting the
             (can-pick <robot> <object>) facts of every *other* robot and checking
             that the problem becomes unsolvable, i.e. no other robot can do it.
  macros     a named sequence of existing actions. Code composes preconditions,
             effects and cost (compose_macro()). Plans that use a macro are expanded
             back to primitive actions and replayed (replay()) before execution.
"""

import json
import re
from typing import Dict, List, Optional, Sequence, Tuple

from mas.slm.client import SLMClient, parse_json

ARM_COST = 80          # sim steps for pick/place, as in the domain (catalog.py)
MAX_COST_TABLE = 20000  # largest precomputed macro-cost table accepted


# ------------------------------------------------------------------ PDDL bits

def _sexprs(text: str) -> List[str]:
    """Top-level parenthesised expressions in `text`."""
    out, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                out.append(text[start:i + 1])
    return out


def _literals(expr: str) -> Tuple[List[str], List[str]]:
    """'(and (a ?x) (not (b ?y)))' -> (['(a ?x)'], ['(b ?y)'])."""
    e = " ".join(expr.split())
    if e.startswith("(and "):
        e = e[5:-1]
    pos, neg = [], []
    for s in _sexprs(e):
        if s.startswith("(not "):
            neg.append(" ".join(_sexprs(s[5:-1])[0].split()))
        else:
            pos.append(s)
    return pos, neg


def _params(text: str) -> List[Tuple[str, str]]:
    """'(?r - robot ?from ?to - location)' -> [('?r','robot'), ('?from','location'), ('?to','location')]."""
    toks = text.strip("() ").split()
    out, pending = [], []
    i = 0
    while i < len(toks):
        if toks[i] == "-":
            out += [(v, toks[i + 1]) for v in pending]
            pending, i = [], i + 2
        else:
            pending.append(toks[i])
            i += 1
    return out + [(v, "object") for v in pending]


def _subst(lit: str, m: Dict[str, str]) -> str:
    return re.sub(r"\?[\w-]+", lambda x: m.get(x.group(0), x.group(0)), lit)


def schemas_from_tools(robots: List[dict]) -> Dict[str, dict]:
    """The planning actions as the MCP servers publish them (same source as domain_pddl)."""
    out = {}
    for r in robots:
        for t in r["tools"]:
            sc = (t.get("meta") or {}).get("pddl")
            if sc and t["name"] != "look":
                pre, _ = _literals(sc["precondition"])
                add, dele = _literals(sc["effect"])
                out.setdefault(t["name"], {"params": _params(sc["parameters"]), "pre": pre,
                                           "add": add, "del": dele, "cost": sc.get("cost", "1")})
    return out


# --------------------------------------------------------------------- macros

class MacroError(ValueError):
    pass


def compose_macro(spec: dict, schemas: Dict[str, dict]) -> dict:
    """Compose a macro from existing actions. Raises MacroError if the spec is not a
    consistent composition (unknown action, wrong arity, type clash, or a step needing
    a fact an earlier step deleted)."""
    name = re.sub(r"[^a-z0-9_]", "_", str(spec.get("name", "macro")).lower())[:40] or "macro"
    steps = spec.get("steps") or []
    if not 2 <= len(steps) <= 6:
        raise MacroError("a macro needs 2-6 steps")
    types: Dict[str, str] = {}
    bound = []
    for st in steps:
        if isinstance(st, dict):
            st = [st.get("action") or st.get("skill")] + list(st.get("args") or [])
        act, args = str(st[0]), [str(a) if str(a).startswith("?") else "?" + str(a) for a in st[1:]]
        if act not in schemas or act == "wait":
            raise MacroError(f"unknown action {act}")
        sc = schemas[act]
        if len(args) != len(sc["params"]):
            raise MacroError(f"{act} takes {len(sc['params'])} arguments, got {len(args)}")
        for a, (_, ty) in zip(args, sc["params"]):
            if types.setdefault(a, ty) != ty:
                raise MacroError(f"{a} used as {types[a]} and {ty}")
        bound.append((act, dict(zip([p for p, _ in sc["params"]], args))))
    pre, add, dele, costs = [], set(), set(), []
    for act, m in bound:
        sc = schemas[act]
        for lit in (_subst(x, m) for x in sc["pre"]):
            if lit in add:
                continue
            if lit in dele:
                raise MacroError(f"{act} needs {lit}, deleted by an earlier step")
            if lit not in pre:
                pre.append(lit)
        for lit in (_subst(x, m) for x in sc["del"]):
            dele.add(lit)
            add.discard(lit)
        for lit in (_subst(x, m) for x in sc["add"]):
            add.add(lit)
            dele.discard(lit)
        costs.append(_subst(sc["cost"], m))
    params = list(types.items())
    fvars = sorted({v for c in costs for v in re.findall(r"\?[\w-]+", c)}, key=[p for p, _ in params].index)
    return {"name": f"macro_{name}", "params": params, "pre": pre, "add": sorted(add), "del": sorted(dele),
            "costs": costs, "cost_vars": fvars, "steps": bound}


def macro_action_pddl(mac: dict) -> str:
    ps = " ".join(f"{v} - {t}" for v, t in mac["params"])
    cost = (f"(cost-{mac['name']} {' '.join(mac['cost_vars'])})" if mac["cost_vars"]
            else str(sum(int(c) for c in mac["costs"])))
    eff = " ".join(mac["add"] + [f"(not {d})" for d in mac["del"]])
    return (f" (:action {mac['name']}\n  :parameters ({ps})\n  :precondition (and {' '.join(mac['pre'])})\n"
            f"  :effect (and {eff} (increase (total-cost) {cost})))")


def macro_cost_table(mac: dict, objects_by_type: Dict[str, List[str]], cost_fn) -> List[str]:
    """Init facts '(= (cost-macro_x a b c) n)' for every tuple of the macro's cost variables.
    cost_fn(term) evaluates one grounded cost term such as '(nav-cost r a b)' or '80'."""
    if not mac["cost_vars"]:
        return []
    types = dict(mac["params"])
    domains = [objects_by_type[types[v]] for v in mac["cost_vars"]]
    size = 1
    for d in domains:
        size *= max(1, len(d))
    if size > MAX_COST_TABLE:
        raise MacroError(f"cost table of {size} entries is too large")
    import itertools
    out = []
    for combo in itertools.product(*domains):
        m = dict(zip(mac["cost_vars"], combo))
        total = sum(cost_fn(_subst(c, m)) for c in mac["costs"])
        out.append(f"(= (cost-{mac['name']} {' '.join(combo)}) {total})")
    return out


def expand(actions: List[Tuple[str, List[str]]], macros: Dict[str, dict]) -> List[Tuple[str, List[str]]]:
    """Replace macro actions in a plan by their primitive steps."""
    out = []
    for name, args in actions:
        mac = macros.get(name)
        if mac is None:
            out.append((name, args))
            continue
        m = dict(zip([p for p, _ in mac["params"]], args))
        for act, binding in mac["steps"]:
            out.append((act, [m[binding[p]] for p in binding]))
    return out


def replay(actions: List[Tuple[str, List[str]]], init: Sequence[str], goal: Sequence[str],
           schemas: Dict[str, dict]) -> Optional[str]:
    """Check a primitive plan against the STRIPS schemas. None if valid, else the error."""
    state = set(init)
    for i, (name, args) in enumerate(actions):
        sc = schemas[name]
        m = dict(zip([p for p, _ in sc["params"]], args))
        for lit in sc["pre"]:
            g = _subst(lit, m)
            if g not in state:
                return f"step {i} {name}{tuple(args)}: precondition {g} false"
        for lit in sc["del"]:
            state.discard(_subst(lit, m))
        for lit in sc["add"]:
            state.add(_subst(lit, m))
    missing = [g for g in goal if g not in state]
    return f"goal not reached: {missing}" if missing else None


# Condition B's macro. A 4-step fetch_and_place (navigate, pick, navigate, place) has three
# location parameters in its cost, so its precomputed cost table is robots x locations^3
# (about 11.6k facts at N=8) and made Fast Downward 10x slower in the smoke test, with the
# same number of expansions. carry = pick, navigate, place needs robots x locations^2.
HAND_MACRO = {"name": "carry",
              "steps": [["pick", "?r", "?o", "?b"], ["navigate_to", "?r", "?b", "?c"],
                        ["place", "?r", "?o", "?c"]]}


# ------------------------------------------------------------------ scheduler

PASSAGE_MARGIN = 15  # sim steps kept free around a robot's estimated time inside a passage


def schedule(steps_by_robot: Dict[str, List[Tuple[str, List[str]]]], loc: Dict[str, list],
             start_place: Dict[str, str], exclusive: set, nav_cost,
             routes: Optional[Dict[str, list]] = None, passages: Optional[set] = None
             ) -> Tuple[int, Dict[Tuple[str, int], int], int]:
    """Timeline of each robot's primitive actions with estimated durations (first come, first
    served). Arm actions at an exclusive receptacle never overlap, and no two robots are inside
    the same exclusive passage at once: routes["a|b"] lists the passages on the path from a to b
    with the fractions of the drive where it enters and leaves them. A conflict is resolved by
    waiting before the action. Returns (makespan, {(robot, action index): wait steps}, conflicts)."""
    routes, passages = routes or {}, passages or set()
    queues = {}
    for r, acts in steps_by_robot.items():
        q = []
        for i, (name, args) in enumerate(acts):
            if name == "navigate_to":
                crossed = [(pid, f0, f1) for pid, f0, f1 in routes.get(f"{args[1]}|{args[2]}", []) if pid in passages]
                q.append((i, nav_cost(loc[args[1]], loc[args[2]]), crossed))
            elif name == "pick":
                q.append((i, ARM_COST, start_place.get(args[1])))
            elif name == "place":
                q.append((i, ARM_COST, args[2]))
            else:
                q.append((i, ARM_COST, None))
        queues[r] = q
    t = {r: 0 for r in queues}
    pos = {r: 0 for r in queues}
    busy: Dict[str, List[Tuple[int, int]]] = {}
    waits, conflicts = {}, 0
    while any(pos[r] < len(queues[r]) for r in queues):
        r = min((r for r in queues if pos[r] < len(queues[r])), key=lambda x: t[x])
        i, dur, rec = queues[r][pos[r]]
        start = t[r]
        if isinstance(rec, list):  # a drive: reserve the passages it crosses
            moved = True
            while moved:
                moved = False
                for pid, f0, f1 in rec:
                    lo, hi = start + f0 * dur - PASSAGE_MARGIN, start + f1 * dur + PASSAGE_MARGIN
                    for a, b in busy.get(pid, []):
                        if lo < b and a < hi:
                            start, moved = int(b - f0 * dur + PASSAGE_MARGIN) + 1, True
            if start > t[r]:
                waits[(r, i)] = start - t[r]
                conflicts += 1
            for pid, f0, f1 in rec:
                busy.setdefault(pid, []).append((start + f0 * dur - PASSAGE_MARGIN, start + f1 * dur + PASSAGE_MARGIN))
            rec = None
        elif rec is not None and rec in exclusive:
            moved = True
            while moved:
                moved = False
                for a, b in busy.get(rec, []):
                    if start < b and a < start + dur:
                        start, moved = b, True
            if start > t[r]:
                waits[(r, i)] = start - t[r]
                conflicts += 1
            busy.setdefault(rec, []).append((start, start + dur))
        t[r] = start + dur
        pos[r] += 1
    return max(t.values(), default=0), waits, conflicts


# ------------------------------------------------------------- SLM proposal

PROPOSAL_SYSTEM = """You help a classical task planner for a team of household robots.
You do not write the plan. You point out structure the planner should know about.

Answer with JSON only, in this form:
{"exclusive": ["<receptacle>"],
 "landmarks": ["(holding <robot> <object>)"],
 "macros": [{"name": "<name>", "steps": [["<action>", "?<var>", ...], ...]}],
 "reasoning": "<one or two sentences>"}

- exclusive: receptacles where only ONE robot can work at a time. Two robots need about
  0.8 m between their bases while their arms are at work, so small receptacles fit one robot
  and long ones may fit two. List only receptacles that will be used by more than one object.
- landmarks: (holding <robot> <object>) when that robot is the ONLY one able to move the
  object, so it must do it. Leave the list empty if any robot could.
- macros: optional. A macro is a fixed sequence of the planner's existing actions that is
  often used together, written with variables. Use only the actions listed. Do not invent
  actions or effects."""


def proposal_prompt(robots: List[dict], state: dict, rec_info: Dict[str, dict],
                    feasibility: Dict[str, Dict[str, dict]], schemas: Dict[str, dict]) -> str:
    lines = ["Robots (base radius about 0.35 m):"]
    for r in robots:
        ar = r["capabilities"].get("arm_reach")
        lines.append(f"- {r['id']} ({r['capabilities'].get('type')}), arm reaches heights "
                     f"{ar['min_height_m']:.2f}-{ar['max_height_m']:.2f} m" if ar else f"- {r['id']}")
    lines.append("\nReceptacles used by the task (size is the footprint, length x width):")
    use: Dict[str, int] = {}
    for o in state["objects"]:
        for p in (o["start_receptacle"], o["goal_receptacle"]):
            if p:
                use[p] = use.get(p, 0) + 1
    for p, n in sorted(use.items()):
        info = rec_info.get(p) or {}
        size = info.get("size_xz")
        lines.append(f"- {p}: {f'{size[0]:.2f} x {size[1]:.2f} m' if size else 'size unknown'}, "
                     f"used by {n} object(s)")
    lines.append("\nObjects (now on -> goes to; which robots passed a trial pick and place):")
    for o in state["objects"]:
        able = [r for r, row in feasibility.items()
                if (row.get(o["name"]) or {}).get("pick") == "OK" and (row.get(o["name"]) or {}).get("place") == "OK"]
        lines.append(f"- {o['name']}: {o['start_receptacle']} -> {o['goal_receptacle']}; "
                     f"can be moved by: {', '.join(able) or 'no robot'}")
    lines.append("\nPlanner actions:")
    for n, sc in schemas.items():
        lines.append(f"- {n}({', '.join(v for v, _ in sc['params'])})")
    return "\n".join(lines)


def propose(slm: SLMClient, robots, state, rec_info, feasibility, schemas) -> dict:
    user = proposal_prompt(robots, state, rec_info, feasibility, schemas)
    call = slm.chat(PROPOSAL_SYSTEM, user)
    data = parse_json(call.output)
    ok = isinstance(data, dict)
    data = data if ok else {}
    places = set(state["places"])
    objects = {o["name"] for o in state["objects"]}
    rids = {r["id"] for r in robots}
    prop = {"exclusive": [], "landmarks": [], "macros": [], "invalid": []}
    for p in data.get("exclusive") or []:
        (prop["exclusive"] if str(p) in places else prop["invalid"]).append(str(p))
    for lm in data.get("landmarks") or []:
        t = str(lm).strip("() ").split()
        if len(t) == 3 and t[0] == "holding" and t[1] in rids and t[2] in objects:
            prop["landmarks"].append(f"(holding {t[1]} {t[2]})")
        else:
            prop["invalid"].append(str(lm))
    for mac in data.get("macros") or []:
        if isinstance(mac, dict):
            prop["macros"].append(mac)
        else:
            prop["invalid"].append(json.dumps(mac))
    return {"proposal": prop, "json_ok": ok, "call": {**call.to_dict(), "system": PROPOSAL_SYSTEM, "user": user}}
