"""Classical planning with unified-planning + Fast Downward (P1, P2).

  classical_only (P1)  the episode's PDDL goal goes straight to the planner
  slm_classical  (P2)  the SLM turns the instruction into a PDDL goal; the
                       planner does allocation and ordering

The domain is assembled from the PDDL schemas in each robot's MCP tool `meta`
(mas/skills/catalog.py), so it always matches the current fleet. The problem
comes from the sim state: each object starts at its own spot `at_<object>`,
goal places are receptacles, and `can-pick` / `can-place` facts come from the
simulation feasibility check (SimHost.feasibility).

Allocation: Fast Downward minimizes *total* cost, which does not reward using
robots in parallel. So we enumerate object->robot assignments (feasible ones
only), solve each with an `(assigned ?r ?o)` constraint, and keep the plan with
the smallest estimated makespan (the busiest robot's cost), then total cost.
With many objects (more than MAX_ALLOCATIONS assignments) a local search on a
cheap tour estimate picks the TOP_ALLOCATIONS most promising assignments first.
"""

import itertools
import math
import re
import time
from typing import Dict, List, Optional, Tuple

from mas.skills.catalog import DOMAIN_FUNCTIONS, DOMAIN_PREDICATES, DOMAIN_TYPES
from mas.slm.client import SLMClient, parse_json

from . import model_aid
from .base import Plan, Step

NAV_BASE_STEPS = 20       # fixed cost of a navigate_to (turning, settling)
NAV_STEPS_PER_M = 40      # sim steps per metre, fitted loosely to observed runs
MAX_ALLOCATIONS = 64      # above this, search allocations on a cost estimate instead of enumerating
TOP_ALLOCATIONS = 8       # allocations handed to the planner after that search


def _inner(expr: str) -> str:
    """'(and a b)' -> 'a b'; anything else unchanged."""
    e = expr.strip()
    return e[5:-1].strip() if e.startswith("(and ") and e.endswith(")") else e


def _macro_cost_function(m: dict) -> str:
    types = dict(m["params"])
    return f"(cost-{m['name']} {' '.join(f'{v} - {types[v]}' for v in m['cost_vars'])}) - number"


def domain_pddl(robots: List[dict], allocation: bool = True, macros: Optional[List[dict]] = None) -> str:
    schemas: Dict[str, dict] = {}
    for r in robots:
        for t in r["tools"]:
            pddl = (t.get("meta") or {}).get("pddl")
            if pddl and t["name"] != "look":  # no perception goals in these tasks
                schemas.setdefault(t["name"], pddl)
    actions = []
    for name, sc in schemas.items():
        pre = _inner(sc["precondition"])
        if allocation and name == "pick":
            pre += " (assigned ?r ?o)"
        actions.append(
            f" (:action {name}\n  :parameters {sc['parameters']}\n"
            f"  :precondition (and {pre})\n"
            f"  :effect (and {_inner(sc['effect'])} (increase (total-cost) {sc.get('cost', '1')})))")
    for m in macros or []:  # E9: composed by model_aid.compose_macro from the actions above
        if allocation:
            m = {**m, "pre": m["pre"] + [p.replace("(can-pick", "(assigned", 1)
                                         for p in m["pre"] if p.startswith("(can-pick")]}
        actions.append(model_aid.macro_action_pddl(m))
    funcs = " ".join(_macro_cost_function(m) for m in macros or [] if m["cost_vars"])
    preds = DOMAIN_PREDICATES + (["(assigned ?r - robot ?o - object)"] if allocation else [])
    return ("(define (domain mas)\n (:requirements :strips :typing :action-costs)\n"
            f" (:types {' '.join(DOMAIN_TYPES)})\n (:predicates {' '.join(preds)})\n"
            f" (:functions {' '.join(DOMAIN_FUNCTIONS)} {funcs} (total-cost) - number)\n"
            + "\n".join(actions) + ")")


def nav_cost(a, b) -> int:
    return int(NAV_BASE_STEPS + NAV_STEPS_PER_M * math.dist((a[0], a[2]), (b[0], b[2])))


class _Problem:
    """Locations, initial facts and goal for one episode state."""

    def __init__(self, state: dict, feasibility: Dict[str, Dict[str, dict]], goal: List[str]):
        self.robots = [r for r in state["robots"] if r["active"]]
        self.objects = state["objects"]
        met = {g["fact"]: g["met"] for g in state.get("goal_status", [])}
        self.loc: Dict[str, List[float]] = {}
        for r in self.robots:
            self.loc[f"start_{r['id']}"] = r["pos"]
        for o in self.objects:
            self.loc[f"at_{o['name']}"] = o["pos"]
            if o.get("goal_receptacle") and o.get("goal_pos"):
                self.loc.setdefault(o["goal_receptacle"], o["goal_pos"])

        held = {r["holding"]: r["id"] for r in self.robots if r.get("holding")}
        self.init = []
        for r in self.robots:
            self.init.append(f"(robot-at {r['id']} start_{r['id']})")
            if r.get("holding"):
                self.init.append(f"(holding {r['id']} {r['holding']})")
            else:
                self.init.append(f"(hand-empty {r['id']})")
        self.todo = []  # objects that still have to be moved
        for o in self.objects:
            done = met.get(f"(obj-at {o['name']} {o['goal_receptacle']})", False)
            if o["name"] in held:
                self.todo.append(o["name"])
            elif done:
                self.init.append(f"(obj-at {o['name']} {o['goal_receptacle']})")
            else:
                self.init.append(f"(obj-at {o['name']} at_{o['name']})")
                self.todo.append(o["name"])
        goal_place = {o["name"]: o["goal_receptacle"] for o in self.objects}
        self.capable: Dict[str, List[str]] = {o: [] for o in self.todo}
        for rid, row in feasibility.items():
            for obj, v in row.items():
                if obj not in goal_place:  # stale entry for another episode
                    continue
                if v.get("pick") == "OK":
                    self.init.append(f"(can-pick {rid} {obj})")
                if v.get("place") == "OK":
                    self.init.append(f"(can-place {rid} {obj} {goal_place[obj]})")
                if obj in self.capable and v.get("pick") == "OK" and v.get("place") == "OK":
                    self.capable[obj].append(rid)
        # objects already in a robot's hand only need that robot to place them
        for obj, rid in held.items():
            if obj in self.capable:
                self.capable[obj] = [rid]
                self.init.append(f"(can-place {rid} {obj} {goal_place[obj]})")
        self.infeasible = [o for o, rs in self.capable.items() if not rs]
        self.goal = [g for g in goal if not any(f" {o} " in g + " " for o in self.infeasible)]

    def text(self, allocation: Dict[str, str], goal: Optional[List[str]] = None) -> str:
        costs = [f"(= (nav-cost {r['id']} {a} {b}) {nav_cost(pa, pb)})"
                 for r in self.robots for a, pa in self.loc.items() for b, pb in self.loc.items() if a != b]
        assigned = [f"(assigned {r} {o})" for o, r in allocation.items()] + getattr(self, "extra_init", [])
        objs = (" ".join(r["id"] for r in self.robots) + " - robot "
                + " ".join(o["name"] for o in self.objects) + " - object "
                + " ".join(self.loc) + " - location")
        return (f"(define (problem episode) (:domain mas)\n (:objects {objs})\n"
                f" (:init {' '.join(self.init + assigned + costs)} (= (total-cost) 0))\n"
                f" (:goal (and {' '.join(self.goal if goal is None else goal)}))\n (:metric minimize (total-cost)))")


def _estimate(prob: "_Problem", alloc: Dict[str, str]) -> Tuple[int, int]:
    """(makespan, total) of an allocation, each robot doing its objects in nearest-first order."""
    goal_loc = {o["name"]: prob.loc.get(o["goal_receptacle"]) for o in prob.objects}
    per = {}
    for r in prob.robots:
        here, cost = prob.loc[f"start_{r['id']}"], 0
        left = [o for o, rr in alloc.items() if rr == r["id"]]
        while left:
            o = min(left, key=lambda x: math.dist(here, prob.loc[f"at_{x}"]))
            left.remove(o)
            cost += nav_cost(here, prob.loc[f"at_{o}"]) + nav_cost(prob.loc[f"at_{o}"], goal_loc[o]) + 160
            here = goal_loc[o]
        per[r["id"]] = cost
    return max(per.values(), default=0), sum(per.values())


def _search_allocations(prob: "_Problem", options: Dict[str, List[str]]) -> List[Dict[str, str]]:
    """Hill-climb over single-object reassignments from several starts; best distinct results."""
    import random
    rng = random.Random(0)
    seen: Dict[tuple, Tuple[int, int]] = {}
    for start in range(16):
        alloc = {o: (rs[0] if start == 0 else rng.choice(rs)) for o, rs in options.items()}
        best = _estimate(prob, alloc)
        improved = True
        while improved:
            improved = False
            for o, rs in options.items():
                for r in rs:
                    if r == alloc[o]:
                        continue
                    cand = {**alloc, o: r}
                    c = _estimate(prob, cand)
                    if c < best:
                        alloc, best, improved = cand, c, True
            seen[tuple(sorted(alloc.items()))] = best
    ranked = sorted(seen.items(), key=lambda kv: kv[1])[:TOP_ALLOCATIONS]
    return [dict(k) for k, _ in ranked]


class ClassicalPlanner:
    def __init__(self, engine: str = "fast-downward-opt"):
        from unified_planning.shortcuts import get_environment
        get_environment().credits_stream = None
        self.engine = engine
        self.expanded = 0  # Fast Downward expanded states, summed over calls (E9 metric)

    def _solve(self, domain: str, problem: str) -> Tuple[Optional[list], str]:
        from unified_planning.io import PDDLReader
        from unified_planning.shortcuts import OneshotPlanner
        pb = PDDLReader().parse_problem_string(domain, problem)
        with OneshotPlanner(name=self.engine) as planner:
            res = planner.solve(pb)
        log = "\n".join(m.message for m in (res.log_messages or []))
        self.expanded += sum(int(x) for x in re.findall(r"Expanded (\d+) state", log)[-1:])
        if res.plan is None:
            return None, res.status.name
        return [(a.action.name, [str(p) for p in a.actual_parameters]) for a in res.plan.actions], res.status.name

    def _solve_allocation(self, domain: str, prob: _Problem, alloc: Dict[str, str]) -> Tuple[Optional[list], str]:
        """With every object assigned, the robots' sub-problems are independent (no shared
        resources in the domain), so each robot is solved alone: same optimal plan per robot,
        far smaller searches when there are many objects."""
        def mine(fact: str, rid: str) -> bool:
            args = fact.strip("()").split()[1:]
            return rid in args or any(alloc.get(a) == rid for a in args)
        init = set(prob.init)
        unowned = [g for g in prob.goal if g not in init and not any(mine(g, r["id"]) for r in prob.robots)]
        if unowned:  # a goal fact no single robot owns: solve jointly
            return self._solve(domain, prob.text(alloc))
        actions, statuses = [], []
        for r in prob.robots:
            goal = [g for g in prob.goal if g not in init and mine(g, r["id"])]
            if not goal:
                continue
            sub = {o: rr for o, rr in alloc.items() if rr == r["id"]}
            acts, status = self._solve(domain, prob.text(sub, goal))
            statuses.append(status)
            if acts is None:
                return None, status
            actions += acts
        return actions, statuses[0] if statuses else "SOLVED_OPTIMALLY"

    @staticmethod
    def _costs(actions, prob: _Problem) -> Dict[str, int]:
        per: Dict[str, int] = {}
        for name, args in actions:
            c = nav_cost(prob.loc[args[1]], prob.loc[args[2]]) if name == "navigate_to" else 80
            per[args[0]] = per.get(args[0], 0) + c
        return per

    def plan_for_goal(self, goal: List[str], robots: List[dict], state: dict,
                      feasibility: Dict[str, Dict[str, dict]], aid: Optional[dict] = None) -> Plan:
        """aid (E9, model_aid.py): {"exclusive": set of receptacles or None, "insert_waits": bool,
        "landmarks": ["(holding r o)"], "macros": [composed macros]}. None: plain planner."""
        t0 = time.time()
        self.expanded = 0
        aid = aid or {}
        out = Plan()
        prob = _Problem(state, feasibility, goal)
        aid_log = {"landmarks_applied": [], "macros_used": 0, "macro_fallbacks": 0,
                   "scheduled_waits": 0, "scheduled_conflicts": 0}
        for lm in aid.get("landmarks", []):  # (holding r o): r must move o
            _, r, o = lm.strip("()").split()
            if o in prob.capable:
                prob.capable[o] = [r] if r in prob.capable[o] else []
                aid_log["landmarks_applied"].append(lm)
        prob.infeasible = [o for o, rs in prob.capable.items() if not rs]
        prob.goal = [g for g in goal if not any(f" {o} " in g + " " for o in prob.infeasible)]
        macros = {m["name"]: m for m in aid.get("macros", [])}
        prob.extra_init = []
        if macros:
            by_type = {"robot": [r["id"] for r in prob.robots], "location": list(prob.loc),
                       "object": [o["name"] for o in prob.objects]}

            def cost_fn(term):
                t = term.strip("()").split()
                return int(t[0]) if len(t) == 1 else nav_cost(prob.loc[t[2]], prob.loc[t[3]])
            for m in macros.values():
                prob.extra_init += model_aid.macro_cost_table(m, by_type, cost_fn)
        start_place = {o["name"]: o.get("start_receptacle") for o in prob.objects}
        exclusive = aid.get("exclusive")
        # a goal that names a place with no placement spot (e.g. only a start
        # receptacle) cannot be expressed, let alone achieved
        declared = set(prob.loc) | {r["id"] for r in prob.robots} | {o["name"] for o in prob.objects}
        bad = [g for g in prob.goal if any(t not in declared for t in g.strip("()").split()[1:])]
        if bad:
            out.parse_ok = False
            out.errors = [f"goal fact names a place with no placement spot: {g}" for g in bad]
            out.planner_calls.append({"engine": self.engine, "latency_s": round(time.time() - t0, 3),
                                      "prompt_tokens": 0, "completion_tokens": 0, "goal": prob.goal})
            return out
        domain = domain_pddl(robots, macros=list(macros.values()))
        plain_domain = domain_pddl(robots) if macros else domain
        schemas = model_aid.schemas_from_tools(robots)
        options = [prob.capable[o] for o in prob.todo if prob.capable[o]]
        todo = [o for o in prob.todo if prob.capable[o]]
        n = 1
        for opt in options:
            n *= len(opt)
        if n <= MAX_ALLOCATIONS:
            allocations = [dict(zip(todo, combo)) for combo in itertools.product(*options)] or [{}]
        else:
            allocations = _search_allocations(prob, dict(zip(todo, options)))

        best, tried = None, []
        for alloc in allocations:
            actions, status = self._solve_allocation(domain, prob, alloc)
            if actions is not None and macros:
                used = sum(a[0] in macros for a in actions)
                actions = model_aid.expand(actions, macros)
                err = None if not aid.get("replay", True) else model_aid.replay(actions, prob.init + [f"(assigned {r} {o})" for o, r in alloc.items()],
                                       prob.goal, schemas)
                if err:  # never execute an unchecked macro plan: plan again without macros
                    aid_log["macro_fallbacks"] += 1
                    actions, status = self._solve_allocation(plain_domain, prob, alloc)
                else:
                    aid_log["macros_used"] += used
            if actions is None:
                tried.append({"allocation": alloc, "status": status})
                continue
            per = self._costs(actions, prob)
            by_robot: Dict[str, list] = {}
            for a in actions:
                by_robot.setdefault(a[1][0], []).append(a)
            if exclusive is not None:
                span, waits, conf = model_aid.schedule(by_robot, prob.loc, start_place, exclusive, nav_cost)
            else:
                span, waits, conf = max(per.values(), default=0), {}, 0
            key = (span, sum(per.values()))
            tried.append({"allocation": alloc, "status": status, "makespan": key[0], "total": key[1],
                          "scheduled_conflicts": conf})
            if best is None or key < best[0]:
                best = (key, alloc, by_robot, waits, conf)

        if best is not None:
            aid_log["scheduled_waits"] = sum(best[3].values()) if aid.get("insert_waits") else 0
            aid_log["scheduled_conflicts"] = best[4]
        record = {"engine": self.engine, "latency_s": round(time.time() - t0, 3),
                  "expanded": self.expanded, "aid": aid_log if aid else None,
                  "prompt_tokens": 0, "completion_tokens": 0, "goal": prob.goal,
                  "infeasible_objects": prob.infeasible, "allocations": tried,
                  "domain": domain, "problem": prob.text(best[1] if best else {})}
        out.planner_calls.append(record)
        if best is None:
            out.parse_ok = False
            out.errors = [f"no plan found ({len(tried)} allocations tried)"]
            return out
        out.raw = [f"{n}({', '.join(a)})" for acts in best[2].values() for n, a in acts]
        waits = best[3] if aid.get("insert_waits") else {}
        for robot, acts in best[2].items():
            for i, (name, args) in enumerate(acts):
                w = waits.get((robot, i), 0)
                while w > 0:  # scheduled wait before an arm action at an exclusive receptacle
                    out.steps.setdefault(robot, []).append(Step("wait", {"steps": min(200, w)}))
                    w -= 200
                if name == "navigate_to":
                    to = args[2]
                    if to.startswith("start_"):
                        continue
                    step = Step("navigate_to", {"target": to[3:] if to.startswith("at_") else to})
                elif name == "pick":
                    step = Step("pick", {"object": args[1]})
                elif name == "place":
                    step = Step("place", {"object": args[1], "place": args[2]})
                else:
                    continue
                out.steps.setdefault(robot, []).append(step)
        if prob.infeasible:
            out.errors = [f"no robot can move {o}; goal dropped" for o in prob.infeasible]
        return out


class ClassicalOnlyPlanner(ClassicalPlanner):
    """P1: the ground-truth PDDL goal, no language model."""
    mode = "classical_only"

    def plan(self, instruction: str, robots: List[dict], state: dict, ctx: dict) -> Plan:
        return self.plan_for_goal(ctx["goal"], robots, state, ctx["feasibility"], aid=ctx.get("aid"))

    def is_landmark(self, landmark: str, robots: List[dict], state: dict,
                    feasibility: Dict[str, Dict[str, dict]], goal: List[str]) -> bool:
        """E9 verification. (holding r o) is a landmark iff the problem without any action that
        achieves it (every pick(r, o, *) needs (can-pick r o), so that fact is removed) is
        unsolvable. Checked with satisficing Fast Downward on the unallocated domain."""
        _, r, o = landmark.strip("()").split()
        prob = _Problem(state, feasibility, goal)
        if o not in prob.todo or prob.capable.get(o) == []:
            return False  # the object does not have to be moved (or cannot be): nothing must hold it
        prob.init = [f for f in prob.init if f != f"(can-pick {r} {o})"]
        prob.extra_init = []
        engine, self.engine = self.engine, "fast-downward"
        try:
            actions, _ = self._solve(domain_pddl(robots, allocation=False), prob.text({}))
        finally:
            self.engine = engine
        return actions is None


GOAL_SYSTEM = """You translate a household instruction into a formal goal for a planner.
Use only these fact forms, with names exactly as listed:
  (obj-at <object> <place>)   the object must end on that place
  (hand-empty <robot>)        the robot must end holding nothing
Answer with JSON only: {"goal": ["(obj-at ...)", "(hand-empty ...)"]}"""


class SLMClassicalPlanner(ClassicalPlanner):
    """P2: the SLM writes the goal, the classical planner does the rest."""
    mode = "slm_classical"

    def __init__(self, slm: SLMClient, engine: str = "fast-downward-opt"):
        super().__init__(engine)
        self.slm = slm

    def plan(self, instruction: str, robots: List[dict], state: dict, ctx: dict) -> Plan:
        objects = [o["name"] for o in state["objects"]]
        places = state["places"]
        rids = [r["id"] for r in state["robots"] if r["active"]]
        user = (f"Objects: {', '.join(objects)}\nPlaces: {', '.join(places)}\n"
                f"Robots: {', '.join(rids)}\n\nInstruction: {instruction}")
        call = self.slm.chat(GOAL_SYSTEM, user)
        data = parse_json(call.output) or {}
        goal, rejected = [], []
        for g in data.get("goal", []) if isinstance(data.get("goal"), list) else []:
            p = str(g).strip("() ").split()
            ok = ((len(p) == 3 and p[0] == "obj-at" and p[1] in objects and p[2] in places)
                  or (len(p) == 2 and p[0] == "hand-empty" and p[1] in rids))
            (goal if ok else rejected).append(f"({' '.join(p)})")
        if not goal:
            out = Plan(parse_ok=False, errors=["SLM produced no usable goal"], raw=call.output)
            out.planner_calls.append({**call.to_dict(), "system": GOAL_SYSTEM, "user": user})
            return out
        out = self.plan_for_goal(goal, robots, state, ctx["feasibility"])
        out.planner_calls.insert(0, {**call.to_dict(), "system": GOAL_SYSTEM, "user": user,
                                     "slm_goal": goal, "rejected_facts": rejected})
        if rejected:
            out.errors.append(f"SLM goal facts rejected: {rejected}")
        return out
