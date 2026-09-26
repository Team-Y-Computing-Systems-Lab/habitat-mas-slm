# habitat-my-mas: SLM + MCP + Classical Planning for Heterogeneous Robot Fleets

Working design document. Sections marked **TODO** are open.

## 1. Idea in one paragraph

A multi-robot system in Habitat where **small language models (SLMs)** form plans, where each robot's
actions are exposed through **MCP servers**, and where a **classical planner** makes plans sound and
executable. Because actions and skills sit behind MCP, the fleet is open: robots can join or leave,
and each one advertises its own skills, so the planner always sees the current fleet. When execution
fails, the system **recovers by replanning**, escalating only as far as needed. The whole system is
evaluated under explicit **resource-constraint scenarios**.

## 2. Positioning

EMOS (Chen et al., 2024, arXiv:2410.22662) is the closest reference. We share its goal of
embodiment-aware multi-robot collaboration in Habitat, but differ in these ways:

| Aspect | EMOS | This work |
|---|---|---|
| Planner model | Large, API-hosted LLM | **SLMs** (≈1–8B, local) |
| Action interface | Hard-coded function calls | **MCP servers** (standard, discoverable) |
| Fleet | Fixed per episode | **Dynamic**: robots can join or leave |
| Capability info | Robot resume generated from URDF, given in the prompt | Skills and capabilities **advertised by each robot's MCP server** |
| Plan correctness | Depends on the LLM | **Classical planner (PDDL)** in the loop |
| Failure handling | Limited | **Recovery escalation ladder** (§6) |
| Constraints | Measures tokens and steps | **Resource-constraint scenarios** as a first-class axis (§7) |

Relation to our previous work:
- IGSC paper (ACM, 10.1145/3797248.3815414): **TODO**: one-line summary. Reused here: SLM planning
  + MCP action wrapper. No classical planner.
- IEEE paper (doc 11476074): **TODO**: one-line summary and what carries over.

**New in this work:** the classical planner module (§5), recovery with replanning (§6) and the
resource-constraint scenarios (§7).

## 3. Architecture

```
                        Natural-language task
                                 │
                   ┌─────────────▼──────────────┐
                   │   Coordinator (SLM)         │  task understanding,
                   │   - goal → PDDL goal        │  allocation, and plan
                   │   - allocation proposal     │  explanation
                   └──────┬───────────────▲──────┘
            PDDL problem  │               │ plan / failure report
                   ┌──────▼───────────────┴──────┐
                   │   Classical Planner Module   │  Fast Downward / pyperplan
                   │   - domain from MCP skills   │  (via unified-planning)
                   │   - plan + validation (VAL)  │
                   └──────┬───────────────▲──────┘
                 per-robot│  sub-plans    │ execution feedback
                   ┌──────▼───────────────┴──────┐
                   │   Execution Monitor          │  dispatch, state tracking,
                   │   + Recovery Manager         │  failure → recovery ladder
                   └──┬─────────┬─────────┬──────┘
                MCP   │         │         │   (MCP client, one session per robot)
             ┌────────▼──┐ ┌────▼──────┐ ┌─▼─────────┐
             │ Fetch MCP │ │Stretch MCP│ │ Spot MCP  │ ...   ← fleet can grow or shrink
             │ tools:    │ │ tools:    │ │ tools:    │
             │ nav, pick │ │ nav, pick │ │ nav, look │
             │ resources:│ │ ...       │ │ ...       │
             │ caps      │ │           │ │           │
             └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
                   └─────────────┼─────────────┘
                     ┌──────────▼──────────┐
                     │  Habitat Sim Host   │  one shared habitat-lab env,
                     │  (multi-agent env)  │  steps all robots together
                     └─────────────────────┘
```

Each MCP server has three jobs:

1. **Registry**: a fleet registry keeps track of which robots are available now.
2. **Skills**: `tools/list` returns the robot's skills, each with typed arguments.
3. **Capabilities**: each robot publishes an MCP *resource* (`robot://<id>/resume`) with its
   capability data, such as reach, sensors and mobility (floors, stairs, flight).

## 4. MCP layer: the two-fold role

### 4a. Dynamic fleet (increase or decrease)
- A **registry** (either its own small MCP server or part of the coordinator) handles
  `register_robot` and `deregister_robot` and emits `fleet_changed` events.
- A fleet change **mid-episode** triggers the replanning path in §6. A robot leaving is treated like
  a failure of all of its pending actions.
- **Habitat constraint:** articulated agents are fixed when the env is configured. The planned
  approach is to pre-spawn a *pool* of N robots. Inactive robots are parked (outside the navmesh, or
  frozen), and joining a robot activates it. **TODO:** confirm this works with habitat-lab's
  multi-agent config.

### 4b. Skill and capability discovery
- Each skill is an MCP tool with a JSON schema. The same metadata also carries the skill's
  **PDDL signature** (parameters, preconditions, effects). This is how the classical planner's
  domain is **generated from the current fleet** instead of written by hand.
- Capability resources come from the URDF, using forward kinematics for workspace and reach and
  sensor configs for field of view. This is similar in spirit to EMOS's resume, but it is served
  by the robot's MCP server, not put in the prompt.

### Example skill descriptor
```json
{
  "name": "pick",
  "robot": "fetch_0",
  "input_schema": {"object": "string"},
  "pddl": {
    "parameters": "(?r - robot ?o - object ?l - location)",
    "precondition": "(and (at ?r ?l) (obj-at ?o ?l) (hand-empty ?r) (reachable ?r ?o))",
    "effect": "(and (holding ?r ?o) (not (obj-at ?o ?l)) (not (hand-empty ?r)))"
  },
  "cost": {"time_s": 8, "energy_j": null}
}
```

## 5. Classical planner module (new)

**Role split:** the SLM handles language and ambiguity, and the classical planner handles
correctness.

| Step | Owner |
|---|---|
| NL task → PDDL goal + object grounding | SLM |
| Domain (actions, predicates, types) | Built from MCP skill descriptors (§4b) |
| Initial state | Read from sim through MCP resources |
| Allocation + ordering | Classical planner (multi-agent PDDL, or SLM allocation followed by per-robot planning) |
| Validation | VAL, or the unified-planning validator |
| Explanation / plan to NL | SLM (optional) |

Planner modes to compare (the ablation axis):
- **P0 SLM-only**: SLM writes the plan directly (EMOS-like baseline, but with an SLM).
- **P1 Classical-only**: goal written in PDDL by hand. This is the upper bound on soundness.
- **P2 Hybrid**: SLM → PDDL goal → classical planner. This is the main system.
- **P3 Hybrid + SLM repair**: when the planner finds no plan or grounding fails, the SLM revises
  the goal or allocation.

Tooling: `unified-planning`, with Fast Downward (optimal and satisficing) and ENHSP (numeric fluents,
needed for resource constraints in §7).

**As built (2026-09-24),** in `mas/planners/classical.py`:
- **Domain** from the MCP tool metadata.
- **Objects start at their own spot** `at_<object>`. This sidesteps objects whose start and goal
  receptacle are the same.
- **`can-pick` / `can-place`** come from simulated trials.
- **Allocation:** every feasible assignment is enumerated, each is solved with an `assigned`
  constraint, and the lowest estimated makespan wins.
- **Engine:** `fast-downward-opt` by default, with `pyperplan` as the fallback.
- **P3 (SLM repair of the goal)** isn't built. P0 instead got a checker plus one repair round
  (`mas/planners/repair.py`).

## 6. Recovery with replanning (new)

Every MCP tool returns a structured result:
`{status: ok|failed, code: UNREACHABLE|GRASP_FAIL|NOT_VISIBLE|TIMEOUT|ROBOT_LOST|..., observed_state}`.

**Escalation ladder.** Each level runs only if the one before it fails:

1. **Retry:** repeat the same action with a perturbation (new approach pose, re-grasp).
2. **Local repair:** replan only the affected robot's remaining sub-plan from the current state.
3. **Reallocation:** give the failed sub-goal to another robot that can do it, chosen through
   capability lookup.
4. **Global replan:** build a new PDDL problem from the current state with the current fleet.
5. **Goal relaxation:** the SLM proposes a partial goal and reports what can't be achieved.

Triggers: an action fails, a precondition the monitor expects no longer holds (plan invalidation),
the fleet changes (§4a), or a resource threshold is crossed (§7).

Questions to study:
- Plan repair vs. replanning from scratch: which costs less and which succeeds more?
- Does the classical planner shorten recovery compared with SLM-only replanning?
- How much does recovery add in tokens, latency and energy?

## 7. Resource-constraint scenarios (to be detailed later)

These are initial candidates. Each scenario becomes an evaluation condition.

| ID | Constraint | How it's modeled | Affects |
|---|---|---|---|
| R1 | Robot battery / energy budget | Numeric fluent per robot, with a cost per skill | Allocation, feasibility |
| R2 | Compute budget for planning | SLM size (1B/3B/8B), token cap, quantization, edge vs. server | Plan quality, latency |
| R3 | Planning time deadline | Timeout on the planner and SLM calls | Choice of planner mode |
| R4 | Fleet attrition | Robots removed at a random or scripted step | Recovery (§6) |
| R5 | Communication limits | Dropped or delayed MCP messages, limited bandwidth | Coordination |
| R6 | Heterogeneity scarcity | Only one robot can do a needed skill | Allocation bottleneck |

**TODO:** decide how energy is measured, both for planning compute (GPU/CPU energy) and for robot
actuation (a proxy per skill). This is also where the IGSC line of work could connect.

## 8. Evaluation

- **Environment:** official habitat-sim / habitat-lab 0.3.1 with official data. The Habitat 3.0 HSSD
  rearrangement episodes use the official two-object task spec; see [datasets.md](datasets.md).
  Until 2026-09-26 the project used EMOS's fork and its Habitat-MAS data; those results are archived.
- **Robots:** Fetch and Stretch (official models); Spot is available but not in a benchmark yet.
- **Tasks:** navigation, cooperative perception, rearrangement (single- and multi-floor), plus new
  **fleet-change** and **failure-injection** episodes.
- **Metrics:**
  - Success rate and sub-goal success rate
  - Plan validity rate
  - Recovery success rate and number of escalation steps
  - Tokens, wall-clock planning latency and planning energy
  - Simulation steps
- **Baselines:** P0 (SLM-only). An EMOS-style run with a large LLM, as a reference point. The
  ablations: without the classical planner, without recovery, and without dynamic discovery (a
  static prompt instead).

Experiment log and findings so far: [findings.md](findings.md).

## 9. Research questions (draft)

- **RQ1:** Can SLMs paired with a classical planner match or beat large-LLM planning (EMOS-style)
  on success while using much less compute?
- **RQ2:** Does MCP-based skill discovery let the system handle fleets that change mid-task without
  retraining or prompt engineering?
- **RQ3:** Which recovery strategy gives the best trade-off between success and cost?
- **RQ4:** How do the resource constraints (R1–R6) change the best configuration?

## 10. Code layout (planned)

```
habitat-my-mas/mas/
  sim_host/     multi-agent habitat-lab env, robot pool, activation and parking
  mcp_servers/  one server per robot type: skill tools + capability resources
  registry/     fleet registration and fleet_changed events
  capability/   URDF + FK → capability resource
  planner/      PDDL domain builder, unified-planning wrapper, validator
  slm/          SLM client (Ollama / vLLM), prompts for grounding and repair
  executor/     dispatch, monitoring, recovery ladder
  scenarios/    resource-constraint + failure-injection configs
  eval/         metrics, runners
```

**Build order:**
1. Sim host + robot pool
2. One MCP server (Fetch) with nav/pick/place
3. PDDL domain builder + planner (P1)
4. SLM grounding (P2)
5. Executor + recovery
6. Dynamic fleet
7. Scenarios + evaluation
