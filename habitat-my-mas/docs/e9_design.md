# E9 design: the SLM improves the planner's model instead of replacing it

Status: **implemented; pilot gate failed (2026-10-29), full run not started** (findings F30–F32). Decisions: both episode sets, the scheduler, 2 robots. Changes made while implementing are in §9. Related work: [related_work.md](related_work.md#e9).

## 0. Two facts that shape the design

1. **There is no interference in the simulator today.** Robot bases are only checked against the
   static navmesh (habitat-lab `step_filter`), so in kinematic mode two robots can drive through
   each other and work at the same receptacle at the same time. A mutex cannot help until
   interference has a cost.
2. **Even after adding it, contention is rare on E8 episodes.** In E8's classical-only traces,
   the two robots used a common receptacle in 10, 19, 36 and 44 of 60 episodes at
   N = 2, 4, 6, 8. They were there *at the same time* in only 3, 0, 2 and 6. On those episodes
   alone, mutexes could change at most about 10% of N=8 episodes. So I propose a second episode
   set with deliberate contention (§2.3) next to the E8 episodes you asked for.

Also: **a mutex in a sequential PDDL plan is meaningless.** A classical plan is a total order,
so two actions never overlap in it. The overlap only appears when we execute each robot's list
in parallel. Interference therefore has to be modelled where time exists: in a scheduler after
the planner (recommended, §3.2), or in a temporal planner (alternative, §3.3).

## 1. Scope of SLM proposals

| Kind | Effect it can have here | In E9? |
|---|---|---|
| Mutex / shared resource: receptacle occupancy | Fewer conflicts and waits during execution; shorter makespan | **Yes, the core of E9** |
| Macro-action `fetch_and_place(r, o, from, to)` | Plans 4× shorter, so a faster Fast Downward search (N=8 takes about 1 minute today) | Yes |
| Landmarks and orderings | Smaller search, *only* when allocation is planned jointly. After allocation each robot's sub-problem is tiny, so they barely matter there | Yes, measured on planning time and expansions with joint planning |
| Soft preferences (PDDL3) | Needs a planner with preference support and a checkable objective | No; later, if the rest works |

The SLM never adds or edits actions or effects. A macro is only a named *sequence* of
existing actions, and its preconditions and effects are computed by code.

## 2. Interference in the simulator

### 2.1 Rule: receptacle occupancy decided by geometry

- **When a robot occupies a receptacle.** While a robot runs the arm phase of `pick`/`place` at
  receptacle R, it occupies its base pose p.
- **When another robot must wait.** A second robot whose own arm pose at R is within
  `D_MIN = 0.8 m` of p waits. With a footprint radius of about 0.3–0.35 m plus an arm-swing
  margin, the two would otherwise collide. The wait is a skill-level `WAIT`: steps are counted,
  and a conflict event is logged.
- **Timeout.** After `T_MAX = 300` steps of waiting the skill fails with `RECEPTACLE_BUSY`.
- **No deadlock.** A robot holds a receptacle only during its arm phase, never while driving, so
  two robots can't end up waiting on each other.
- **Where it lives.** The rule is in our skill layer (`mas/sim_host/skills.py`), not in
  habitat-lab, so the official code stays untouched.

### 2.2 What is "true"

Because the rule is geometric, a large receptacle (a long counter or dining area) can host two
robots at once and a small one (a chair or nightstand) cannot. Whether R is exclusive is a
property of the scene that can be checked (§4). The hand-written upper bound (B) uses this
ground truth, and SLM proposals can be right or wrong about it.

Also logged: **base overlap**, meaning steps where the bases are closer than 0.6 m anywhere
(kinematic mode lets them pass through). This is the collision count.

### 2.3 Episodes

- **E8 set:** `hssd_n{2,4,6,8}`, the same 60 episodes. Feasibility trials are cached from E8.
- **Contention set `hssd_hub_n{4,6,8}` (proposed):** the same episodes and objects, but all goal
  spots sampled on 1–2 "hub" receptacles of the scene (like "set the table" or "fill the
  shelf"). Built by the same generator with a hub option; new feasibility trials needed.

## 3. Planner side

### 3.1 Inputs the SLM sees, and what it returns

**Prompt.** Receptacles with their size (from the receptacle's bounding box) and position; which
objects start and end where; robot footprints and arm reach; the feasibility facts.

**Output: JSON only, checked against a schema.**

```json
{"exclusive": ["chair_0", "nightstand_1"],
 "shared_ok": ["counter_0"],
 "macros": [{"name": "fetch_and_place", "steps": ["navigate_to", "pick", "navigate_to", "place"]}],
 "landmarks": ["(holding ? bowl_0)"],
 "orderings": [["(obj-at bowl_0 shelves_0)", "(obj-at cup_0 shelves_0)"]]}
```

**Allowed names.** Only names from the episode are accepted; anything else is a rejected
proposal.

### 3.2 Recommended: timeline scheduler after Fast Downward

The planner stays as it is: allocation search, then Fast Downward per robot. A new step turns
each robot's action list into a timeline and computes start times:
- **Durations:** today's `nav_cost`, plus a fixed arm cost.
- **Mutual exclusion:** occupancy of each receptacle marked exclusive is respected by inserting
  waits (list scheduling).
- **Allocation search objective:** the *scheduled* makespan, waits included, replaces today's
  makespan estimate.
- **Executor:** the schedule's waits are passed to the executor as `wait` steps.

The mutex set is an *input*. The planner stays sound whatever the set is, because the simulator
enforces the real rule anyway. A wrong set only costs time: unneeded waits if it is too strict,
or conflicts at run time if it is too loose.

### 3.3 Alternative: temporal planning

PDDL 2.1 durative actions with `(free ?r)` resources, solved by a temporal planner through
unified-planning (for example `up-tamer`). It's more principled but slower, adds a new engine,
and changes P1 itself. I'd keep it as a later comparison.

## 4. Automatic verification of each proposal (condition D)

| Proposal | Accepted only if | Cost |
|---|---|---|
| `R` exclusive / `R` shared | Agrees with the simulator. Sample the arm poses the pick/place skill would use at R (its own pose planner); R is truly exclusive if no two poses that reach are ≥ `D_MIN` apart. Cached per scene and receptacle. | Seconds per receptacle, once |
| Fact landmark `L` | The problem with every action that achieves `L` removed is **unsolvable** (Fast Downward). A sound test: every plan then contains `L`, as you suggested. Disjunctive landmarks ("some robot holds o") are compiled to a single fact. | One planner call per landmark |
| Ordering `L1 before L2` | Compile "L2 before L1" with a flag; accepted if that problem is unsolvable (the order is necessary). Otherwise rejected, and the extra cost of enforcing it is logged. | One call |
| Macro | Steps are existing actions with consistent parameters; code composes preconditions and effects; the composition must be consistent (no step needs something an earlier step deleted). Primitive actions stay in the domain, so nothing becomes unsolvable. Final plans are expanded to primitives and replayed with unified-planning's plan validator. | Milliseconds |
| Anything | Rejected if it makes a solvable problem unsolvable (then the plain domain is used and the case is counted). | One call |

## 5. Conditions

| | Planner knows about interference? | Additions used | Soundness |
|---|---|---|---|
| **A** plain | No: schedule ignores occupancy; the executor waits when it must | None | By construction |
| **B** hand-written | Yes: true exclusive set from the geometry check, plus a hand-written `fetch_and_place` macro and landmarks found by Fast Downward's own landmark generator | All, correct | Upper bound |
| **C** SLM, unverified | The SLM's sets as given | All SLM proposals as hard constraints | Can fail: unsolvable problems fall back to A and are counted |
| **D** SLM, verified | Only proposals that pass §4 | Accepted ones | Kept |
| **E** SLM as hint | No hard constraints; the executor enforces occupancy as in A | Proposals only order the allocation search (seeds) and break ties under a fixed budget (e.g. 8 allocations) | By construction |

One SLM call per episode and model, cached and shared by C, D and E, so they differ only in how
the proposal is used.

## 6. Metrics

- **Planning:** wall time, Fast Downward expanded states (parsed from its log; checked that it
  is available), number of allocations tried.
- **Execution:** makespan (sim steps to finish), number of waits and wait steps, conflicts,
  `RECEPTACLE_BUSY` failures, base-overlap steps.
- **Task:** success, sub-goals, success on ceiling episodes (as in E8).
- **Proposal step:** tokens in/out, latency, JSON validity.
- **Proposal quality, per kind:** accepted / rejected, and *wrong* (disagrees with ground truth).
  Mutexes are wrong in two ways, false-exclusive (costs time) and false-shared (causes
  conflicts). Also how often C became unsolvable.
- **Tests:** paired across conditions on the same episodes. Wilcoxon signed-rank for makespan,
  planning time and expansions; exact McNemar for success.

## 7. Scale, runtime and gates

- **Planned matrix.** Per episode set and N: A and B (no SLM), plus C, D and E × {qwen3:4b, qwen3:8b,
  gemma3:27b}, so 11 runs. 4 N values × 60 episodes gives 44 runs and 2,640 episodes on the E8
  set.
- **Runtime estimate.** E8 averaged about 49 s per episode (48 runs in about 39.5 h).
  - **E8 set:** about 36 h of runs. The SLM proposal step adds about 2 h (720 calls; gemma3:27b
    about 30 s each). Feasibility trials are reused from E8.
  - **Hub set (N = 4, 6, 8):** 33 more runs, about 27 h, plus about 3 h of new trials.
  - **Total:** about 2.5–3 days of compute, run as the systemd user service `habitat-e9` in
    stages, so partial results are usable.
- **Implementation estimate.** About 2–3 working days: occupancy rule, scheduler, verifiers,
  proposal prompt, hub generator, runner.
- **Gate 1, before the full run.** A pilot of A vs B on 10 hub episodes at N=8. If B's makespan
  and conflicts are not clearly better than A's, mutexes cannot help, and I come back to you
  before running anything long.
- **Gate 2.** The geometry check for "R exclusive" is compared against what the simulator does
  on those 10 episodes (every observed conflict should be at an exclusive R).

## 8. Questions for you

1. Run the **hub contention set** as well, or only the E8 episodes? (I recommend both; see §0.)
2. Use the **scheduler** (§3.2) for the first round and keep temporal planning (§3.3) for later?
3. Keep **2 robots**, or add a third (a second Fetch) for more contention? This needs a
   habitat-lab config with 3 agents and a 3-robot task spec, about half a day more.

## 9. Changes made while implementing (2026-10-29)

- **Planner for every condition: P1 (true goal).** Conditions then differ only in the model
  additions, not in goal translation.
- **Orderings dropped from round 1.** Checking them needs grounded constraint compilation, and in
  this domain no ordering between goal facts is ever necessary, so a verifier would reject all of
  them.
- **Landmarks are `(holding r o)` only.** Verified by deleting `(can-pick r o)` (the only way to
  achieve it) and checking with Fast Downward that the problem becomes unsolvable. Used as an
  allocation constraint ("r must move o"). Caveat from the smoke test: the proposal prompt
  already lists which robots passed the trial for each object, so a correct landmark here is
  mostly the SLM reading that list back.
- **Macro costs.** Fast Downward rejects summed action costs, so a macro's cost is a new function
  precomputed by code for every tuple of its cost variables. The 4-step `fetch_and_place` needs
  robots × locations³ facts. In the smoke test it made planning 10× slower with the same number
  of expansions. Condition B therefore uses the 3-step `carry` (pick, navigate, place), with
  robots × locations² facts. SLM macros are accepted up to 20k table entries.
- **Hubs are large.** The hub generator picks the receptacles with the most reachable spots, and
  those are usually big enough for two robots. Across the 51 hub episodes with 8 objects, 21 of
  145 contended receptacles are exclusive (in 20 episodes), mostly small cabinets, shelves and a
  washer-dryer rather than the hubs. The pilot gate decides whether that is enough contention.
- **Hub coverage.** 59, 57 and 51 of the 60 episodes at N = 4, 6, 8
  (`scripts/e9_hub_n{N}_episodes.txt`). Comparisons are paired within each set.
