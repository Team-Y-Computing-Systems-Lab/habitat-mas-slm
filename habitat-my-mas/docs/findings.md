# Findings

Companion to [design.md](design.md) and [results.md](results.md). This is a running log of what we
learned from experiments, with evidence and likely causes. Newest entries go at the top.

**Note:** entries dated before 2026-09-26 were measured on EMOS's fork of habitat-lab. Their result
paths now live under `results/emos_fork_era/`.

---

## 2026-09-26 · Standalone on official Habitat (EMOS removed)

**Why:** the project had been running on EMOS's fork of habitat-lab, because the existing
`habitat` env had it installed, and it also used EMOS's released data. The goal is a standalone
system that depends only on official Habitat, with EMOS as related work.

**What changed:**

| Before (EMOS fork) | Now (official 0.3.1) |
|---|---|
| EMOS benchmark and robot configs | our own config (`mas/sim_host/configs/hssd_fetch_stretch.yaml`) built from official agent, sensor, task and dataset configs |
| Habitat-MAS episodes and robot line-ups (EMOS release) | official `hab3-episodes` (HSSD, validation split), official task spec `multi_agent_tidy_house`; habitat places the robots |
| EMOS per-robot navigation | official `OracleNavAction` |
| EMOS oracle pick/place (IK toward the object, magic grasp) | **our own arm controller**: turn so the arm can reach, move the gripper about 4 cm per step with IK on the robot's **official URDF** (pybullet, `mas/sim_host/ik.py`), then grasp or release with the official grasp manager |
| EMOS arm-only models for Stretch and Spot | not needed: IK runs on the full official URDF; only Fetch has an official arm-only model |
| Drone | removed; there's no official drone |
| Env `habitat` (EMOS clone) | new env `habitat-mas`: habitat-sim 0.3.1 (conda) + habitat-lab v0.3.1 (official GitHub tag) |

**Checks:**
- **Arm model vs habitat:** after fitting the frame transform once per robot, the IK model and
  habitat agree on every link position to under 1 mm.
- **IK solver:** converges to under 1 mm on reachable targets, about 4 ms per solve.
- **Skills on 4 episodes (first object each):**
  - Fetch completed 4 of 4 pick-and-places, all confirmed by habitat's own goal check.
  - Stretch completed 2 of 4. Its two failures are an object at 0.36 m and a goal spot at
    0.33 m, below its lowest reach of 0.38 m; both fail cleanly as `OUT_OF_REACH`.
- **Smoke tests:** robot pool and MCP end to end both pass.
- **Planners:** classical, SLM-only and SLM + classical all run end to end.

**F14. On official robot models the embodiment gap is reversed.**
- With EMOS's arm models, Stretch couldn't reach *high* shelves (max 1.22 m).
- From the official URDFs: Fetch reaches 0.05–1.76 m, and Stretch reaches 0.38–1.70 m, so Stretch
  now fails on *low* objects.

Allocation still matters, but which robot suits which object changed. That's a reminder that
embodiment findings depend on the robot model used.

**Things we had to build ourselves, and why:**

- **Arm IK:** official `ArmEEAction` drives motors, which do nothing in kinematic mode, and
  habitat re-applies stored joint values (`fix_joint_values`) every step. Our controller sets both.
- **Frame fit:** habitat reports link positions at each link's centre of mass, while pybullet's IK
  works on URDF link frames with the base at its centre of mass. The frame is fitted from all
  links' centres of mass (exact).
- **Numerical Jacobian:** pybullet's analytic Jacobian uses another frame convention, so we
  compute it numerically in habitat's frame.
- **Turning before a grasp:** Stretch's arm points sideways, so pick/place first turn the base
  to the heading with the smallest turn from which IK reaches the target.
- **Getting unstuck:** turning stops when the heading stops improving, since a blocked base
  jitters in place. The rest pose is clamped to joint limits, because the start pose can sit
  slightly outside them.

**Consequence:** every earlier number (E1–E5, F1–F13) was measured on the EMOS fork, with different
controllers, episodes and robot models. Those results are archived in `results/emos_fork_era/` and
must not be mixed with new runs. The qualitative findings (e.g. P2 ≫ P0 for small models) need to
be re-established by the new matrix run.

---

## 2026-09-24 · Planner matrix: P0 vs P1 vs P2, five models, 3 repeats

**Setup:**
- Benchmark `replica_manipulation`, first 30 episodes, fleet Fetch + Stretch.
- 3 repeats of every configuration; videos for repeat 1 only.
- `scripts/run_experiments.sh`, aggregated by `scripts/analyze_matrix.py` into
  `results/matrix/summary.md`, with reports in `results/matrix/index.html`.

| Planner | Model | Success | Sub-goals | Tokens in / out | Plan time | Main failures |
|---|---|---|---|---|---|---|
| P1 classical only | none | **89%** (87–90) | 97% | 0 | 0.4 s | 3 episodes no robot can do |
| P2 SLM + classical | qwen3:4b-instruct | **90%** | 98% | 184 / 51 | 0.9 s | same 3 episodes |
| P2 SLM + classical | gemma2:2b | **90%** | 98% | 190 / 55 | 2.3 s | same 3 episodes |
| P2 SLM + classical | llama3.2:3b | 86% (83–87) | 96% | 191 / 49 | 1.5 s | +1 wrong goal |
| P2 SLM + classical | qwen3:0.6b | 80% | 92% | 524 / 47 | 1.6 s | 4 unusable goals |
| P2 SLM + classical | smollm2:1.7b | 33% | 81% | 200 / 39 | 0.8 s | incomplete goals |
| P0 SLM only, `reach_v2` | qwen3:4b-instruct | 72% (70–73) | 91% | 1113 / 646 | 2.9 s | `OUT_OF_REACH` |
| P0 + rewrite | qwen3:4b-instruct | 73% | 92% | 1113 / 647 | 2.8 s | `OUT_OF_REACH` |
| P0 + rewrite + repair | qwen3:4b-instruct | 80% | 93% | 1578 / 938 | 4.1 s | `OUT_OF_REACH` |
| P0 + rewrite + repair | gemma2:2b | 20% | 68% | 2517 / 401 | 12.5 s | reach, `NOT_HOLDING` |
| P0 + rewrite + repair | smollm2:1.7b | 7% | 32% | 2759 / 422 | 4.1 s | reach |
| P0 + rewrite + repair | llama3.2:3b | 3% | 39% | 2341 / 460 | 8.2 s | `NOT_HOLDING`, reach |
| P0 + rewrite + repair | qwen3:0.6b | 0% | 50% | 3708 / 693 | 5.6 s | invented skill names |

Ranges are min–max over the 3 repeats; a single number means all 3 repeats agreed.

**F9. SLM + classical (P2) reaches the ceiling set by the robots' own skills, with a 4B model and even a 2B one.**
- P1 and P2 fail on the *same* 3 episodes (54, 149, 152). In each, the simulated trial shows that
  *no* robot can move one of the objects, so ~90% is the ceiling for these skills. It isn't a
  planner limit.
- P2 with qwen3:4b and gemma2:2b wrote exactly the right goal in 30/30 episodes.
- P2 needs about 6× fewer input tokens and 3–5× less planning time than the best P0.

*Why:* the SLM only has to do what it is good at here, turning an instruction into a goal.
Allocation, ordering and reach constraints go to the planner, which cannot get them wrong.

**F10. The classical planner uses both robots in parallel; the SLM rarely does.**
- P1 and P2 used both robots in 16/30 episodes; the best P0 in 5/30.
- Successful P2 episodes took 473 sim steps on average, against 580 for P0, about 20% faster.

*Why:* the allocation step picks the assignment with the smallest estimated makespan. The SLM,
once told about reach, plays safe and gives everything to Fetch (F4).

**F11. For P0, rewriting navigation did nothing here; one repair round added 7 points.**
- The rewrite (73% vs 72%) helps only when two robots head for the same receptacle, which the
  "Fetch does everything" plans rarely produce.
- Repair ran in 10/30 episodes. It fixed the checker's problem in 3 of them, and all 3 then
  succeeded. In the other 7 the problem was still there after repair (one of those episodes
  succeeded anyway).

*Implication:* small SLMs don't use feedback reliably. A plan they can't repair is better handed
to the classical planner (a P0 → P2 fallback).

**F12. Small models can translate goals but cannot plan.**

| Model | P0 (writes the whole plan) | P2 (writes the goal) |
|---|---|---|
| gemma2:2b | 20% | 90% |
| llama3.2:3b | 3% | 86% |
| qwen3:0.6b | 0% | 80% |
| smollm2:1.7b | 7% | 33% |

The P0 failure modes differ by model:
- **qwen3:0.6b** invents skill names such as `"place(object, place)"`, so the robots' tools
  reject 45 calls per run.
- **llama3.2:3b** places objects it never picked up (`NOT_HOLDING`).
- **smollm2** ignores reach.

In P2, smollm2 wrote a *complete* goal in only 1/30 episodes: it usually lists just the first
object. qwen3:0.6b produced no usable goal in 4/30.
*Implication for a Jetson:* **P2 with gemma2:2b** matches the 4B model at 90%, at 2.3 s per plan
on this GPU. It is the candidate to deploy. qwen3:0.6b is the fallback for the smallest devices.

**F13. Repeats are nearly identical.**
Most configurations gave the same number in all 3 repeats; the largest spread is 4 points
(llama3.2 P2: 83–87%). Earlier single-run differences (F7) came mostly from code changes between
runs, not from randomness. With 30 episodes, one episode is 3.3 points, so differences under
about 5 points are not meaningful.

### Bugs found during the matrix (fixed)

- **Unparseable goal crashed the whole run.** llama3.2 wrote a goal naming `tvstand_0`, a place
  that exists only as an object's *starting* spot. That produced unparseable PDDL, and the error
  killed the whole run at episode 57 in every repeat. Now such a goal fails that episode only
  ("no placement spot"), and any planner error is recorded as a failed episode. The three llama3.2
  P2 runs were redone.
- **Wrong failure code for invented skill names (not fixed).** A call to a non-existent skill
  shows up as `BAD_TOOL_RESULT`; `INVALID_STEP` would be clearer. The outcome is correct.

### Next

1. **P0 → P2 fallback:** if the checker still finds problems after repair, hand the goal to the
   classical planner.
2. **Harder instructions** (referential and underspecified), where the SLM's grounding matters
   more than in these explicit ones.
3. **Recovery with replanning (design §6).** In the 3 infeasible episodes it could at least
   report the dropped goal, and the classical planner can replan after an execution failure.
4. **HSSD benchmarks** (`hssd_height_man`, `hssd_dist_man`) for more scene variety.

---

## 2026-09-24 · A reach check that accounts for furniture

**Question:** F6 showed that a height-only reach model misses furniture. Can a geometric check
do better?

**Method:** `scripts/calibrate_reach.py` covers the 10 episodes × 2 objects × 2 robots, 40 pairs.
Before acting, it makes three predictions for each pair:

1. **Height range:** the object's height lies within the arm's forward-kinematics height range.
2. **IK only:** put the robot at the approach pose its navigation would use, facing the object.
   Solve inverse kinematics for the gripper and accept if it ends within 0.15 m, the
   magic-grasp distance.
3. **IK + collision:** as 2, and the arm in that pose must not intersect the scene.

Then it really navigates and picks, and places if the pick worked. The data is in
`results/reach_calibration/replica_manipulation.jsonl`.

| Predictor | Pick accuracy (n=40) | Wrongly says "can" | Wrongly says "cannot" | Place accuracy (n=26) |
|---|---|---|---|---|
| Height range | **82%** | 3 | 4 | **92%** |
| IK only | 65% | 1 | 13 | n/a |
| IK + collision | 38% | 0 | 25 | 23% |

**Finding F8: the "smarter" geometric checks are worse than the height range.**
Adding collision flagged 25 of 40 pairs, and most of them were picks that succeeded.
*Why:*
- The arm routinely brushes furniture while the magic grasp still works, so "any contact"
  is far too strict.
- For Stretch, a single IK solve is off by about 1 m even for objects Stretch *does* pick. The
  telescoping arm and lift aren't moved the way the oracle controller moves them.
- For Fetch, navigation ends in a different pose from the modelled approach, so IK comes out
  0.2–0.3 m short for objects Fetch actually picks.

A trustworthy geometric check would need a real motion planner.

**What we use instead:** a simulated trial, `SimHost.feasibility`. For every arm robot and
task object it runs navigate + pick, then navigate + place if the pick worked, in the real
scene, and then resets the episode. This is the simulation stand-in for a motion planner's
feasibility check.
- **Cost:** about 20–25 s of wall time per episode (2 robots × 2 objects, about 650 sim steps).
  It is cached per episode, and the trial steps are *not* counted in the episode's `sim_steps`.
- **Used by:** the P0 checker (`--repair`) and both classical modes, as `can-pick` /
  `can-place` facts.
- **Caveat:** in the real run the approach point is sampled again, so a trial result is a
  strong hint, not a guarantee.

### Bugs found while building this (fixed before the matrix run)

- **Stale navigation state on the same episode.** Resetting the *same* episode kept habitat's
  cached approach points and "already arrived" flags. They are now cleared on every reset
  (`SimHost._clear_action_caches`).
- **Navigation livelock.** Blocked short of its approach point, habitat's navigation spins in
  place forever. `navigate_to` now ends after 60 still steps once the robot has moved. It counts
  as arrived within 1 m of the target, and fails with `NAV_STUCK` otherwise.
- **Feasibility run on the wrong episode.** The first version checked the *previously* loaded
  episode. The check now takes the episode ID and the runner asserts that it matches.

---

## 2026-09-23 · Prompt variants for the SLM-only planner

### Setup

| | |
|---|---|
| Benchmark | `replica_manipulation` (Habitat-MAS, ReplicaCAD), first 10 episodes: 119, 146, 106, 152, 163, 54, 43, 150, 40, 95 |
| Fleet | `fetch_0` (7-DoF arm) + `stretch_0` (telescoping arm on a lift) |
| Model | `qwen3:4b-instruct` via Ollama, temperature 0, one planning call per episode |
| Task | move 2 objects to their goal receptacles; grippers empty at the end |
| Instructions | template, "explicit" level (every object and place named) |
| Recovery | none: a robot stops at its first failed step |
| Arm reach | from URDF forward kinematics over 2000 random joint configurations: **Fetch 0.00–1.84 m**, **Stretch 0.12–1.22 m** above the floor |

A manual check matches the kinematic estimate. Fetch picked objects 0.2–1.6 m above the floor.
Stretch picked objects at 0.05 m and 1.0 m, but failed at every object above ~1.27 m.

### Prompt variants

All variants are defined in `mas/planners/slm_only.py` (`--prompt`).

| Variant | What the SLM is given, in addition to the previous row |
|---|---|
| `base` | skills (from MCP), which object is on which place, the instruction |
| `reach` | each robot's arm reach, each object's height and its goal spot's height, and the rule "only give an object to a robot that can reach both" |
| `reach_alloc` | `reach`, plus the SLM must first fill an allocation table (object, heights, `can_reach`, `assigned_to`) and then write the plan |
| `reach_v2` | `reach`, but each object is written as one line (*"now on X at h m → goes to Y (goal spot h m)"*), plus a fixed four-step template: `navigate_to(obj), pick(obj), navigate_to(goal), place(obj, goal)` |
| `reach_hint` | `reach_v2`, plus the **system** compares the heights and tells the SLM which robots can handle each object |

### Results (10 episodes each, same episodes, same code)

| Variant | Success | Sub-goals | Picks that navigate to the object | Fetch does everything | Both robots used | Avg. sim steps | Tokens in / out | Plan latency |
|---|---|---|---|---|---|---|---|---|
| `base` | 10% | 75% | 0/20 | 0/10 | 10/10 | 341 | 885 / 432 | 2.0 s |
| `reach` | 40% | 73% | 0/20 | 8/10 | 2/10 | 451 | 1079 / 602 | 2.7 s |
| `reach_alloc` | 20% | 80% | 0/20 | 1/10 | 9/10 | 404 | 1197 / 461 | 2.1 s |
| **`reach_v2`** | **90%** | **98%** | 0/20 | **10/10** | 0/10 | **579** | 1114 / 599 | 2.7 s |
| `reach_hint` | 70% | 93% | 4/20 | 8/10 | 2/10 | 536 | 1133 / 591 | 2.7 s |

Per episode (✓ = success):

| Episode | base | reach | reach_alloc | reach_v2 | reach_hint |
|---|---|---|---|---|---|
| 119 | · | · | · | ✓ | · |
| 146 | · | · | · | ✓ | ✓ |
| 106 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 152 | · | · | · | ✓ | ✓ |
| 163 | · | · | ✓ | ✓ | ✓ |
| 54 | · | · | · | · | · |
| 43 | · | ✓ | · | ✓ | · |
| 150 | · | ✓ | · | ✓ | ✓ |
| 40 | · | ✓ | · | ✓ | ✓ |
| 95 | · | · | · | ✓ | ✓ |

Every variant produced a valid plan in 10/10 episodes: real robots, real skills, correct arguments.
**Plan format is not the problem. Choosing the right robot and the right target is.**

The runs are in `results/prompt_variants/replica_manipulation/slm_only-<variant>__qwen3-4b-instruct__20260923-15*`.
Open `results/prompt_variants/index.html` to compare them, or each run's `report.html` for plans, traces and videos.

### Findings

**F1. Without body information, the SLM splits work evenly, and that is wrong here.**
In `base`, all 10 plans gave one object to each robot. In 9 of 10 episodes Stretch got an object
or goal spot above its 1.22 m reach, which failed as `OUT_OF_REACH`.
*Why:* the prompt contains nothing that distinguishes the two arms. The model falls back on a
generic "divide the work" habit, which also looks efficient because it runs in parallel.

**F2. Giving the numbers helps, but the SLM compares them unreliably.**
`reach` raised success from 10% to 40%. Its 6 failed episodes break down as follows (one episode
can show more than one error):

- **Picking at the wrong place (4 episodes: 54, 95, 146, 119).** The robot navigated to the
  object's *goal* receptacle, or to another object's place, and then called `pick`. In 119 it
  also skipped a `place` and picked while still holding something (`HAND_FULL`).
- **Wrong allocation (2 episodes: 146, 163).** Stretch got an object at 1.34 m, or a goal spot
  at 1.33 m, both above its 1.22 m reach.
- **Ambiguous navigation (1 episode: 152).** Two objects were on the same cabinet and the robot
  grasped the wrong one (see F5).

*Why:* the height information shifted the model's attention to "who can reach", and it got
sloppier about "where is the object now". It confuses the source and destination places that
sit next to each other in the prompt. The allocation errors are two-sided interval checks gone
wrong. A 4B model has to do one per object–robot pair inside a single generation, and nothing
checks the answer. `reach_v2` fixes most of the first category by putting "now on X → goes to Y"
on one line per object.

**F3. The structured "allocation table" made things worse: the model filled it in to justify its choice.**
`reach_alloc` scored 20%. Checked against the kinematic truth, the table's `can_reach` column was
wrong in 13 of 20 rows. Yet the plan followed the table in 20 of 20 rows. The errors all point the
same way: the model had already decided "one object per robot" and wrote `can_reach` to match.
For example, it claimed Stretch could reach a softball at 1.61 m (max 1.22 m).
*Why:* a table with one `assigned_to` per object nudges the model toward a balanced split. The
reasoning fields are filled in *after* that decision, as justification, not as a check. For small
models, "reason in a structured field first" is not a verifier.

**F4. The best prompt (`reach_v2`, 90%) won by never using Stretch.**
Every `reach_v2` plan gave both objects to Fetch. That is always safe in this benchmark, because
Fetch's reach contains Stretch's. But it throws away parallelism: 579 sim steps on average,
against 341 for `base`, about 70% longer.
*Why:* the per-object lines together with the rule "if only one robot can, that robot must do it"
make the model conservative. When unsure, it picks the robot with the larger range.
*Caveat:* this shortcut only works because one robot dominates the other. With complementary
robots (one reaches only low shelves, another only high ones), "give it all to the bigger arm"
fails. **The 90% overstates the SLM's ability to allocate.**

**F5. Telling the SLM who can reach what (`reach_hint`) allowed parallel work, which exposed an execution ambiguity.**
With the height comparison done by the system, the SLM split work in 2/10 episodes and success
was 70%. Two of the three failures were `WRONG_OBJECT`. Both robots were sent to the same cabinet
*by receptacle name*, e.g. `navigate_to(wall_cabinet_02_0)`, and one grasped the other robot's
object lying next to its own.
*Why:* the SLM almost never navigates to the object itself. Only 0–4 of 20 picks do so in any
variant, even though the prompt says "navigate_to the OBJECT itself before pick". Going to "the
cabinet" is the model's natural phrasing. Our `navigate_to(place)` then has to guess which object
on that place is meant. That guess is fine for one robot and ambiguous for two.

**F6. A height-only reach model is incomplete.**
Episode 54 failed in *every* variant. The racquetball sits on a TV-stand shelf 0.07 m above the
floor. That is inside Fetch's kinematic range (from 0.00 m), but the arm can't get into the
shelf, so the pick fails with `OUT_OF_REACH`.
*Why:* reach is computed from the arm alone, ignoring the furniture around the object and the
robot's approach pose.

**F7. Results vary between identical runs.**
The same `base` configuration scored 20% and then 10%: in episode 163 the model produced a
different plan. Ollama is not fully deterministic even at temperature 0, because GPU batching
changes floating-point order. With 10 episodes, one episode is 10 points.
*Implication:* use ≥50 episodes and ≥3 repeats before comparing variants that are close.

### Worked example: episode 150

*"Tidy up: put toy_airplane_0 on chair_01_0; put strawberry_0 on wall_cabinet_01_0."*

| Object | Now on (height) | Goes to (goal spot height) | Fetch can? | Stretch can? |
|---|---|---|---|---|
| toy_airplane_0 | table_02_0 (0.91 m) | chair_01_0 (0.38 m) | yes | yes |
| strawberry_0 | wall_cabinet_02_0 (**1.29 m**) | wall_cabinet_01_0 (0.71 m) | yes | **no** (max 1.22 m) |

**`base`, failed.** The SLM's reasoning: *"fetch_0 … pick up toy_airplane_0 … stretch_0 can
navigate to wall_cabinet_02_0 to pick up strawberry_0"*. It had no height information, so it split
evenly.

```
fetch_0    navigate_to(table_02_0)          OK   [78]
fetch_0    pick(toy_airplane_0)             OK   [64]
stretch_0  navigate_to(wall_cabinet_02_0)   OK   [248]
fetch_0    navigate_to(chair_01_0)          OK   [135]
fetch_0    place(toy_airplane_0, chair_01_0) OK  [59]
stretch_0  pick(strawberry_0)               OUT_OF_REACH [174]   ← 1.29 m > 1.22 m
```
Video: [`base` ep 150](../results/prompt_variants/replica_manipulation/slm_only-base__qwen3-4b-instruct__20260923-154618/episodes/150/video.mp4)

**`reach_v2`, succeeded, sequentially.** The SLM's reasoning: *"stretch_0's arm reaches 0.12–1.22 m …
cannot reach it (1.29 > 1.22) … fetch_0 must pick both objects"*. The comparison is right. But
Stretch *could* have carried the toy airplane at the same time, and the plan left it idle.

```
fetch_0  navigate_to(table_02_0) → pick(toy_airplane_0) → navigate_to(chair_01_0) → place(...)
fetch_0  navigate_to(wall_cabinet_02_0) → pick(strawberry_0) → navigate_to(wall_cabinet_01_0) → place(...)
all OK, 723 sim steps
```
Video: [`reach_v2` ep 150](../results/prompt_variants/replica_manipulation/slm_only-reach_v2__qwen3-4b-instruct__20260923-155022/episodes/150/video.mp4)

The ideal plan gives the toy airplane to Stretch and the strawberry to Fetch, in parallel, in
roughly `base`'s ~420 steps with both objects delivered. No prompt variant found it.

### What to do next

Prompting alone got the SLM from 10% to 90%, but F4–F6 show the limits. These are the next steps,
roughly in order of cost:

1. **Adopt `reach_v2` as the SLM-only (P0) prompt** for future comparisons. It is the fair
   "best effort" SLM-only baseline.
2. **Normalize navigation targets (system fix, F5).** Before execution, rewrite
   `navigate_to(place)` followed by `pick(obj)` into `navigate_to(obj)`. Separately, make
   `navigate_to(place)` return an `AMBIGUOUS_TARGET` error when several objects on it could be
   meant, instead of guessing.
3. **Check the plan and give the SLM one repair round (F2, F3).** A cheap programmatic check
   covers reach, one object per robot, and pick-before-place. Its errors go back to the SLM once,
   for repair. This checks the model's reasoning instead of trusting it.
4. **Collision-aware reach (F6).** Add an MCP tool `can_reach(object)` that runs the robot's IK
   toward the object from its approach pose, so capabilities include the furniture around the
   object.
5. **Classical planner (P1/P2).** Its `can-reach` preconditions and cost-optimal allocation
   address F2–F4 directly. It should find the parallel plan for episode 150 that no prompt found.
6. **More data (F7).** Use 50 episodes × 3 repeats, add HSSD `height_man` / `dist_man`, and try
   smaller models (`qwen3:0.6b`, `llama3.2:3b`) with the Jetson in mind.

### Reproduce

```
conda activate habitat && python -m mas.sim_host.server --benchmark replica_manipulation
for v in base reach reach_alloc reach_v2 reach_hint; do
  .venv/bin/python -m mas.eval.run --planner slm_only --prompt $v --model qwen3:4b-instruct --episodes 10
done
```

Runs made before the resolver fix (see below) are kept in `results/prompt_variants/_superseded/` and excluded
from the index.

### Bugs found and fixed during these experiments

These affected earlier numbers, which is why those runs are superseded.

- **Navigating to a receptacle went to its goal spot.** `navigate_to(<receptacle>)` sent an
  empty-handed robot to the receptacle's *goal* spot, not to the object waiting there. It now
  resolves by intent: a robot holding an object goes to that object's goal spot, and an
  empty-handed robot goes to an object waiting there that hasn't been delivered yet.
- **Sub-goals were counted too early.** Sub-goals were counted from receptacle-level facts,
  which are already true when an object merely *starts* on its goal receptacle. They are now
  checked by habitat on the sim geometry.
- **The sim host crashed on pick/place.** It crashed when the last robot in a step was picking
  or placing, because habitat uses the last action's return value as the observations.
