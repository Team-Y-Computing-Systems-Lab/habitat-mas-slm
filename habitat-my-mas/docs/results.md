# Results format

Each experiment has its own folder (`results/matrix/`, `results/prompt_variants/`, ...) with its
own `index.html`. Runs made on EMOS's fork before 2026-09-26 are in `results/emos_fork_era/`. Every planner mode writes the same format, so runs can be compared directly:

- **P0 `slm_only`:** the SLM writes the plan.
- **P1 `classical_only`:** PDDL goal → classical planner.
- **P2 `slm_classical`:** SLM → PDDL → classical planner.

## Layout

```
results/
  index.csv, index.html          one row per run (compare modes/models here)
  <benchmark>/<planner>-<config>__<model>__<timestamp>[__r<repeat>]/
    config.json                  benchmark, planner, prompt variant, model, episodes, fleet, instruction level, recovery
    records.jsonl                one record per episode (below)
    summary.json                 averages over episodes + failure-code counts
    report.html                  plan, execution trace and video for every episode
    episodes/<id>/plan.json      full planner output: raw SLM text, prompt, parsed steps
    episodes/<id>/video.mp4      top-down map (trails, objects, goals ✕) + each robot's camera
    episodes/<id>/video_final.png
```

Regenerate the HTML at any time with `.venv/bin/python -m mas.eval.report results`.

## Episode record (`records.jsonl`)

| Field | Meaning |
|---|---|
| `instruction` | natural-language task given to the planner (template, "explicit" level) |
| `goal` | ground-truth PDDL goal facts from the episode (never shown to the planner) |
| `initial_facts`, `final_facts` | symbolic state before and after |
| `plan` | `{robot: [{skill, args}]}`, as the planner produced it |
| `plan_errors` | validation errors against each robot's MCP tools |
| `config`, `repeat` | configuration tag (e.g. `reach_v2+norm+repair`, `fast-downward-opt`) and repeat index |
| `pipeline` | P0 only: navigation rewrites, checker problems before/after, whether the repair round ran |
| `feasibility` | simulated reach trials: `{robot: {object: {"pick": code, "place": code}}}` |
| `trace` | every executed skill call: robot, skill, args, status, code, message, sim steps, wall times |
| `execution_wall_s` | wall-clock time of execution |
| `metrics` | see below |
| `video` | path relative to the run directory |

## Metrics

| Metric | Meaning |
|---|---|
| `success` | habitat's `pddl_success`, checked on the sim's geometry. **The headline number.** |
| `subgoal_rate` | fraction of goal facts that hold, each checked by habitat on the sim geometry |
| `success_symbolic` | every goal fact holds in our tracked symbolic state (receptacle-level, so it can be true too early when an object starts on its goal receptacle; for debugging the planner only) |
| `plan_parse_ok`, `plan_valid` | output parsed; every step uses a real robot, skill and arguments |
| `plan_steps`, `steps_ok`, `steps_failed`, `failure_codes` | execution outcome |
| `sim_steps` | simulation steps until all robots finished |
| `prompt_tokens`, `completion_tokens` | SLM tokens, summed over all calls (P0 with repair makes 2 calls) |
| `planning_latency_s` | SLM calls plus classical solving time |
| `plan_rewrites`, `repaired`, `check_problems_before/after` | P0 normalization and repair activity |
| `feasibility_sim_steps`, `feasibility_wall_s` | cost of the simulated reach trials (0 when read from the cache); not included in `sim_steps` |

Recovery is off in these runs (`config.recovery = "none"`): a robot stops at its first failed step.

Prompt variants for `slm_only` (`--prompt base|reach|reach_alloc|reach_v2|reach_hint`) and what we
learned from them are in [findings.md](findings.md). Runs made before a known bug fix are moved
to `results/_superseded/` and left out of the index; `mas.eval.report` rebuilds the index from
the run folders on disk.
