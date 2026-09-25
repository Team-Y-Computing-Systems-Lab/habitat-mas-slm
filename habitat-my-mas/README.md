# habitat-my-mas

A heterogeneous multi-robot system in [Habitat](https://aihabitat.org/). **Small language models
(SLMs)** plan the work, each robot's skills are exposed through **MCP servers**, and a
**classical planner** (PDDL + Fast Downward) makes plans that are sound and executable.

The benchmark is Habitat-MAS from EMOS ([Chen et al., 2024](https://arxiv.org/abs/2410.22662)).
Our design differs from EMOS: see [docs/design.md](docs/design.md).

Three planner modes are compared on the same episodes:

| Mode | Who plans | Flag |
|---|---|---|
| **P0 SLM-only** | the SLM writes every robot's step list | `--planner slm_only` |
| **P1 classical-only** | the episode's PDDL goal goes straight to the classical planner (no language model) | `--planner classical_only` |
| **P2 SLM + classical** | the SLM turns the instruction into a PDDL goal; the classical planner allocates and orders | `--planner slm_classical` |

Every run stores per-episode records, an HTML report and a video: top-down map plus each robot's camera.

## Status

| Part | State |
|---|---|
| Sim host: shared Habitat env, robot pool, join/leave, video | working |
| MCP server per robot: skills as tools, PDDL schema in each tool's metadata, capabilities as resources | working |
| Skills: `navigate_to`, `pick`, `place`, `look`, `reset_arm`, `wait` | working (Fetch, Stretch, Spot); the drone navigates only in its own benchmarks |
| P0 SLM-only, with plan normalization and one check-and-repair round | working |
| Furniture-aware reach check (simulated trial pick/place per robot and object) | working |
| P1 classical-only, P2 SLM + classical (unified-planning + Fast Downward) | working |
| Recovery with replanning, resource-constraint scenarios | designed, not built (design.md §6–7) |
| MP3D multi-floor benchmarks | need the full Matterport3D download |

Findings so far: [docs/findings.md](docs/findings.md).

## How it fits together

```
 .venv (Python 3.12)                                   habitat conda env (Python 3.9)
┌──────────────────────────────────────────────┐      ┌─────────────────────────────┐
│ mas.eval.run  (experiment runner)            │      │ mas.sim_host.server          │
│   ├─ planner: slm_only | classical | slm+cl  │ HTTP │   SimHost: habitat-lab env,  │
│   │    ├─ SLM via Ollama (OpenAI API)        │ JSON │   robot pool, skill runners, │
│   │    └─ unified-planning + Fast Downward   │◄────►│   scheduler (all robots step │
│   └─ MCP client ── stdio ──┐                 │      │   together), recorder        │
│ mas.mcp_servers.robot_server (one per robot) ├─────►│                              │
└──────────────────────────────────────────────┘      └─────────────────────────────┘
```

Two Python environments are needed because habitat-sim 0.3.1 runs on Python 3.9, while the MCP SDK
needs Python ≥ 3.10. They talk over HTTP, so the planner side can also run on another machine,
such as a Jetson, with `--sim-url`.

## Repository layout

```
mas/
  sim_host/     host.py (env, pool, state, feasibility), skills.py (skill runners),
                scheduler.py, server.py / client.py (HTTP), recorder.py (video), naming.py
  skills/       catalog.py (skills + PDDL schemas), robots.py (robot traits)
  mcp_servers/  robot_server.py (one MCP server per robot)
  slm/          client.py (OpenAI-compatible client, JSON parsing)
  planners/     slm_only.py (P0 + prompt variants), repair.py (normalize, check),
                classical.py (P1, P2), base.py (shared plan format)
  eval/         run.py (runner), instructions.py, report.py (HTML reports)
scripts/        run_experiments.sh (full matrix), calibrate_reach.py
tests/          smoke_sim_host.py, smoke_mcp_robots.py
docs/           design.md, datasets.md, results.md, findings.md
results/        experiment outputs (see docs/results.md)
data -> …       symlink to the Habitat / EMOS data folder
```

## Setup

Tested on Ubuntu 22.04, an NVIDIA RTX PRO 6000 (driver 580), conda, Ollama 0.22.

### 1. Simulator environment (`habitat`, Python 3.9)

We use EMOS's fork of habitat-lab, because it adds the Habitat-MAS robots, actions and configs:

```bash
git clone -b embodied_mas https://github.com/SgtVincent/EMOS.git
conda create -n habitat python=3.9 cmake=3.14.0 -y
conda activate habitat
conda install habitat-sim=0.3.1 withbullet -c conda-forge -c aihabitat -y
pip install -e EMOS/habitat-lab
pip install imageio imageio-ffmpeg opencv-python     # video recording
```

Check: `python -c "import habitat, habitat_sim; print(habitat_sim.__version__)"` prints `0.3.1`.

### 2. Data

This follows [EMOS's data instructions](https://github.com/SgtVincent/EMOS/tree/embodied_mas/habitat-mas#download-data):

```bash
python -m habitat_sim.utils.datasets_download --data-path /path/to/emos_data \
  --uids hssd-hab hab3-episodes habitat_humanoids hab_spot_arm ycb hab3_bench_assets rearrange_task_assets
# plus the Habitat-MAS robot configs and episodes from EMOS's Google Drive link,
# merged into /path/to/emos_data (robots/, datasets/, robots/robot_configs/)
ln -s /path/to/emos_data data            # from the repository root
```

MP3D (for the multi-floor tasks) needs the Matterport3D terms of use signed. Only the example
scene is used here. See [docs/datasets.md](docs/datasets.md) for what is and isn't on disk.

### 3. Planner environment (`.venv`, Python ≥ 3.10)

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 4. SLMs (Ollama)

```bash
# install: https://ollama.com/download
ollama pull qwen3:4b-instruct          # main model
ollama pull qwen3:0.6b llama3.2:3b gemma2:2b smollm2:1.7b   # Jetson-sized comparison
```

Any OpenAI-compatible server works (`--slm-url`), e.g. llama.cpp's `llama-server` or vLLM.

### 5. Check that it all works

```bash
conda activate habitat
python -m mas.sim_host.server --benchmark replica_pool4 &      # 4-robot pool
python tests/smoke_sim_host.py                                  # any Python
.venv/bin/python tests/smoke_mcp_robots.py                      # MCP end to end
kill %1
```

## The classical planner: what's installed

Everything comes from pip, in `requirements.txt`. Nothing else is needed on x86_64 Linux:

| Package | What it is |
|---|---|
| `unified-planning` | Python planning library. Parses the PDDL we generate and runs the engines |
| `up-fast-downward` | the [Fast Downward](https://www.fast-downward.org/) engine. The x86_64 Linux wheel ships compiled binaries. Engines: `fast-downward` (satisficing) and `fast-downward-opt` (optimal, **our default**) |
| `up-pyperplan` | a pure-Python engine, the fallback. It ignores action costs, so plans can contain detours |

Choose the engine with `--engine fast-downward-opt | fast-downward | pyperplan`.

On other platforms, such as a Jetson (aarch64), pip may have to build Fast Downward from source.
That needs `cmake` and a C++17 compiler (`sudo apt install cmake g++`). If the build fails, use
`--engine pyperplan`. We have not yet tested on a Jetson.

How the planner is set up (`mas/planners/classical.py`):

- **Domain:** assembled from the PDDL schemas inside each robot's MCP tool `meta`, so it always
  matches the current fleet.
- **Problem:** built from the sim state. `can-pick` / `can-place` facts come from the
  simulated feasibility check.
- **Allocation:** every feasible object→robot assignment is solved, and the plan with the
  smallest estimated makespan is kept. This is how parallel plans are preferred.

## Running

Start the simulator once, and leave it running:

```bash
conda activate habitat
python -m mas.sim_host.server --benchmark replica_manipulation      # port 8765
```

Benchmarks (`mas/sim_host/benchmarks.py`): `replica_manipulation`, `replica_perception`,
`hssd_height_man`, `hssd_dist_man`, `hssd_height_per`, `replica_pool4`.

Then run a planner from the planner environment:

```bash
# P0: SLM only (best prompt + rewrite + one check/repair round)
.venv/bin/python -m mas.eval.run --planner slm_only --prompt reach_v2 --normalize --repair --episodes 10
# P1: classical only
.venv/bin/python -m mas.eval.run --planner classical_only --episodes 10
# P2: SLM writes the goal, classical planner does the rest
.venv/bin/python -m mas.eval.run --planner slm_classical --model qwen3:4b-instruct --episodes 10
```

| Flag | Meaning |
|---|---|
| `--model` | Ollama model (default `qwen3:4b-instruct`) |
| `--prompt` | P0 prompt variant: `base`, `reach`, `reach_alloc`, `reach_v2` (default), `reach_hint` |
| `--normalize` | P0: rewrite `navigate_to(place)` + `pick(obj)` into `navigate_to(obj)` |
| `--repair` | P0: check the plan (order, holding, simulated reach, goals) and give the SLM one repair round |
| `--engine` | classical engine (default `fast-downward-opt`) |
| `--episodes N` / `--episode-ids …` | which episodes |
| `--repeat K` | repeat index, recorded in the run id |
| `--no-video` | skip video recording |
| `--sim-url`, `--slm-url` | sim host and SLM server addresses (defaults: localhost) |

Each run writes to `results/<benchmark>/<planner>-<config>__<model>__<time>/`:
- `records.jsonl`, `summary.json`
- `report.html`: plans, traces and videos
- `episodes/<id>/`: `video.mp4`, `plan.json`, with the full prompts, SLM output and PDDL

Across runs, `results/index.html` is the overview. The format is in [docs/results.md](docs/results.md).

The feasibility check tries each robot/object pair in simulation, about 20 s per episode for
2 robots × 2 objects. It is cached in `results/feasibility/`, so repeated runs pay it once.

## Reproducing the experiments

| Experiment | Command | Output |
|---|---|---|
| Prompt variants for P0 (findings, 2026-09-23) | `for v in base reach reach_alloc reach_v2 reach_hint; do .venv/bin/python -m mas.eval.run --planner slm_only --prompt $v --episodes 10 --out results/prompt_variants; done` | `results/prompt_variants/` |
| Reach-check calibration (height vs IK vs trial) | `conda activate habitat && python scripts/calibrate_reach.py --episodes 10` | `results/reach_calibration/` |
| Full matrix: P0 ablations, P1, P2 × models × repeats | `bash scripts/run_experiments.sh` (env vars: `EPISODES=30 REPEATS=3 MAIN_MODEL=… SMALL_MODELS="…"`) | `results/matrix/` |

`run_experiments.sh` starts and stops its own sim host. With the defaults (30 episodes, 3
repeats, 5 models) it takes a few hours on one GPU. Repeat 1 of every configuration finishes
before any repeat 2, so a partial run is still usable. Run
`.venv/bin/python -m mas.eval.report <dir>` to regenerate the HTML at any time.

The SLM runs use temperature 0, but Ollama is not bit-for-bit deterministic on GPU. Compare
configurations using the repeats, not single runs.

## Running the planner on a Jetson

Habitat stays on the workstation. On the Jetson, run Ollama (or `llama-server`) and the
`.venv` side:

```bash
.venv/bin/python -m mas.eval.run --planner slm_classical --model qwen3:0.6b \
  --sim-url http://<workstation>:8765 --slm-url http://localhost:11434/v1 --engine pyperplan
```

For that, start the sim host with `--host 0.0.0.0` so it accepts remote connections. Findings on
which small models work: [docs/findings.md](docs/findings.md).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Address already in use` when starting the sim host | another sim host is running: `pgrep -af mas.sim_host.server`, or use `--port` |
| Every skill returns `SIM_UNREACHABLE` | the sim host is not running or crashed; check its log |
| The drone never finishes `navigate_to` in `replica_pool4` | known issue with EMOS's 4-robot config; the drone works in `replica_perception` |
| `NAV_STUCK` / "approach point blocked" messages | habitat's navigation can spin in place when blocked; our skill ends it after 60 still steps and counts it as arrived when within 1 m |
| Many `[Error] … navmesh … not found` lines at startup | harmless: ReplicaCAD scenes `sc4_*` that no episode uses |
| MP3D benchmarks fail to load | only the example MP3D scene is present; see docs/datasets.md |

## Documentation

- [docs/design.md](docs/design.md): the idea, architecture, and how it differs from EMOS
- [docs/datasets.md](docs/datasets.md): data inventory, input format, processing pipeline
- [docs/results.md](docs/results.md): result files and metrics
- [docs/findings.md](docs/findings.md): experiment log, with evidence and explanations

## References

- EMOS / Habitat-MAS: J. Chen et al., *EMOS: Embodiment-aware Heterogeneous Multi-robot Operating
  System with LLM Agents*, arXiv:2410.22662, 2024. Code: https://github.com/SgtVincent/EMOS
- Our previous work: IEEE Xplore document 11476074, and ACM IGSC 2026,
  doi:10.1145/3797248.3815414 (see docs/design.md §2)
