"""Benchmarks we can run: our own configs on official habitat-lab 0.3.1.

Each config (mas/sim_host/configs/*.yaml) uses official agent, sensor, task
and dataset configs, and official data only (see README "Data").
"""

import os

_CONFIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")

BENCHMARKS = {
    # Habitat 3.0 HSSD rearrangement episodes (hab3-episodes, val), official
    # two-object task spec `multi_agent_tidy_house`; Fetch + Stretch
    "hssd_fetch_stretch": os.path.join(_CONFIGS, "hssd_fetch_stretch.yaml"),
}

# E8: the same official episodes with N target objects instead of 2 (mas/datasets/scale_episodes.py)
_GEN = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "generated", "e8")
OVERRIDES = {}
for _n in (4, 6, 8):
    BENCHMARKS[f"hssd_n{_n}"] = BENCHMARKS["hssd_fetch_stretch"]
    OVERRIDES[f"hssd_n{_n}"] = [
        f"habitat.dataset.data_path={_GEN}/hssd_n{_n}.json.gz",
        f"habitat.task.task_spec_base_path={_GEN}",
        f"habitat.task.task_spec=tidy_house_n{_n}",
    ]

# articulated_agent_type -> short robot-id prefix
ROBOT_PREFIX = {
    "FetchRobot": "fetch",
    "StretchRobot": "stretch",
    "SpotRobot": "spot",
}
