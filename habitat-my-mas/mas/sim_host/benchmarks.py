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

# articulated_agent_type -> short robot-id prefix
ROBOT_PREFIX = {
    "FetchRobot": "fetch",
    "StretchRobot": "stretch",
    "SpotRobot": "spot",
}
