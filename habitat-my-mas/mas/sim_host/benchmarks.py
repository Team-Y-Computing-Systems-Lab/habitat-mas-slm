"""Short names for the Habitat-MAS benchmark configs we can run locally.

Configs come from the EMOS habitat-lab fork installed in the `habitat` env.
MP3D configs are left out until the full Matterport3D scenes are downloaded
(see docs/datasets.md).
"""

BENCHMARKS = {
    # HSSD, single floor (main evaluation)
    "hssd_height_man": "benchmark/multi_agent/config_height_man.yaml",
    "hssd_dist_man": "benchmark/multi_agent/config_dist_man.yaml",
    "hssd_height_per": "benchmark/multi_agent/config_height_per.yaml",
    # ReplicaCAD (development)
    "replica_manipulation": "benchmark/multi_agent/config_fetch_stretch_man.yaml",
    "replica_perception": "benchmark/multi_agent/config_spot_drone_per.yaml",
    # All four robot types in one scene: the robot pool for fleet-change work.
    # Its robot_configs file is not in the data release, so starts are random.
    "replica_pool4": "benchmark/multi_agent/config_multi_agent_man.yaml",
}

EXTRA_OVERRIDES = {
    "replica_pool4": [
        "habitat.dataset.randomize_agent_start=1",
        # debug sensor that needs arm IK; the drone has none and it crashes
        "~habitat.task.lab_sensors.arm_workspace_rgb_sensor",
    ],
}

# articulated_agent_type -> short robot-id prefix
ROBOT_PREFIX = {
    "FetchRobot": "fetch",
    "StretchRobot": "stretch",
    "SpotRobot": "spot",
    "DJIDrone": "drone",
}
