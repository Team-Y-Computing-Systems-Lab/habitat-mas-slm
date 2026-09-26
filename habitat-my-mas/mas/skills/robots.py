"""Static traits per robot type, published in each robot's capability resource.

Arm workspace and reach (the `can-reach` facts) will come from URDF + forward
kinematics (design.md §4b, mas/capability/); until then they are "unknown".
"""

ROBOT_TRAITS = {
    "FetchRobot": {"mobility": "wheeled", "can_climb_stairs": False, "arm": "7-DoF arm, suction/magic grasp"},
    "StretchRobot": {"mobility": "wheeled", "can_climb_stairs": False, "arm": "telescoping arm on a lift"},
    "SpotRobot": {"mobility": "legged", "can_climb_stairs": True, "arm": "6-DoF arm on the body"},
    "DJIDrone": {"mobility": "flying", "can_climb_stairs": True, "arm": None},
}
