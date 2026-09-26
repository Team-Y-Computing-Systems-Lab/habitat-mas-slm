"""Static traits per robot type, published in each robot's capability resource.

Arm reach comes from forward kinematics on each robot's official URDF
(SimHost.arm_reach); feasibility per object from simulated trials.
"""

ROBOT_TRAITS = {
    "FetchRobot": {"mobility": "wheeled", "can_climb_stairs": False, "arm": "7-DoF arm, suction/magic grasp"},
    "StretchRobot": {"mobility": "wheeled", "can_climb_stairs": False, "arm": "telescoping arm on a lift"},
    "SpotRobot": {"mobility": "legged", "can_climb_stairs": True, "arm": "6-DoF arm on the body"},
}
