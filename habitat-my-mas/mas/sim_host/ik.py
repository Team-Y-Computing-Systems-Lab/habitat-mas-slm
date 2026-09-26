"""Arm inverse kinematics on a robot's official URDF (pybullet), our own controller.

Official habitat-lab ships an arm-only IK model only for Fetch, and its
`ArmEEAction` drives motors, which do nothing in kinematic mode. So pick/place
use this instead: the robot's full official URDF is loaded into a headless
pybullet world with its base fixed; joints are matched to habitat by name;
IK moves only the arm joints (head and wheels stay at their current values).

Frames: habitat reports each link at its centre of mass, in the robot's root
frame (`sim_obj.transformation`), with y up. `calibrate()` fits the rigid
transform from pybullet's frame to habitat's from all links' centres of mass
(exact up to float error), so `fk()` / `ik()` work directly in habitat's
robot-local frame.
"""

from typing import Dict, Sequence

import numpy as np
import pybullet as p


def _kabsch(a: np.ndarray, b: np.ndarray):
    """R, t minimizing |R a + t - b|."""
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, cb - r @ ca


class UrdfArmIK:
    def __init__(self, urdf_path: str, arm_joint_names: Sequence[str], ee_link_name: str):
        self.cid = p.connect(p.DIRECT)
        self.body = p.loadURDF(urdf_path, useFixedBase=True, physicsClientId=self.cid)
        self.joint_index: Dict[str, int] = {}
        self.link_index: Dict[str, int] = {}
        self.movable, self.lower, self.upper = [], {}, {}
        for j in range(p.getNumJoints(self.body, physicsClientId=self.cid)):
            info = p.getJointInfo(self.body, j, physicsClientId=self.cid)
            name, jtype, lo, hi = info[1].decode(), info[2], info[8], info[9]
            self.joint_index[name] = j
            self.link_index[info[12].decode()] = j
            if jtype != p.JOINT_FIXED:
                self.movable.append(j)
                if hi <= lo:  # continuous joint
                    lo, hi = -np.pi, np.pi
                self.lower[j], self.upper[j] = lo, hi
        missing = [a for a in arm_joint_names if a not in self.joint_index]
        if missing:
            raise ValueError(f"arm joints not in {urdf_path}: {missing}")
        self.arm = [self.joint_index[a] for a in arm_joint_names]
        # a robot's arm list can include head joints (Stretch); IK never moves those
        self.solve = {j for j, a in zip(self.arm, arm_joint_names) if "head" not in a}
        self.ee = self.link_index[ee_link_name]
        self.r, self.t = np.eye(3), np.zeros(3)  # pybullet world -> habitat robot-local

    # -- state ---------------------------------------------------------------
    def set_joints(self, positions: Dict[str, float]) -> None:
        for name, value in positions.items():
            j = self.joint_index.get(name)
            if j is not None and j in self.lower:
                p.resetJointState(self.body, j, float(value), physicsClientId=self.cid)

    def arm_q(self) -> np.ndarray:
        return np.array([p.getJointState(self.body, j, physicsClientId=self.cid)[0] for j in self.arm])

    def set_arm(self, q: Sequence[float]) -> None:
        for j, v in zip(self.arm, q):
            p.resetJointState(self.body, j, float(v), physicsClientId=self.cid)

    def limits(self):
        return (np.array([self.lower[j] for j in self.arm]), np.array([self.upper[j] for j in self.arm]))

    def _com(self, j: int) -> np.ndarray:
        return np.array(p.getLinkState(self.body, j, computeForwardKinematics=1, physicsClientId=self.cid)[0])

    def calibrate(self, habitat_local_com: Dict[str, Sequence[float]]) -> float:
        """Fit pybullet -> habitat frame from link centres of mass (same joint state
        on both sides). Returns the worst residual in metres."""
        names = [n for n in habitat_local_com if n in self.link_index]
        a = np.array([self._com(self.link_index[n]) for n in names])
        b = np.array([habitat_local_com[n] for n in names], dtype=float)
        self.r, self.t = _kabsch(a, b)
        return float(np.linalg.norm(a @ self.r.T + self.t - b, axis=1).max())

    # -- kinematics (habitat robot-local frame, end effector centre of mass) --
    def fk(self, q: Sequence[float] = None) -> np.ndarray:
        if q is not None:
            self.set_arm(q)
        return self.r @ self._com(self.ee) + self.t

    def ik(self, target: Sequence[float], iterations: int = 30, damping: float = 0.05,
           max_joint_step: float = 0.2) -> np.ndarray:
        """Arm joint values that bring the end effector (centre of mass) toward
        `target` (habitat robot-local), by damped least squares on the arm
        Jacobian. Only arm joints move; they stay within their limits. The
        model's joint state is restored afterwards."""
        target = np.asarray(target, float)
        lo, hi = self.limits()
        q0 = self.arm_q()
        q = q0.copy()
        cols = [k for k, j in enumerate(self.arm) if j in self.solve]
        eps = 1e-4
        for _ in range(iterations):
            cur = self.fk(q)
            err = target - cur
            if np.linalg.norm(err) < 1e-3:
                break
            # numerical Jacobian in habitat's frame (pybullet's analytic one uses
            # another frame convention)
            jac = np.zeros((3, len(cols)))
            for c, k in enumerate(cols):
                dq_k = q.copy()
                dq_k[k] += eps
                jac[:, c] = (self.fk(dq_k) - cur) / eps
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damping ** 2 * np.eye(3), err)
            dq = np.clip(dq, -max_joint_step, max_joint_step)
            q[cols] = np.clip(q[cols] + dq, lo[cols], hi[cols])
        self.set_arm(q0)
        return q

    def sample_reach(self, n: int = 2000, seed: int = 0) -> np.ndarray:
        """End-effector positions (habitat robot-local) over random arm configurations."""
        rng = np.random.default_rng(seed)
        lo, hi = self.limits()
        q0 = self.arm_q()
        pts = []
        for _ in range(n):
            q = q0.copy()
            for k, j in enumerate(self.arm):
                if j in self.solve:
                    q[k] = lo[k] + rng.random() * (hi[k] - lo[k])
            pts.append(self.fk(q))
        self.set_arm(q0)
        return np.array(pts)
