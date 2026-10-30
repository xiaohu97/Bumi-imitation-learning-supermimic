"""Noetix Bumi 21-DoF constants for SuperMimic (no Isaac Lab imports here).

Plant = the Bumi Mimic task of noetix_bumi_lab (``Mimic-Bumi-v0``, BUMI_21DOF_CFG, commit 2454535c): URDF
``bumi.urdf``, implicit PD with their gains / effort / velocity limits, command delay 0-4 physics steps,
zero-point noise +-0.02 on every joint, per-joint action scale 0.25 * effort / stiffness.  Setpoints are
clipped to the URDF limits (no controller limits are published).
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DESC_DIR = os.path.join(REPO, "assets/noetix_bumi")
URDF_PATH = os.path.join(DESC_DIR, "urdf/bumi.urdf")

CONTROL_DT = 0.02
SIM_DT = 0.005
DECIMATION = 4
ROOT_BODY = "base_link"
TORSO_BODY = "waist_yaw_link"  # torso COM randomization (the Bumi Mimic task's anchor body)

# Command delay in physics steps (BUMI_21DOF_CFG DelayedImplicitActuatorCfg min/max_delay).
MIN_DELAY, MAX_DELAY = 0, 4

# The 14 bodies of the Bumi Mimic task (bumi_flat_env_cfg.py commands.motion.body_names).
TRACKED_BODIES = [
    "base_link", "l_leg_pitch_link", "l_leg_yaw_link", "l_knee_pitch_link", "l_ankle_roll_link",
    "r_leg_pitch_link", "r_leg_yaw_link", "r_knee_pitch_link", "r_ankle_roll_link", "waist_yaw_link",
    "l_arm_roll_link", "l_elbow_pitch_link", "r_arm_roll_link", "r_elbow_pitch_link",
]  # fmt: skip
FEET_BODIES = ["l_ankle_roll_link", "r_ankle_roll_link"]

# BUMI_21DOF_CFG.init_state; unlisted joints are 0.
DEFAULT_JOINT_POS = {
    "l_leg_pitch_joint": -0.1495, "l_knee_pitch_joint": 0.3215, "l_ankle_pitch_joint": -0.1720,
    "r_leg_pitch_joint": -0.1495, "r_knee_pitch_joint": 0.3215, "r_ankle_pitch_joint": -0.1720,
    "l_arm_roll_joint": 0.3, "r_arm_roll_joint": -0.3,
}  # fmt: skip

# BUMI_21DOF_CFG actuators: (effort_limit_sim, velocity_limit_sim, stiffness, damping); armature only on the ankles.
ACTUATORS = {
    "legs": {
        ".*_leg_yaw_joint": (27.0, 9.0, 60.0, 2.5), ".*_leg_roll_joint": (27.0, 12.0, 60.0, 3.0),
        ".*_leg_pitch_joint": (60.0, 12.0, 60.0, 3.0), ".*_knee_pitch_joint": (60.0, 12.0, 45.0, 2.0),
    },
    "waist": {"waist_yaw_joint": (27.0, 9.0, 53.0, 3.4)},
    "feet": {".*_ankle_pitch_joint": (13.0, 12.0, 10.0, 0.5), ".*_ankle_roll_joint": (13.0, 12.0, 10.0, 0.5)},
    "arms": {
        ".*_arm_pitch_joint": (5.5, 50.0, 12.0, 0.4), ".*_arm_roll_joint": (5.5, 50.0, 12.0, 0.4),
        ".*_arm_yaw_joint": (5.5, 50.0, 12.0, 0.4), ".*_elbow_pitch_joint": (5.5, 50.0, 12.0, 0.4),
    },
}  # fmt: skip
ANKLE_ARMATURE = 0.01

# Zero-point (encoder offset) noise half-width (Mimic-Bumi-v0 randomize_joint_default_pos +-0.02, all joints).
JOINT_OFFSET_RANGE = 0.02

# noetix_bumi_lab BUMI_21DOF_NAMES (Isaac Sim 5.0 articulation order) -- checked against the URDF below.
ISAAC_JOINT_NAMES = [
    "l_leg_pitch_joint", "r_leg_pitch_joint", "waist_yaw_joint", "l_leg_roll_joint", "r_leg_roll_joint",
    "l_arm_pitch_joint", "r_arm_pitch_joint", "l_leg_yaw_joint", "r_leg_yaw_joint", "l_arm_roll_joint",
    "r_arm_roll_joint", "l_knee_pitch_joint", "r_knee_pitch_joint", "l_arm_yaw_joint", "r_arm_yaw_joint",
    "l_ankle_pitch_joint", "r_ankle_pitch_joint", "l_elbow_pitch_joint", "r_elbow_pitch_joint",
    "l_ankle_roll_joint", "r_ankle_roll_joint",
]  # fmt: skip


def _gains_of(name: str) -> tuple[float, float, float, float]:
    import re

    for group in ACTUATORS.values():
        for pat, v in group.items():
            if re.fullmatch(pat, name):
                return v
    raise KeyError(name)


class URDFTree:
    """Joint/link tree of the URDF, with the Isaac Sim 5.0 articulation ordering.

    Isaac orders links breadth-first with siblings sorted by name (BUMI_21DOF_NAMES: the legs come before
    the waist although the URDF lists the waist first).  Depth-first in document order = GMR MJCF order.
    """

    def __init__(self, path: str = URDF_PATH):
        root = ET.parse(path).getroot()
        self.links = [l.get("name") for l in root.findall("link")]
        self.joints = {}
        children = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            ax = j.find("axis")
            lim = j.find("limit")
            info = {
                "name": j.get("name"),
                "type": j.get("type"),
                "parent": j.find("parent").get("link"),
                "child": j.find("child").get("link"),
                "xyz": [float(x) for x in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()],
                "rpy": [float(x) for x in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()],
                "axis": [float(x) for x in (ax.get("xyz") if ax is not None else "1 0 0").split()],
                "limit": (float(lim.get("lower")), float(lim.get("upper"))) if lim is not None else None,
            }
            self.joints[info["child"]] = info
            children.setdefault(info["parent"], []).append(info["child"])
        self.root = [l for l in self.links if l not in self.joints][0]
        bfs, queue = [], [self.root]
        while queue:
            l = queue.pop(0)
            bfs.append(l)
            queue.extend(sorted(children.get(l, [])))
        dfs = []

        def _dfs(l):
            dfs.append(l)
            for c in children.get(l, []):
                _dfs(c)

        _dfs(self.root)
        self.links_bfs, self.links_dfs = bfs, dfs
        self.joint_names_bfs = [self.joints[l]["name"] for l in bfs[1:]]
        self.joint_names_dfs = [self.joints[l]["name"] for l in dfs[1:]]


TREE = URDFTree()
assert TREE.joint_names_bfs == ISAAC_JOINT_NAMES, TREE.joint_names_bfs
# Canonical joint order everywhere in this package = Isaac articulation order (the env maps by name anyway).
JOINT_NAMES = TREE.joint_names_bfs
BODY_NAMES = TREE.links_bfs
NUM_J = len(JOINT_NAMES)
# per-joint action scale 0.25 * effort / stiffness (BUMI_21DOF_ACTION_SCALE)
ACTION_SCALE = [0.25 * _gains_of(n)[0] / _gains_of(n)[2] for n in JOINT_NAMES]


def default_joint_pos_list() -> list[float]:
    return [DEFAULT_JOINT_POS.get(n, 0.0) for n in JOINT_NAMES]


def deploy_limits_lists() -> tuple[list[float], list[float]]:
    lim = {j["name"]: j["limit"] for j in TREE.joints.values()}
    return [lim[n][0] for n in JOINT_NAMES], [lim[n][1] for n in JOINT_NAMES]


def joint_offset_ranges() -> list[float]:
    return [JOINT_OFFSET_RANGE] * NUM_J
