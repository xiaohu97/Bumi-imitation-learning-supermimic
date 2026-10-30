"""Derive Noetix Bumi's SMPL-X IK config (general_motion_retargeting/ik_configs/smplx_to_noetix_bumi.json) from smplx_to_g1.json.

GMR sets each robot link's target rotation to R_human * O (O = the config's rotation offset).  For a human pose
that puts G1 at q = 0, R_human = R_G1(q=0) O_G1^T, so the offset that puts the robot in the equivalent pose q_eq is

    O_robot = O_G1 * R_G1(q=0)^T * R_robot(q_eq)

q_eq = the robot pose that looks like G1's zero pose (G1: arms down, forearms pointing forward).

Scales: GMR places every target at  s_root * p_root + s_i * (p_i - p_root).  Root = G1's 0.9 x straight-leg root height
ratio; leg entries = G1's scale x the ratio of the link's distance to the root link (Bumi base_link sits 4 cm below its
hip joints, G1's pelvis at them); arms = G1's 0.8 x arm length ratio (distance-to-root ratios are meaningless for links
near the pelvis height, e.g. G1's elbow).

The result is already in the repo; rerun only to change the derivation.  G1's model is not in the repo, take it from GMR
(https://github.com/YanjieZe/GMR, assets/unitree_g1):

    python retarget/derive_smplx_ik_from_g1.py --g1_xml <gmr>/assets/unitree_g1/g1_mocap_29dof.xml
"""

import argparse
import copy
import json
import os

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IK = os.path.join(ROOT, "general_motion_retargeting/ik_configs")
ROBOT_XML = os.path.join(ROOT, "assets/noetix_bumi/noetix_bumi.xml")
G1_ROOT_H = 0.792  # straight-leg pelvis height above the sole
G1_ARM = ["left_shoulder_pitch_link", "left_elbow_link", "left_wrist_yaw_link"]
ROOT_H = 0.480  # Bumi
ARM = ["l_arm_pitch_link", "l_elbow_pitch_link", "l_hand_link"]
Q_EQ = {"l_elbow_pitch_joint": -np.pi / 2, "r_elbow_pitch_joint": -np.pi / 2}
LINKS = {  # G1 link -> robot link
    "pelvis": "base_link",
    "left_hip_roll_link": "l_leg_roll_link", "right_hip_roll_link": "r_leg_roll_link",
    "left_knee_link": "l_knee_pitch_link", "right_knee_link": "r_knee_pitch_link",
    "left_toe_link": "l_toe_link", "right_toe_link": "r_toe_link",
    "torso_link": "waist_yaw_link",
    "left_shoulder_yaw_link": "l_arm_yaw_link", "right_shoulder_yaw_link": "r_arm_yaw_link",
    "left_elbow_link": "l_elbow_pitch_link", "right_elbow_link": "r_elbow_pitch_link",
    "left_wrist_yaw_link": "l_hand_link", "right_wrist_yaw_link": "r_hand_link",
}  # fmt: skip


def pose(xml, q_named):
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    d.qpos[:] = 0.0
    d.qpos[3] = 1.0
    for n, v in q_named.items():
        d.qpos[m.joint(n).qposadr[0]] = v
    mujoco.mj_kinematics(m, d)
    return m, d


def chain_length(m, d, chain):
    p = [d.xpos[m.body(b).id] for b in chain]
    return sum(np.linalg.norm(p[i + 1] - p[i]) for i in range(len(p) - 1))


def offsets(cfg, mg, dg, mr, dr):
    out = {}
    for lg, lr in LINKS.items():
        Rg = dg.xmat[mg.body(lg).id].reshape(3, 3)
        Rr = dr.xmat[mr.body(lr).id].reshape(3, 3)
        O_g1 = R.from_quat(np.asarray(cfg["ik_match_table1"][lg][4])[[1, 2, 3, 0]])
        out[lg] = (O_g1 * R.from_matrix(Rg.T @ Rr)).as_quat()[[3, 0, 1, 2]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g1_xml", required=True, help="GMR's assets/unitree_g1/g1_mocap_29dof.xml")
    a = ap.parse_args()
    mg, dg = pose(a.g1_xml, {})
    mr, dr = pose(ROBOT_XML, Q_EQ)
    root_link = LINKS["pelvis"]
    root_scale = 0.9 * ROOT_H / G1_ROOT_H
    arm_ratio = chain_length(mr, dr, ARM) / chain_length(mg, dg, G1_ARM)

    g1 = json.load(open(os.path.join(IK, "smplx_to_g1.json")))
    rot = offsets(g1, mg, dg, mr, dr)
    pg0, pr0 = dg.xpos[mg.body("pelvis").id], dr.xpos[mr.body(root_link).id]
    scale = {}
    for lg, lr in LINKS.items():
        human = g1["ik_match_table1"][lg][0]
        if lg in ("pelvis", "torso_link"):
            scale[human] = root_scale  # torso: position weight 0 in both tables
        elif "shoulder" in lg or "elbow" in lg or "wrist" in lg:
            scale[human] = g1["human_scale_table"][human] * arm_ratio
        else:
            dgl = np.linalg.norm(dg.xpos[mg.body(lg).id] - pg0)
            drl = np.linalg.norm(dr.xpos[mr.body(lr).id] - pr0)
            scale[human] = g1["human_scale_table"][human] * drl / dgl
            print(f"{lg:24s} -> {lr:22s} d_G1 {dgl:.3f} d_robot {drl:.3f} scale {scale[human]:.4f}")

    out = copy.deepcopy(g1)
    out["robot_root_name"] = root_link
    out["human_scale_table"] = {k: round(float(scale[k]), 4) for k in g1["human_scale_table"]}
    for table in ("ik_match_table1", "ik_match_table2"):
        new = {}
        for lg, entry in g1[table].items():
            e = list(entry)
            e[3] = [0.0, 0.0, 0.0]  # G1's +-2 cm toe shift is G1-specific
            e[4] = [round(float(x), 6) for x in rot[lg]]
            new[LINKS[lg]] = e
        out[table] = new
    path = os.path.join(IK, "smplx_to_noetix_bumi.json")
    json.dump(out, open(path, "w"), indent=4)
    print("root scale", round(root_scale, 4), "arm length ratio", round(arm_ratio, 4))
    print("wrote", path)


if __name__ == "__main__":
    main()
