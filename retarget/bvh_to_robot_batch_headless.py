"""Headless batch BVH -> robot retargeting (no viewer), one process per file.

Writes, per BVH, ``<out_dir>/<name>.csv`` in the LAFAN1_Retargeting_Dataset layout
(one row per source frame: root pos xyz [m], root quat xyzw, joint positions [rad] in MJCF order)
and ``<name>.pkl`` in the GMR format of ``bvh_to_robot.py``.

    python retarget/bvh_to_robot_batch_headless.py --bvh_dir <lafan1 dir> --out_dir <dir> --workers 16
"""

import argparse
import glob
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# First-frame warm start (root at the scaled Hips target, joints at a standing pose).  GMR's IK is
# incremental with a few iterations per frame; started from qpos0 far from the first human frame,
# it can converge to a pitched-over "quadruped" local minimum and never leave it.
WARM_POSE = {
    # noetix_bumi_lab BUMI_21DOF_CFG init_state, elbows slightly bent
    "noetix_bumi": {
        "l_leg_pitch_joint": -0.1495, "r_leg_pitch_joint": -0.1495, "l_knee_pitch_joint": 0.3215, "r_knee_pitch_joint": 0.3215,
        "l_ankle_pitch_joint": -0.172, "r_ankle_pitch_joint": -0.172,
        "l_arm_roll_joint": 0.3, "r_arm_roll_joint": -0.3, "l_elbow_pitch_joint": -0.3, "r_elbow_pitch_joint": -0.3,
    },
}


def warm_start(rt, robot, frame):
    import copy

    rt.update_targets(copy.deepcopy(frame))
    pos, quat = rt.scaled_human_data[rt.human_root_name]
    q = rt.configuration.data.qpos.copy()
    q[:3], q[3:7] = pos, quat
    for name, v in WARM_POSE.get(robot, {}).items():
        q[rt.model.jnt_qposadr[rt.model.joint(name).id]] = v
    rt.configuration.update(q)


def _one(job):
    bvh_file, robot, fmt, out_dir = job
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    name = os.path.splitext(os.path.basename(bvh_file))[0]
    csv_path = os.path.join(out_dir, name + ".csv")
    if os.path.exists(csv_path):
        return name, 0, 0.0, "skip"
    t0 = time.time()
    frames, h = load_bvh_file(bvh_file, format=fmt)
    rt = GMR(src_human=f"bvh_{fmt}", tgt_robot=robot, actual_human_height=h, verbose=False)
    warm_start(rt, robot, frames[0])
    q = np.array([rt.retarget(f)[0] for f in frames])  # (T, 7 + nj), root quat wxyz
    root_pos, root_wxyz, dof = q[:, :3], q[:, 3:7], q[:, 7:]
    root_xyzw = root_wxyz[:, [1, 2, 3, 0]]
    np.savetxt(csv_path + ".tmp", np.concatenate((root_pos, root_xyzw, dof), 1), delimiter=",", fmt="%.6f")
    os.replace(csv_path + ".tmp", csv_path)
    with open(os.path.join(out_dir, name + ".pkl"), "wb") as f:
        pickle.dump({"fps": 30, "root_pos": root_pos, "root_rot": root_xyzw, "dof_pos": dof,
                     "local_body_pos": None, "link_body_list": None}, f)
    return name, len(q), time.time() - t0, "ok"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvh_dir", required=True)
    ap.add_argument("--robot", default="noetix_bumi")
    ap.add_argument("--format", default="lafan1")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None, help="subset of file stems")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.bvh_dir, "*.bvh")))
    if a.only:
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in set(a.only)]
    jobs = [(f, a.robot, a.format, a.out_dir) for f in files]
    with Pool(a.workers) as p:
        for name, n, dt, st in p.imap_unordered(_one, jobs):
            print(f"{st:4s} {name:32s} {n:6d} frames {dt:7.1f} s", flush=True)
