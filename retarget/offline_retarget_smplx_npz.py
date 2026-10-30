"""Offline GMR retarget of a per-frame SMPL-X skeleton (gmr_smplx_results.npz) to Noetix Bumi, with gap filling and
zero-phase smoothing:

1. gaps (shot cuts, title cards, close-ups -- frames given in --segments) are replaced in SMPL-X space: every joint
   rotation and the root are slerped, the translation lerped, between the last frame before and the first frame
   after the gap (smoothstep timing); leading/trailing gaps hold the nearest valid frame;
2. FastSMPLXAdapter (SMPL-X FK) -> GMR IK, --max_iter iterations per stage; the first frame is solved to convergence
   from a warm start (root at the scaled pelvis target, joints at a standing pose -- from qpos0 the IK can fall into a
   pitched-over "quadruped" minimum and stay there);
3. zero-phase Butterworth low-pass (--cutoff Hz) on root position, root quaternion and joints, then joint limits.

Output: CSV in the LAFAN1_Retargeting_Dataset layout (root pos, root quat xyzw, joints in MJCF order), 30 fps,
not grounded (next step: ground_video_retarget.py).

    python retarget/offline_retarget_smplx_npz.py --npz examples/langzi/gmr_smplx_results.npz \
        --segments examples/langzi/segments.json --n_frames 2240 --out motion_raw.csv
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# standing pose for the first-frame warm start (as bvh_to_robot_batch_headless.py)
WARM_POSE = {
    "noetix_bumi": {
        "l_leg_pitch_joint": -0.1495, "r_leg_pitch_joint": -0.1495, "l_knee_pitch_joint": 0.3215, "r_knee_pitch_joint": 0.3215,
        "l_ankle_pitch_joint": -0.172, "r_ankle_pitch_joint": -0.172,
        "l_arm_roll_joint": 0.3, "r_arm_roll_joint": -0.3, "l_elbow_pitch_joint": -0.3, "r_elbow_pitch_joint": -0.3,
    },
}  # fmt: skip


def warm_start(gmr, robot, frame):
    import copy

    gmr.update_targets(copy.deepcopy(frame))
    pos, quat = gmr.scaled_human_data[gmr.human_root_name]
    q = gmr.configuration.data.qpos.copy()
    q[:3], q[3:7] = pos, quat
    for name, v in WARM_POSE.get(robot, {}).items():
        q[gmr.model.jnt_qposadr[gmr.model.joint(name).id]] = v
    gmr.configuration.update(q)


def fill_gaps(rotvecs, trans, gaps, n):
    """rotvecs (N, J, 3) local rotations (root first), trans (N, 3); gaps [[a, b), ...] -> filled copies."""
    rv, tr = rotvecs.copy(), trans.copy()
    valid = np.ones(n, dtype=bool)
    for a, b in gaps:
        valid[a:b] = False
    for a, b in gaps:
        a, b = max(0, a), min(n, b)
        prev, nxt = a - 1, b
        if prev < 0 and nxt >= n:
            raise ValueError("no valid frame")
        if prev < 0 or nxt >= n:
            src = nxt if prev < 0 else prev
            rv[a:b], tr[a:b] = rv[src], tr[src]
            continue
        u = (np.arange(a, b) - prev) / float(nxt - prev)
        s = u * u * (3 - 2 * u)
        for j in range(rv.shape[1]):
            sl = Slerp([0.0, 1.0], R.from_rotvec(np.stack((rv[prev, j], rv[nxt, j]))))
            rv[a:b, j] = sl(s).as_rotvec()
        tr[a:b] = tr[prev] + s[:, None] * (tr[nxt] - tr[prev])
    return rv, tr, valid


def lowpass(x, fps, cutoff):
    b, a = butter(2, cutoff / (0.5 * fps))
    return filtfilt(b, a, x, axis=0, padlen=min(len(x) - 1, 30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--robot", default="noetix_bumi")
    ap.add_argument("--segments", default=None, help="json with 'gaps': [[start, end), ...] in video frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--body_models", default=os.path.join(ROOT, "assets", "body_models"),
                    help="folder holding smplx/SMPLX_NEUTRAL.npz (see assets/body_models/README.md)")
    ap.add_argument("--max_iter", type=int, default=20)
    ap.add_argument("--cutoff", type=float, default=6.0, help="Hz, zero-phase 2nd-order Butterworth (0 = off)")
    ap.add_argument("--n_frames", type=int, default=0, help="pad to this many frames by holding the last one "
                    "(the npz ends at the last frame with a person)")
    a = ap.parse_args()

    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.params import IK_CONFIG_DICT
    from smplx_adapter import FastSMPLXAdapter

    z = np.load(a.npz)
    assert str(z["coord_system"]) == "y-up", z["coord_system"]
    fps = float(z["mocap_frame_rate"])
    n = z["pose_body"].shape[0]
    rotvecs = np.concatenate((z["root_orient"][:, None], z["pose_body"].reshape(n, 21, 3)), 1).astype(np.float64)
    trans = z["trans"].astype(np.float64)
    if a.n_frames > n:
        pad = a.n_frames - n
        rotvecs = np.concatenate((rotvecs, np.repeat(rotvecs[-1:], pad, 0)))
        trans = np.concatenate((trans, np.repeat(trans[-1:], pad, 0)))
        n = a.n_frames
    betas = z["betas"].astype(np.float64)
    gaps = json.load(open(a.segments))["gaps"] if a.segments else []
    rotvecs, trans, valid = fill_gaps(rotvecs, trans, gaps, n)
    print(f"{n} frames @ {fps:g} fps, {int((~valid).sum())} gap frames filled", flush=True)

    adapter = FastSMPLXAdapter(a.body_models, coord_fix="yup_to_zup")
    gmr = GMR(src_human="smplx", tgt_robot=a.robot, actual_human_height=None, verbose=False)
    gmr.max_iter = a.max_iter
    # GMR's scale table assumes a 1.8 m human; the height estimate comes from the first shape coefficient
    assumed = float(json.load(open(IK_CONFIG_DICT["smplx"][a.robot]))["human_height_assumption"])
    height = 1.66 + 0.1 * float(betas[0])
    gmr.human_scale_table = {k: v * height / assumed for k, v in gmr.human_scale_table.items()}
    print(f"human height {height:.3f} m (betas[0] {betas[0]:+.3f})", flush=True)

    qpos = np.zeros((n, gmr.model.nq))
    for t in range(n):
        Rl = R.from_rotvec(rotvecs[t]).as_matrix()
        pos, quat = adapter(Rl[1:], Rl[0], trans[t], betas)
        frame = adapter.as_dict(pos, quat)
        if t == 0:
            warm_start(gmr, a.robot, frame)
        reps = 30 if t == 0 else 1
        for _ in range(reps):
            q, _ = gmr.retarget(frame)
        qpos[t] = q
        if t % 300 == 0:
            print(f"  ik frame {t}/{n}", flush=True)

    quat = qpos[:, 3:7].copy()  # wxyz
    for t in range(1, n):
        if np.dot(quat[t], quat[t - 1]) < 0:
            quat[t] = -quat[t]
    if a.cutoff > 0:
        qpos[:, :3] = lowpass(qpos[:, :3], fps, a.cutoff)
        quat = lowpass(quat, fps, a.cutoff)
        qpos[:, 7:] = lowpass(qpos[:, 7:], fps, a.cutoff)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    m = gmr.model
    lo = np.array([m.jnt_range[j][0] for j in range(1, m.njnt)])
    hi = np.array([m.jnt_range[j][1] for j in range(1, m.njnt)])
    n_out = int(((qpos[:, 7:] < lo - 1e-6) | (qpos[:, 7:] > hi + 1e-6)).sum())
    qpos[:, 7:] = np.clip(qpos[:, 7:], lo, hi)
    out = np.concatenate((qpos[:, :3], quat[:, [1, 2, 3, 0]], qpos[:, 7:]), 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savetxt(a.out, out, delimiter=",", fmt="%.6f")
    np.save(os.path.splitext(a.out)[0] + "_valid.npy", valid)
    print(f"wrote {a.out} ({n} x {out.shape[1]}), clamped {n_out} joint values back into range", flush=True)


if __name__ == "__main__":
    main()
