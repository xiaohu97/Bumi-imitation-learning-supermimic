"""Offline checks of the Bumi port (CPU, no Isaac):

1. FK (URDF parse, Isaac joint order) vs the Isaac-recorded body poses of noetix_bumi_lab mimic_data/bumi/2.npz
   (all 22 bodies; this also checks the breadth-first / sorted-siblings body order) -- only with --npz <that file>;
2. FK vs MuJoCo (noetix_bumi.xml, CSV/MJCF joint order) on a retargeted clip -- needs ``mujoco``;
3. MotionLib + MDP dimensions on a motion list.

    python supermimic/check_offline.py [--npz <noetix_bumi_lab>/mimic_data/bumi/2.npz] [--motions supermimic/motions/bumi_lafan40.txt]
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from supermimic_bumi.kinematics import Kinematics  # noqa: E402
from supermimic_bumi.mdp import CRITIC_EXTRA, INV_DIM, MDP, OBS_DIM, STATE_DIM, XI_DIM  # noqa: E402
from supermimic_bumi.motion import GOAL_FRAME_DIM, MotionLib, load_clip  # noqa: E402
from supermimic_bumi.robot import BODY_NAMES, JOINT_NAMES, NUM_J, TREE, default_joint_pos_list  # noqa: E402
from supermimic_bumi.rotations import axis_angle_to_mat, quat_to_mat  # noqa: E402

ap = argparse.ArgumentParser()
ROOT = os.path.dirname(HERE)
ap.add_argument("--npz", default=None)
ap.add_argument("--csv", default=os.path.join(ROOT, "examples/langzi/langzi_bumi.csv"))
ap.add_argument("--xml", default=os.path.join(ROOT, "assets/noetix_bumi/noetix_bumi.xml"))
ap.add_argument("--motions", default=os.path.join(HERE, "motions", "langzi.txt"))
a = ap.parse_args()

# 1. FK vs Isaac NPZ.  The NPZ was recorded with an older leg geometry than bumi.urdf, so the joint origins are
#    taken from the NPZ itself (parent-relative, constant to 1e-7); joint order, axes and body order are what is checked.
if a.npz:
    d = np.load(a.npz)
    t = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32)  # noqa: E731
    kin_all = Kinematics("cpu", tracked=BODY_NAMES)
    bp, bq, bv, bw = t(d["body_pos_w"]), t(d["body_quat_w"]), t(d["body_lin_vel_w"]), t(d["body_ang_vel_w"])
    assert bp.shape[1] == len(BODY_NAMES), bp.shape
    Rw = quat_to_mat(bq)
    n_diff = 0
    for i, n in enumerate(BODY_NAMES[1:], 1):
        p = kin_all.parent[i]
        rel = (Rw[:, p].transpose(-1, -2) @ (bp[:, i] - bp[:, p]).unsqueeze(-1)).squeeze(-1)
        assert rel.std(0).max() < 1e-5, n
        # fixed joint rotation R_off = R_p^T R_c Rot(axis, q)^T must be constant if JOINT_NAMES[j] drives this body
        q_i = t(d["joint_pos"])[:, kin_all.joint_idx[i]]
        R_off = Rw[:, p].transpose(-1, -2) @ Rw[:, i] @ axis_angle_to_mat(kin_all.axes[i], q_i).transpose(-1, -2)
        assert (R_off - R_off[0]).abs().max() < 1e-4, f"{n}: joint {JOINT_NAMES[kin_all.joint_idx[i]]} does not drive it"
        dev_p, dev_R = (rel.mean(0) - kin_all.off_p[i]).norm(), (R_off[0] - kin_all.off_R[i]).abs().max()
        if dev_p > 1e-3 or dev_R > 1e-3:
            n_diff += 1
        kin_all.off_p[i], kin_all.off_R[i] = rel.mean(0), R_off[0]
    P, Rm, V, W = kin_all.forward(bp[:, 0], Rw[:, 0], bv[:, 0], bw[:, 0], t(d["joint_pos"]), t(d["joint_vel"]))
    e_p = (P - bp).norm(dim=-1)
    e_R = (Rm - Rw).abs().amax((-1, -2))
    e_w = (W - bw).norm(dim=-1)
    print(f"[1] FK vs Isaac NPZ ({bp.shape[0]} frames, {bp.shape[1]} bodies; joint frames taken from the NPZ, {n_diff} differ from "
          f"bumi.urdf): pos max {e_p.max():.2e} m, "
          f"rot max {e_R.max():.2e}, ang vel max {e_w.max():.2e} rad/s")
    assert e_p.max() < 1e-4 and e_R.max() < 1e-3, "joint order / axes / body order mismatch with Isaac"
else:
    print("[1] skipped (no --npz)")

# 2. FK vs MuJoCo on a retargeted clip
try:
    import mujoco
except ImportError:
    mujoco = None
if mujoco is not None:
    m = mujoco.MjModel.from_xml_path(a.xml)
    md = mujoco.MjData(m)
    mj_names = [m.joint(i).name for i in range(1, m.njnt)]
    assert mj_names == TREE.joint_names_dfs, mj_names
    _, rp, rq, q = load_clip(a.csv)
    kin = Kinematics("cpu", tracked=BODY_NAMES)
    idx = torch.arange(0, rp.shape[0], 97)
    P = kin.forward(rp[idx], quat_to_mat(rq[idx]), torch.zeros(len(idx), 3), torch.zeros(len(idx), 3), q[idx], torch.zeros_like(q[idx]))[0]
    x = np.loadtxt(a.csv, delimiter=",")
    err = 0.0
    for k, f in enumerate(idx.tolist()):
        md.qpos[:3], md.qpos[3:7], md.qpos[7:] = x[f, :3], x[f, [6, 3, 4, 5]], x[f, 7:]
        mujoco.mj_kinematics(m, md)
        mj = np.stack([md.xpos[m.body(b).id] for b in BODY_NAMES])
        err = max(err, float(np.abs(mj - P[k].numpy()).max()))
    print(f"[2] FK vs MuJoCo on {os.path.basename(a.csv)} ({len(idx)} frames): pos max {err:.2e} m")
    assert err < 1e-4
else:
    print("[2] skipped (no mujoco in this env)")

# 3. motion library + MDP dims
kin = Kinematics("cpu")
motion = MotionLib.from_list(a.motions, kin, "cpu")
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list()))
lo, hi = kin.q_lo, kin.q_hi
out = ((motion.q < lo - 1e-3) | (motion.q > hi + 1e-3)).float().mean(0)
print(f"[3] {len(motion.names)} clips, {motion.num_frames} frames = {motion.num_frames * 0.02 / 3600:.2f} h; NUM_J {NUM_J}, "
      f"STATE {STATE_DIM}, OBS {OBS_DIM}, XI {XI_DIM}, INV {INV_DIM}, CRITIC_EXTRA {CRITIC_EXTRA}, GOAL/frame {GOAL_FRAME_DIM}")
print("    ref joints outside URDF limits (frame frac):", {n: round(float(f), 3) for n, f in zip(JOINT_NAMES, out) if f > 0})
print(f"    ref root z p1/50/99: {np.percentile(motion.root_p[:, 2].numpy(), [1, 50, 99]).round(3)}, "
      f"max |qd| {motion.qd.abs().max():.1f} rad/s (p99.9 {np.percentile(motion.qd.abs().numpy(), 99.9):.1f})")
S = motion.ref_state(torch.arange(0, motion.num_frames, 1000))
r, terms = mdp.reward(S, torch.arange(0, motion.num_frames, 1000), torch.zeros(len(S[0]), NUM_J), torch.zeros(len(S[0]), NUM_J), terms=True)
print(f"    reward of the reference itself (tracking terms at max, minus limit/rate terms): {r.mean() / 0.1:.3f} of max, "
      f"e_body_pos {terms['e_body_pos'].max():.1e}")
print("OK")
