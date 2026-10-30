"""Isaac-side checks of the Bumi plant (run before the first training; ~2-3 GB VRAM).

1. articulation as configured: joint order, PD gains / effort / velocity limits / armature in PhysX, mass;
2. FK of the package (URDF parse) vs PhysX body poses after stepping reference states (all 22 bodies);
3. (informational) hold the default pose from the reference start of a walking clip for 2 s.

    python supermimic/check_sim.py --headless
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--motions", default=os.path.join(HERE, "motions", "langzi.txt"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

import isaaclab.sim as sim_utils

from supermimic_bumi.env import TrackingSim, make_sim_cfg
from supermimic_bumi.kinematics import Kinematics
from supermimic_bumi.mdp import MDP
from supermimic_bumi.motion import MotionLib
from supermimic_bumi.robot import ACTUATORS, BODY_NAMES, JOINT_NAMES, default_joint_pos_list

dev = args.device  # --device cpu runs PhysX on the CPU (small VRAM footprint when the GPU is busy)
kin = Kinematics(dev)
motion = MotionLib.from_list(args.motions, kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
C = len(motion.names)
sim = sim_utils.SimulationContext(make_sim_cfg(dev))
env = TrackingSim(sim, C, motion, mdp, eval_mode=True)
rob = env.robot
ok = True

# 1. configuration as applied
print("[1] Isaac joint order:", list(rob.joint_names), "| mapped" if env.jids is not None else "| = JOINT_NAMES")
print(f"    body order = BODY_NAMES: {list(rob.body_names) == BODY_NAMES}; total mass {float(rob.root_physx_view.get_masses()[0].sum()):.3f} kg")
d = rob.data
for n in JOINT_NAMES:
    i = rob.joint_names.index(n)
    print(f"    {n:22s} kp {float(d.joint_stiffness[0, i]):6.1f} kd {float(d.joint_damping[0, i]):5.2f} "
          f"effort {float(d.joint_effort_limits[0, i]):5.1f} vel {float(d.joint_vel_limits[0, i]):5.1f} "
          f"armature {float(d.joint_armature[0, i]):.3f} limits [{float(d.joint_pos_limits[0, i, 0]):+.2f}, {float(d.joint_pos_limits[0, i, 1]):+.2f}]")
import re  # noqa: E402

for group in ACTUATORS.values():
    for pat, (eff, vel, kp, kd) in group.items():
        for n in JOINT_NAMES:
            if re.fullmatch(pat, n):
                i = rob.joint_names.index(n)
                got = [float(x[0, i]) for x in (d.joint_effort_limits, d.joint_vel_limits, d.joint_stiffness, d.joint_damping)]
                if max(abs(a - b) for a, b in zip(got, (eff, vel, kp, kd))) > 1e-3:
                    print(f"    MISMATCH {n}: {got} vs {(eff, vel, kp, kd)}")
                    ok = False

# 2. FK vs PhysX
kin_all = Kinematics(dev, tracked=BODY_NAMES)
bids = [rob.body_names.index(b) for b in BODY_NAMES]
err = 0.0
for k in range(5):
    env.step(torch.zeros(C, len(JOINT_NAMES), device=dev))
    S = env.state()
    P = kin_all.forward(*S)[0]
    err = max(err, float((P - (d.body_link_pos_w[:, bids] - env.scene.env_origins[:, None])).norm(dim=-1).max()))
print(f"[2] FK vs PhysX body positions over 5 steps x {C} envs: max {err:.2e} m")
ok &= err < 1e-3

# 3. (informational) hold the default pose from the first clip's first frame for 2 s (delay 2, no noise)
c = 0
ids = torch.arange(C, device=dev)
env.reset(ids, motion.clip_start.clone())
for _ in range(100):
    out = env.step(torch.zeros(C, len(JOINT_NAMES), device=dev))
    env.f = torch.minimum(env.f, motion.clip_start)  # keep the reference frame fixed
S = env.state()
z, tilt = float(S[0][c, 2]), float(torch.rad2deg(torch.arccos(S[1][c, 2, 2].clamp(-1, 1))))
print(f"[3] after 2 s holding the default pose: base z {z:.3f} m (reference start {float(motion.root_p[motion.clip_start[c], 2]):.3f}), "
      f"tilt {tilt:.1f} deg  (informational: zero action on kp-10 ankles need not stay up)")
print("CHECK_SIM", "OK" if ok else "FAILED", flush=True)
os._exit(0 if ok else 1)
