"""Roll out a SuperMimic Bumi checkpoint on new reference clips (e.g. retargeted videos) and record it for rendering.

Same plant and policy settings as evaluate.py (friction 1.0, no randomization, no observation/action noise, command
delay 2 physics steps, deterministic policy).  Unlike evaluate.py, a terminated clip does not stop: the rollout keeps
running for --fall_s seconds (so the failure is visible), then the robot is reset to the reference state of the
current frame and tracking resumes.  Each termination is printed and stored.

    python supermimic/rollout_clip.py --headless --ckpt logs/<run>/ckpt_040000.pt \
        --clips <clip.csv> [...] --out traj.npz

Clips: GMR noetix_bumi CSV at 30 fps (root pos, root quat xyzw, joints in MJCF order), grounded with
retarget/ground_video_retarget.py.  The npz has evaluate.py's --record layout plus ``tracking`` (False while falling after a
termination), ``reset`` (step at which the robot was put back on the reference) and per-clip ``terminations``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--clips", type=str, nargs="+", required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--fall_s", type=float, default=0.6, help="seconds shown after a termination before the reset")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import torch

import isaaclab.sim as sim_utils

import supermimic_bumi.networks as networks
from supermimic_bumi.env import TrackingSim, make_sim_cfg
from supermimic_bumi.kinematics import Kinematics
from supermimic_bumi.mdp import MAX_STEP_REWARD, MDP, OBS_DIM
from supermimic_bumi.motion import GOAL_FRAME_DIM, MotionLib
from supermimic_bumi.networks import Policy
from supermimic_bumi.robot import CONTROL_DT, JOINT_NAMES, default_joint_pos_list
from supermimic_bumi.rotations import mat_to_quat

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cpu" if args.device == "cpu" else "cuda:0"
kin = Kinematics(dev)
motion = MotionLib(args.clips, kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
C = len(motion.names)


@torch.no_grad()
def main():
    ck = torch.load(args.ckpt, map_location=dev)
    a_ = ck["args"]
    K, Hh = a_["goal_frames"], a_["hist"]
    networks.AMP_DTYPE = torch.bfloat16 if a_.get("amp", "none") == "bf16" else None
    pi = Policy(OBS_DIM, GOAL_FRAME_DIM * K, hist=Hh).to(dev)
    pi.load_state_dict(ck["pi"])
    pi.eval()

    sim = sim_utils.SimulationContext(make_sim_cfg(dev))
    env = TrackingSim(sim, C, motion, mdp, eval_mode=True)
    ids = torch.arange(C, device=dev)
    env.reset(ids, motion.clip_start.clone())
    clip_end = motion.clip_end.clone()
    fall_steps = int(round(args.fall_s / CONTROL_DT))
    since_term = torch.full((C,), -1, dtype=torch.long, device=dev)  # -1: tracking
    finished = torch.zeros(C, dtype=torch.bool, device=dev)
    fresh = torch.ones(C, dtype=torch.bool, device=dev)  # refill the observation history
    ret = torch.zeros(C, device=dev)
    n_track = torch.zeros(C, device=dev)
    e_pos_sum = torch.zeros(C, device=dev)
    hist = None
    terms_log = [[] for _ in range(C)]
    traj = {k: [] for k in ("root_p", "root_quat", "q", "f", "alive", "tracking", "reset")}
    step = 0
    while not bool(finished.all()):
        S = env.state()
        o = mdp.obs(S, env.a_prev, env.xi, env.f)
        if hist is None:
            hist = o.unsqueeze(1).repeat(1, Hh, 1)
        else:
            hist = torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
            hist[fresh] = o[fresh].unsqueeze(1).repeat(1, Hh, 1)
        reset_now = fresh.clone()
        fresh[:] = False
        a = pi(hist, motion.goal(env.f, K))
        tracking = since_term < 0
        traj["root_p"].append(S[0].cpu()), traj["root_quat"].append(mat_to_quat(S[1]).cpu())
        traj["q"].append(S[4].cpu()), traj["f"].append(env.f.cpu()), traj["alive"].append((~finished).cpu())
        traj["tracking"].append(tracking.cpu()), traj["reset"].append(reset_now.cpu())
        f_before = env.f.clone()
        out = env.step(a)
        step += 1
        live = ~finished & tracking
        ret += torch.where(live, out["r"], 0.0)
        n_track += live.float()
        e_pos_sum += torch.where(live, out["terms"]["e_body_pos"], 0.0)
        new_term = out["term"] & live
        for c in new_term.nonzero().flatten().tolist():
            fr = int(f_before[c] - motion.clip_start[c])
            p, R = S[0][c], S[1][c]
            rp, rR = motion.root_p[f_before[c]], motion.root_R[f_before[c]]
            info = {"t": round(fr * CONTROL_DT, 2), "e_z": round(float((rp[2] - p[2]).abs()), 3),
                    "e_tilt": round(float((rR[2, 2] - R[2, 2]).abs()), 3)}  # fmt: skip
            terms_log[c].append(info)
            print(f"[rollout] {motion.names[c]}: terminated at {info['t']:.2f} s (e_z {info['e_z']}, e_tilt {info['e_tilt']})", flush=True)
        since_term[new_term] = 0
        since_term[since_term >= 0] += 1
        finished |= env.f >= clip_end
        back = (since_term > fall_steps) & ~finished
        if bool(back.any()):
            bids = back.nonzero().flatten()
            env.reset(bids, env.f[bids].clone())
            since_term[bids] = -1
            fresh[bids] = True
        env.f = torch.minimum(env.f, clip_end - 1)
    L = (motion.clip_len - 1).float()
    res = {}
    for c in range(C):
        res[motion.names[c]] = {
            "seconds": round(float(motion.clip_len[c]) * CONTROL_DT, 2),
            "norm_return": round(float(ret[c] / (L[c] * MAX_STEP_REWARD)), 4),
            "tracked_fraction": round(float(n_track[c] / L[c]), 4),
            "mean_body_pos_err_m": round(float((e_pos_sum[c] / n_track[c].clamp(min=1)).sqrt()), 4),
            "terminations": terms_log[c],
        }
    print(json.dumps(res, indent=1))
    np.savez_compressed(args.out, names=np.array(motion.names), **{k: torch.stack(v).numpy() for k, v in traj.items()},
                        ref_root_p=motion.root_p.cpu().numpy(), ref_root_R=motion.root_R.cpu().numpy(), ref_q=motion.q.cpu().numpy(),
                        clip_start=motion.clip_start.cpu().numpy(), clip_len=motion.clip_len.cpu().numpy(),
                        summary=np.array(json.dumps(res)), ckpt=np.array(os.path.abspath(args.ckpt)),
                        joint_names=np.array(JOINT_NAMES))  # fmt: skip
    json.dump(res, open(os.path.splitext(args.out)[0] + "_summary.json", "w"), indent=1)
    print("wrote", args.out, flush=True)


main()
os._exit(0)  # app.close() hangs after headless runs
