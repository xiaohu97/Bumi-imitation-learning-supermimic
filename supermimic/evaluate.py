"""Checkpoint evaluation (paper Appendix F.1) on every motion of the training library.

Adapted from supermimic_vibed (MIT, Chong Zhang) ``evaluate.py``.  Each motion plays once from its
first frame: friction 1.0, no randomization, no observation/action noise, command delay fixed at 2
physics steps (middle of the 0-4 training range), deterministic policy.  A motion fails if the
rollout triggers the training termination.  Normalized return = undiscounted reward sum / maximum.

    python supermimic/evaluate.py --headless --run_dir logs/<run> --watch
    python supermimic/evaluate.py --headless --run_dir logs/<run> --ckpt <ckpt.pt> [--record traj.npz]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", type=str, required=True)
parser.add_argument("--watch", action="store_true")
parser.add_argument("--ckpt", type=str, default=None, help="evaluate one checkpoint")
parser.add_argument("--record", type=str, default=None, help="save trajectories (npz) for rendering")
parser.add_argument("--motions", type=str, default=None, help="default: the run's motions.txt")
parser.add_argument("--shard", type=str, default="0/1", help="i/n: evaluate every n-th checkpoint starting at i (parallel evals)")
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
from supermimic_bumi.robot import default_joint_pos_list
from supermimic_bumi.rotations import mat_to_quat

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cpu" if args.device == "cpu" else "cuda:0"  # cpu: smoke tests
kin = Kinematics(dev)
motion = MotionLib.from_list(args.motions or os.path.join(args.run_dir, "motions.txt"), kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
C = len(motion.names)


@torch.no_grad()
def evaluate(env: TrackingSim, ck: dict, record: str | None = None) -> dict:
    a_ = ck["args"]
    K, Hh = a_["goal_frames"], a_["hist"]
    networks.AMP_DTYPE = torch.bfloat16 if a_.get("amp", "none") == "bf16" else None
    pi = Policy(OBS_DIM, GOAL_FRAME_DIM * K, hist=Hh).to(dev)
    pi.load_state_dict(ck["pi"])
    pi.eval()
    ids = torch.arange(C, device=dev)
    env.done_eval[:] = False
    env.failed_eval[:] = False
    env.reset(ids, motion.clip_start.clone())
    clip_end = motion.clip_end.clone()
    ret = torch.zeros(C, device=dev)
    hist = None
    traj = {"root_p": [], "root_quat": [], "q": [], "f": [], "alive": []}
    t0, steps, total = time.time(), 0, int(motion.clip_len.max())
    while not bool(env.done_eval.all()):
        steps += 1
        if steps % 1000 == 0:
            print(f"  [eval] step {steps}/{total}, {int(env.done_eval.sum())}/{C} clips done, {time.time() - t0:.0f}s", flush=True)
        S = env.state()
        o = mdp.obs(S, env.a_prev, env.xi, env.f)
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        a = pi(hist, motion.goal(env.f, K))
        alive = ~env.done_eval.clone()
        if record:
            traj["root_p"].append(S[0].cpu()), traj["root_quat"].append(mat_to_quat(S[1]).cpu())
            traj["q"].append(S[4].cpu()), traj["f"].append(env.f.cpu()), traj["alive"].append(alive.cpu())
        out = env.step(a)
        ret += torch.where(alive, out["r"], 0.0)
        env.f = torch.minimum(env.f, clip_end - 1)  # finished clips keep stepping; keep f+1 in range
    norm = ret / ((motion.clip_len - 1).float() * MAX_STEP_REWARD)
    res = {
        "norm_return": norm.mean().item(),
        "norm_return_framew": (ret.sum() / ((motion.clip_len - 1).sum() * MAX_STEP_REWARD)).item(),
        "n_failed": int(env.failed_eval.sum().item()),
        "failed": [motion.names[i] for i in env.failed_eval.nonzero().flatten().tolist()],
        "per_clip": {motion.names[i]: round(norm[i].item(), 4) for i in range(C)},
    }
    if record:
        np.savez_compressed(record, names=np.array(motion.names), **{k: torch.stack(v).numpy() for k, v in traj.items()},
                            ref_root_p=motion.root_p.cpu().numpy(), ref_root_R=motion.root_R.cpu().numpy(), ref_q=motion.q.cpu().numpy(),
                            clip_start=motion.clip_start.cpu().numpy(), clip_len=motion.clip_len.cpu().numpy())  # fmt: skip
    return res


sim = sim_utils.SimulationContext(make_sim_cfg(dev))
env = TrackingSim(sim, C, motion, mdp, eval_mode=True)
if args.ckpt:
    ck = torch.load(args.ckpt, map_location=dev)
    res = evaluate(env, ck, args.record)
    print(json.dumps({k: v for k, v in res.items() if k != "per_clip"}, indent=1))
    json.dump(res, open(args.ckpt.replace(".pt", "_eval.json"), "w"), indent=1)
    os._exit(0)  # app.close() hangs after headless runs
out_csv = os.path.join(args.run_dir, "eval.csv")
done = set()
if os.path.exists(out_csv):
    done = {int(l.split(",")[0]) for l in open(out_csv).read().strip().splitlines()[1:]}
elif args.shard == "0/1":  # sharded runs: the launcher writes the header
    open(out_csv, "w").write("iteration,train_wall_h,saved_at,eval_at,norm_return,norm_return_framew,n_failed,failed\n")
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

tb = SummaryWriter(args.run_dir)
while True:
    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
    ckpts = sorted(glob.glob(os.path.join(args.run_dir, "ckpt_*.pt")))[shard_i::shard_n]
    for c in [c for c in ckpts if int(os.path.basename(c)[5:11]) not in done]:
        it = int(os.path.basename(c)[5:11])
        ck = torch.load(c, map_location=dev)
        t0 = time.time()
        res = evaluate(env, ck)
        line = (f"{it},{ck['wall_time'] / 3600:.4f},{ck['saved_at']},{time.strftime('%Y-%m-%d %H:%M:%S')},{res['norm_return']:.4f},"
                f"{res['norm_return_framew']:.4f},{res['n_failed']},{'|'.join(res['failed'])}\n")
        open(out_csv, "a").write(line)
        json.dump(res, open(c.replace(".pt", "_eval.json"), "w"), indent=1)
        tb.add_scalar("eval/norm_return", res["norm_return"], it)
        tb.add_scalar("eval/n_failed", res["n_failed"], it)
        tb.flush()
        print(f"[eval] it {it} ({ck['wall_time'] / 3600:.2f} h): norm return {res['norm_return']:.4f}, failed {res['n_failed']}/{C} "
              f"({time.time() - t0:.0f}s) {res['failed']}", flush=True)
        done.add(it)
    if not args.watch:
        break
    time.sleep(60)
tb.close()
os._exit(0)  # app.close() hangs after headless runs
