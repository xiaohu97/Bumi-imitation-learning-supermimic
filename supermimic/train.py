"""SuperMimic training on Noetix Bumi 21-DoF (Algorithm A.1): collect -> fit world model -> BPTT policy + critic.

Adapted from supermimic_vibed
(MIT, Chong Zhang) ``train.py``; simulator = Isaac Lab 2.x / PhysX with the noetix_bumi_lab Mimic plant.

    python supermimic/train.py --headless --motions supermimic/motions/bumi_lafan40.txt \
        --log_dir logs/<run>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--iterations", type=int, default=320_000)
parser.add_argument("--buffer_steps", type=int, default=1000)
parser.add_argument("--utd", type=float, default=0.4, help="updates per vectorized simulator step")
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--goal_frames", type=int, default=10)
parser.add_argument("--hist", type=int, default=10)
parser.add_argument("--wm_window", type=int, default=8)
parser.add_argument("--wm_batches", type=int, default=10)
parser.add_argument("--wm_batch", type=int, default=2048)
parser.add_argument("--pi_batch", type=int, default=4096)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--action_noise", type=float, default=0.1)
parser.add_argument("--critic_epochs", type=int, default=5)
parser.add_argument("--critic_minibatches", type=int, default=4)
parser.add_argument("--no_critic", action="store_true")
parser.add_argument("--warmup_steps", type=int, default=64, help="sim steps before any update")
parser.add_argument("--wm_only_iters", type=int, default=500, help="world-model-only iterations at the start")
parser.add_argument("--lr_decay_iters", type=int, default=250_000)
parser.add_argument("--ckpt_every", type=int, default=2000)
parser.add_argument("--log_every", type=int, default=50)
parser.add_argument("--motions", type=str, default=os.path.join(HERE, "motions", "bumi_lafan40.txt"))
parser.add_argument("--log_dir", type=str, required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--init", type=str, default=None, help="fine-tune: policy + world model from this checkpoint")
parser.add_argument("--init_critic", type=str, default=None, help="fine-tune: critic from this full checkpoint (last.pt)")
parser.add_argument("--start_iter", type=int, default=0, help="fine-tune: iteration counter / lr schedule start")
parser.add_argument("--w_foot_impact", type=float, default=0.0, help="foot impact penalty weight (RewardCfg)")
parser.add_argument("--w_foot_slip", type=float, default=0.0, help="foot slip penalty weight (RewardCfg)")
parser.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16"], help="autocast dtype for MLP matmuls")
parser.add_argument("--no_compile", action="store_true")
parser.add_argument("--grad_ckpt", action="store_true", help="recompute imagined steps in backward (saves BPTT memory)")
parser.add_argument("--profile_mem", action="store_true", help="print peak CUDA memory after the first policy updates")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.utils.tensorboard import SummaryWriter

import isaaclab.sim as sim_utils

import supermimic_bumi.networks as networks
from supermimic_bumi.buffer import ReplayBuffer
from supermimic_bumi.env import TrackingSim, make_sim_cfg
from supermimic_bumi.kinematics import Kinematics
from supermimic_bumi.mdp import CRITIC_EXTRA, INV_DIM, MAX_STEP_REWARD, MDP, OBS_DIM, RewardCfg, normalize_xi, unflatten_state
from supermimic_bumi.motion import GOAL_FRAME_DIM, MotionLib
from supermimic_bumi.networks import Critic, Policy, WorldModel
from supermimic_bumi.robot import NUM_J, default_joint_pos_list

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(args.seed)
networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
dev = "cpu" if args.device == "cpu" else args.device if args.device.startswith("cuda") else "cuda:0"  # cpu: smoke tests
os.makedirs(args.log_dir, exist_ok=True)
json.dump(vars(args), open(os.path.join(args.log_dir, "args.json"), "w"), indent=2)

kin = Kinematics(dev)
motion = MotionLib.from_list(args.motions, kin, dev)
open(os.path.join(args.log_dir, "motions.txt"), "w").write("\n".join(motion.files) + "\n")
print(f"[supermimic] {len(motion.names)} clips, {motion.num_frames} frames ({motion.num_frames * 0.02 / 3600:.2f} h) at 50 Hz", flush=True)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev), RewardCfg(w_foot_impact=args.w_foot_impact, w_foot_slip=args.w_foot_slip))
K, Hh, HZ = args.goal_frames, args.hist, args.horizon
GOAL_DIM = GOAL_FRAME_DIM * K
CRITIC_IN = INV_DIM + CRITIC_EXTRA + GOAL_DIM

wm = WorldModel().to(dev)
pi = Policy(OBS_DIM, GOAL_DIM, hist=Hh).to(dev)
critic = Critic(CRITIC_IN).to(dev)
opt_wm = torch.optim.Adam(wm.parameters(), lr=args.lr)
opt_pi = torch.optim.Adam(pi.parameters(), lr=args.lr, betas=(0.9, 0.95))
opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr)
start_it, wall_offset, sim_steps = 0, 0.0, 0
if args.resume:
    ck = torch.load(args.resume, map_location=dev)
    wm.load_state_dict(ck["wm"]), pi.load_state_dict(ck["pi"]), critic.load_state_dict(ck["critic"])
    opt_wm.load_state_dict(ck["opt_wm"]), opt_pi.load_state_dict(ck["opt_pi"]), opt_v.load_state_dict(ck["opt_v"])
    start_it, wall_offset, sim_steps = ck["iteration"], ck["wall_time"], ck["sim_steps"]
elif args.init:
    ck = torch.load(args.init, map_location=dev)
    wm.load_state_dict(ck["wm"]), pi.load_state_dict(ck["pi"])
    if args.init_critic:
        critic.load_state_dict(torch.load(args.init_critic, map_location=dev)["critic"])
    start_it = args.start_iter
    print(f"[supermimic] fine-tune from {args.init} (iteration {ck['iteration']}), critic from {args.init_critic}, "
          f"counter starts at {start_it}", flush=True)

# per-component floors on the error normalization (pos, rot, lin vel, ang vel, q, qd increments)
ERR_FLOOR = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                       torch.full((NUM_J,), 3e-3), torch.full((NUM_J,), 5e-2))).to(dev)  # fmt: skip


def _wm_rollout(s, a, xi):
    S = unflatten_state(s[:, 0])
    preds = []
    for j in range(args.wm_window):
        S = wm.step(S, a[:, j], normalize_xi(xi[:, j]))
        preds.append(S)
    return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6))


def _wm_err(S_hat, s1, mask):
    std = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
    e = (wm.state_error(S_hat, unflatten_state(s1)) / std) ** 2
    e = e.mean(-1)
    return (e * mask).sum() / s1.shape[0], e[:, -1]


def _imag_step(S, o_hist, a_prev, f, xi, xi_n, g):
    a = pi(o_hist, g)
    S1 = wm.step(S, a, xi_n)
    f1 = f + 1
    o1 = mdp.add_noise(mdp.obs(S1, a, xi, f1))
    o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
    return S1, a, o_hist, f1, motion.goal(f1, K)


def _reward_term(S, f, a, a_prev):
    return mdp.reward(S, f, a, a_prev), mdp.soft_termination(S, f)


def _critic_input(S, f, xi_n, g):
    return torch.cat((mdp.critic_features(S, f, xi_n), g), -1)


cmp = (lambda fn: fn) if args.no_compile else torch.compile
wm_rollout, wm_err, imag_step_c, reward_term, critic_input_c = map(cmp, (_wm_rollout, _wm_err, _imag_step, _reward_term, _critic_input))


def imag_step(*xs):
    if not args.grad_ckpt:
        return imag_step_c(*xs)
    # the observation noise must be identical in the recomputation
    return checkpoint(imag_step_c, *xs, use_reentrant=False, preserve_rng_state=True)


def lr_at(it: int) -> float:
    if it >= args.lr_decay_iters:
        return args.lr * 0.03
    return args.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * it / args.lr_decay_iters)))


def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)


def wm_update():
    w = buffer.sample_windows(args.wm_batch, args.wm_window)
    S_hat = wm_rollout(w["s"], w["a"], w["xi"])
    loss, e_last = wm_err(S_hat, w["s1"], w["mask"])
    opt_wm.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
    opt_wm.step()
    m = w["mask"][:, -1]
    return loss.detach() / args.wm_window, (e_last.detach() * m).sum() / m.sum().clamp(min=1)


def sample_valid_starts(B: int):
    starts, hists, n = [], [], 0
    while n < B:
        s, h = buffer.sample_starts(2 * B, Hh)
        f = s["f"]
        ok = f + HZ <= motion.clip_end[motion.clip_id[f]]
        idx = ok.nonzero(as_tuple=False).squeeze(-1)[: B - n]
        starts.append({k: v[idx] for k, v in s.items()}), hists.append({k: v[idx] for k, v in h.items()})
        n += len(idx)
    cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}  # noqa: E731
    return cat(starts), cat(hists)


def policy_update():
    s, h = sample_valid_starts(args.pi_batch)
    S = unflatten_state(s["s"])
    xi, f, a_prev = s["xi"], s["f"], s["a_prev"]
    xi_n = normalize_xi(xi)
    with torch.no_grad():
        o_hist = mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"])
    o_hist = mdp.add_noise(o_hist)
    set_requires_grad(wm, False)
    set_requires_grad(critic, False)
    B = args.pi_batch
    g0 = motion.goal(f, K)
    g = g0
    S0 = S
    Ss, As, Aps, Fs, Gs = [], [], [], [], []
    for t in range(HZ):
        Aps.append(a_prev)
        S, a_prev, o_hist, f, g = imag_step(S, o_hist, a_prev, f, xi, xi_n, g)
        Ss.append(S), As.append(a_prev), Fs.append(f), Gs.append(g)
    flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731
    S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
    f_all = flat(Fs)
    r, d = reward_term(S_all, f_all, flat(As), flat(Aps))
    r, d = r.view(HZ, B), d.view(HZ, B)
    alive = torch.cumprod(1 - d, 0)
    c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
    disc = args.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
    J = (disc * c_t * r).sum(0)
    c = alive[-1]
    r_sum = (c_t * r).detach().sum(0)
    rews, dones = r.detach(), d.detach()
    if not args.no_critic:
        J = J + (args.gamma**HZ) * c * critic(_critic_input(S, f, xi_n, g))
    loss = -J.mean()
    opt_pi.zero_grad(set_to_none=True)
    loss.backward()
    gn = nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
    nn.utils.clip_grad_value_(pi.parameters(), 0.1)
    if torch.isfinite(gn):
        opt_pi.step()
    set_requires_grad(wm, True)
    set_requires_grad(critic, True)
    v_loss = torch.zeros((), device=dev)
    if not args.no_critic:
        with torch.no_grad():
            S_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1) for i in range(6))
            f_c = torch.cat((s["f"].unsqueeze(0), torch.stack(Fs)), 0).flatten()
            g_c = torch.cat((g0.unsqueeze(0), torch.stack([x.detach() for x in Gs])), 0).flatten(0, 1)
            xi_c = xi_n.unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
            X = critic_input_c(S_c, f_c, xi_c, g_c).view(HZ + 1, B, -1)
        critic.norm.update(X)
        with torch.no_grad():
            V = critic(X)
            R = V[-1]
            targets = []
            for t in range(HZ - 1, -1, -1):
                R = rews[t] + args.gamma * (1 - dones[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
                targets.append(R)
            targets = torch.stack(targets[::-1])
        Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
        n = Xf.shape[0]
        mb = n // args.critic_minibatches
        for _ in range(args.critic_epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(args.critic_minibatches):
                idx = perm[i * mb : (i + 1) * mb]
                v_loss = ((critic(Xf[idx]) - Yf[idx]) ** 2).mean()
                opt_v.zero_grad(set_to_none=True)
                v_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                opt_v.step()
    return loss.detach(), v_loss.detach(), (r_sum / HZ / 0.1).mean(), gn.detach(), c.mean().detach()


# ------------------------------------------------------------------------------------------------
sim = sim_utils.SimulationContext(make_sim_cfg(dev))
env = TrackingSim(sim, args.num_envs, motion, mdp, gamma=args.gamma)
buffer = ReplayBuffer(args.buffer_steps, args.num_envs, dev)
obs_hist = torch.zeros(args.num_envs, Hh, OBS_DIM, device=dev)
print(f"[supermimic] sim up: {args.num_envs} envs, buffer {args.buffer_steps} steps, "
      f"mem {torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated", flush=True)


@torch.no_grad()
def collect(n_steps: int, stats: dict):
    global obs_hist, sim_steps
    for _ in range(n_steps):
        S = env.state()
        o = mdp.add_noise(mdp.obs(S, env.a_prev, env.xi, env.f))
        fresh = env.ep_step == 0
        obs_hist = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(obs_hist), torch.cat((obs_hist[:, 1:], o.unsqueeze(1)), 1))
        g = motion.goal(env.f, K)
        pi.obs_norm.update(o)
        pi.goal_norm.update(g)
        a = pi(obs_hist, g)
        a = torch.clamp(a + args.action_noise * torch.randn_like(a), -20, 20)
        out = env.step(a)
        buffer.add(out)
        S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
        wm.in_norm.update(wm.inputs(S0, out["a"], normalize_xi(out["xi"])))
        wm.out_norm.update(wm.targets(S0, S1))
        sim_steps += args.num_envs
        stats["r"].append(out["r"].mean() / MAX_STEP_REWARD)
        stats["e_pos"].append(out["terms"]["e_body_pos"].mean())
        stats["imp"].append(out["terms"]["foot_impact"].mean())
        stats["slip"].append(out["terms"]["foot_slip"].mean())
        if "ep_len" in out:
            stats["ep_len"].append(out["ep_len"].mean())
            stats["ep_ret"].append(out["ep_ret_norm"].mean())
            stats["term"].append(out["term"].float().sum())
            stats["done"].append(out["done"].float().sum())


stats = {"r": [], "e_pos": [], "imp": [], "slip": [], "ep_len": [], "ep_ret": [], "term": [], "done": []}
collect(args.warmup_steps if start_it == 0 else max(args.warmup_steps, args.wm_window + 1), stats)
log_f = open(os.path.join(args.log_dir, "train_log.csv"), "a")
tb = SummaryWriter(args.log_dir)
if log_f.tell() == 0:
    log_f.write("iteration,wall_h,sim_steps,wm_loss,wm_err8,pi_loss,v_loss,imag_rew,pi_gradnorm,imag_survival,sim_rew,"
                "e_body_pos,ep_len,ep_ret_norm,term_frac,it_per_s,lr,mem_gib\n")
t_start = time.time() - wall_offset
t_log = time.time()
step_acc = 0.0
agg = {"wm": [], "wm8": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": []}
for it in range(start_it, args.iterations):
    lr = lr_at(it)
    for o in (opt_wm, opt_pi, opt_v):
        for gparam in o.param_groups:
            gparam["lr"] = lr
    step_acc += 1.0 / args.utd
    n = int(step_acc)
    step_acc -= n
    collect(n, stats)
    for _ in range(args.wm_batches):
        l, e8 = wm_update()
    agg["wm"].append(l), agg["wm8"].append(e8)
    if it >= args.wm_only_iters:
        pl, vl, ir, gn, surv = policy_update()
        agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir), agg["gn"].append(gn), agg["surv"].append(surv)
        if args.profile_mem and it < args.wm_only_iters + 3:
            print(f"[mem] it {it}: peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB allocated, "
                  f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB reserved", flush=True)
    if (it + 1) % args.log_every == 0:
        mean = lambda x: float(sum(float(v) for v in x) / max(len(x), 1))  # noqa: E731
        ips = args.log_every / (time.time() - t_log)
        t_log = time.time()
        wall_h = (time.time() - t_start) / 3600
        term_frac = float(sum(float(v) for v in stats["term"]) / max(sum(float(v) for v in stats["done"]), 1))
        mem = torch.cuda.max_memory_reserved() / 2**30
        row = [it + 1, wall_h, sim_steps, mean(agg["wm"]), mean(agg["wm8"]), mean(agg["pi"]), mean(agg["v"]), mean(agg["ir"]),
               mean(agg["gn"]), mean(agg["surv"]), mean(stats["r"]), mean(stats["e_pos"]), mean(stats["ep_len"]), mean(stats["ep_ret"]),
               term_frac, ips, lr, mem]  # fmt: skip
        log_f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
        log_f.flush()
        for k, v in zip(("wm/loss", "wm/err8", "pi/loss", "critic/v_loss", "pi/imag_rew", "pi/gradnorm", "pi/imag_survival",
                         "sim/rew_per_s", "sim/e_body_pos", "sim/ep_len", "sim/ep_ret_norm", "sim/term_frac", "perf/it_per_s", "perf/lr",
                         "perf/mem_gib"), row[3:]):  # fmt: skip
            tb.add_scalar(k, v, it + 1)
        tb.add_scalar("sim/foot_impact", mean(stats["imp"]), it + 1)
        tb.add_scalar("sim/foot_slip", mean(stats["slip"]), it + 1)
        print(f"it {it + 1} | {wall_h:.3f} h | wm {row[3]:.4f} e8 {row[4]:.4f} | pi {row[5]:.3f} v {row[6]:.4f} imag_r {row[7]:.3f} "
              f"surv {row[9]:.3f} | sim_r {row[10]:.3f} e_pos {row[11]:.4f} ep_len {row[12]:.1f} ep_ret {row[13]:.3f} "
              f"term {term_frac:.2f} imp {mean(stats['imp']):.4f} slip {mean(stats['slip']):.4f} | {ips:.2f} it/s | {mem:.1f} GiB", flush=True)  # fmt: skip
        stats = {k: [] for k in stats}
        agg = {k: [] for k in agg}
    if (it + 1) % args.ckpt_every == 0 or it + 1 == args.iterations:
        ck = {"wm": wm.state_dict(), "pi": pi.state_dict(), "critic": critic.state_dict(), "opt_wm": opt_wm.state_dict(),
              "opt_pi": opt_pi.state_dict(), "opt_v": opt_v.state_dict(), "iteration": it + 1,
              "wall_time": time.time() - t_start, "sim_steps": sim_steps, "args": vars(args),
              "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
        last = os.path.join(args.log_dir, "last.pt")
        torch.save(ck, last + ".tmp")
        os.replace(last + ".tmp", last)
        small = {k: ck[k] for k in ("pi", "wm", "iteration", "wall_time", "sim_steps", "args", "saved_at")}
        path = os.path.join(args.log_dir, f"ckpt_{it + 1:06d}.pt")
        torch.save(small, path + ".tmp")
        os.replace(path + ".tmp", path)

log_f.close()
tb.close()
# app.close() hangs after headless Isaac Lab runs and keeps the GPU (and blocks eval_after_train.sh)
os._exit(0)
