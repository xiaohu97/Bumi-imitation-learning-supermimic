"""Analytic MDP heads shared by the simulator and the world model (paper Appendix A).

Adapted from supermimic_vibed (MIT, Chong Zhang) ``mdp.py`` for Noetix Bumi (21 joints).
State tuple ``S = (p, R, v, w, q, qd)``: base_link position, rotation matrix, world linear/angular
velocity of the base_link frame, joint positions and velocities.

Sim context xi = [friction (1), joint zero-point offsets (21), waist_yaw_link COM offset (3), command delay
in physics steps (1)].  The delay (0-4 steps) is part of the Bumi Mimic plant; it is per-episode hidden
state of the actuator, so the world model is given it like the other randomized parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .kinematics import Kinematics
from .motion import MotionLib
from .robot import CONTROL_DT, FEET_BODIES, MAX_DELAY, MIN_DELAY, NUM_J, TRACKED_BODIES, joint_offset_ranges
from .rotations import mat_to_6d, mat_to_quat, quat_to_mat, rot_angle_sq, yaw_mat

NB = len(TRACKED_BODIES)
STATE_DIM = 3 + 4 + 3 + 3 + 2 * NUM_J
OBS_DIM = NUM_J + NUM_J + 3 + 3 + NUM_J + 6
XI_DIM = 1 + NUM_J + 3 + 1
XI_OFF = slice(1, 1 + NUM_J)
XI_COM = slice(1 + NUM_J, 4 + NUM_J)
XI_DELAY = 4 + NUM_J

# Table A.1 sensor noise (uniform half-widths), plus the reference-relative orientation
OBS_NOISE = torch.cat(
    (
        torch.full((NUM_J,), 0.01),
        torch.full((NUM_J,), 0.5),
        torch.full((3,), 0.2),
        torch.full((3,), 0.05),
        torch.zeros(NUM_J),
        torch.full((6,), 0.05),
    )
)
FEET = [TRACKED_BODIES.index(b) for b in FEET_BODIES]
MAX_STEP_REWARD = (4 * 1.0 + 2 * 0.5) * CONTROL_DT


@dataclass
class RewardCfg:
    body_pos_std: float = 0.3
    body_ori_std: float = 0.4
    body_linvel_std: float = 1.0
    body_angvel_std: float = 3.14
    root_pos_std: float = 0.3
    root_ori_std: float = 0.4
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    soft_limit_factor: float = 0.9
    term_height: float = 0.25
    term_tilt: float = 0.8
    # foot impact (2026-10-28): near the ground, penalize a foot coming down faster than the reference foot
    # (squared excess over a margin).  0 = off (original objective).  Bumi's ankle-roll origin sits 0.051 m above
    # the floor in stance (gates shifted by -0.004 from the 0.055 m ankle height they were tuned on): full weight below z_lo, fading to 0 at z_hi.
    w_foot_impact: float = 0.0
    impact_margin: float = 0.1
    impact_z_lo: float = 0.076
    impact_z_hi: float = 0.136
    # foot slip: while both the robot foot and the reference foot are near the ground (ankle-roll origin within
    # slip_z_lo..slip_z_hi), penalize the robot foot's horizontal velocity relative to the reference foot
    # (squared): sliding a planted foot, or shuffling where the reference steps.  0 = off.
    w_foot_slip: float = 0.0
    slip_z_lo: float = 0.051
    slip_z_hi: float = 0.071


def flatten_state(S) -> torch.Tensor:
    p, R, v, w, q, qd = S
    return torch.cat((p, mat_to_quat(R), v, w, q, qd), -1)


def unflatten_state(x: torch.Tensor):
    j = 13 + NUM_J
    return x[..., 0:3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13], x[..., 13:j], x[..., j : j + NUM_J]


def invariant_features(S) -> torch.Tensor:
    """Heading- and position-invariant state features: height, tilt (6D), body-frame velocities, joints."""
    p, R, v, w, q, qd = S
    H = yaw_mat(R)
    Rt = R.transpose(-1, -2)
    return torch.cat(
        (
            p[..., 2:3],
            mat_to_6d(H.transpose(-1, -2) @ R),
            (Rt @ v.unsqueeze(-1)).squeeze(-1),
            (Rt @ w.unsqueeze(-1)).squeeze(-1),
            q,
            qd,
        ),
        -1,
    )


INV_DIM = 1 + 6 + 3 + 3 + 2 * NUM_J


class MDP:
    def __init__(self, kin: Kinematics, motion: MotionLib, default_q: torch.Tensor, cfg: RewardCfg = RewardCfg()):
        self.kin, self.motion, self.cfg = kin, motion, cfg
        self.default_q = default_q
        mid, half = (kin.q_lo + kin.q_hi) / 2, (kin.q_hi - kin.q_lo) / 2
        self.soft_lo, self.soft_hi = mid - half * cfg.soft_limit_factor, mid + half * cfg.soft_limit_factor
        self.obs_noise = OBS_NOISE.to(default_q.device)

    # ---------------- observation ----------------
    def obs(self, S, a_prev, xi, f) -> torch.Tensor:
        """Noise-free proprioceptive observation (+ reference-relative root orientation)."""
        p, R, v, w, q, qd = S
        Rt = R.transpose(-1, -2)
        g_b = -R[..., 2, :]  # R^T (0,0,-1)
        rel = mat_to_6d(Rt @ self.motion.root_R[f])
        return torch.cat((q - self.default_q - xi[..., XI_OFF], qd, (Rt @ w.unsqueeze(-1)).squeeze(-1), g_b, a_prev, rel), -1)

    def add_noise(self, o: torch.Tensor) -> torch.Tensor:
        return o + (2 * torch.rand_like(o) - 1) * self.obs_noise

    # ---------------- reward / termination ----------------
    def _ref_bodies(self, S, f):
        m = self.motion
        p, R = S[0], S[1]
        rp, rR = m.root_p[f], m.root_R[f]
        # reference bodies re-anchored to the robot's root (xy + yaw), as in BeyondMimic
        dR = yaw_mat(R @ rR.transpose(-1, -2))
        anchor = torch.cat((p[..., :2], rp[..., 2:3]), -1)
        ref_bp = anchor.unsqueeze(-2) + ((dR.unsqueeze(-3) @ (m.body_p[f] - rp.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1))
        return ref_bp, dR

    def reward(self, S, f, a, a_prev, terms: bool = False):
        """Per-step reward (already multiplied by dt) comparing S with reference frame f."""
        c, m = self.cfg, self.motion
        p, R, v, w, q, qd = S
        bp, bR, bv, bw = self.kin.forward(p, R, v, w, q, qd)
        rp, rR = m.root_p[f], m.root_R[f]
        ref_bp, dR = self._ref_bodies(S, f)
        ref_bR = dR.unsqueeze(-3) @ m.body_R[f]
        e_pos = ((ref_bp - bp) ** 2).sum(-1).mean(-1)
        e_ori = rot_angle_sq(ref_bR, bR).mean(-1)
        e_lv = ((m.body_v[f] - bv) ** 2).sum(-1).mean(-1)
        e_av = ((m.body_w[f] - bw) ** 2).sum(-1).mean(-1)
        e_rp = ((rp - p) ** 2).sum(-1)
        e_ro = rot_angle_sq(rR, R)
        r_track = (
            torch.exp(-e_pos / c.body_pos_std**2)
            + torch.exp(-e_ori / c.body_ori_std**2)
            + torch.exp(-e_lv / c.body_linvel_std**2)
            + torch.exp(-e_av / c.body_angvel_std**2)
            + 0.5 * torch.exp(-e_rp / c.root_pos_std**2)
            + 0.5 * torch.exp(-e_ro / c.root_ori_std**2)
        )
        r_rate = ((a - a_prev) ** 2).sum(-1)
        r_lim = (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
        fz, fvz, ref_fvz = bp[..., FEET, 2], bv[..., FEET, 2], m.body_v[f][..., FEET, 2]
        gate = ((c.impact_z_hi - fz) / (c.impact_z_hi - c.impact_z_lo)).clamp(0.0, 1.0)
        r_imp = (gate * torch.relu(ref_fvz - fvz - c.impact_margin) ** 2).sum(-1)
        on = lambda zz: ((c.slip_z_hi - zz) / (c.slip_z_hi - c.slip_z_lo)).clamp(0.0, 1.0)  # noqa: E731
        ref_fv = (dR.unsqueeze(-3) @ m.body_v[f][..., FEET, :].unsqueeze(-1)).squeeze(-1)  # yaw-aligned like ref_bp
        r_slip = (on(fz) * on(m.body_p[f][..., FEET, 2]) * ((bv[..., FEET, :2] - ref_fv[..., :2]) ** 2).sum(-1)).sum(-1)
        r = (r_track + c.w_action_rate * r_rate + c.w_joint_limit * r_lim - c.w_foot_impact * r_imp
             - c.w_foot_slip * r_slip) * CONTROL_DT
        if terms:
            return r, {
                "track": r_track * CONTROL_DT,
                "e_body_pos": e_pos,
                "e_body_ori": e_ori,
                "e_root_pos": e_rp,
                "action_rate": r_rate,
                "joint_limit": r_lim,
                "foot_impact": r_imp,
                "foot_slip": r_slip,
            }
        return r

    def termination(self, S, f) -> torch.Tensor:
        p, R = S[0], S[1]
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        e_z = (rp[..., 2] - p[..., 2]).abs()
        e_tilt = (rR[..., 2, 2] - R[..., 2, 2]).abs()
        return (e_z > self.cfg.term_height) | (e_tilt > self.cfg.term_tilt)

    def soft_termination(self, S, f, tz: float = 0.02, tt: float = 0.04) -> torch.Tensor:
        """Differentiable termination probability for imagined rollouts."""
        p, R = S[0], S[1]
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        e_z = (rp[..., 2] - p[..., 2]).abs()
        e_tilt = (rR[..., 2, 2] - R[..., 2, 2]).abs()
        alive = (1 - torch.sigmoid((e_z - self.cfg.term_height) / tz)) * (1 - torch.sigmoid((e_tilt - self.cfg.term_tilt) / tt))
        return 1 - alive

    def critic_features(self, S, f, xi_n) -> torch.Tensor:
        """Critic input: invariant state, sim context, and robot-vs-reference errors (goal appended by caller)."""
        m = self.motion
        p, R, v, w, q, qd = S
        Ht = yaw_mat(R).transpose(-1, -2)
        rp, rR = m.root_p[f], m.root_R[f]
        bp = self.kin.forward(p, R, v, w, q, qd)[0]
        ref_bp, _ = self._ref_bodies(S, f)
        e_body = (Ht.unsqueeze(-3) @ (ref_bp - bp).unsqueeze(-1)).squeeze(-1).flatten(-2)
        e_root = (Ht @ (rp - p).unsqueeze(-1)).squeeze(-1)
        rel = mat_to_6d(R.transpose(-1, -2) @ rR)
        return torch.cat((invariant_features(S), m.q[f] - q, e_body, e_root, rel, xi_n), -1)


CRITIC_EXTRA = NUM_J + NB * 3 + 3 + 6 + XI_DIM

_OFF_RANGE = torch.tensor(joint_offset_ranges())


def normalize_xi(xi: torch.Tensor) -> torch.Tensor:
    mid_d, half_d = (MIN_DELAY + MAX_DELAY) / 2, max((MAX_DELAY - MIN_DELAY) / 2, 0.5)
    return torch.cat(
        (
            (xi[..., :1] - 0.75) / 0.45,
            xi[..., XI_OFF] / _OFF_RANGE.to(xi.device),
            xi[..., XI_COM] / 0.05,
            (xi[..., XI_DELAY : XI_DELAY + 1] - mid_d) / half_d,
        ),
        -1,
    )
