"""Differentiable forward kinematics of Noetix Bumi in torch, built from the training URDF.

Adapted from supermimic_vibed (MIT, Chong Zhang) ``kinematics.py``: the same FK computes tracked-body
poses/velocities for the reference motion, for logged simulator states and for world-model
predictions, so the analytic reward head matches the reward the simulator transitions carry.
Link frames are URDF link frames (= Isaac Lab body frames; the Bumi URDF has no fixed joints).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .robot import BODY_NAMES, JOINT_NAMES, TRACKED_BODIES, TREE, deploy_limits_lists
from .rotations import axis_angle_to_mat


def _rpy_to_mat(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class Kinematics:
    def __init__(self, device: str, tracked: list[str] = TRACKED_BODIES):
        self.device = device
        jidx = {n: i for i, n in enumerate(JOINT_NAMES)}
        self.body_names = list(BODY_NAMES)  # BFS: parents precede children
        self.parent, self.joint_idx = [], []
        axes, offs_p, offs_R = [], [], []
        for b in self.body_names:
            if b == TREE.root:
                self.parent.append(-1), self.joint_idx.append(-1)
                axes.append([1.0, 0.0, 0.0]), offs_p.append([0.0, 0.0, 0.0]), offs_R.append(np.eye(3))
                continue
            j = TREE.joints[b]
            assert j["type"] in ("revolute", "continuous"), j
            self.parent.append(self.body_names.index(j["parent"]))
            self.joint_idx.append(jidx[j["name"]])
            ax = np.array(j["axis"])
            axes.append(ax / np.linalg.norm(ax)), offs_p.append(j["xyz"]), offs_R.append(_rpy_to_mat(*j["rpy"]))
        f32 = dict(dtype=torch.float32, device=device)
        self.axes = torch.tensor(np.array(axes), **f32)
        self.off_p = torch.tensor(np.array(offs_p), **f32)
        self.off_R = torch.tensor(np.array(offs_R), **f32)
        self.num_bodies = len(self.body_names)
        self.tracked_idx = [self.body_names.index(n) for n in tracked]
        lo, hi = deploy_limits_lists()
        self.q_lo = torch.tensor(lo, **f32)
        self.q_hi = torch.tensor(hi, **f32)

    def forward(self, p, R, v, w, q, qd, all_bodies: bool = False):
        """Body world positions, rotations, linear (link-origin) and angular velocities.

        Args:
            p, R, v, w: root (base_link) position (B,3), rotation (B,3,3), world lin/ang velocity (B,3).
            q, qd: joint positions/velocities (B,NUM_J) in JOINT_NAMES order.
        """
        n = self.num_bodies
        P, Rs, V, W = [None] * n, [None] * n, [None] * n, [None] * n
        for i in range(n):
            par = self.parent[i]
            if par < 0:
                P[i], Rs[i], V[i], W[i] = p, R, v, w
                continue
            j = self.joint_idx[i]
            Rp = Rs[par]
            pi = P[par] + (Rp @ self.off_p[i].unsqueeze(-1)).squeeze(-1)
            R_off = Rp @ self.off_R[i]
            Ri = R_off @ axis_angle_to_mat(self.axes[i], q[..., j])
            axis_w = (R_off @ self.axes[i].unsqueeze(-1)).squeeze(-1)
            wi = W[par] + axis_w * qd[..., j : j + 1]
            vi = V[par] + torch.cross(W[par], pi - P[par], dim=-1)
            P[i], Rs[i], V[i], W[i] = pi, Ri, vi, wi
        idx = range(n) if all_bodies else self.tracked_idx
        return (
            torch.stack([P[i] for i in idx], -2),
            torch.stack([Rs[i] for i in idx], -3),
            torch.stack([V[i] for i in idx], -2),
            torch.stack([W[i] for i in idx], -2),
        )
