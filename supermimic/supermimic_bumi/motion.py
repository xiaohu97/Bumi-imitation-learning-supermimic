"""Reference motion library resampled to 50 Hz, with FK of the tracked bodies.

Adapted from supermimic_vibed (MIT, Chong Zhang) ``motion.py``.  Sources:

* ``*.csv``: LAFAN1_Retargeting_Dataset layout at 30 fps (root pos, root quat xyzw, joints in URDF
  depth-first order = GMR MJCF order) -- the GMR noetix_bumi retargets;
* ``*.npz``: BeyondMimic motion files (50 fps, Isaac Lab joint/body order), e.g. noetix_bumi_lab mimic_data.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch

from .kinematics import Kinematics
from .robot import CONTROL_DT, JOINT_NAMES, TREE
from .rotations import mat_to_6d, quat_slerp, quat_to_mat, so3_log, yaw_mat

BIN_FRAMES = 50  # 1 s bins for return-based reset sampling
_DFS_TO_BFS = [TREE.joint_names_dfs.index(n) for n in JOINT_NAMES]


def load_clip(path: str):
    """-> (fps, root pos (T,3), root quat wxyz (T,4), joints (T,NUM_J) in JOINT_NAMES order)."""
    if path.endswith(".csv"):
        raw = torch.tensor(np.loadtxt(path, delimiter=","), dtype=torch.float32)
        assert raw.shape[1] == 7 + len(JOINT_NAMES), (path, raw.shape)
        q_xyzw = raw[:, 3:7]
        return 30.0, raw[:, :3], torch.cat((q_xyzw[:, 3:4], q_xyzw[:, :3]), -1), raw[:, 7:][:, _DFS_TO_BFS]
    d = np.load(path)
    fps = float(np.asarray(d["fps"]).reshape(-1)[0])
    t = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32)  # noqa: E731
    return fps, t(d["body_pos_w"][:, 0]), t(d["body_quat_w"][:, 0]), t(d["joint_pos"])


class MotionLib:
    def __init__(self, files: list[str], kin: Kinematics, device: str):
        self.device = device
        self.files = list(files)
        self.names = [os.path.splitext(os.path.basename(f))[0] for f in files]
        ps, qs, js, lens = [], [], [], []
        for f in files:
            src_fps, p, quat, j = load_clip(f)
            T_src = p.shape[0]
            dur = (T_src - 1) / src_fps
            n = int(np.floor(dur / CONTROL_DT + 1e-6)) + 1
            t = torch.arange(n, dtype=torch.float32) * CONTROL_DT * src_fps
            i0 = torch.clamp(t.floor().long(), max=T_src - 2)
            a = (t - i0.float()).clamp(0, 1)
            p = p[i0] * (1 - a[:, None]) + p[i0 + 1] * a[:, None]
            j = j[i0] * (1 - a[:, None]) + j[i0 + 1] * a[:, None]
            quat = quat_slerp(quat[i0], quat[i0 + 1], a)
            ps.append(p), qs.append(quat), js.append(j)
            lens.append(n)
        self.clip_len = torch.tensor(lens, device=device)
        self.clip_start = torch.cumsum(
            torch.cat((torch.zeros(1, dtype=torch.long), torch.tensor(lens[:-1], dtype=torch.long))), 0
        ).to(device)
        self.clip_end = self.clip_start + self.clip_len - 1  # last valid frame (inclusive)
        self.num_frames = int(sum(lens))
        self.clip_id = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(lens)]).to(device)

        root_p, root_R, root_v, root_w, jq, jqd = [], [], [], [], [], []
        for p, quat, j in zip(ps, qs, js):
            R = quat_to_mat(quat)
            v = _gradient(p, CONTROL_DT)
            qd = _gradient(j, CONTROL_DT)
            w = torch.zeros_like(p)
            w[1:-1] = so3_log(R[2:] @ R[:-2].transpose(-1, -2)) / (2 * CONTROL_DT)
            w[0] = so3_log(R[1] @ R[0].T) / CONTROL_DT
            w[-1] = so3_log(R[-1] @ R[-2].T) / CONTROL_DT
            root_p.append(p), root_R.append(R), root_v.append(v), root_w.append(w), jq.append(j), jqd.append(qd)
        cat = lambda xs: torch.cat(xs).to(device)  # noqa: E731
        self.root_p, self.root_R, self.root_v, self.root_w = cat(root_p), cat(root_R), cat(root_v), cat(root_w)
        self.q, self.qd = cat(jq), cat(jqd)
        with torch.no_grad():
            bp, bR, bv, bw = [], [], [], []
            for s in range(0, self.num_frames, 65536):
                e = min(s + 65536, self.num_frames)
                out = kin.forward(self.root_p[s:e], self.root_R[s:e], self.root_v[s:e], self.root_w[s:e], self.q[s:e], self.qd[s:e])
                bp.append(out[0]), bR.append(out[1]), bv.append(out[2]), bw.append(out[3])
            self.body_p, self.body_R, self.body_v, self.body_w = torch.cat(bp), torch.cat(bR), torch.cat(bv), torch.cat(bw)
        Rt = self.root_R.transpose(-1, -2)
        self.root_v_local = (Rt @ self.root_v.unsqueeze(-1)).squeeze(-1)
        self.root_w_local = (Rt @ self.root_w.unsqueeze(-1)).squeeze(-1)
        bin_clip, bin_start = [], []
        for c in range(len(lens)):
            for b in range(int(np.ceil(lens[c] / BIN_FRAMES))):
                bin_clip.append(c)
                bin_start.append(int(self.clip_start[c]) + b * BIN_FRAMES)
        self.bin_clip = torch.tensor(bin_clip, device=device)
        self.bin_start = torch.tensor(bin_start, device=device)
        self.num_bins = len(bin_start)
        self.frame_bin = torch.zeros(self.num_frames, dtype=torch.long, device=device)
        for b in range(self.num_bins):
            s = bin_start[b]
            e = min(s + BIN_FRAMES, int(self.clip_end[bin_clip[b]]) + 1)
            self.frame_bin[s:e] = b

    @staticmethod
    def from_list(list_file: str, kin: Kinematics, device: str) -> "MotionLib":
        """Text file with one path (or glob) per line, relative to the list file; '#' comments."""
        base = os.path.dirname(os.path.abspath(list_file))
        files = []
        for line in open(list_file):
            line = line.split("#")[0].strip()
            if not line:
                continue
            pat = line if os.path.isabs(line) else os.path.join(base, line)
            hits = sorted(glob.glob(pat))
            assert hits, f"no file matches {pat}"
            files.extend(hits)
        return MotionLib(files, kin, device)

    def future_frames(self, f: torch.Tensor, K: int) -> torch.Tensor:
        end = self.clip_end[self.clip_id[f]]
        fut = f.unsqueeze(-1) + torch.arange(1, K + 1, device=f.device)
        return torch.minimum(fut, end.unsqueeze(-1))

    def goal(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """K future reference frames relative to the current reference frame (reference-only goal)."""
        fut = self.future_frames(f, K)
        H = yaw_mat(self.root_R[f])
        Ht = H.transpose(-1, -2).unsqueeze(1)
        dp = (Ht @ (self.root_p[fut] - self.root_p[f].unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
        rot = mat_to_6d(Ht @ self.root_R[fut])
        feats = torch.cat((dp, rot, self.root_v_local[fut], self.root_w_local[fut], self.q[fut], self.qd[fut] * 0.1), -1)
        return feats.reshape(f.shape[0], -1)

    def ref_state(self, f: torch.Tensor):
        return self.root_p[f], self.root_R[f], self.root_v[f], self.root_w[f], self.q[f], self.qd[f]


GOAL_FRAME_DIM = 3 + 6 + 3 + 3 + 2 * len(JOINT_NAMES)


def _gradient(x: torch.Tensor, dt: float) -> torch.Tensor:
    g = torch.zeros_like(x)
    g[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    g[0] = (x[1] - x[0]) / dt
    g[-1] = (x[-1] - x[-2]) / dt
    return g
