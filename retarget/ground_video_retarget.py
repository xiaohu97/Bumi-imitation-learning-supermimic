"""Ground a video retarget (offline_retarget_smplx_npz.py output) per frame.

A monocular estimate's vertical root translation drifts over a video (several cm per shot), so the one constant offset
that ground_retargets.py takes from mocap (median stance level) leaves whole shots floating.  Here the stance level is
tracked over time: the lowest foot-mesh point, as a 1 s running minimum (flight phases shorter than that stay
airborne), smoothed (sigma 5 frames); it is subtracted from the root height.  Then, as in ground_retargets.py,
residual penetration of any robot mesh is lifted out (lightly smoothed, never below the frame's own penetration).

    python retarget/ground_video_retarget.py --in motion_raw.csv --out motion.csv
"""

import argparse
import os
import sys

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d, minimum_filter1d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from general_motion_retargeting.params import ROBOT_XML_DICT  # noqa: E402

FEET = {"noetix_bumi": ("l_ankle_roll_link", "r_ankle_roll_link")}
FPS = 30


def lowest_points(m, d, x, feet):
    verts = []
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_bodyid[g] > 0:
            mid = m.geom_dataid[g]
            Rg = np.zeros(9)
            mujoco.mju_quat2Mat(Rg, m.geom_quat[g])
            v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            verts.append((m.geom_bodyid[g], v @ Rg.reshape(3, 3).T + m.geom_pos[g]))
    zmin, zfoot = np.zeros(len(x)), np.zeros(len(x))
    for t in range(len(x)):
        d.qpos[:3], d.qpos[3:7], d.qpos[7:] = x[t, :3], x[t, [6, 3, 4, 5]], x[t, 7:]
        mujoco.mj_kinematics(m, d)
        za, zf = np.inf, np.inf
        for b, v in verts:
            z = (v @ d.xmat[b].reshape(3, 3).T + d.xpos[b])[:, 2].min()
            za = min(za, z)
            if b in feet:
                zf = min(zf, z)
        zmin[t], zfoot[t] = za, zf
    return zmin, zfoot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="noetix_bumi", choices=sorted(FEET))
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window_s", type=float, default=1.0)
    a = ap.parse_args()
    m = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT[a.robot]))
    d = mujoco.MjData(m)
    feet = {m.body(n).id for n in FEET[a.robot]}
    x = np.loadtxt(a.inp, delimiter=",")
    zmin, zfoot = lowest_points(m, d, x, feet)
    stance = gaussian_filter1d(minimum_filter1d(zfoot, int(round(a.window_s * FPS)) | 1, mode="nearest"), 5.0, mode="nearest")
    pen = np.maximum(0.0, stance - zmin)
    lift = np.maximum(gaussian_filter1d(pen, sigma=2.0), pen)
    y = x.copy()
    y[:, 2] += lift - stance
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savetxt(a.out, y, delimiter=",", fmt="%.6f")
    print(f"stance level {stance.min():+.3f}..{stance.max():+.3f} m (range {np.ptp(stance) * 100:.1f} cm), "
          f"lifted frames {(pen > 0.01).mean():.2f}, max lift {pen.max():.3f} m -> {a.out}")


if __name__ == "__main__":
    main()
