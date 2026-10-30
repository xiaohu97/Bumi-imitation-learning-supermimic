"""Ground GMR retargets of LAFAN1 on Noetix Bumi (needs mujoco).

GMR leaves the stance foot ~3 cm above the floor and, on lying segments (fallAndGetUp, ground*),
pushes the robot body into the floor.  Per clip:

1. lower the root by the median per-frame lowest foot point (stance level -> 0);
2. per frame, lift the root by the remaining penetration of the lowest mesh point, lightly smoothed
   (sigma 2 frames, never below the frame's own penetration).  No dilation: lying segments penetrate
   up to 0.4 m, and dilating that lifts the neighbouring standing frames off the floor.

    python retarget/ground_retargets.py --in_dir <gmr csv dir> --out_dir <dir>
"""

import argparse
import glob
import os
from multiprocessing import Pool

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d

XML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets/noetix_bumi/noetix_bumi.xml")
FPS = 30


def _lowest(m, d, verts, feet):
    zz, zf = [], []
    for g, v in verts:
        b = m.geom_bodyid[g]
        Rb = d.xmat[b].reshape(3, 3)
        Rg = np.zeros(9)
        mujoco.mju_quat2Mat(Rg, m.geom_quat[g])
        z = (v @ (Rb @ Rg.reshape(3, 3)).T + d.xpos[b] + Rb @ m.geom_pos[g])[:, 2].min()
        zz.append(z)
        if b in feet:
            zf.append(z)
    return min(zz), min(zf)


def run(job):
    src, out_dir = job
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    verts = []
    for g in range(m.ngeom):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            verts.append((g, m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]))
    feet = {m.body(f"{s}_ankle_roll_link").id for s in ("l", "r")}
    x = np.loadtxt(src, delimiter=",")
    zmin, zfoot = np.zeros(len(x)), np.zeros(len(x))
    for t in range(len(x)):
        d.qpos[:3], d.qpos[3:7], d.qpos[7:] = x[t, :3], x[t, [6, 3, 4, 5]], x[t, 7:]
        mujoco.mj_kinematics(m, d)
        zmin[t], zfoot[t] = _lowest(m, d, verts, feet)
    c = np.median(zfoot)
    pen = np.maximum(0.0, -(zmin - c))
    lift = np.maximum(gaussian_filter1d(pen, sigma=2.0), pen)  # smoothing must not reintroduce penetration
    y = x.copy()
    y[:, 2] += lift - c
    name = os.path.basename(src)
    np.savetxt(os.path.join(out_dir, name), y, delimiter=",", fmt="%.6f")
    return name[:-4], c, pen.max(), (pen > 0.01).mean()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    jobs = [(f, a.out_dir) for f in sorted(glob.glob(os.path.join(a.in_dir, "*.csv")))]
    with Pool(16) as p:
        for name, c, pmax, pfrac in p.imap(run, jobs):
            print(f"{name:28s} stance {c:+.3f} m  max lift {pmax:.3f} m  lifted frames {pfrac:.2f}", flush=True)
