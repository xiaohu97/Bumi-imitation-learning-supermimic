"""Render Isaac Lab rollouts recorded by ``evaluate.py --record`` to MP4: policy solid, reference as a blue ghost.

Run in the retargeting env (mujoco + imageio-ffmpeg, CPU; EGL rendering).  The recording is replayed kinematically
in MuJoCo (noetix_bumi.xml = bumi.urdf kinematics) -- the physics is Isaac's, MuJoCo only draws it.

    MUJOCO_GL=egl python supermimic/render_traj.py --traj logs/<run>/video/traj_040000.npz \
        --eval logs/<run>/ckpt_040000_eval.json --out highlights.mp4 [--clips walk1_subject1:5 ...]

Without --clips: one window per motion category, chosen on the reference (walk: from 5 s; run/sprint: fastest root;
jumps: largest root-height swing; fallAndGetUp: centred on the lowest root; dance/fight: most joint motion).
``name:start`` picks the window start by hand.  ``--ghost anchored`` draws the reference at the robot's xy and heading
(the frame of the body tracking reward); the default ``world`` shows the global drift as well.
"""

from __future__ import annotations

import argparse
import json
import os

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

XML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets/noetix_bumi/noetix_bumi.xml")
# JOINT_NAMES order of the recording (supermimic_bumi.robot, Isaac articulation order)
JOINT_NAMES = [
    "l_leg_pitch_joint", "r_leg_pitch_joint", "waist_yaw_joint", "l_leg_roll_joint", "r_leg_roll_joint",
    "l_arm_pitch_joint", "r_arm_pitch_joint", "l_leg_yaw_joint", "r_leg_yaw_joint", "l_arm_roll_joint",
    "r_arm_roll_joint", "l_knee_pitch_joint", "r_knee_pitch_joint", "l_arm_yaw_joint", "r_arm_yaw_joint",
    "l_ankle_pitch_joint", "r_ankle_pitch_joint", "l_elbow_pitch_joint", "r_elbow_pitch_joint",
    "l_ankle_roll_joint", "r_ankle_roll_joint",
]  # fmt: skip
DEFAULT_CLIPS = ["walk1_subject1", "run1_subject2", "sprint1_subject2", "jumps1_subject1", "dance2_subject1",
                 "fight1_subject2", "fallAndGetUp1_subject1"]  # fmt: skip


def scene_model(xml: str = XML) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(xml)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           rgb1=[0.22, 0.26, 0.30], rgb2=[0.15, 0.18, 0.21], width=512, height=512, mark=mujoco.mjtMark.mjMARK_EDGE,
                           markrgb=[0.4, 0.45, 0.5])  # fmt: skip
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.45, 0.55, 0.65], rgb2=[0.08, 0.1, 0.12], width=512, height=512)  # fmt: skip
    mat = spec.add_material(name="grid", texrepeat=[2, 2], texuniform=True, reflectance=0.1)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="grid", group=1)
    spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.6, 0.6, 0.6], castshadow=1)
    return spec.compile()


def pick_window(name: str, rp: np.ndarray, rq: np.ndarray, seconds: float) -> int:
    """Start frame (50 Hz, clip-relative) of the window shown for this clip."""
    n, w = len(rp), int(seconds * 50)
    if n <= w:
        return 0
    starts = np.arange(0, n - w, 25)
    win = lambda x: np.array([x[s : s + w].mean() for s in starts])  # noqa: E731
    cat = name.split("_")[0].rstrip("0123456789")
    if cat == "walk":
        return min(250, n - w)
    if cat in ("run", "sprint"):
        v = np.r_[0, np.linalg.norm(np.diff(rp[:, :2], axis=0), axis=1) * 50]
        return int(starts[win(v).argmax()])
    if cat == "jumps":
        return int(starts[np.array([np.ptp(rp[s : s + w, 2]) for s in starts]).argmax()])
    if cat == "fallAndGetUp":
        return int(np.clip(rp[:, 2].argmin() - w // 2, 0, n - w))
    qd = np.r_[0, np.abs(np.diff(rq, axis=0)).sum(1) * 50]
    return int(starts[win(qd).argmax()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--eval", default=None, help="ckpt_*_eval.json for the per-clip return in the caption")
    ap.add_argument("--clips", nargs="*", default=DEFAULT_CLIPS, help="name or name:start_seconds")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--label", default="SuperMimic Noetix Bumi 21-DoF")
    ap.add_argument("--ghost", choices=["world", "anchored"], default="world",
                    help="anchored: reference moved to the robot's xy and heading, as the body reward compares them")
    a = ap.parse_args()

    z = np.load(a.traj)
    names = [str(x) for x in z["names"]]
    per_clip = json.load(open(a.eval))["per_clip"] if a.eval else {}
    m = scene_model()
    d = mujoco.MjData(m)
    gd = mujoco.MjData(m)
    qadr = np.array([m.joint(n).qposadr[0] for n in JOINT_NAMES])
    ren = mujoco.Renderer(m, a.height, a.width)
    vis = mujoco.MjvOption()
    vis.geomgroup[:] = 0
    vis.geomgroup[1] = 1  # visual meshes only (the collision copies are group 0)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 2.3, 135.0, -12.0
    font = ImageFont.load_default(size=max(16, a.height // 34))
    small = ImageFont.load_default(size=max(13, a.height // 46))
    writer = imageio.get_writer(a.out, fps=25, codec="libx264", quality=8, macro_block_size=8)
    T = z["root_p"].shape[0]
    for spec_ in a.clips:
        name, _, st = spec_.partition(":")
        c = names.index(name)
        s0, L = int(z["clip_start"][c]), int(z["clip_len"][c])
        rp_all, rR_all, rq_all = z["ref_root_p"][s0 : s0 + L], z["ref_root_R"][s0 : s0 + L], z["ref_q"][s0 : s0 + L]
        w0 = int(float(st) * 50) if st else pick_window(name, rp_all, rq_all, a.seconds)
        steps = [t for t in range(T) if z["alive"][t, c]]
        alive_until = len(steps)  # the recording stops a clip at its end or its termination
        ok = alive_until >= L - 1
        print(f"{name:26s} window {w0 / 50:6.1f}-{(w0 + a.seconds * 50) / 50:6.1f} s of {L / 50:.1f} s, "
              f"rollout alive {alive_until / 50:.1f} s, return {per_clip.get(name, float('nan')):.3f}", flush=True)
        for t in range(w0, min(w0 + int(a.seconds * 50), alive_until), 2):  # 25 fps
            f = int(z["f"][t, c]) - s0  # reference frame of this step (clip-relative)
            d.qpos[:3], d.qpos[3:7] = z["root_p"][t, c], z["root_quat"][t, c]
            d.qpos[qadr] = z["q"][t, c]
            mujoco.mj_kinematics(m, d)
            gp, gR = rp_all[f].astype(np.float64), rR_all[f].astype(np.float64)
            if a.ghost == "anchored":
                Rr = d.xmat[1].reshape(3, 3)
                dyaw = np.arctan2(Rr[1, 0], Rr[0, 0]) - np.arctan2(gR[1, 0], gR[0, 0])
                cz, sz = np.cos(dyaw), np.sin(dyaw)
                gR = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ gR
                gp = np.r_[d.qpos[:2], gp[2]]
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, gR.reshape(-1))
            gd.qpos[:3], gd.qpos[3:7] = gp, q
            gd.qpos[qadr] = rq_all[f]
            mujoco.mj_kinematics(m, gd)
            cam.lookat[:] = d.qpos[:3] * np.array([1, 1, 0]) + np.array([0, 0, 0.42])
            ren.update_scene(d, cam, scene_option=vis)
            n0 = ren.scene.ngeom
            mujoco.mjv_addGeoms(m, gd, vis, mujoco.MjvPerturb(), int(mujoco.mjtCatBit.mjCAT_DYNAMIC), ren.scene)
            for gi in range(n0, ren.scene.ngeom):
                ren.scene.geoms[gi].rgba[:] = (0.25, 0.55, 1.0, 0.3)
            img = Image.fromarray(ren.render())
            dr = ImageDraw.Draw(img)
            dr.text((16, 12), f"{name}    t = {t / 50:5.1f} s", fill=(255, 255, 255), font=font)
            dr.text((16, 18 + a.height // 26), f"{a.label}    solid: policy (Isaac Lab PhysX rollout)    blue: reference"
                    + ("  (moved to the robot's xy + heading)" if a.ghost == "anchored" else ""),
                    fill=(220, 220, 220), font=small)  # fmt: skip
            dr.text((16, 22 + a.height // 26 + a.height // 40),
                    f"{os.path.basename(a.traj).replace('traj_', 'ckpt ').replace('.npz', '')}    clip return "
                    f"{per_clip.get(name, float('nan')):.3f}    {'tracked to the end' if ok else 'terminated'}",
                    fill=(220, 220, 220), font=small)  # fmt: skip
            writer.append_data(np.asarray(img))
    writer.close()
    ren.close()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
