"""Side-by-side demo video: source video + human estimate + retargeted reference (left), SuperMimic rollout (right).

Run in the retargeting env (mujoco, cv2, imageio-ffmpeg; EGL).  --video/--overlay/--crop come from the human pose
estimator's run, --traj from ``rollout_clip.py``; the robot is replayed kinematically in MuJoCo from the Isaac Lab
recording (physics is Isaac's).  Output frames are aligned 1:1 with the source video (30 fps).

    MUJOCO_GL=egl python supermimic/render_video_demo.py --traj traj.npz --video <source.mp4> --overlay <mesh_overlay.mp4> \
        --crop <cropped_source.mp4> --segments examples/langzi/segments.json --plain --out demo.mp4

Layout (1920x1080): top-left source video, bottom-left mesh overlay (cropped panel) + retargeted reference (blue
robot), right column the policy rollout.  Gap frames (title cards, close-ups) show the cropped
source without a mesh; the reference there is the interpolated transition.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_traj import JOINT_NAMES as BUMI_JOINT_NAMES  # noqa: E402
from render_traj import XML  # noqa: E402
from render_traj import scene_model  # noqa: E402

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
SC = 2  # Noto Sans CJK SC face in the collection
W, H = 1920, 1080
CTRL_HZ = 50.0


def font(size):
    return ImageFont.truetype(FONT, size, index=SC)


def label(img: Image.Image, xy, text, size=26, fill=(255, 255, 255), bg=(0, 0, 0, 150), pad=6):
    dr = ImageDraw.Draw(img, "RGBA")
    f = font(size)
    x0, y0, x1, y1 = dr.textbbox(xy, text, font=f)
    dr.rectangle((x0 - pad, y0 - pad, x1 + pad, y1 + pad), fill=bg)
    dr.text(xy, text, font=f, fill=fill)


def fit(frame_bgr, w, h):
    """Letterbox a BGR frame into w x h (RGB)."""
    fh, fw = frame_bgr.shape[:2]
    s = min(w / fw, h / fh)
    nw, nh = int(round(fw * s)), int(round(fh * s))
    out = np.zeros((h, w, 3), np.uint8)
    r = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    y, x = (h - nh) // 2, (w - nw) // 2
    out[y : y + nh, x : x + nw] = r[..., ::-1]
    return out


class RobotView:
    def __init__(self, xml, joint_names, w, h, tint=None, distance=1.75, elevation=-8.0, azimuth=90.0, look_z=0.40):
        self.m = scene_model(xml)
        if tint is not None:
            for g in range(self.m.ngeom):
                if self.m.geom_group[g] == 1 and self.m.geom_bodyid[g] > 0:
                    self.m.geom_rgba[g] = tint
        self.d = mujoco.MjData(self.m)
        self.qadr = np.array([self.m.joint(n).qposadr[0] for n in joint_names])
        self.look_z = look_z
        self.ren = mujoco.Renderer(self.m, h, w)
        self.vis = mujoco.MjvOption()
        self.vis.geomgroup[:] = 0
        self.vis.geomgroup[1] = 1
        self.cam = mujoco.MjvCamera()
        self.cam.distance, self.cam.elevation, self.cam.azimuth = distance, elevation, azimuth
        self.look = None

    def render(self, p, quat_wxyz, q):
        self.d.qpos[:3], self.d.qpos[3:7] = p, quat_wxyz
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        tgt = np.array([p[0], p[1], self.look_z])
        self.look = tgt if self.look is None else self.look + 0.06 * (tgt - self.look)  # ~0.5 s camera lag
        self.cam.lookat[:] = self.look
        self.ren.update_scene(self.d, self.cam, scene_option=self.vis)
        return self.ren.render()


def mat_to_quat(Rm):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(Rm, dtype=np.float64).reshape(-1))
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--zoom", type=float, default=1.0, help="camera distance divisor on top of the root-height scaling")
    ap.add_argument("--video", required=True)
    ap.add_argument("--overlay", required=True)
    ap.add_argument("--crop", required=True)
    ap.add_argument("--segments", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--plain", action="store_true", help="neutral captions: no method, checkpoint or plant names")
    a = ap.parse_args()

    z = np.load(a.traj)
    summ = json.loads(str(z["summary"]))
    clip = str(z["names"][0])
    s0, L = int(z["clip_start"][0]), int(z["clip_len"][0])
    ckpt = os.path.basename(str(z["ckpt"])).replace(".pt", "")
    seg = json.load(open(a.segments))
    gaps, moves = seg["gaps"], seg["moves"]
    root_p, root_q, qs = z["root_p"][:, 0], z["root_quat"][:, 0], z["q"][:, 0]
    tracking, fr = z["tracking"][:, 0], z["f"][:, 0] - s0
    ref_p, ref_R, ref_q = z["ref_root_p"][s0 : s0 + L], z["ref_root_R"][s0 : s0 + L], z["ref_q"][s0 : s0 + L]
    T = len(root_p)

    # fixed camera azimuth: in front of the dancer (mean reference heading over the clip, + 180 deg)
    hx = ref_R[:, :2, 0].mean(0)
    az = np.degrees(np.arctan2(hx[1], hx[0])) + 180.0
    # distances tuned at the root height 0.47 m, scaled by the reference root height
    k = float(np.median(ref_p[:, 2])) / 0.47
    kd = k / a.zoom
    names = [str(n) for n in z["joint_names"]] if "joint_names" in z else BUMI_JOINT_NAMES  # recording's joint order
    sim_view = RobotView(XML, names, 960, 1080 - 60, azimuth=az, distance=2.1 * kd, look_z=0.40 * k)
    ref_view = RobotView(XML, names, 560, 540, tint=(0.45, 0.65, 1.0, 1.0), azimuth=az, distance=1.75 * kd, look_z=0.40 * k)

    cap_v, cap_o, cap_c = cv2.VideoCapture(a.video), cv2.VideoCapture(a.overlay), cv2.VideoCapture(a.crop)
    fps = cap_v.get(cv2.CAP_PROP_FPS)
    n = int(cap_v.get(cv2.CAP_PROP_FRAME_COUNT))
    if a.max_frames:
        n = min(n, a.max_frames)
    terms = summ[clip]["terminations"]
    term_t = np.array([e["t"] for e in terms]) if terms else np.zeros(0)
    writer = imageio.get_writer(a.out, fps=fps, codec="libx264", quality=None, macro_block_size=8,
                                ffmpeg_params=["-crf", "18", "-preset", "medium"])  # fmt: skip
    f_title = font(30)
    res = summ[clip]
    stats = (f"整段 {res['seconds']:.1f} s · 终止 {len(terms)} 次 · 归一化回报 {res['norm_return']:.3f} · "
             f"身体位置 RMS 误差 {res['mean_body_pos_err_m'] * 100:.1f} cm")
    txt = dict(human="WHAM 人体估计", ref="GMR 重定向参考 (Bumi)", sim="SuperMimic 策略 · Isaac Lab PhysX 仿真",
               gap_h="镜头切换/特写：不做估计", gap_r="参考：插值过渡",
               fail="跟踪失败 (终止条件触发) → 0.6 s 后从参考状态重置", back="已重置，继续跟踪")  # fmt: skip
    if a.plain:
        txt = dict(human="识别到的人体动作", ref="机器人参考动作", sim="仿真效果", gap_h="镜头切换/特写",
                   gap_r="过渡动作", fail="摔倒 → 0.6 s 后重置", back="已重置")  # fmt: skip
        stats = f"全程 {res['seconds']:.1f} s · 摔倒 {len(terms)} 次"
    for k in range(n):
        ok_v, fv = cap_v.read()
        ok_o, fo = cap_o.read()
        ok_c, fc = cap_c.read()
        if not ok_v:
            break
        in_gap = any(g0 <= k < g1 for g0, g1 in gaps)
        move = next((m for m, m0, m1 in moves if m0 <= k < m1), None)
        t = k / fps
        s = min(int(round(t * CTRL_HZ)), T - 1)
        canvas = Image.new("RGB", (W, H), (18, 20, 24))
        # top-left: source video
        canvas.paste(Image.fromarray(fit(fv, 960, 540)), (0, 0))
        label(canvas, (14, 10), "原视频", 26)
        # bottom-left: mesh overlay (or the plain crop in gaps) + reference robot
        left = fc if (in_gap or not ok_o) else fo
        if left is fo and ok_c:
            left = fo.copy()
            left[:68] = fc[:68]  # the pipeline's FPS/latency HUD (rows 24-59)
        canvas.paste(Image.fromarray(fit(left, 400, 540)), (0, 540))
        rf = min(int(round(t * CTRL_HZ)), L - 1)
        img_ref = ref_view.render(ref_p[rf], mat_to_quat(ref_R[rf]), ref_q[rf])
        canvas.paste(Image.fromarray(img_ref), (400, 540))
        label(canvas, (14, 552), txt["human"], 22)
        label(canvas, (414, 552), txt["ref"], 22)
        if in_gap:
            label(canvas, (14, 1040), txt["gap_h"], 20, fill=(255, 220, 120))
            label(canvas, (414, 1040), txt["gap_r"], 20, fill=(255, 220, 120))
        # right: policy rollout in Isaac Lab
        img_sim = sim_view.render(root_p[s], root_q[s], qs[s])
        canvas.paste(Image.fromarray(img_sim), (960, 60))
        dr = ImageDraw.Draw(canvas)
        dr.rectangle((960, 0, W, 60), fill=(10, 12, 16))
        dr.text((976, 12), txt["sim"], font=f_title, fill=(255, 255, 255))
        y_move = 76
        if not a.plain:
            label(canvas, (976, 76), f"{ckpt}  延迟 2 步  无随机化  t = {t:5.1f} s", 20, fill=(220, 220, 220))
            y_move = 112
        if move:
            label(canvas, (976, y_move), f"动作：{move}", 24, fill=(255, 255, 255))
        if not tracking[s]:
            label(canvas, (976, 990), txt["fail"], 22, fill=(255, 110, 110))
        elif len(term_t) and np.any((t - term_t > 0.6) & (t - term_t < 1.6)):
            label(canvas, (976, 990), txt["back"], 22, fill=(255, 200, 120))
        label(canvas, (976, 1040), stats, 20, fill=(200, 230, 200))
        # thin separators
        dr.line((960, 0, 960, H), fill=(60, 64, 72), width=2)
        dr.line((0, 540, 960, 540), fill=(60, 64, 72), width=2)
        dr.line((400, 540, 400, H), fill=(60, 64, 72), width=2)
        writer.append_data(np.asarray(canvas))
        if k % 300 == 0:
            print(f"  frame {k}/{n}", flush=True)
    writer.close()
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
