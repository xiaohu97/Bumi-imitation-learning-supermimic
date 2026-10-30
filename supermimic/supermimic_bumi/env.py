"""Batched Noetix Bumi motion-tracking simulator on Isaac Lab 2.x / PhysX.

Adapted from supermimic_vibed (MIT, Chong Zhang) ``env.py`` (Isaac Lab 3 + Newton/MuJoCo-Warp there).
A thin wrapper: apply the joint setpoint, step PhysX (4 x 5 ms), label
the transition with the analytic reward/termination of :mod:`mdp`, reset episodes to reference states drawn
by return-based bin sampling (paper Appendix D.1).

Plant = noetix_bumi_lab Mimic-Bumi-v0: BUMI_21DOF_CFG (bumi.urdf, implicit PD with their gains / limits,
self-collisions on), one command delay per env drawn from 0-4 physics steps, setpoints clipped to the URDF
limits, per-joint action scale 0.25 * effort / stiffness.

Must be imported after the Isaac Sim app is launched.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab.actuators import ImplicitActuator, ImplicitActuatorCfg
from isaaclab.utils import DelayBuffer

from .mdp import MAX_STEP_REWARD, MDP, XI_COM, XI_DELAY, XI_DIM, XI_OFF, flatten_state
from .motion import BIN_FRAMES, MotionLib
from .robot import (
    ACTION_SCALE,
    ACTUATORS,
    ANKLE_ARMATURE,
    DECIMATION,
    DEFAULT_JOINT_POS,
    JOINT_NAMES,
    MAX_DELAY,
    MIN_DELAY,
    NUM_J,
    SIM_DT,
    TORSO_BODY,
    URDF_PATH,
    joint_offset_ranges,
)
from .rotations import mat_to_quat, quat_to_mat, so3_exp

FRICTION_RANGE = (0.3, 1.2)
FRICTION_BUCKETS = 64  # PhysX material count stays bounded, as in Isaac Lab's material event


class DelayedImplicitActuator(ImplicitActuator):
    """Implicit PD with delayed setpoints (noetix_bumi_lab ``actuators/delayed_implicit_actuator_pd.py``, BSD-3)."""

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)

    def reset(self, env_ids):
        super().reset(env_ids)
        n = self._num_envs if env_ids is None or env_ids == slice(None) else len(env_ids)
        lag = torch.randint(self.cfg.min_delay, self.cfg.max_delay + 1, (n,), dtype=torch.int, device=self._device)
        for buf in (self.positions_delay_buffer, self.velocities_delay_buffer, self.efforts_delay_buffer):
            buf.set_time_lag(lag, env_ids)
            buf.reset(env_ids)

    def compute(self, control_action, joint_pos, joint_vel):
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class DelayedImplicitActuatorCfg(ImplicitActuatorCfg):
    class_type: type = DelayedImplicitActuator
    min_delay: int = 0
    max_delay: int = 0


def robot_cfg() -> ArticulationCfg:
    """BUMI_21DOF_CFG (noetix_bumi_lab assets/robots/bumi/bumi.py), delays set to MIN/MAX_DELAY."""
    actuators = {}
    for group, joints in ACTUATORS.items():
        pick = lambda i: {pat: v[i] for pat, v in joints.items()}  # noqa: E731
        actuators[group] = DelayedImplicitActuatorCfg(
            joint_names_expr=list(joints), effort_limit_sim=pick(0), velocity_limit_sim=pick(1), stiffness=pick(2),
            damping=pick(3), armature={pat: ANKLE_ARMATURE for pat in joints} if group == "feet" else None,
            min_delay=MIN_DELAY, max_delay=MAX_DELAY,
        )  # fmt: skip
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            fix_base=False,
            asset_path=URDF_PATH,
            activate_contact_sensors=True,
            replace_cylinders_with_capsules=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, retain_accelerations=False, linear_damping=0.0, angular_damping=0.0,
                max_linear_velocity=1000.0, max_angular_velocity=1000.0, max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
            ),
        ),  # fmt: skip
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.65), joint_pos={n: DEFAULT_JOINT_POS.get(n, 0.0) for n in JOINT_NAMES}
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators=actuators,
    )


@configclass
class TrackingSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        # multiply with 1.0: the pair friction is the robot's randomized value
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply", static_friction=1.0, dynamic_friction=1.0
        ),
    )
    robot: ArticulationCfg = robot_cfg()
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DistantLightCfg(intensity=3000.0))


def make_sim_cfg(device: str) -> sim_utils.SimulationCfg:
    cfg = sim_utils.SimulationCfg(dt=SIM_DT, render_interval=DECIMATION, device=device)
    cfg.physx.gpu_max_rigid_patch_count = 10 * 2**15
    return cfg


class TrackingSim:
    def __init__(self, sim: sim_utils.SimulationContext, num_envs: int, motion: MotionLib, mdp: MDP, gamma: float = 0.99,
                 eval_mode: bool = False, max_episode_steps: int = 500, reset_noise: bool = True, eval_delay: int = 2):
        self.sim = sim
        self.device = sim.device
        self.scene = InteractiveScene(TrackingSceneCfg(num_envs=num_envs, env_spacing=2.5))
        sim.reset()
        self.robot = self.scene["robot"]
        names = list(self.robot.joint_names)
        assert sorted(names) == sorted(JOINT_NAMES), names
        perm = [names.index(n) for n in JOINT_NAMES]  # canonical -> Isaac joint index
        self.jperm = torch.tensor(perm, device=sim.device)
        self.jids = None if perm == list(range(NUM_J)) else perm
        if self.jids is not None:
            print(f"[supermimic_bumi] Isaac joint order differs from JOINT_NAMES, mapping by name: {names}", flush=True)
        self.N = num_envs
        self.motion, self.mdp, self.gamma = motion, mdp, gamma
        self.eval_mode = eval_mode
        self.max_episode_steps = max_episode_steps if not eval_mode else 10**9
        self.reset_noise = reset_noise and not eval_mode
        dev = self.device
        self.default_q = self.robot.data.default_joint_pos[0, self.jperm].clone()
        self.action_scale = torch.tensor(ACTION_SCALE, device=dev)
        assert torch.allclose(self.default_q, mdp.default_q, atol=1e-5), (self.default_q, mdp.default_q)
        self.q_lo, self.q_hi = mdp.kin.q_lo, mdp.kin.q_hi
        self.off_range = torch.tensor(joint_offset_ranges(), device=dev)
        self.torso_id = self.robot.find_bodies(TORSO_BODY)[0][0]
        self.eval_delay = eval_delay

        # ---- sim context xi = [friction, joint offsets, torso com offset, delay] ----
        self.xi = torch.zeros(self.N, XI_DIM, device=dev)
        if eval_mode:
            self.xi[:, 0] = 1.0
        else:
            b = torch.randint(0, FRICTION_BUCKETS, (self.N,), device=dev).float()
            self.xi[:, 0] = FRICTION_RANGE[0] + (FRICTION_RANGE[1] - FRICTION_RANGE[0]) * b / (FRICTION_BUCKETS - 1)
            com = torch.stack(
                (
                    torch.empty(self.N, device=dev).uniform_(-0.025, 0.025),
                    torch.empty(self.N, device=dev).uniform_(-0.05, 0.05),
                    torch.empty(self.N, device=dev).uniform_(-0.05, 0.05),
                ),
                -1,
            )
            self.xi[:, XI_COM] = com
            coms = self.robot.root_physx_view.get_coms().clone()
            coms[:, self.torso_id, :3] += com.cpu()
            self.robot.root_physx_view.set_coms(coms, torch.arange(self.N, device="cpu"))
        mats = self.robot.root_physx_view.get_material_properties().clone()
        mu = self.xi[:, 0].cpu()
        mats[:, :, 0] = mu[:, None]
        mats[:, :, 1] = mu[:, None]
        mats[:, :, 2] = 0.0
        self.robot.root_physx_view.set_material_properties(mats, torch.arange(self.N, device="cpu"))

        # ---- episode bookkeeping ----
        self.f = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_step = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.a_prev = torch.zeros(self.N, NUM_J, device=dev)
        self.start_bin = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_disc_ret = torch.zeros(self.N, device=dev)
        self.ep_ret = torch.zeros(self.N, device=dev)
        self.G_max = MAX_STEP_REWARD / (1 - gamma)
        self.bin_G = torch.full((motion.num_bins,), 0.0, device=dev)
        self.bin_w_avg = torch.full((motion.num_bins,), 1.0 / motion.num_bins, device=dev)
        self.done_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.failed_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.reset(torch.arange(self.N, device=dev))

    # ------------------------------------------------------------------ state
    def state(self):
        d = self.robot.data
        pose = d.root_link_pose_w
        vel = d.root_link_vel_w
        p = pose[:, :3] - self.scene.env_origins
        R = quat_to_mat(pose[:, 3:7])  # Isaac Lab 2.x quaternions are wxyz
        return (p, R, vel[:, :3].clone(), vel[:, 3:6].clone(), d.joint_pos[:, self.jperm], d.joint_vel[:, self.jperm])

    # ------------------------------------------------------------------ reset
    def sample_frames(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.motion
        soft = torch.softmax(-25.0 * self.bin_G / self.G_max, 0)
        self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
        probs = 0.5 / m.num_bins + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        bins = torch.multinomial(probs, n, replacement=True)
        start = m.bin_start[bins]
        end = torch.minimum(start + BIN_FRAMES, m.clip_end[m.bin_clip[bins]])  # keep >= 1 step before the clip end
        f = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        return f, bins

    def reset(self, env_ids: torch.Tensor, frames: torch.Tensor | None = None):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        dev = self.device
        if frames is None:
            frames, bins = self.sample_frames(n)
        else:
            bins = self.motion.frame_bin[frames]
        p, R, v, w, q, qd = (x.clone() for x in self.motion.ref_state(frames))
        if self.reset_noise:
            u = lambda lo, hi: torch.empty(n, device=dev).uniform_(lo, hi)  # noqa: E731
            p = p + torch.stack((u(-0.05, 0.05), u(-0.05, 0.05), u(-0.01, 0.01)), -1)
            R = so3_exp(torch.stack((u(-0.1, 0.1), u(-0.1, 0.1), u(-0.2, 0.2)), -1)) @ R
            v = v + torch.stack((u(-0.5, 0.5), u(-0.5, 0.5), u(-0.2, 0.2)), -1)
            w = w + torch.stack((u(-0.52, 0.52), u(-0.52, 0.52), u(-0.78, 0.78)), -1)
            q = q + torch.empty_like(q).uniform_(-0.1, 0.1)
        q = torch.clamp(q, self.q_lo, self.q_hi)
        pose = torch.cat((p + self.scene.env_origins[env_ids], mat_to_quat(R)), -1)
        # actuator reset first (draws per-group delays, clears delay history), then the state
        self.robot.reset(env_ids)
        self.robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim(torch.cat((v, w), -1), env_ids=env_ids)
        self.robot.write_joint_state_to_sim(q, qd, joint_ids=self.jids, env_ids=env_ids)
        self.robot.set_joint_position_target(q, joint_ids=self.jids, env_ids=env_ids)
        # one delay per env shared by all actuator groups, recorded in xi
        lag = torch.full((n,), self.eval_delay, dtype=torch.int, device=dev) if self.eval_mode else \
            torch.randint(MIN_DELAY, MAX_DELAY + 1, (n,), dtype=torch.int, device=dev)
        for act in self.robot.actuators.values():
            for buf in (act.positions_delay_buffer, act.velocities_delay_buffer, act.efforts_delay_buffer):
                buf.set_time_lag(lag, env_ids)
        self.xi[env_ids, XI_DELAY] = lag.float()
        self.f[env_ids] = frames
        self.ep_step[env_ids] = 0
        self.a_prev[env_ids] = 0.0
        self.start_bin[env_ids] = bins
        self.ep_disc_ret[env_ids] = 0.0
        self.ep_ret[env_ids] = 0.0
        if not self.eval_mode:
            self.xi[env_ids, XI_OFF] = (2 * torch.rand(n, NUM_J, device=dev) - 1) * self.off_range

    # ------------------------------------------------------------------ step
    def step(self, a: torch.Tensor) -> dict:
        S = self.state()
        f = self.f.clone()
        xi = self.xi.clone()
        a_prev = self.a_prev.clone()
        ep_step = self.ep_step.clone()
        # commanded setpoint in the robot's (offset) encoder frame, clipped like the deploy controller
        cmd = torch.clamp(self.default_q + self.action_scale * a, self.q_lo, self.q_hi)
        self.robot.set_joint_position_target(cmd + xi[:, XI_OFF], joint_ids=self.jids)
        for _ in range(DECIMATION):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(SIM_DT)
        S1 = self.state()
        f1 = f + 1
        r, terms = self.mdp.reward(S1, f1, a, a_prev, terms=True)
        term = self.mdp.termination(S1, f1)
        clip_end = self.motion.clip_end[self.motion.clip_id[f]]
        timeout = (f1 >= clip_end) | (ep_step + 1 >= self.max_episode_steps)
        done = term | timeout
        out = {
            "s": flatten_state(S), "a": a.clone(), "a_prev": a_prev, "f": f, "s1": flatten_state(S1), "r": r,
            "term": term, "done": done, "ep_step": ep_step, "xi": xi, "terms": terms,
        }  # fmt: skip
        self.ep_disc_ret += (self.gamma ** ep_step.float()) * r
        self.ep_ret += r
        self.f = f1
        self.ep_step = ep_step + 1
        self.a_prev = a.clone()
        if self.eval_mode:
            self.failed_eval |= term & ~self.done_eval
            self.done_eval |= done
            return out
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            # bin return estimate; truncated episodes bootstrap with their mean reward
            T = self.ep_step[ids].float()
            boot = torch.where(term[ids], 0.0, (self.gamma**T) * (self.ep_ret[ids] / T) / (1 - self.gamma))
            G = self.ep_disc_ret[ids] + boot
            b = self.start_bin[ids]
            self.bin_G.index_put_((b,), 0.8 * self.bin_G[b] + 0.2 * G)
            out["ep_len"] = T
            out["ep_ret_norm"] = self.ep_ret[ids] / (T * MAX_STEP_REWARD)
            self.reset(ids)
        return out
