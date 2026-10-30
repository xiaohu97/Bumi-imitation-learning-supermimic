"""SMPL-X body pose -> joint frames for GMR, without skinning a mesh."""
import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES

# Y-up (skeleton world) -> Z-up (MuJoCo world): R_x(+90 deg).
YUP_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


class FastSMPLXAdapter:
    """Replaces the legacy tail_params_to_smplx_frame() (a full SMPL-X forward
    over 10475 vertices plus a 55-joint scipy loop, ~5 ms, ~22 ms under GIL
    contention) with a 22-joint forward-kinematics chain (~0.3 ms).

    SMPL-X joint positions only depend on the rest joints J(betas), which are
    linear in betas, and on the joint rotations; pose blend shapes move vertices,
    not joints. Every GMR IK config only references the 22 body joints.
    """

    NUM_BODY_JOINTS = 22

    def __init__(self, smplx_folder, coord_fix="yup_to_zup"):
        body_model = smplx.create(str(smplx_folder), "smplx", gender="neutral", use_pca=False)
        n = self.NUM_BODY_JOINTS
        with torch.no_grad():
            j_reg = body_model.J_regressor[:n].double()
            self._j0 = (j_reg @ body_model.v_template.double()).numpy()
            self._jdirs = torch.einsum("jv,vcb->jcb", j_reg, body_model.shapedirs.double()).numpy()
        self.num_betas = int(self._jdirs.shape[-1])
        self.parents = body_model.parents[:n].numpy()
        self.names = list(JOINT_NAMES[:n])
        self.coord = YUP_TO_ZUP if coord_fix in ("auto", "yup_to_zup") else np.eye(3)

    def rest_joints(self, betas):
        b = np.zeros(self.num_betas)
        betas = np.asarray(betas, dtype=np.float64).reshape(-1)[:self.num_betas]
        b[:betas.shape[0]] = betas
        return self._j0 + self._jdirs @ b

    def __call__(self, body_rotmats, root_rotmat, transl, betas):
        """(positions (22, 3), quaternions wxyz (22, 4)) in the Z-up world."""
        n = self.NUM_BODY_JOINTS
        rest = self.rest_joints(betas)
        local = np.empty((n, 3, 3))
        local[0] = self.coord @ np.asarray(root_rotmat, dtype=np.float64)
        local[1:] = np.asarray(body_rotmats, dtype=np.float64)[:n - 1]
        # Snap to the nearest rotations (the legacy path did this implicitly via
        # axis-angle): GPU-side products can leave the inputs slightly non-orthonormal.
        local = R.from_matrix(local).as_matrix()
        glob_r = np.empty((n, 3, 3))
        glob_t = np.empty((n, 3))
        glob_r[0], glob_t[0] = local[0], rest[0]
        for i in range(1, n):
            p = self.parents[i]
            glob_r[i] = glob_r[p] @ local[i]
            glob_t[i] = glob_r[p] @ (rest[i] - rest[p]) + glob_t[p]
        positions = glob_t + self.coord @ np.asarray(transl, dtype=np.float64)
        quats = R.from_matrix(glob_r).as_quat()[:, [3, 0, 1, 2]]
        return positions, quats

    def as_dict(self, positions, quats):
        """GMR's human-frame format: {joint name: (position, quaternion wxyz)}."""
        return {name: (positions[i], quats[i]) for i, name in enumerate(self.names)}
