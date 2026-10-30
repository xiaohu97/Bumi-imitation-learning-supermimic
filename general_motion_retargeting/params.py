import pathlib

HERE = pathlib.Path(__file__).parent
IK_CONFIG_ROOT = HERE / "ik_configs"
ASSET_ROOT = HERE / ".." / "assets"

ROBOT_XML_DICT = {
    "noetix_bumi": ASSET_ROOT / "noetix_bumi" / "noetix_bumi.xml",
}

IK_CONFIG_DICT = {
    # offline_retarget_smplx_npz.py: per-frame SMPL-X skeleton
    "smplx": {
        "noetix_bumi": IK_CONFIG_ROOT / "smplx_to_noetix_bumi.json",
    },
    # bvh_to_robot_batch_headless.py: LAFAN1 training clips
    "bvh_lafan1": {
        "noetix_bumi": IK_CONFIG_ROOT / "bvh_lafan1_to_noetix_bumi.json",
    },
}
