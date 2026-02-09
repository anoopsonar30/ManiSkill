import sapien
import torch

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env

from .peg_insertion_side import PegInsertionSideEnv, INTRINSICS_REAL


@register_env("PegInsertionSide-Global", max_episode_steps=150)
class PegInsertionSideGlobalAndWristEnv(PegInsertionSideEnv):
    """
    Same as PegInsertionSide-v1 but with an additional global base_camera
    mounted on cam_mount (with per-episode pose randomization).

    Sensors:
      - base_camera: global camera (from _default_sensor_configs)
      - hand_camera: wrist camera (from panda_wristcam agent)
    """

    SUPPORTED_ROBOTS = ["panda_wristcam"]

    def __init__(self, *args, robot_uids="panda_wristcam", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        return CameraConfig(
            "base_camera",
            sapien.Pose(),  # Identity local pose; world pose set via cam_mount
            224, 224,
            intrinsic=torch.from_numpy(INTRINSICS_REAL),
            near=0.01, far=100,
            mount=self.cam_mount,
        )
