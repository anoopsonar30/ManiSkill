from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import sapien
import torch
from scipy.spatial.transform import Rotation as R

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs import Pose

from .panda import Panda

if TYPE_CHECKING:
    from mani_skill.envs.sapien_env import BaseEnv

# urdf -> sapien camera pose
_base_rotation = R.from_euler('y', -np.pi / 2)
_roll_correction = R.from_euler('x', -np.pi / 2)
_combined = _base_rotation * _roll_correction
_quat_xyzw = _combined.as_quat()  # scipy: [x, y, z, w]
CAMERA_POSE_Q = [_quat_xyzw[3], _quat_xyzw[0], _quat_xyzw[1], _quat_xyzw[2]]  # sapien: [w, x, y, z]


INTRINSICS_WRIST_REAL = np.array([
        [436.40466309,   0.          , 322.48730469],
        [  0.          , 435.8762207 , 245.15719604],
        [  0.          , 0.          , 1.        ]
    ])
INTRINSICS_WRIST_REAL[0, -1] -= (640 - 480) / 2
INTRINSICS_WRIST_REAL[:-1] *= 224.0 / 480.0 


@register_agent()
class PandaWristCam(Panda):
    """Panda arm robot with wrist camera attached to gripper.
    
    Uses fr3_agilex_gripper_wristcam.urdf which includes a wrist_camera_origin
    link positioned from the real robot's camera calibration.
    """

    uid = "panda_wristcam"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/fr3/fr3_agilex_gripper_wristcam.urdf"

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="hand_camera",
                pose=sapien.Pose(p=[0, 0, 0], q=CAMERA_POSE_Q),
                width=224,
                height=224,
                intrinsic=torch.from_numpy(INTRINSICS_WRIST_REAL),
                near=0.01,
                far=100,
                mount=self.robot.links_map["wrist_camera_origin"],
            )
        ]

    def randomize_wrist_camera_pose(
        self,
        env: "BaseEnv",
        env_idx: torch.Tensor,
        position_range: float = 0.003,  # ±3mm
        angle_range_deg: float = 1.0,   # ±1 degree
    ) -> None:
        """Randomize the wrist camera's local pose with small perturbations.
        
        Applies per-environment random perturbations to the hand_camera sensor's
        local pose relative to its mount (wrist_camera_origin link).
        
        Args:
            env: The environment instance (needed to access sensors)
            env_idx: Tensor of environment indices being reset
            position_range: Max position perturbation in meters (default ±3mm)
            angle_range_deg: Max rotation perturbation in degrees (default ±1°)
        """
        b = len(env_idx)
        
        # Generate random rotation and position perturbations
        max_angle = np.deg2rad(angle_range_deg)
        axes = np.random.randn(b, 3)
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        angles = np.random.uniform(-max_angle, max_angle, size=(b, 1))
        rotvecs = axes * angles
        delta_quats = R.from_rotvec(rotvecs).as_quat()  
        delta_quats = np.roll(delta_quats, 1, axis=1)   # scipy: [x, y, z, w] -> sapien: [w, x, y, z]
        delta_positions = np.random.uniform(-position_range, position_range, size=(b, 3))
        
        base_pose = sapien.Pose(p=[0, 0, 0], q=CAMERA_POSE_Q)
        
        hand_camera = env._sensors.get("hand_camera")
        if hand_camera is None:
            return
        
        # Access the underlying render cameras and set per-env local poses
        render_cameras = hand_camera.camera._render_cameras
        env_idx_np = env_idx.cpu().numpy()
        
        for i, idx in enumerate(env_idx_np):
            perturb_pose = sapien.Pose(
                p=delta_positions[i].astype(np.float32),
                q=delta_quats[i].astype(np.float32),
            )
            # Apply perturbation to base pose
            new_pose = base_pose * perturb_pose
            render_cameras[idx].local_pose = new_pose
        
        # Clear cached local pose so it gets recomputed
        hand_camera.camera._cached_local_pose = None
