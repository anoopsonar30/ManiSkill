from typing import Any, Dict, Union
from pathlib import Path
import os.path as osp
import glob
import os

import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat
from sapien.render import RenderBodyComponent
from scipy.spatial.transform import Rotation as R

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array


# Load table texture files for domain randomization (relative to cwd)
TABLE_TEXTURE_DIR = Path(os.getcwd()) / "assets/curated_table_textures"
TABLE_TEXTURES = sorted(glob.glob(str(TABLE_TEXTURE_DIR / "*.png")))
print(f"### Loaded {len(TABLE_TEXTURES)} curated table textures")

# Load peg/object texture files for domain randomization
PEG_TEXTURE_DIR = Path(os.getcwd()) / "assets/object_textures"
PEG_TEXTURES = sorted(glob.glob(str(PEG_TEXTURE_DIR / "*.png")))
print(f"### Loaded {len(PEG_TEXTURES)} object textures for peg randomization")

# Load dome lighting textures
EXRS_DOME_LIGHTING_DIR = Path(os.getcwd()) / "assets/dome_light_textures"
EXRS_DOME_LIGHTINGS = sorted(glob.glob(str(EXRS_DOME_LIGHTING_DIR / "*.exr")))
print(f"### Loaded {len(EXRS_DOME_LIGHTINGS)} curated dome lighting textures")

# Load real camera calibration data
camera_data = Path(os.getcwd()) / "assets/calibration_data.npz"
camera_data = np.load(camera_data)
REAL_POSE = np.eye(4)
REAL_POSE[:3, 3] = camera_data["translations"]
REAL_POSE[:3, :3] = (R.from_matrix(camera_data["rotations"]) * R.from_euler('zyx', [90, 0, 90], degrees=True)).as_matrix()
INTRINSICS_REAL = camera_data["K"]

INTRINSICS_REAL[0, -1] -= (1280 - 720) / 2
INTRINSICS_REAL[:-1] *= 224.0 / 720.0  # adjustment factor after centercropping


@register_env("LiftPegUprightReal-v1", max_episode_steps=75)
class LiftPegUprightRealEnv(BaseEnv):
    """
    **Task Description:**
    A simple task where the objective is to move a peg laying on the table to any upright position on the table.
    This variant includes domain randomization for sim-to-real transfer.

    **Randomizations:**
    - the peg's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]
      relative to the spawn center. It is placed flat along its length on the table
    - the peg dimensions are randomized +/-10% around nominal values per environment
    - the peg's visual texture is randomized from a set of object textures (assets/object_textures)
    - the table texture is randomized per environment from curated table textures (same as PickCube)
    - the camera pose has small perturbations (+/-1cm position, +/-3deg rotation) around a
      calibrated real-world pose (same as PickCube)
    - EXR dome lighting is randomized per environment with random ambient light levels

    **Peg Dimensions (nominal, +/-10%):**
    - Face: 40mm x 50mm rectangular cross-section
    - Length: 150mm

    **Success Conditions:**
    - the absolute value of the peg's y euler angle is within 0.08 of pi/2 and the z position
      of the peg is within 0.005 of its half-length.
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/LiftPegUpright-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Union[Panda, Fetch]

    # Nominal peg dimensions: 40mm x 50mm face, 150mm length (+/-10% randomized per env)
    peg_half_length_nom = 0.075    # 150mm / 2, along x-axis (length direction)
    peg_half_face_h_nom = 0.020    # 40mm / 2, along y-axis
    peg_half_face_w_nom = 0.025    # 50mm / 2, along z-axis
    peg_dim_variation = 0.10       # +/-10% variation

    def __init__(
        self,
        *args,
        robot_uids="panda",
        num_envs=1,
        reconfiguration_freq=None,
        robot_init_qpos_noise=0.02,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

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

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=[0.4, 1.6, 0.8],
            target=[-0.2, 0.6, 0.1]
        )
        return CameraConfig("render_camera", pose, 512, 512, fov=1, near=0.01, far=100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        robot_initial_pose = sapien.Pose(p=[-0.615, 0, 0])
        cam_initial_pose = robot_initial_pose * sapien.Pose(REAL_POSE)
        cam_mount_builder = self.scene.create_actor_builder()
        cam_mount_builder.initial_pose = cam_initial_pose
        self.cam_mount = cam_mount_builder.build_kinematic("camera_mount")

        model_dir = Path(osp.dirname(__file__)).parent.parent.parent / "utils" / "scene_builder" / "table" / "assets"
        table_model_file = str(model_dir / "table.glb")
        scale = 1.75
        base_table_length = 58 * 0.0254   # 1.4732m
        base_table_width = 40 * 0.0254    # 0.7112m
        base_table_height = 0.9196429
        table_length = base_table_length * scale
        table_width = base_table_width * scale
        table_height = base_table_height * scale
        table_pose = sapien.Pose(q=euler2quat(0, 0, np.pi / 2))
        table_initial_pose = sapien.Pose(p=[-0.12, 0, -table_height], q=euler2quat(0, 0, np.pi / 2))


        tables = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(
                pose=sapien.Pose(p=[0, 0, 0.9196429 / 2]),
                half_size=(table_length / 2, table_width / 2, 0.9196429 / 2),
            )
            builder.add_visual_from_file(
                filename=table_model_file, scale=[scale] * 3, pose=table_pose
            )
            builder.initial_pose = table_initial_pose
            builder.set_scene_idxs([i])
            table = builder.build_kinematic(name=f"table-workspace_{i}")
            self.remove_from_state_dict_registry(table)
            tables.append(table)

        self.table = Actor.merge(tables, name="table-workspace")
        self.add_to_state_dict_registry(self.table)

        # Apply random textures to each table
        if len(TABLE_TEXTURES) > 0:
            for i, obj in enumerate(self.table._objs):
                texture_idx = self._batched_episode_rng[i].randint(0, len(TABLE_TEXTURES))
                texture_path = TABLE_TEXTURES[texture_idx]
                texture = sapien.render.RenderTexture2D(
                    filename=texture_path,
                    mipmap_levels=1,
                )
                render_body_component: RenderBodyComponent = obj.find_component_by_type(RenderBodyComponent)
                if render_body_component is not None:
                    for render_shape in render_body_component.render_shapes:
                        for part in render_shape.parts:
                            part.material.set_base_color([1, 1, 1, 1])
                            part.material.set_base_color_texture(texture)
                            part.material.set_metallic(0.0)
                            part.material.set_roughness(0.8)
                            part.material.set_specular(0.0)
                            part.material.set_normal_texture(None)
                            part.material.set_metallic_texture(None)
                            part.material.set_roughness_texture(None)
                            part.material.set_emission_texture(None)

        # Compute table dimensions from first table
        aabb = (
            tables[0]._objs[0]
            .find_component_by_type(sapien.render.RenderBodyComponent)
            .compute_global_aabb_tight()
        )
        self.table_length = aabb[1, 0] - aabb[0, 0]
        self.table_width = aabb[1, 1] - aabb[0, 1]
        self.table_height = aabb[1, 2] - aabb[0, 2]

        # Build ground (shared across all environments) - collision only, no visual
        floor_width = 100
        if self.scene.parallel_in_single_scene:
            floor_width = 500
        self.ground = build_ground(
            self.scene, floor_width=floor_width, altitude=-self.table_height
        )
        # Hide the ground visual (grid texture) - keep collision for physics
        for obj in self.ground._objs:
            render_comp = obj.find_component_by_type(RenderBodyComponent)
            if render_comp is not None:
                obj.remove_component(render_comp)

        # Store table scene reference for compatibility and use TableSceneBuilder for robot init
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.table = self.table
        self.table_scene.ground = self.ground
        self.table_scene.table_height = self.table_height
        self.table_scene.table_length = self.table_length
        self.table_scene.table_width = self.table_width
        self.table_scene.scene_objects = [self.table, self.ground]

        # per-env pegs with random textures and +/-10% size randomization
        v = self.peg_dim_variation
        peg_half_lengths = self._batched_episode_rng.uniform(
            self.peg_half_length_nom * (1 - v), self.peg_half_length_nom * (1 + v)
        )
        peg_half_face_hs = self._batched_episode_rng.uniform(
            self.peg_half_face_h_nom * (1 - v), self.peg_half_face_h_nom * (1 + v)
        )
        peg_half_face_ws = self._batched_episode_rng.uniform(
            self.peg_half_face_w_nom * (1 - v), self.peg_half_face_w_nom * (1 + v)
        )
        # Store per-env dimensions as tensors for use in episode init, eval, and reward
        from mani_skill.utils import common
        self.peg_half_sizes = common.to_tensor(
            np.vstack([peg_half_lengths, peg_half_face_hs, peg_half_face_ws])
        ).T.to(self.device)  # (num_envs, 3): [half_length, half_face_h, half_face_w]

        pegs = []
        for i in range(self.num_envs):
            hl = peg_half_lengths[i]
            hh = peg_half_face_hs[i]
            hw = peg_half_face_ws[i]

            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=[hl, hh, hw])

            # Create material: glossy printed cardboard look (like a toothpaste box)
            peg_material = sapien.render.RenderMaterial(
                base_color=[1, 1, 1, 1],  # White base so texture shows properly
                metallic=0.0,
                roughness=0.15,
                specular=0.5,
            )
            if len(PEG_TEXTURES) > 0:
                texture_idx = self._batched_episode_rng[i].randint(0, len(PEG_TEXTURES))
                peg_texture = sapien.render.RenderTexture2D(
                    filename=PEG_TEXTURES[texture_idx],
                    mipmap_levels=1,
                )
                peg_material.set_base_color_texture(peg_texture)

            builder.add_box_visual(half_size=[hl, hh, hw], material=peg_material)

            builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
            builder.set_scene_idxs([i])
            peg = builder.build(f"peg_{i}")
            self.remove_from_state_dict_registry(peg)
            pegs.append(peg)

        self.peg = Actor.merge(pegs, "peg")
        self.add_to_state_dict_registry(self.peg)

    def _load_lighting(self, options: Dict):
        """EXR dome lighting with randomized ambient levels."""
        if len(EXRS_DOME_LIGHTINGS) > 0:
            for i in range(self.num_envs):
                self.scene.sub_scenes[i].set_environment_map(
                    EXRS_DOME_LIGHTINGS[self._batched_episode_rng[i].randint(0, len(EXRS_DOME_LIGHTINGS))]
                )
                self.scene.sub_scenes[i].render_system.ambient_light = (
                    np.array([1, 1, 1]) * self._batched_episode_rng[i].uniform(0.05, 0.2)
                )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            link0_pose = self.agent.robot.links[0].pose[env_idx]

            # Apply nominal camera pose (REAL_POSE) relative to link_0
            nominal_pose = Pose.create(sapien.Pose(REAL_POSE))
            cam_pose = link0_pose * nominal_pose

            # Apply random perturbation: position +/-1cm, rotation up to +/-3 degrees
            max_angle = np.deg2rad(3)
            axes = np.random.randn(b, 3)
            axes /= np.linalg.norm(axes, axis=1, keepdims=True)
            angles = np.random.uniform(-max_angle, max_angle, size=(b, 1))
            rotvecs = axes * angles
            delta_quats = R.from_rotvec(rotvecs).as_quat()
            delta_quats = np.roll(delta_quats, 1, axis=1)
            delta_positions = np.random.uniform(-0.01, 0.01, size=(b, 3))
            perturbation = Pose.create_from_pq(
                p=torch.from_numpy(delta_positions).float().to(self.device),
                q=torch.from_numpy(delta_quats).float().to(self.device),
            )
            cam_pose = cam_pose * perturbation
            self.cam_mount.set_pose(cam_pose)

            # Randomize wrist camera if using panda_wristcam
            if self.robot_uids == "panda_wristcam":
                self.agent.randomize_wrist_camera_pose(env=self, env_idx=env_idx)

            # Peg spawn: same workspace region as PickCube (visible to calibrated camera)
            robot_base_x = -0.615 + 7 * 0.0254    # -0.4372m
            robot_base_y = 1.200032 - 14 * 0.0254  # 0.8444m
            peg_spawn_center_x = robot_base_x + 23 * 0.0254  # in front of robot
            peg_spawn_center_y = robot_base_y
            peg_spawn_half_size = 0.1  # +/-10cm range

            xyz = torch.zeros((b, 3))
            xyz[:, 0] = peg_spawn_center_x + (torch.rand((b,)) * 2 - 1) * peg_spawn_half_size
            xyz[:, 1] = peg_spawn_center_y + (torch.rand((b,)) * 2 - 1) * peg_spawn_half_size
            # Resting height: half_face_h (local y) maps to world z after pi/2 x-rotation
            xyz[:, 2] = self.peg_half_sizes[env_idx, 1]

            q = euler2quat(np.pi / 2, 0, 0)  # Rotate to lie flat on table
            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.peg.set_pose(obj_pose)

    def evaluate(self):
        q = self.peg.pose.q
        qmat = rotation_conversions.quaternion_to_matrix(q)
        euler = rotation_conversions.matrix_to_euler_angles(qmat, "XYZ")
        is_peg_upright = (
            torch.abs(torch.abs(euler[:, 2]) - np.pi / 2) < 0.08
        )  # 0.08 radians of difference permitted
        close_to_table = torch.abs(self.peg.pose.p[:, 2] - self.peg_half_sizes[:, 0]) < 0.005
        return {
            "success": is_peg_upright & close_to_table,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                obj_pose=self.peg.pose.raw_pose,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        # rotation reward as cosine similarity between peg direction vectors
        # peg center of mass to end of peg, (1,0,0), rotated by peg pose rotation
        # dot product with its goal orientation: (0,0,1) or (0,0,-1)
        qmats = rotation_conversions.quaternion_to_matrix(self.peg.pose.q)
        vec = torch.tensor([1.0, 0, 0], device=self.device)
        goal_vec = torch.tensor([0, 0, 1.0], device=self.device)
        rot_vec = (qmats @ vec).view(-1, 3)
        # abs since (0,0,-1) is also valid, values in [0,1]
        rot_rew = (rot_vec @ goal_vec).view(-1).abs()
        reward = rot_rew

        # position reward using common maniskill distance reward pattern
        # giving reward in [0,1] for moving center of mass toward half length above table
        z_dist = torch.abs(self.peg.pose.p[:, 2] - self.peg_half_sizes[:, 0])
        reward += 1 - torch.tanh(5 * z_dist)

        # small reward to motivate initial reaching
        # initially, we want to reach and grip peg
        to_grip_vec = self.peg.pose.p - self.agent.tcp.pose.p
        to_grip_dist = torch.linalg.norm(to_grip_vec, axis=1)
        reaching_rew = 1 - torch.tanh(5 * to_grip_dist)
        # reaching reward granted if gripping block
        reaching_rew[self.agent.is_grasping(self.peg)] = 1
        # weight reaching reward less
        reaching_rew = reaching_rew / 5
        reward += reaching_rew

        reward[info["success"]] = 3
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        max_reward = 3.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
